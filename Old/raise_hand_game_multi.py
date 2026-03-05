import time
import random
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import mediapipe as mp


@dataclass
class PlayerState:
    idx: int
    score: int = 0
    answered_this_round: bool = False
    hold_start_left: Optional[float] = None
    hold_start_right: Optional[float] = None


@dataclass
class GameConfig:
    round_seconds: float = 5.0
    hold_seconds: float = 0.20
    cooldown_seconds: float = 0.0  # per-player cooldown not needed if "1 answer per round"
    min_raise_confidence: float = 0.55
    wrist_y_threshold_ratio: float = 0.55  # wrist must be above this line


class RaiseHandGameMulti:
    def __init__(self, cam: int, width: int, height: int, players: int, mirror: bool, cfg: GameConfig):
        self.players = players
        self.mirror = mirror
        self.cfg = cfg

        self.cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=min(2, max(2, players)),  # still realistically 2
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.states: List[PlayerState] = [PlayerState(i + 1) for i in range(players)]

        self.round_idx = 0
        self.target = "LEFT"
        self.deadline = 0.0

        self.last_feedback = ""
        self.feedback_until = 0.0

        self._new_round()

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _set_feedback(self, text: str, seconds: float = 0.9):
        self.last_feedback = text
        self.feedback_until = time.time() + seconds

    def _new_round(self):
        self.round_idx += 1
        self.target = random.choice(["LEFT", "RIGHT"])
        self.deadline = time.time() + self.cfg.round_seconds
        for st in self.states:
            st.answered_this_round = False
            st.hold_start_left = None
            st.hold_start_right = None

    def _process_choice(self, player_i: int, choice: str):
        st = self.states[player_i]
        if st.answered_this_round:
            return

        st.answered_this_round = True
        if choice == self.target:
            st.score += 1
            self._set_feedback(f"J{st.idx}: ✅ CORRECTO (+1)", 0.8)
        else:
            self._set_feedback(f"J{st.idx}: ❌ INCORRECTO", 0.8)

    def _lane_index(self, x: int, w: int) -> int:
        lane_w = w / self.players
        i = int(x // lane_w)
        return max(0, min(self.players - 1, i))

    def reset_scores(self):
        for st in self.states:
            st.score = 0
        self.round_idx = 0
        self._set_feedback("Scores reseteados", 0.8)
        self._new_round()

    def run(self):
        if not self.cap.isOpened():
            raise RuntimeError("No pude abrir la webcam. Prueba --cam 0/1 y revisa permisos.")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            if self.mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            now = time.time()

            # HUD top
            cv2.rectangle(frame, (0, 0), (w, 95), (0, 0, 0), -1)
            remaining = max(0.0, self.deadline - now)
            cv2.putText(frame, f"Ronda: {self.round_idx}   Tiempo: {remaining:0.1f}s   (ESC salir | R reset)",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

            target_txt = "LEVANTA MANO IZQUIERDA" if self.target == "LEFT" else "LEVANTA MANO DERECHA"
            cv2.putText(frame, f"OBJETIVO GLOBAL: {target_txt}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 3)

            # timeout -> new round
            if now >= self.deadline:
                self._set_feedback("⏱️ TIEMPO AGOTADO (nueva ronda)", 0.7)
                self._new_round()

            # lanes + player status
            lane_w = w / self.players
            for i in range(1, self.players):
                x = int(i * lane_w)
                cv2.line(frame, (x, 95), (x, h), (40, 40, 40), 2)

            y_thr = int(h * self.cfg.wrist_y_threshold_ratio)
            cv2.line(frame, (0, y_thr), (w, y_thr), (80, 80, 80), 2)
            cv2.putText(frame, "Mano arriba de esta linea = cuenta",
                        (20, y_thr - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

            for i, st in enumerate(self.states):
                x1 = int(i * lane_w) + 15
                status = "OK" if st.answered_this_round else "PEND"
                cv2.putText(frame, f"J{st.idx} Pts:{st.score} [{status}]",
                            (x1, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            # detect hands
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            # current raised flags per player
            raised_left = [False] * self.players
            raised_right = [False] * self.players

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label  # "Left" / "Right"
                    conf = hd.classification[0].score

                    wx = int(lm.landmark[0].x * w)
                    wy = int(lm.landmark[0].y * h)

                    logical = label

                    # assign this hand to a lane by wrist x
                    p_i = self._lane_index(wx, w)

                    # draw
                    cv2.circle(frame, (wx, wy), 8, (255, 255, 255), -1)
                    cv2.putText(frame, f"{logical} J{p_i+1} ({conf:.2f})",
                                (wx + 10, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                    is_raised = (conf >= self.cfg.min_raise_confidence) and (wy < y_thr)

                    if is_raised:
                        if logical == "Left":
                            raised_left[p_i] = True
                        else:
                            raised_right[p_i] = True

            # hold + decision per player
            for i, st in enumerate(self.states):
                if st.answered_this_round:
                    st.hold_start_left = None
                    st.hold_start_right = None
                    continue

                # LEFT
                if raised_left[i]:
                    if st.hold_start_left is None:
                        st.hold_start_left = now
                    elif (now - st.hold_start_left) >= self.cfg.hold_seconds:
                        self._process_choice(i, "LEFT")
                        st.hold_start_left = None
                        st.hold_start_right = None
                else:
                    st.hold_start_left = None

                # RIGHT
                if raised_right[i]:
                    if st.hold_start_right is None:
                        st.hold_start_right = now
                    elif (now - st.hold_start_right) >= self.cfg.hold_seconds:
                        self._process_choice(i, "RIGHT")
                        st.hold_start_left = None
                        st.hold_start_right = None
                else:
                    st.hold_start_right = None

            # feedback
            if now < self.feedback_until and self.last_feedback:
                cv2.putText(frame, self.last_feedback, (20, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # practical warning
            if self.players > 2:
                cv2.putText(frame, "AVISO: deteccion simultanea confiable suele ser 1-2 manos (limitacion del modelo).",
                            (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("Raise Hand Game (Multi)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                self.reset_scores()

        self.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--mirror", type=int, default=1, help="1=modo espejo (webcam), 0=sin espejo")
    ap.add_argument("--line", type=float, default=0.55, help="umbral de altura de muñeca (0-1), menor = mas alto requerido")
    ap.add_argument("--hold", type=float, default=0.20, help="segundos que debe mantenerse la mano arriba")
    args = ap.parse_args()

    if args.players < 1:
        raise ValueError("--players debe ser >= 1")

    cfg = GameConfig(
        round_seconds=args.seconds,
        wrist_y_threshold_ratio=args.line,
        hold_seconds=args.hold
    )
    game = RaiseHandGameMulti(args.cam, args.width, args.height, args.players, mirror=bool(args.mirror), cfg=cfg)
    game.run()


if __name__ == "__main__":
    main()