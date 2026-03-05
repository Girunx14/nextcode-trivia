import time
import random
import argparse
from dataclasses import dataclass
from typing import Optional, List

import cv2
import mediapipe as mp


@dataclass
class PlayerState:
    idx: int
    score: int = 0
    answered_this_round: bool = False
    hold_start_left: Optional[float] = None
    hold_start_right: Optional[float] = None

    # UI: last detected hand label in this lane
    detected_hand: str = "—"   # "IZQ", "DER", "—"


@dataclass
class GameConfig:
    round_seconds: float = 5.0
    hold_seconds: float = 0.20
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
            max_num_hands=2,  # en práctica: 1-2 manos simultáneas confiable
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.states: List[PlayerState] = [PlayerState(i + 1) for i in range(players)]

        # Round state
        self.round_idx = 0
        self.target = "LEFT"          # "LEFT" or "RIGHT"
        self.deadline = 0.0
        self.wait_next = False        # when True, waits keypress to advance

        # feedback text pinned until next
        self.round_result_text = "Presiona N / ESPACIO para iniciar"

        self._new_round(init=True)

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _new_round(self, init: bool = False):
        self.round_idx += 0 if init else 1
        self.target = random.choice(["LEFT", "RIGHT"])
        self.deadline = time.time() + self.cfg.round_seconds
        self.wait_next = False

        # reset per-round player flags
        for st in self.states:
            st.answered_this_round = False
            st.hold_start_left = None
            st.hold_start_right = None
            st.detected_hand = "—"

        if init:
            self.round_result_text = "Listo. Levanta mano IZQ o DER para responder."
        else:
            self.round_result_text = "Nueva ronda: responde antes de que acabe el tiempo."

    def _lane_index(self, x: int, w: int) -> int:
        lane_w = w / self.players
        i = int(x // lane_w)
        return max(0, min(self.players - 1, i))

    def _process_choice(self, player_i: int, choice: str):
        st = self.states[player_i]
        if st.answered_this_round:
            return

        st.answered_this_round = True
        if choice == self.target:
            st.score += 1
            self.round_result_text = f"J{st.idx}: ✅ CORRECTO (+1). Presiona N/ESPACIO para la siguiente."
        else:
            self.round_result_text = f"J{st.idx}: ❌ INCORRECTO. Presiona N/ESPACIO para la siguiente."

        # Pausa la ronda hasta que presiones tecla
        self.wait_next = True

    def reset_scores(self):
        for st in self.states:
            st.score = 0
        self.round_idx = 0
        self.round_result_text = "Scores reseteados. Presiona N/ESPACIO para iniciar."
        self.wait_next = True  # obligar a arrancar manualmente

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

            # --- TOP HUD ---
            cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)

            # Timer only runs if not waiting next
            remaining = max(0.0, self.deadline - now) if not self.wait_next else 0.0

            cv2.putText(
                frame,
                f"Ronda: {self.round_idx}  |  Tiempo: {remaining:0.1f}s  |  (ESC salir | R reset | N/Espacio next)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            target_txt = "MANO IZQUIERDA" if self.target == "LEFT" else "MANO DERECHA"
            status_txt = "PAUSA (NEXT manual)" if self.wait_next else "EN JUEGO"
            cv2.putText(
                frame,
                f"OBJETIVO: {target_txt}   |   Estado: {status_txt}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                self.round_result_text,
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # --- Lane separators + per-player HUD (detected hand) ---
            lane_w = w / self.players
            for i in range(1, self.players):
                x = int(i * lane_w)
                cv2.line(frame, (x, 130), (x, h), (40, 40, 40), 2)

            y_thr = int(h * self.cfg.wrist_y_threshold_ratio)
            cv2.line(frame, (0, y_thr), (w, y_thr), (80, 80, 80), 2)
            cv2.putText(
                frame,
                "Mano arriba de esta linea = cuenta",
                (20, y_thr - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                2,
            )

            for i, st in enumerate(self.states):
                x1 = int(i * lane_w) + 15
                status = "OK" if st.answered_this_round else "PEND"
                cv2.putText(
                    frame,
                    f"J{st.idx}  Pts:{st.score}  [{status}]  Detectado:{st.detected_hand}",
                    (x1, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (200, 200, 200),
                    2,
                )

            # --- TIMEOUT handling (only if in game) ---
            if (not self.wait_next) and (now >= self.deadline):
                self.round_result_text = "⏱️ TIEMPO AGOTADO. Presiona N/ESPACIO para la siguiente."
                self.wait_next = True

            # --- Detect hands ONLY if in game ---
            raised_left = [False] * self.players
            raised_right = [False] * self.players

            if not self.wait_next:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)

                # Reset "detected" each frame (we’ll fill if any hand seen in lane)
                for st in self.states:
                    st.detected_hand = "—"

                if res.multi_hand_landmarks and res.multi_handedness:
                    for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                        label = hd.classification[0].label   # "Left" / "Right" (raw from model)
                        conf = hd.classification[0].score

                        # wrist landmark = 0
                        wx = int(lm.landmark[0].x * w)
                        wy = int(lm.landmark[0].y * h)

                        # Assign hand to player lane by wrist x
                        p_i = self._lane_index(wx, w)

                        # IMPORTANT:
                        # Para que “derecha/izquierda” coincida con lo que tú sientes,
                        # lo más consistente es NO hacer swap aquí, y usar --mirror solo para la vista.
                        logical = label  # keep as-is
                        # Si algún día quieres swap, lo hacemos con un flag separado.

                        # UI label in Spanish
                        hand_es = "IZQ" if logical == "Left" else "DER"
                        self.states[p_i].detected_hand = hand_es

                        # draw wrist point + label
                        cv2.circle(frame, (wx, wy), 8, (255, 255, 255), -1)
                        cv2.putText(
                            frame,
                            f"{hand_es} ({conf:.2f}) J{p_i+1}",
                            (wx + 10, wy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                        )

                        is_raised = (conf >= self.cfg.min_raise_confidence) and (wy < y_thr)

                        if is_raised:
                            if logical == "Left":
                                raised_left[p_i] = True
                            else:
                                raised_right[p_i] = True

                # Hold + decision per player
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

            cv2.imshow("Raise Hand Game (Multi + Next Manual)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key in (ord("r"), ord("R")):
                self.reset_scores()
            if key in (ord("n"), ord("N"), 32):  # N or SPACE
                if self.wait_next:
                    # advance to next round
                    self.round_idx += 1
                    self.target = random.choice(["LEFT", "RIGHT"])
                    self.deadline = time.time() + self.cfg.round_seconds
                    self.wait_next = False
                    self.round_result_text = "Nueva ronda: responde antes de que acabe el tiempo."
                    for st in self.states:
                        st.answered_this_round = False
                        st.hold_start_left = None
                        st.hold_start_right = None
                        st.detected_hand = "—"

        self.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--mirror", type=int, default=1, help="1=espejo visual, 0=sin espejo")
    ap.add_argument("--line", type=float, default=0.55, help="umbral muñeca (0-1). Menor = mano más arriba")
    ap.add_argument("--hold", type=float, default=0.20, help="segundos sosteniendo mano arriba para contar")
    args = ap.parse_args()

    if args.players < 1:
        raise ValueError("--players debe ser >= 1")

    cfg = GameConfig(
        round_seconds=args.seconds,
        wrist_y_threshold_ratio=args.line,
        hold_seconds=args.hold,
    )

    game = RaiseHandGameMulti(
        cam=args.cam,
        width=args.width,
        height=args.height,
        players=args.players,
        mirror=bool(args.mirror),
        cfg=cfg,
    )
    game.run()


if __name__ == "__main__":
    main()