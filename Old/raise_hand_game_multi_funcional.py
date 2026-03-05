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
    detected_hand: str = "—"   # "IZQ", "DER", "—"


@dataclass
class GameConfig:
    round_seconds: float = 5.0
    hold_seconds: float = 0.20
    min_raise_confidence: float = 0.55
    wrist_y_threshold_ratio: float = 0.55  # "raised" if wrist is above this line

    # NEXT button
    next_hold_seconds: float = 0.25


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
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.states: List[PlayerState] = [PlayerState(i + 1) for i in range(players)]

        self.round_idx = 0
        self.target = "LEFT"
        self.deadline = 0.0
        self.wait_next = True  # start paused

        self.round_result_text = "Presiona el botón SIGUIENTE para iniciar."

        self.next_hover_start: Optional[float] = None
        self.next_rect: Optional[Tuple[int, int, int, int]] = None  # x1,y1,x2,y2

        self._start_new_round(reset_only=True)

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _lane_index(self, x: int, w: int) -> int:
        lane_w = w / self.players
        i = int(x // lane_w)
        return max(0, min(self.players - 1, i))

    def _set_next_rect(self, w: int, h: int):
        bw = int(w * 0.34)
        bh = 85
        pad = 18
        x1 = (w - bw) // 2
        x2 = x1 + bw
        y2 = h - pad
        y1 = y2 - bh
        self.next_rect = (x1, y1, x2, y2)

    def _rect_contains(self, rect, x: int, y: int) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _draw_next_button(self, frame, enabled: bool, hover: bool):
        if self.next_rect is None:
            return
        x1, y1, x2, y2 = self.next_rect

        cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)

        if not enabled:
            border = (120, 120, 120)
            thick = 2
        else:
            border = (0, 255, 255)
            thick = 4

        if hover and enabled:
            border = (0, 255, 0)
            thick = 6

        cv2.rectangle(frame, (x1, y1), (x2, y2), border, thick)

        label = "SIGUIENTE"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.15, 3)
        cx = (x1 + x2) // 2 - tw // 2
        cy = (y1 + y2) // 2 + th // 2
        cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3)

        if enabled:
            cv2.putText(frame, "Baja las manos y toca el boton con el dedo indice",
                        (20, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    def _start_new_round(self, reset_only: bool = False):
        if not reset_only:
            self.round_idx += 1

        self.target = random.choice(["LEFT", "RIGHT"])
        self.deadline = time.time() + self.cfg.round_seconds
        self.wait_next = False
        self.next_hover_start = None

        for st in self.states:
            st.answered_this_round = False
            st.hold_start_left = None
            st.hold_start_right = None
            st.detected_hand = "—"

        self.round_result_text = "Nueva ronda: responde antes de que acabe el tiempo."

    def reset_scores(self):
        for st in self.states:
            st.score = 0
        self.round_idx = 0
        self.round_result_text = "Scores reseteados. Presiona SIGUIENTE para iniciar."
        self.wait_next = True
        self.next_hover_start = None

        for st in self.states:
            st.answered_this_round = False
            st.hold_start_left = None
            st.hold_start_right = None
            st.detected_hand = "—"

    def _process_choice(self, player_i: int, choice: str):
        st = self.states[player_i]
        if st.answered_this_round:
            return

        st.answered_this_round = True
        if choice == self.target:
            st.score += 1
            self.round_result_text = f"J{st.idx}: ✅ CORRECTO (+1). Baja manos y presiona SIGUIENTE."
        else:
            self.round_result_text = f"J{st.idx}: ❌ INCORRECTO. Baja manos y presiona SIGUIENTE."

        self.wait_next = True
        self.next_hover_start = None

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
            if self.next_rect is None:
                self._set_next_rect(w, h)

            now = time.time()
            y_thr = int(h * self.cfg.wrist_y_threshold_ratio)

            # --- TOP HUD ---
            cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)

            remaining = max(0.0, self.deadline - now) if not self.wait_next else 0.0
            cv2.putText(
                frame,
                f"Ronda: {self.round_idx}  |  Tiempo: {remaining:0.1f}s  |  (ESC salir | R reset)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            target_txt = "MANO IZQUIERDA" if self.target == "LEFT" else "MANO DERECHA"
            status_txt = "PAUSA (usa SIGUIENTE)" if self.wait_next else "EN JUEGO"
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

            # --- Lanes & line ---
            lane_w = w / self.players
            for i in range(1, self.players):
                x = int(i * lane_w)
                cv2.line(frame, (x, 130), (x, h), (40, 40, 40), 2)

            cv2.line(frame, (0, y_thr), (w, y_thr), (80, 80, 80), 2)
            cv2.putText(frame, "Mano arriba de esta linea = cuenta",
                        (20, y_thr - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

            for i, st in enumerate(self.states):
                x1 = int(i * lane_w) + 15
                status = "OK" if st.answered_this_round else "PEND"
                cv2.putText(frame, f"J{st.idx}  Pts:{st.score}  [{status}]  Detectado:{st.detected_hand}",
                            (x1, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

            # Timeout -> pause
            if (not self.wait_next) and (now >= self.deadline):
                self.round_result_text = "⏱️ TIEMPO AGOTADO. Baja las manos y presiona SIGUIENTE."
                self.wait_next = True
                self.next_hover_start = None

            # --- Hand detection ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            for st in self.states:
                st.detected_hand = "—"

            raised_left = [False] * self.players
            raised_right = [False] * self.players

            any_hand_raised = False
            any_index_in_next = False  # NEW: index tip inside NEXT rect (with hands down)

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label  # "Left" / "Right"
                    conf = hd.classification[0].score

                    # wrist (0) and index tip (8)
                    wx = int(lm.landmark[0].x * w)
                    wy = int(lm.landmark[0].y * h)
                    ix = int(lm.landmark[8].x * w)
                    iy = int(lm.landmark[8].y * h)

                    p_i = self._lane_index(wx, w)

                    logical = label  # no swap
                    hand_es = "IZQ" if logical == "Left" else "DER"
                    self.states[p_i].detected_hand = hand_es

                    # draw wrist + index tip
                    cv2.circle(frame, (wx, wy), 7, (255, 255, 255), -1)
                    cv2.circle(frame, (ix, iy), 9, (255, 255, 255), -1)
                    cv2.circle(frame, (ix, iy), 13, (0, 255, 0), 2)

                    cv2.putText(frame, f"{hand_es} ({conf:.2f}) J{p_i+1}",
                                (wx + 10, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                    is_raised = (conf >= self.cfg.min_raise_confidence) and (wy < y_thr)
                    if is_raised:
                        any_hand_raised = True
                        if logical == "Left":
                            raised_left[p_i] = True
                        else:
                            raised_right[p_i] = True

                    # NEXT press logic: only when paused, require "hands down" (index below line)
                    if self.wait_next and self.next_rect is not None:
                        if (conf >= self.cfg.min_raise_confidence) and (iy >= y_thr) and self._rect_contains(self.next_rect, ix, iy):
                            any_index_in_next = True

            # --- Main logic ---
            if not self.wait_next:
                # answer detection
                for i, st in enumerate(self.states):
                    if st.answered_this_round:
                        st.hold_start_left = None
                        st.hold_start_right = None
                        continue

                    if raised_left[i]:
                        if st.hold_start_left is None:
                            st.hold_start_left = now
                        elif (now - st.hold_start_left) >= self.cfg.hold_seconds:
                            self._process_choice(i, "LEFT")
                            st.hold_start_left = None
                            st.hold_start_right = None
                    else:
                        st.hold_start_left = None

                    if raised_right[i]:
                        if st.hold_start_right is None:
                            st.hold_start_right = now
                        elif (now - st.hold_start_right) >= self.cfg.hold_seconds:
                            self._process_choice(i, "RIGHT")
                            st.hold_start_left = None
                            st.hold_start_right = None
                    else:
                        st.hold_start_right = None

                # draw NEXT disabled
                self._draw_next_button(frame, enabled=False, hover=False)

            else:
                # Pause: enable NEXT only if NO hands are raised
                next_enabled = not any_hand_raised
                hover = False

                if next_enabled and any_index_in_next:
                    hover = True
                    if self.next_hover_start is None:
                        self.next_hover_start = now
                    elif (now - self.next_hover_start) >= self.cfg.next_hold_seconds:
                        self._start_new_round(reset_only=False)
                else:
                    self.next_hover_start = None

                self._draw_next_button(frame, enabled=next_enabled, hover=hover)

            cv2.imshow("Raise Hand Game (Multi + Next Button Index)", frame)

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
    ap.add_argument("--mirror", type=int, default=1, help="1=espejo visual, 0=sin espejo")
    ap.add_argument("--line", type=float, default=0.55, help="umbral muñeca (0-1). Menor = mano más arriba")
    ap.add_argument("--hold", type=float, default=0.20, help="segundos sosteniendo mano arriba para contar")
    ap.add_argument("--next-hold", type=float, default=0.25, help="segundos sosteniendo indice en SIGUIENTE")
    args = ap.parse_args()

    if args.players < 1:
        raise ValueError("--players debe ser >= 1")

    cfg = GameConfig(
        round_seconds=args.seconds,
        wrist_y_threshold_ratio=args.line,
        hold_seconds=args.hold,
        next_hold_seconds=args.next_hold
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