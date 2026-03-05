import time
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class Button:
    name: str
    rect: Tuple[int, int, int, int]  # x1,y1,x2,y2

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class GameConfig:
    round_seconds: float = 5.0
    hold_click_seconds: float = 0.25
    cooldown_seconds: float = 0.35
    target_font_scale: float = 1.2


class HandButtonGame:
    def __init__(self, cam_index: int, width: int, height: int, cfg: GameConfig):
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.cfg = cfg

        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.score = 0
        self.round_idx = 0

        self.target: Optional[str] = None
        self.round_deadline = 0.0

        self.hovering_button: Optional[str] = None
        self.hover_start: Optional[float] = None
        self.cooldown_until: float = 0.0

        self.last_feedback: str = ""
        self.feedback_until: float = 0.0

        self.btn_left: Optional[Button] = None
        self.btn_right: Optional[Button] = None

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _layout_buttons(self, w: int, h: int):
        pad = 30
        btn_w = (w - pad * 3) // 2
        btn_h = 120
        y2 = h - pad
        y1 = y2 - btn_h

        left = (pad, y1, pad + btn_w, y2)
        right = (pad * 2 + btn_w, y1, pad * 2 + btn_w * 2, y2)

        self.btn_left = Button("LEFT", left)
        self.btn_right = Button("RIGHT", right)

    def _new_round(self):
        self.round_idx += 1
        self.target = random.choice(["LEFT", "RIGHT"])
        self.round_deadline = time.time() + self.cfg.round_seconds
        self.hovering_button = None
        self.hover_start = None
        self.last_feedback = ""
        self.feedback_until = 0.0

    def _draw_button(self, frame, btn: Button, label: str, is_target: bool, is_hover: bool):
        x1, y1, x2, y2 = btn.rect

        # Base
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)

        # Border states
        if is_target:
            border = (0, 255, 255)   # target highlight
            thick = 4
        else:
            border = (120, 120, 120)
            thick = 2

        if is_hover:
            border = (0, 255, 0)     # hover highlight
            thick = 6

        cv2.rectangle(frame, (x1, y1), (x2, y2), border, thick)

        # Label centered
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cx = (x1 + x2) // 2 - tw // 2
        cy = (y1 + y2) // 2 + th // 2
        cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    def _set_feedback(self, text: str, seconds: float = 1.0):
        self.last_feedback = text
        self.feedback_until = time.time() + seconds

    def _process_choice(self, choice: str):
        if choice == self.target:
            self.score += 1
            self._set_feedback("✅ CORRECTO (+1)", 1.0)
        else:
            self._set_feedback("❌ INCORRECTO", 1.0)

        self.cooldown_until = time.time() + self.cfg.cooldown_seconds
        self._new_round()

    def run(self):
        if not self.cap.isOpened():
            raise RuntimeError("No pude abrir la webcam. Prueba --cam 0/1 y revisa permisos.")

        # initialize first frame layout + round
        self._new_round()

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if self.btn_left is None:
                self._layout_buttons(w, h)

            now = time.time()

            # HUD top bar
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
            remaining = max(0.0, self.round_deadline - now)
            cv2.putText(frame, f"Ronda: {self.round_idx}   Score: {self.score}   Tiempo: {remaining:0.1f}s",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Target instruction
            target_txt = "TOCA IZQ" if self.target == "LEFT" else "TOCA DER"
            cv2.putText(frame, f"OBJETIVO: {target_txt}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, self.cfg.target_font_scale, (0, 255, 255), 3)

            # If time runs out => next round
            if now >= self.round_deadline:
                self._set_feedback("⏱️ TIEMPO AGOTADO", 0.8)
                self._new_round()

            # Hand detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            index_xy = None

            if res.multi_hand_landmarks:
                # We use first detected hand
                lm = res.multi_hand_landmarks[0]
                # Index finger tip = 8
                ix = int(lm.landmark[8].x * w)
                iy = int(lm.landmark[8].y * h)
                index_xy = (ix, iy)

                # Draw cursor
                cv2.circle(frame, (ix, iy), 10, (255, 255, 255), -1)
                cv2.circle(frame, (ix, iy), 14, (0, 255, 0), 2)

            # Determine hover / click by hold in button
            hover_btn_name = None
            if index_xy and now >= self.cooldown_until:
                ix, iy = index_xy
                if self.btn_left.contains(ix, iy):
                    hover_btn_name = "LEFT"
                elif self.btn_right.contains(ix, iy):
                    hover_btn_name = "RIGHT"

                if hover_btn_name is not None:
                    if self.hovering_button != hover_btn_name:
                        self.hovering_button = hover_btn_name
                        self.hover_start = now
                    else:
                        # same button - check hold
                        if self.hover_start is not None and (now - self.hover_start) >= self.cfg.hold_click_seconds:
                            self._process_choice(hover_btn_name)
                            self.hovering_button = None
                            self.hover_start = None
                else:
                    self.hovering_button = None
                    self.hover_start = None
            else:
                self.hovering_button = None
                self.hover_start = None

            # Draw buttons
            self._draw_button(frame, self.btn_left, "IZQUIERDA", self.target == "LEFT", self.hovering_button == "LEFT")
            self._draw_button(frame, self.btn_right, "DERECHA", self.target == "RIGHT", self.hovering_button == "RIGHT")

            # Feedback
            if now < self.feedback_until and self.last_feedback:
                cv2.putText(frame, self.last_feedback, (w // 2 - 220, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # Help line
            cv2.putText(frame, "ESC salir | R reset score", (20, h - 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow("Hand Button Game", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord('r'), ord('R')):
                self.score = 0
                self.round_idx = 0
                self._set_feedback("Scores reseteados", 0.8)
                self._new_round()

        self.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=5.0, help="segundos por ronda")
    ap.add_argument("--hold", type=float, default=0.25, help="segundos para considerar 'click' por mantener dedo")
    args = ap.parse_args()

    cfg = GameConfig(round_seconds=args.seconds, hold_click_seconds=args.hold)
    game = HandButtonGame(args.cam, args.width, args.height, cfg)
    game.run()


if __name__ == "__main__":
    main()