import time
import random
import argparse
from dataclasses import dataclass
from typing import Optional, List

import cv2
import mediapipe as mp


@dataclass
class GameConfig:
    round_seconds: float = 5.0
    cooldown_seconds: float = 0.6
    min_raise_confidence: float = 0.55

    # Raise detection thresholds:
    # "Raised" if wrist is above this fraction of frame height
    # (smaller y = higher on screen)
    wrist_y_threshold_ratio: float = 0.55

    # Require raised state to be held this many seconds to count
    hold_seconds: float = 0.20


class RaiseHandGame:
    def __init__(self, cam: int, width: int, height: int, cfg: GameConfig, mirror: bool):
        self.cfg = cfg
        self.mirror = mirror

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

        self.score = 0
        self.round_idx = 0
        self.target: str = "LEFT"
        self.deadline: float = 0.0

        self.cooldown_until: float = 0.0
        self.hold_start_left: Optional[float] = None
        self.hold_start_right: Optional[float] = None

        self.last_feedback: str = ""
        self.feedback_until: float = 0.0

        self._new_round()

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _new_round(self):
        self.round_idx += 1
        self.target = random.choice(["LEFT", "RIGHT"])
        self.deadline = time.time() + self.cfg.round_seconds
        self.hold_start_left = None
        self.hold_start_right = None

    def _set_feedback(self, text: str, seconds: float = 0.9):
        self.last_feedback = text
        self.feedback_until = time.time() + seconds

    def _process_choice(self, choice: str):
        if choice == self.target:
            self.score += 1
            self._set_feedback("✅ CORRECTO (+1)")
        else:
            self._set_feedback("❌ INCORRECTO")

        self.cooldown_until = time.time() + self.cfg.cooldown_seconds
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

            # HUD
            cv2.rectangle(frame, (0, 0), (w, 95), (0, 0, 0), -1)
            remaining = max(0.0, self.deadline - now)
            cv2.putText(frame, f"Ronda: {self.round_idx}   Score: {self.score}   Tiempo: {remaining:0.1f}s",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            target_txt = "LEVANTA MANO IZQUIERDA" if self.target == "LEFT" else "LEVANTA MANO DERECHA"
            cv2.putText(frame, f"OBJETIVO: {target_txt}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 3)

            # time out
            if now >= self.deadline:
                self._set_feedback("⏱️ TIEMPO AGOTADO", 0.7)
                self._new_round()

            # draw threshold line
            y_thr = int(h * self.cfg.wrist_y_threshold_ratio)
            cv2.line(frame, (0, y_thr), (w, y_thr), (80, 80, 80), 2)
            cv2.putText(frame, "Levanta la mano por encima de esta linea",
                        (20, y_thr - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

            # detect hands
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            left_raised = False
            right_raised = False

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label  # "Left" or "Right"
                    conf = hd.classification[0].score

                    # wrist landmark = 0
                    wx = int(lm.landmark[0].x * w)
                    wy = int(lm.landmark[0].y * h)

                    # If mirrored, swap logical left/right
                    logical = label
                    if self.mirror:
                        logical = "Left" if label == "Right" else "Right"

                    # Draw wrist point + label
                    cv2.circle(frame, (wx, wy), 8, (255, 255, 255), -1)
                    cv2.putText(frame, f"{logical} ({conf:.2f})", (wx + 10, wy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                    is_raised = (conf >= self.cfg.min_raise_confidence) and (wy < y_thr)

                    if logical == "Left" and is_raised:
                        left_raised = True
                    if logical == "Right" and is_raised:
                        right_raised = True

            # decision logic (hold + cooldown)
            if now >= self.cooldown_until:
                # LEFT
                if left_raised:
                    if self.hold_start_left is None:
                        self.hold_start_left = now
                    elif (now - self.hold_start_left) >= self.cfg.hold_seconds:
                        self._process_choice("LEFT")
                        self.hold_start_left = None
                        self.hold_start_right = None
                else:
                    self.hold_start_left = None

                # RIGHT
                if right_raised:
                    if self.hold_start_right is None:
                        self.hold_start_right = now
                    elif (now - self.hold_start_right) >= self.cfg.hold_seconds:
                        self._process_choice("RIGHT")
                        self.hold_start_left = None
                        self.hold_start_right = None
                else:
                    self.hold_start_right = None
            else:
                self.hold_start_left = None
                self.hold_start_right = None

            # Show raised state
            cv2.putText(frame, f"Estado: L={'UP' if left_raised else '--'}  R={'UP' if right_raised else '--'}",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)

            # feedback
            if now < self.feedback_until and self.last_feedback:
                cv2.putText(frame, self.last_feedback, (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            cv2.putText(frame, "ESC salir | R reset", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

            cv2.imshow("Raise Hand Game", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
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
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--mirror", type=int, default=1, help="1=modo espejo (webcam), 0=sin espejo")
    ap.add_argument("--line", type=float, default=0.55, help="umbral de altura de muñeca (0-1), menor = mas alto requerido")
    ap.add_argument("--hold", type=float, default=0.20, help="segundos que debe mantenerse la mano arriba")
    args = ap.parse_args()

    cfg = GameConfig(
        round_seconds=args.seconds,
        wrist_y_threshold_ratio=args.line,
        hold_seconds=args.hold,
    )
    game = RaiseHandGame(args.cam, args.width, args.height, cfg, mirror=bool(args.mirror))
    game.run()


if __name__ == "__main__":
    main()