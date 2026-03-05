import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# ----------------------------
# Config / Helpers
# ----------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class GestureConfig:
    # Threshold for normalized nose offset to decide LEFT/RIGHT
    # (nose_x - face_center_x) / face_width
    left_thresh: float = -0.18
    right_thresh: float = 0.18

    # How long the head must stay beyond threshold to count as a choice (seconds)
    hold_time: float = 0.35

    # Cooldown after a registered choice (seconds)
    cooldown: float = 0.70

    # Exponential moving average for stability (0..1), higher = smoother
    ema_alpha: float = 0.35

    # Consider CENTER if within this deadzone (abs(value) < deadzone)
    deadzone: float = 0.10


@dataclass
class PlayerState:
    idx: int
    score: int = 0
    yaw_ema: float = 0.0
    last_dir: str = "CENTER"     # current direction based on yaw_ema
    last_choice: Optional[str] = None  # last registered choice
    hold_start: Optional[float] = None
    cooldown_until: float = 0.0
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x1,y1,x2,y2
    visible: bool = False


class HeadGestureDetector:
    """
    Approximates head turn left/right using nose position relative to face bbox center.
    Uses EMA + hold-time + cooldown to avoid false triggers.
    """

    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg

    def _dir_from_yaw(self, yaw: float) -> str:
        if abs(yaw) < self.cfg.deadzone:
            return "CENTER"
        if yaw <= self.cfg.left_thresh:
            return "LEFT"
        if yaw >= self.cfg.right_thresh:
            return "RIGHT"
        # intermediate zone: treat as CENTER-ish to avoid jitter
        return "CENTER"

    def update_player(self, p: PlayerState, yaw_raw: float, now: float) -> Optional[str]:
        # EMA smoothing
        p.yaw_ema = (self.cfg.ema_alpha * yaw_raw) + ((1.0 - self.cfg.ema_alpha) * p.yaw_ema)
        current_dir = self._dir_from_yaw(p.yaw_ema)
        p.last_dir = current_dir

        # Cooldown active => no new choice
        if now < p.cooldown_until:
            p.hold_start = None
            return None

        # Register choice only for LEFT/RIGHT
        if current_dir in ("LEFT", "RIGHT"):
            if p.hold_start is None:
                p.hold_start = now
            else:
                if (now - p.hold_start) >= self.cfg.hold_time:
                    # Choice registered
                    p.last_choice = current_dir
                    p.cooldown_until = now + self.cfg.cooldown
                    p.hold_start = None
                    return current_dir
        else:
            # returned to center
            p.hold_start = None

        return None


class FaceGameMVP:
    def __init__(self, num_players: int, cam_index: int = 0, width: int = 1280, height: int = 720):
        self.num_players = num_players
        self.cam_index = cam_index
        self.width = width
        self.height = height

        self.cfg = GestureConfig()
        self.gesture = HeadGestureDetector(self.cfg)

        self.players: List[PlayerState] = [PlayerState(idx=i + 1) for i in range(num_players)]

        self.mp_face = mp.solutions.face_detection
        # model_selection=0 is short-range; 1 is full-range. For webcam, 0 is usually better.
        self.face_det = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Simple “question” placeholder
        self.question = "TEST: Inclina la cabeza a IZQ o DER"
        self.left_label = "IZQUIERDA"
        self.right_label = "DERECHA"
        self.last_event = "Sin eventos"
        self.last_event_time = 0.0

    def close(self):
        try:
            self.face_det.close()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _draw_hud(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Jugadores objetivo: {self.num_players} | Presiona ESC para salir | R para reset scores",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Question area
        cv2.putText(frame, self.question, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Options bottom
        cv2.rectangle(frame, (0, h - 110), (w // 2, h), (20, 20, 20), -1)
        cv2.rectangle(frame, (w // 2, h - 110), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"<< {self.left_label}", (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
        cv2.putText(frame, f"{self.right_label} >>", (w // 2 + 30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)

        # Last event
        if (time.time() - self.last_event_time) < 2.0:
            cv2.putText(frame, self.last_event, (w - 520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    def _draw_player_panel(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        panel_w = 320
        x0 = w - panel_w

        cv2.rectangle(frame, (x0, 80), (w, h - 110), (10, 10, 10), -1)
        cv2.putText(frame, "SCORES / ESTADO", (x0 + 15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y = 150
        for p in self.players:
            status = "VISIBLE" if p.visible else "NO FACE"
            cv2.putText(frame, f"J{p.idx}  Pts:{p.score}  {status}", (x0 + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.putText(frame, f"Dir:{p.last_dir}  yaw:{p.yaw_ema:+.3f}", (x0 + 15, y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)
            y += 70

    def _bbox_from_rel(self, rel_bbox, w: int, h: int) -> Tuple[int, int, int, int]:
        x = int(rel_bbox.xmin * w)
        y = int(rel_bbox.ymin * h)
        bw = int(rel_bbox.width * w)
        bh = int(rel_bbox.height * h)
        x1 = clamp(x, 0, w - 1)
        y1 = clamp(y, 0, h - 1)
        x2 = clamp(x + bw, 0, w - 1)
        y2 = clamp(y + bh, 0, h - 1)
        return x1, y1, x2, y2

    def _compute_yaw_proxy(self, bbox: Tuple[int, int, int, int], keypoints, w: int, h: int) -> float:
        """
        keypoints: list of 6 FaceDetection keypoints, where index 2 is nose tip in mediapipe.
        We'll use nose_x vs bbox_center_x normalized by bbox width.
        """
        x1, y1, x2, y2 = bbox
        bw = max(1, (x2 - x1))
        cx = (x1 + x2) / 2.0

        # Nose tip is keypoint[2] (in mp FaceDetection)
        nose = keypoints[2]
        nose_x = nose.x * w

        yaw = (nose_x - cx) / bw  # normalized offset
        return float(clamp(yaw, -1.0, 1.0))

    def reset_scores(self):
        for p in self.players:
            p.score = 0
            p.last_choice = None
            p.hold_start = None
            p.cooldown_until = 0.0
        self.last_event = "Scores reseteados"
        self.last_event_time = time.time()

    def run(self):
        if not self.cap.isOpened():
            raise RuntimeError("No pude abrir la webcam. Cambia --cam o revisa permisos.")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            # mirror for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Reset visibility
            for p in self.players:
                p.visible = False
                p.face_bbox = None

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_det.process(rgb)

            detections = []
            if results.detections:
                for det in results.detections:
                    bbox = self._bbox_from_rel(det.location_data.relative_bounding_box, w, h)
                    score = det.score[0] if det.score else 0.0
                    keypoints = det.location_data.relative_keypoints
                    x1, y1, x2, y2 = bbox
                    area = (x2 - x1) * (y2 - y1)
                    cx = (x1 + x2) / 2.0
                    detections.append({
                        "bbox": bbox,
                        "score": score,
                        "keypoints": keypoints,
                        "area": area,
                        "cx": cx
                    })

            # Pick top-N by area, then sort left-to-right for stable assignment
            detections.sort(key=lambda d: d["area"], reverse=True)
            detections = detections[:self.num_players]
            detections.sort(key=lambda d: d["cx"])

            now = time.time()

            # Assign detections to players (J1..JN)
            for i, d in enumerate(detections):
                p = self.players[i]
                p.visible = True
                p.face_bbox = d["bbox"]

                yaw_raw = self._compute_yaw_proxy(d["bbox"], d["keypoints"], w, h)
                choice = self.gesture.update_player(p, yaw_raw, now)

                # If a choice is registered, add score (for now just increment)
                if choice is not None:
                    p.score += 1
                    self.last_event = f"J{p.idx} eligio {choice} (+1)"
                    self.last_event_time = now

            # Draw face boxes and labels
            for i, d in enumerate(detections):
                bbox = d["bbox"]
                x1, y1, x2, y2 = bbox
                p = self.players[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"J{p.idx}", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # show nose point
                nose = d["keypoints"][2]
                nx, ny = int(nose.x * w), int(nose.y * h)
                cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)

            # HUD
            self._draw_hud(frame)
            self._draw_player_panel(frame)

            # Info if not enough faces
            if len(detections) < self.num_players:
                cv2.putText(frame, f"Detectados: {len(detections)}/{self.num_players}. Acercate / mejora luz.",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            cv2.imshow("Head Game MVP", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key in (ord('r'), ord('R')):
                self.reset_scores()

        self.close()


def main():
    parser = argparse.ArgumentParser(description="MVP Head-gesture game (multi-players) - OpenCV + MediaPipe")
    parser.add_argument("--players", type=int, default=1, help="Numero de jugadores objetivo (1..N)")
    parser.add_argument("--cam", type=int, default=0, help="Indice de camara (0 por defecto)")
    parser.add_argument("--width", type=int, default=1280, help="Ancho de captura")
    parser.add_argument("--height", type=int, default=720, help="Alto de captura")
    args = parser.parse_args()

    if args.players < 1 or args.players > 6:
        raise ValueError("Usa --players entre 1 y 6 para este MVP (por rendimiento/legibilidad).")

    app = FaceGameMVP(num_players=args.players, cam_index=args.cam, width=args.width, height=args.height)
    app.run()


if __name__ == "__main__":
    main()