import time
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class Button:
    name: str  # "LEFT" or "RIGHT"
    rect: Tuple[int, int, int, int]  # x1,y1,x2,y2

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class PlayerState:
    idx: int
    score: int = 0
    armed: bool = False
    ready_hold_start: Optional[float] = None

    hovering_button: Optional[str] = None
    hover_start: Optional[float] = None
    cooldown_until: float = 0.0

    last_choice: Optional[str] = None


@dataclass
class GameConfig:
    round_seconds: float = 5.0
    hold_click_seconds: float = 0.25
    cooldown_seconds: float = 0.35

    # "Levantar mano" = índice entra a ready zone (parte superior) y se mantiene
    ready_zone_ratio: float = 0.30    # top 30% of screen
    ready_hold_seconds: float = 0.25  # must hold in ready zone to arm


class HandButtonGameMulti:
    def __init__(self, cam_index: int, width: int, height: int, players: int, cfg: GameConfig):
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.players = players
        self.cfg = cfg

        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.mp_hands = mp.solutions.hands
        # mediapipe hands is realistically strong for 1-2 hands; we still allow N lanes, but detection is capped.
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=min(2, players),   # hard cap for stability
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.states: List[PlayerState] = [PlayerState(idx=i + 1) for i in range(players)]

        self.round_idx = 0
        self.target: str = "LEFT"
        self.round_deadline = 0.0

        self.last_feedback = ""
        self.feedback_until = 0.0

        self.btns_left: List[Button] = []
        self.btns_right: List[Button] = []

        self._new_round()

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _set_feedback(self, text: str, seconds: float = 1.0):
        self.last_feedback = text
        self.feedback_until = time.time() + seconds

    def _new_round(self):
        self.round_idx += 1
        self.target = random.choice(["LEFT", "RIGHT"])
        self.round_deadline = time.time() + self.cfg.round_seconds

        # reset arming / hover for all players
        for st in self.states:
            st.armed = False
            st.ready_hold_start = None
            st.hovering_button = None
            st.hover_start = None
            st.cooldown_until = 0.0
            st.last_choice = None

    def _layout_buttons(self, w: int, h: int):
        self.btns_left.clear()
        self.btns_right.clear()

        pad = 18
        btn_h = 110
        y2 = h - pad
        y1 = y2 - btn_h

        lane_w = w // self.players
        inner_pad = 16

        for i in range(self.players):
            lane_x1 = i * lane_w
            lane_x2 = (i + 1) * lane_w

            # two buttons inside lane
            total_inner_w = (lane_x2 - lane_x1) - inner_pad * 3
            btn_w = total_inner_w // 2

            lx1 = lane_x1 + inner_pad
            lx2 = lx1 + btn_w

            rx1 = lx2 + inner_pad
            rx2 = rx1 + btn_w

            self.btns_left.append(Button("LEFT", (lx1, y1, lx2, y2)))
            self.btns_right.append(Button("RIGHT", (rx1, y1, rx2, y2)))

    def _player_from_x(self, x: int, w: int) -> int:
        lane_w = w // self.players
        idx = int(x // lane_w)
        return max(0, min(self.players - 1, idx))

    def _draw_top_hud(self, frame, now: float):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 92), (0, 0, 0), -1)

        remaining = max(0.0, self.round_deadline - now)
        cv2.putText(frame, f"Ronda: {self.round_idx}   Tiempo: {remaining:0.1f}s   (ESC salir | R reset)",
                    (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        target_txt = "TOCA IZQ" if self.target == "LEFT" else "TOCA DER"
        cv2.putText(frame, f"OBJETIVO GLOBAL: {target_txt}  |  Para responder: LEVANTA MANO (zona superior) -> ARMADO",
                    (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # READY ZONE line
        ready_y = int(h * self.cfg.ready_zone_ratio)
        cv2.line(frame, (0, ready_y), (w, ready_y), (80, 80, 80), 2)
        cv2.putText(frame, "READY ZONE (levanta la mano aqui)", (20, ready_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    def _draw_lane_dividers(self, frame):
        h, w = frame.shape[:2]
        lane_w = w // self.players
        for i in range(1, self.players):
            x = i * lane_w
            cv2.line(frame, (x, 92), (x, h), (40, 40, 40), 2)

    def _draw_player_status(self, frame):
        h, w = frame.shape[:2]
        lane_w = w // self.players
        for i, st in enumerate(self.states):
            x1 = i * lane_w + 15
            y1 = 110
            status = "ARMADO" if st.armed else "NO ARMADO"
            cv2.putText(frame, f"J{i+1}  Pts:{st.score}  {status}",
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if st.armed else (200, 200, 200), 2)

    def _draw_button(self, frame, btn: Button, label: str, is_target: bool, is_hover: bool, armed: bool):
        x1, y1, x2, y2 = btn.rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)

        # border priority
        border = (120, 120, 120)
        thick = 2

        if armed:
            border = (150, 150, 150)

        if is_target:
            border = (0, 255, 255)
            thick = 4

        if is_hover:
            border = (0, 255, 0)
            thick = 6

        cv2.rectangle(frame, (x1, y1), (x2, y2), border, thick)

        # label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cx = (x1 + x2) // 2 - tw // 2
        cy = (y1 + y2) // 2 + th // 2
        cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        # lock hint
        if not armed:
            cv2.putText(frame, "LEVANTA MANO",
                        (x1 + 12, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2)

    def _process_choice(self, player_i: int, choice: str):
        st = self.states[player_i]
        st.last_choice = choice

        if choice == self.target:
            st.score += 1
            self._set_feedback(f"J{player_i+1}: ✅ CORRECTO (+1)", 0.9)
        else:
            self._set_feedback(f"J{player_i+1}: ❌ INCORRECTO", 0.9)

        # after answering, disarm + cooldown (avoid double click)
        st.armed = False
        st.ready_hold_start = None
        st.cooldown_until = time.time() + self.cfg.cooldown_seconds

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

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if not self.btns_left:
                self._layout_buttons(w, h)

            now = time.time()
            self._draw_top_hud(frame, now)
            self._draw_lane_dividers(frame)
            self._draw_player_status(frame)

            # round timeout -> new round
            if now >= self.round_deadline:
                self._set_feedback("⏱️ TIEMPO AGOTADO (nueva ronda)", 0.7)
                self._new_round()

            # detect hands
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            # per-player cursor from index tip (best-effort: at most 2 hands)
            cursors = [None] * self.players  # (x,y)
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    ix = int(lm.landmark[8].x * w)  # index tip
                    iy = int(lm.landmark[8].y * h)
                    p_i = self._player_from_x(ix, w)
                    cursors[p_i] = (ix, iy)

            ready_y = int(h * self.cfg.ready_zone_ratio)

            # update arming + hover/click for each player
            for i, st in enumerate(self.states):
                cur = cursors[i]

                # draw cursor if present
                if cur:
                    cx, cy = cur
                    cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
                    cv2.circle(frame, (cx, cy), 14, (0, 255, 0), 2)

                # cannot interact during cooldown
                if now < st.cooldown_until:
                    st.hovering_button = None
                    st.hover_start = None
                    st.ready_hold_start = None
                    continue

                # 1) arming logic (raise hand => cursor in ready zone)
                if not st.armed:
                    if cur and cur[1] < ready_y:
                        if st.ready_hold_start is None:
                            st.ready_hold_start = now
                        elif (now - st.ready_hold_start) >= self.cfg.ready_hold_seconds:
                            st.armed = True
                            st.ready_hold_start = None
                            self._set_feedback(f"J{i+1}: ARMADO ✅", 0.4)
                    else:
                        st.ready_hold_start = None

                    # if not armed, skip click logic
                    st.hovering_button = None
                    st.hover_start = None
                    continue

                # 2) click logic (only if armed)
                hover_btn = None
                if cur:
                    x, y = cur
                    if self.btns_left[i].contains(x, y):
                        hover_btn = "LEFT"
                    elif self.btns_right[i].contains(x, y):
                        hover_btn = "RIGHT"

                if hover_btn:
                    if st.hovering_button != hover_btn:
                        st.hovering_button = hover_btn
                        st.hover_start = now
                    else:
                        if st.hover_start and (now - st.hover_start) >= self.cfg.hold_click_seconds:
                            self._process_choice(i, hover_btn)
                            st.hovering_button = None
                            st.hover_start = None
                else:
                    st.hovering_button = None
                    st.hover_start = None

            # draw buttons per lane
            for i, st in enumerate(self.states):
                self._draw_button(frame, self.btns_left[i], "IZQUIERDA",
                                  is_target=(self.target == "LEFT"),
                                  is_hover=(st.hovering_button == "LEFT"),
                                  armed=st.armed)
                self._draw_button(frame, self.btns_right[i], "DERECHA",
                                  is_target=(self.target == "RIGHT"),
                                  is_hover=(st.hovering_button == "RIGHT"),
                                  armed=st.armed)

            # feedback
            if now < self.feedback_until and self.last_feedback:
                cv2.putText(frame, self.last_feedback, (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # warning about >2 players
            if self.players > 2:
                cv2.putText(frame, "AVISO: MediaPipe Hands es confiable para 1-2 manos simultaneas (no N).",
                            (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("Hand Button Game (Multi + Arm)", frame)
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
    ap.add_argument("--players", type=int, default=1, help="Cantidad de jugadores (lanes). Recomendado 1-2.")
    ap.add_argument("--seconds", type=float, default=5.0, help="segundos por ronda")
    ap.add_argument("--hold", type=float, default=0.25, help="segundos para considerar click por mantener dedo en boton")
    ap.add_argument("--ready-hold", type=float, default=0.25, help="segundos en READY ZONE para armar")
    ap.add_argument("--ready-zone", type=float, default=0.30, help="porcentaje superior (0-1) como READY ZONE")
    args = ap.parse_args()

    if args.players < 1:
        raise ValueError("--players debe ser >= 1")

    cfg = GameConfig(
        round_seconds=args.seconds,
        hold_click_seconds=args.hold,
        ready_hold_seconds=args.ready_hold,
        ready_zone_ratio=args.ready_zone,
    )

    game = HandButtonGameMulti(args.cam, args.width, args.height, args.players, cfg)
    game.run()


if __name__ == "__main__":
    main()