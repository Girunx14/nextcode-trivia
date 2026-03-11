import time
import random
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import cv2
import mediapipe as mp
import unicodedata

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '¿': '', '¡': '!', '✅': '[OK]', '❌': '[X]', '⏱️': '[TIEMPO]', '⏱': '[TIEMPO]',
        'ñ': 'n', 'Ñ': 'N'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def draw_panel(frame, x1, y1, x2, y2, alpha=0.70, color=(0, 0, 0)):
    """Dibuja un rectángulo semitransparente."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_text_with_shadow(frame, text, pos, font, scale, color, thickness, center=False):
    """Auxiliar para texto con sombra paralela elegante y AA."""
    text = clean_text(text)
    if center:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = int(pos[0] - tw / 2)
        y = int(pos[1] + th / 2)
    else:
        x, y = int(pos[0]), int(pos[1])
    # Shadow
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, (15, 15, 15), thickness, cv2.LINE_AA)
    # Text
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return (x, y)

def draw_wrapped_text(frame, text, x, y, max_width, font, scale, color, thickness, line_gap=8):
    """Dibuja texto con salto de línea por ancho aproximado, con sombra y AA."""
    text = clean_text(text)
    words = text.split()
    line = ""
    y_cursor = y

    for w in words:
        test = (line + " " + w).strip()
        (tw, th), _ = cv2.getTextSize(test, font, scale, thickness)
        if tw <= max_width:
            line = test
        else:
            draw_text_with_shadow(frame, line, (x, y_cursor), font, scale, color, thickness)
            y_cursor += th + line_gap
            line = w

    if line:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        draw_text_with_shadow(frame, line, (x, y_cursor), font, scale, color, thickness)
        y_cursor += th + line_gap

    return y_cursor

# -----------------------------
# Banco de preguntas (A/B)
# Cada item: {"q": "...", "left": "...", "right": "...", "ans": "LEFT"/"RIGHT"}
# -----------------------------
QUESTION_BANK: Dict[str, List[Dict[str, str]]] = {
    "Python": [
        {"q": "¿Qué estructura es INMUTABLE?", "left": "tuple", "right": "list", "ans": "LEFT"},
        {"q": "¿Qué operador es de igualdad?", "left": "==", "right": "=", "ans": "LEFT"},
        {"q": "¿Qué devuelve len('abc')?", "left": "3", "right": "2", "ans": "LEFT"},
        {"q": "¿Qué palabra define una función?", "left": "def", "right": "func", "ans": "LEFT"},
        {"q": "¿Cuál es un diccionario?", "left": "{'a':1}", "right": "['a',1]", "ans": "LEFT"},
        {"q": "¿Qué hace break?", "left": "sale del loop", "right": "salta iteración", "ans": "LEFT"},
        {"q": "¿Qué devuelve type(5)?", "left": "int", "right": "str", "ans": "LEFT"},
        {"q": "¿Qué es True and False?", "left": "False", "right": "True", "ans": "LEFT"},
        {"q": "¿Cuál imprime en consola?", "left": "print()", "right": "echo()", "ans": "LEFT"},
        {"q": "¿Qué índice es el primero?", "left": "0", "right": "1", "ans": "LEFT"},
        {"q": "¿Qué crea una lista?", "left": "[]", "right": "()", "ans": "LEFT"},
        {"q": "¿Qué hace .append(x)?", "left": "agrega al final", "right": "ordena", "ans": "LEFT"},
    ],
    "Web": [
        {"q": "¿Qué es HTML?", "left": "estructura", "right": "base de datos", "ans": "LEFT"},
        {"q": "¿Qué es CSS?", "left": "estilos", "right": "servidor", "ans": "LEFT"},
        {"q": "¿Qué es HTTP?", "left": "protocolo", "right": "lenguaje", "ans": "LEFT"},
        {"q": "¿Qué status es 'Not Found'?", "left": "404", "right": "200", "ans": "LEFT"},
        {"q": "¿Qué se ejecuta en el navegador?", "left": "JavaScript", "right": "PHP", "ans": "LEFT"},
        {"q": "¿Qué etiqueta hace un link?", "left": "<a>", "right": "<p>", "ans": "LEFT"},
        {"q": "¿GET manda datos en...?", "left": "URL", "right": "body", "ans": "LEFT"},
        {"q": "¿POST manda datos en...?", "left": "body", "right": "URL", "ans": "LEFT"},
        {"q": "¿CORS afecta...?", "left": "origen", "right": "RAM", "ans": "LEFT"},
        {"q": "¿Qué es JSON?", "left": "formato datos", "right": "compilador", "ans": "LEFT"},
    ],
    "IA": [
        {"q": "¿Qué es 'overfitting'?", "left": "memoriza", "right": "generaliza", "ans": "LEFT"},
        {"q": "¿Qué es un 'dataset'?", "left": "conjunto datos", "right": "un CPU", "ans": "LEFT"},
        {"q": "¿Accuracy mide...?", "left": "aciertos", "right": "tiempo", "ans": "LEFT"},
        {"q": "¿Train/Test sirve para...?", "left": "evaluar", "right": "colorear", "ans": "LEFT"},
        {"q": "¿Qué es 'features'?", "left": "variables", "right": "servidores", "ans": "LEFT"},
        {"q": "¿Clasificación predice...?", "left": "clases", "right": "texturas", "ans": "LEFT"},
        {"q": "¿Regresión predice...?", "left": "números", "right": "colores", "ans": "LEFT"},
        {"q": "¿Qué es 'label'?", "left": "respuesta", "right": "cámara", "ans": "LEFT"},
    ],
}


# -----------------------------
# UI helpers
# -----------------------------
@dataclass
class RectButton:
    key: str
    label: str
    rect: Tuple[int, int, int, int]  # x1,y1,x2,y2

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class PlayerState:
    idx: int
    score: int = 0
    attempts: int = 0
    answered_this_round: bool = False
    hold_start_left: Optional[float] = None
    hold_start_right: Optional[float] = None
    detected_hand: str = "—"  # "IZQ", "DER", "—"


@dataclass
class GameConfig:
    # Answering
    hold_seconds: float = 0.20
    min_raise_confidence: float = 0.55
    wrist_y_threshold_ratio: float = 0.55  # "raised" if wrist above this line

    # Next button
    next_hold_seconds: float = 0.25

    # Modes
    fixed_questions: int = 10
    speed_seconds: int = 120
    question_seconds: float = 15.0


class QuizRaiseHandGame:
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

        # Flow
        self.phase = "SELECT_MODE"  # SELECT_MODE -> SELECT_TOPIC -> PLAY -> PAUSE_NEXT -> END
        self.mode = None            # "FIXED10" or "SPEED2M"
        self.topic = None

        # Quiz state
        self.question_pool: List[Dict[str, str]] = []
        self.q_index = 0
        self.current_q: Optional[Dict[str, str]] = None

        # Timing
        self.round_deadline: float = 0.0   # used in both modes per question (to keep pace)
        self.speed_end: float = 0.0        # only in speed mode

        # UI: buttons
        self.buttons: List[RectButton] = []
        self.btn_hover_key: Optional[str] = None
        self.btn_hover_start: Optional[float] = None

        # Next button (bottom)
        self.next_rect: Optional[Tuple[int, int, int, int]] = None
        self.next_hover_start: Optional[float] = None

        # Messages
        self.banner = "Elige modalidad"
        self.result_text = ""

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

    def _rect_contains(self, rect, x: int, y: int) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _set_next_rect(self, w: int, h: int):
        bw = int(w * 0.34)
        bh = 85
        pad = 18
        x1 = (w - bw) // 2
        x2 = x1 + bw
        y2 = h - pad
        y1 = y2 - bh
        self.next_rect = (x1, y1, x2, y2)

    def _draw_button(self, frame, btn: RectButton, enabled: bool, hover: bool):
        x1, y1, x2, y2 = btn.rect
        
        # Fondo premium
        bg_color = (35, 30, 30) if not hover else (55, 45, 45)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)

        # Borde elegante
        if not enabled:
            border = (100, 100, 100)
            thick = 1
        else:
            border = (220, 170, 50) # Dorado/Cyan
            thick = 2
        if hover and enabled:
            border = (120, 255, 120) # Verde vibrante
            thick = 3

        cv2.rectangle(frame, (x1, y1), (x2, y2), border, thick, cv2.LINE_AA)

        if hover and enabled:
            draw_panel(frame, x1+thick, y1+thick, x2-thick, y2-thick, alpha=0.1, color=(255,255,255))

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        draw_text_with_shadow(frame, btn.label, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, center=True)

    def _draw_next(self, frame, enabled: bool, hover: bool):
        if self.next_rect is None:
            return
        btn = RectButton("NEXT", "SIGUIENTE", self.next_rect)
        self._draw_button(frame, btn, enabled, hover)
        if enabled:
            x1, y1, x2, y2 = self.next_rect
            draw_text_with_shadow(frame, "Baja las manos y manten el indice apuntando en SIGUIENTE",
                                  (max(20, x1 - 100), y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.6, (220, 220, 220), 1)

    def _build_mode_buttons(self, w: int, h: int):
        self.buttons.clear()
        bw = int(w * 0.42)
        bh = 95
        gap = 25
        x1 = (w - bw) // 2
        y_top = 210
        self.buttons.append(RectButton("FIXED10", "Modo 1: 10 preguntas", (x1, y_top, x1 + bw, y_top + bh)))
        self.buttons.append(RectButton("SPEED2M", "Modo 2: 2 minutos (mas preguntas)", (x1, y_top + bh + gap, x1 + bw, y_top + 2 * bh + gap)))

    def _build_topic_buttons(self, w: int, h: int):
        self.buttons.clear()
        topics = list(QUESTION_BANK.keys())
        bw = int(w * 0.40)
        bh = 80
        gap = 18
        x1 = (w - bw) // 2
        y = 200
        for t in topics:
            self.buttons.append(RectButton(f"TOPIC:{t}", f"Tema: {t}", (x1, y, x1 + bw, y + bh)))
            y += bh + gap

    def _start_quiz(self):
        # Reset player stats
        for st in self.states:
            st.score = 0
            st.attempts = 0
            st.answered_this_round = False
            st.hold_start_left = None
            st.hold_start_right = None
            st.detected_hand = "—"

        # Build question list
        pool = QUESTION_BANK[self.topic][:]
        random.shuffle(pool)
        if self.mode == "FIXED10":
            self.question_pool = pool[: min(self.cfg.fixed_questions, len(pool))]
        else:
            # speed: keep big pool, we will cycle through it
            self.question_pool = pool if pool else []
        self.q_index = 0
        self.current_q = None

        # Time
        now = time.time()
        self.speed_end = now + self.cfg.speed_seconds if self.mode == "SPEED2M" else 0.0

        self.phase = "PLAY"
        self._load_next_question()
        self.result_text = ""

    def _load_next_question(self):
        if self.mode == "FIXED10":
            if self.q_index >= len(self.question_pool):
                self.phase = "END"
                return
            self.current_q = self.question_pool[self.q_index]
            self.q_index += 1
        else:
            # speed mode: cycle questions
            if not self.question_pool:
                self.phase = "END"
                return
            self.current_q = self.question_pool[self.q_index % len(self.question_pool)]
            self.q_index += 1

        # per-question deadline
        self.round_deadline = time.time() + self.cfg.question_seconds
        for st in self.states:
            st.answered_this_round = False
            st.hold_start_left = None
            st.hold_start_right = None

        self.phase = "PLAY"

    def _end_now(self):
        self.phase = "END"

    def _winner_text(self) -> str:
        # FIXED10: by score
        if self.mode == "FIXED10":
            best = max(self.states, key=lambda s: s.score)
            tied = [s for s in self.states if s.score == best.score]
            if len(tied) > 1:
                return f"Empate en {best.score} puntos."
            return f"Gana J{best.idx} con {best.score} puntos."
        # SPEED2M: by attempts, then score
        best_attempts = max(s.attempts for s in self.states)
        top = [s for s in self.states if s.attempts == best_attempts]
        if len(top) == 1:
            return f"Gana J{top[0].idx}: {top[0].attempts} preguntas, {top[0].score} correctas."
        best_score = max(s.score for s in top)
        top2 = [s for s in top if s.score == best_score]
        if len(top2) == 1:
            return f"Gana J{top2[0].idx}: {top2[0].attempts} preguntas, {top2[0].score} correctas."
        return "Empate total."

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

            # Build buttons per phase (once per frame is fine for MVP)
            if self.phase == "SELECT_MODE":
                self.banner = "Selecciona modalidad (con dedo indice)"
                self._build_mode_buttons(w, h)
            elif self.phase == "SELECT_TOPIC":
                self.banner = f"Selecciona tema (Modo: {self.mode})"
                self._build_topic_buttons(w, h)
            else:
                self.buttons.clear()

            # ---------------- HUD Top ----------------
            draw_panel(frame, 0, 0, w, 110, alpha=0.85, color=(20, 15, 15))
            draw_text_with_shadow(frame, self.banner, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 235, 200), 2)

            if self.mode:
                draw_text_with_shadow(frame, f"Modo: {self.mode}  |  Tema: {self.topic or '—'}",
                                      (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 220, 255), 1)

            # line threshold (Elegante)
            cv2.line(frame, (0, y_thr), (w, y_thr), (100, 60, 60), 2, cv2.LINE_AA)
            draw_text_with_shadow(frame, "Zona de Respuesta (Mano Arriba de esta linea = IZQ / DER)",
                                  (20, y_thr - 10), cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 150, 150), 1)

            # ---------------- Hand detection ----------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            # reset detected labels
            for st in self.states:
                st.detected_hand = "—"

            raised_left = [False] * self.players
            raised_right = [False] * self.players

            any_hand_raised = False

            # index (for UI button pressing)
            index_points: List[Tuple[int, int, float]] = []  # (ix, iy, conf)

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label
                    conf = hd.classification[0].score

                    wx = int(lm.landmark[0].x * w)
                    wy = int(lm.landmark[0].y * h)
                    ix = int(lm.landmark[8].x * w)
                    iy = int(lm.landmark[8].y * h)

                    # draw index tip cursor (Glowing style)
                    cv2.circle(frame, (ix, iy), 8, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (ix, iy), 14, (0, 200, 255), 2, cv2.LINE_AA)

                    index_points.append((ix, iy, conf))

                    p_i = self._lane_index(wx, w)
                    logical = label  # no swap
                    hand_es = "IZQ" if logical == "Left" else "DER"
                    self.states[p_i].detected_hand = hand_es

                    # draw wrist
                    cv2.circle(frame, (wx, wy), 6, (200, 255, 200), -1, cv2.LINE_AA)
                    draw_text_with_shadow(frame, f"{hand_es} ({conf:.2f}) J{p_i+1}",
                                          (wx + 12, wy - 12), cv2.FONT_HERSHEY_DUPLEX, 0.6, (150, 255, 150), 1)

                    is_raised = (conf >= self.cfg.min_raise_confidence) and (wy < y_thr)
                    if is_raised:
                        any_hand_raised = True
                        if logical == "Left":
                            raised_left[p_i] = True
                        else:
                            raised_right[p_i] = True

            # ---------------- Phase logic ----------------
            # A) Selection screens: press buttons with index
            if self.phase in ("SELECT_MODE", "SELECT_TOPIC"):
                # detect hovering on any button
                hover_key = None
                for (ix, iy, conf) in index_points:
                    if conf < self.cfg.min_raise_confidence:
                        continue
                    for btn in self.buttons:
                        if btn.contains(ix, iy):
                            hover_key = btn.key
                            break
                    if hover_key:
                        break

                if hover_key != self.btn_hover_key:
                    self.btn_hover_key = hover_key
                    self.btn_hover_start = now if hover_key else None
                else:
                    if hover_key and self.btn_hover_start and (now - self.btn_hover_start) >= self.cfg.next_hold_seconds:
                        # Commit selection
                        if self.phase == "SELECT_MODE":
                            self.mode = hover_key
                            self.topic = None
                            self.phase = "SELECT_TOPIC"
                            self.btn_hover_key = None
                            self.btn_hover_start = None
                        else:
                            # TOPIC
                            self.topic = hover_key.split("TOPIC:", 1)[1]
                            self.phase = "PLAY"
                            self.btn_hover_key = None
                            self.btn_hover_start = None
                            self._start_quiz()

                # draw buttons
                for btn in self.buttons:
                    hover = (btn.key == self.btn_hover_key)
                    self._draw_button(frame, btn, enabled=True, hover=hover)

                # show players info
                lane_w = w / self.players
                for i, st in enumerate(self.states):
                    x1 = int(i * lane_w) + 15
                    draw_text_with_shadow(frame, f"J{st.idx} Detectado:{st.detected_hand}",
                                          (x1, 140), cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 220), 1)

            # B) Play / Pause / End
            else:
                # draw per-player scoreboard
                lane_w = w / self.players
                for i in range(1, self.players):
                    x = int(i * lane_w)
                    cv2.line(frame, (x, 110), (x, h), (80, 80, 80), 1, cv2.LINE_AA)

                for i, st in enumerate(self.states):
                    x1 = int(i * lane_w) + 15
                    metrics = f"J{st.idx} | Pts: {st.score} | Int: {st.attempts} | Mano: {st.detected_hand}"
                    draw_text_with_shadow(frame, metrics,
                                          (x1, 140), cv2.FONT_HERSHEY_DUPLEX, 0.6, (230, 230, 230), 1)

                if self.phase == "END":
                    draw_text_with_shadow(frame, "FIN DEL JUEGO", (20, 220), cv2.FONT_HERSHEY_DUPLEX, 1.4, (100, 255, 255), 3)
                    draw_text_with_shadow(frame, self._winner_text(), (20, 280), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
                    draw_text_with_shadow(frame, "Presiona R para reiniciar interactividad", (20, 330),
                                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 1)
                else:
                    # Timer display (sleek)
                    if self.phase == "PLAY":
                        remaining = max(0.0, self.round_deadline - now)
                        draw_text_with_shadow(frame, f"Tiempo: {remaining:0.1f}s",
                                              (20, 210), cv2.FONT_HERSHEY_DUPLEX, 0.75, (200, 255, 200), 2)
                    else:
                        draw_text_with_shadow(frame, "Tiempo: PAUSA",
                                              (20, 210), cv2.FONT_HERSHEY_DUPLEX, 0.75, (180, 180, 180), 1)

                    if self.mode == "SPEED2M":
                        rem2 = max(0.0, self.speed_end - now)
                        draw_text_with_shadow(frame, f"Global (2 min): {rem2:0.1f}s",
                                              (20, 245), cv2.FONT_HERSHEY_DUPLEX, 0.75, (100, 200, 255), 1)
                        if now >= self.speed_end:
                            self._end_now()

                    # Caja central de pregunta (más elegante)
                    px1, py1, px2, py2 = max(20, int((w - 900) / 2)), 270, min(w - 20, w - max(20, int((w - 900) / 2))), 470
                    draw_panel(frame, px1, py1, px2, py2, alpha=0.85, color=(30, 25, 35))
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (200, 150, 50), 2, cv2.LINE_AA)

                    if self.current_q:
                        q = self.current_q["q"]
                        left = self.current_q["left"]
                        right = self.current_q["right"]

                        y = py1 + 40
                        y = draw_wrapped_text(frame, f"PREGUNTA: {q}", px1 + 25, y, (px2 - px1) - 50,
                                              cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

                        y += 20
                        draw_text_with_shadow(frame, f"IZQ: {left}", (px1 + 25, y),
                                              cv2.FONT_HERSHEY_DUPLEX, 0.8, (180, 220, 255), 2)
                        y += 35
                        draw_text_with_shadow(frame, f"DER: {right}", (px1 + 25, y),
                                              cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 180, 180), 2)

                    # Timeouts -> pause and require Next
                    if self.phase == "PLAY" and now >= self.round_deadline:
                        self.phase = "PAUSE_NEXT"
                        self.result_text = "⏱️ TIEMPO AGOTADO. Baja manos y presiona SIGUIENTE."
                        self.next_hover_start = None

                    # Answer detection in PLAY
                    if self.phase == "PLAY":
                        for i, st in enumerate(self.states):
                            if st.answered_this_round:
                                continue

                            # LEFT
                            if raised_left[i]:
                                if st.hold_start_left is None:
                                    st.hold_start_left = now
                                elif (now - st.hold_start_left) >= self.cfg.hold_seconds:
                                    st.answered_this_round = True
                                    st.attempts += 1
                                    if self.current_q and self.current_q["ans"] == "LEFT":
                                        st.score += 1
                                        self.result_text = f"J{st.idx}: ✅ CORRECTO (+1). Baja manos y presiona SIGUIENTE."
                                    else:
                                        self.result_text = f"J{st.idx}: ❌ INCORRECTO. Baja manos y presiona SIGUIENTE."
                                    self.phase = "PAUSE_NEXT"
                                    self.next_hover_start = None
                                    break
                            else:
                                st.hold_start_left = None

                            # RIGHT
                            if raised_right[i]:
                                if st.hold_start_right is None:
                                    st.hold_start_right = now
                                elif (now - st.hold_start_right) >= self.cfg.hold_seconds:
                                    st.answered_this_round = True
                                    st.attempts += 1
                                    if self.current_q and self.current_q["ans"] == "RIGHT":
                                        st.score += 1
                                        self.result_text = f"J{st.idx}: ✅ CORRECTO (+1). Baja manos y presiona SIGUIENTE."
                                    else:
                                        self.result_text = f"J{st.idx}: ❌ INCORRECTO. Baja manos y presiona SIGUIENTE."
                                    self.phase = "PAUSE_NEXT"
                                    self.next_hover_start = None
                                    break
                            else:
                                st.hold_start_right = None

                    # Pause screen + Next button
                    if self.phase == "PAUSE_NEXT":
                        draw_text_with_shadow(frame, self.result_text, (max(20, int((w - 900) / 2)), 510),
                                              cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

                        next_enabled = not any_hand_raised
                        hover = False
                        in_next = False

                        if self.next_rect is not None:
                            for (ix, iy, conf) in index_points:
                                if conf < self.cfg.min_raise_confidence:
                                    continue
                                # requiere indice ABAJO de la linea (manos abajo)
                                if iy >= y_thr and self._rect_contains(self.next_rect, ix, iy):
                                    in_next = True
                                    break

                        if next_enabled and in_next:
                            hover = True
                            if self.next_hover_start is None:
                                self.next_hover_start = now
                            elif (now - self.next_hover_start) >= self.cfg.next_hold_seconds:
                                # load next question
                                self._load_next_question()
                                if self.phase != "END":
                                    self.phase = "PLAY"
                                self.next_hover_start = None
                        else:
                            self.next_hover_start = None

                        self._draw_next(frame, enabled=next_enabled, hover=hover)
                    else:
                        # show disabled Next (visual)
                        self._draw_next(frame, enabled=False, hover=False)

            # Show
            cv2.imshow("Quiz (Raise Hand + Next Index)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key in (ord("r"), ord("R")):
                # reset to mode selection
                self.mode = None
                self.topic = None
                self.phase = "SELECT_MODE"
                self.round_idx = 0
                self.q_index = 0
                self.current_q = None
                self.question_pool = []
                self.banner = "Elige modalidad"
                self.result_text = ""
                self.btn_hover_key = None
                self.btn_hover_start = None
                self.next_hover_start = None
                for st in self.states:
                    st.score = 0
                    st.attempts = 0
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
    ap.add_argument("--mirror", type=int, default=1, help="1=espejo visual, 0=sin espejo")
    ap.add_argument("--line", type=float, default=0.55, help="umbral muñeca (0-1). Menor = mano más arriba")
    ap.add_argument("--hold", type=float, default=0.20, help="segundos sosteniendo mano arriba para contar respuesta")
    ap.add_argument("--next-hold", type=float, default=0.25, help="segundos sosteniendo indice en SIGUIENTE")
    ap.add_argument("--time", type=float, default=15.0, help="tiempo en segundos para responder a cada pregunta")
    args = ap.parse_args()

    cfg = GameConfig(
        wrist_y_threshold_ratio=args.line,
        hold_seconds=args.hold,
        next_hold_seconds=args.next_hold,
        question_seconds=args.time,
    )

    app = QuizRaiseHandGame(
        cam=args.cam,
        width=args.width,
        height=args.height,
        players=max(1, args.players),
        mirror=bool(args.mirror),
        cfg=cfg,
    )
    app.run()


if __name__ == "__main__":
    main()