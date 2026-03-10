# NextCode Trivia (Club Hand Quiz)

Un juego interactivo de trivia multijugador que utiliza la cámara web y visión por computadora (MediaPipe) para detectar los movimientos de las manos. Los jugadores responden a las preguntas levantando la mano izquierda o derecha y navegan por el menú utilizando su dedo índice como puntero.

## Requisitos Previos

- **Python 3.11.x** (Versión requerida y especificada en `pyproject.toml`).
- Una cámara web conectada y funcional.

## Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto en tu entorno local tras haberlo clonado:

1. **Abrir una terminal** y navegar a la carpeta del proyecto.

2. **Crear y activar un entorno virtual** (recomendado para aislar las dependencias):

   ```bash
   py -3.11 -m venv venv
   # Activar en Windows
   .\venv\Scripts\activate
   ```

3. **Instalar las dependencias**:
   El proyecto utiliza un archivo `requirements.txt` pre-generado que incluye librerías fundamentales como `mediapipe`, `opencv-python` y `numpy`.
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución del Proyecto

Para iniciar el juego con las configuraciones por defecto (Cámara principal, resolución 1280x720, 2 jugadores), simplemente ejecuta este comando:

```bash
python raise_hand_game_multi.py
```

### Argumentos Opcionales por Consola

Puedes personalizar la ejecución del juego con banderas (flags) adiccionales:

- `--cam`: Índice de la cámara web (por defecto `0`). Si no detecta tu cámara, prueba con `1`.
- `--width`: Ancho de la ventana de la cámara (por defecto `1280`).
- `--height`: Alto de la ventana de la cámara (por defecto `720`).
- `--players`: Número de jugadores simultáneos dividiendo la pantalla en carriles (por defecto `2`).
- `--mirror`: Efecto espejo visual (`1` para activar, `0` para desactivar; por defecto `1`).
- `--line`: Altura de la línea umbral para detectar si la mano se ha levantado, calculada como proporción del alto de la pantalla (valor de `0.0` a `1.0`, por defecto `0.55`). Menor valor = tienes que alzar la mano más arriba.
- `--hold`: Tiempo en segundos que debes mantener la mano alzada para que actúe como respuesta válida (por defecto `0.20`).
- `--next-hold`: Tiempo en segundos que debes mantener el dedo índice sobre el botón "Siguiente" o en los menús para pulsarlo (por defecto `0.25`).

**Ejemplo de ejecución personalizada para 1 jugador y cambiando la duración para responder:**

```bash
python raise_hand_game_multi.py --players 1 --hold 0.50
```

## ¿Cómo Funciona y Cómo se Juega?

1. **Selección de Modo y Tema**: Utilizando tu dedo **índice**, debes apuntar a los botones virtuales en la pantalla. Mantén la punta del dedo sobre el botón por un cuarto de segundo. Podrás elegir entre un **Modo de 10 preguntas** o un **Modo por tiempo de 2 minutos**, y seguidamente el tema de las preguntas (Python, Desarrollo Web o IA).
2. **Sistema de Respuestas**: Durante el juego, la pantalla se divide verticalmente en secciones (carriles) equivalentes al número de jugadores. Aparecerá una pregunta en el centro con una opción a la Izquierda (IZQ) y otra a la Derecha (DER).
3. **Levantar la Mano**: Para elegir una respuesta, debes levantar físicamente tu **mano izquierda** (corresponde a la opción Izquierda) o tu **mano derecha** (corresponde a la opción Derecha) **por encima de la línea gris horizontal** que se muestra. Un modelo de IA con `mediapipe` detectará tanto las coordenadas de la muñeca (para ver si pasaste la línea o umbral) como si la mano pertenece al lado derecho o izquierdo de tu cuerpo.
4. **Validación**: Mantén la mano alzada momentáneamente y el juego validará automáticamente la respuesta, otorgando puntos si ha sido correcta.
5. **Avanzar a la Siguiente**: Posterior a saber si tu respuesta fue correcta o si el tiempo terminó, debes bajar las manos por debajo de la línea y usar un dedo índice para dar clic al botón inferior que dice "SIGUIENTE", pasando a la próxima ronda.
