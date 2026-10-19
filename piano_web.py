import argparse
from pathlib import Path
import cv2
import mediapipe as np_mp # Just to be safe or avoid collision
import numpy as np
import threading
import time
from flask import Flask, render_template_string, Response

app = Flask(__name__)

DIRECTORIO_APP = Path(__file__).resolve().parent
MODELO_DEFAULT = DIRECTORIO_APP / "models" / "hand_landmarker.task"
CONFIANZA = 0.5
UMBRAL_PRESIONAR = 140.0
UMBRAL_SOLTAR = 160.0
UMBRAL_PULGAR = (135.0, 155.0)
PESO_ANGULO_ACTUAL = 0.6
MARCOS_PARA_REINICIAR = 8

DEDOS = {
    "pulgar": ((1, 2, 3), (2, 3, 4)),
    "indice": ((5, 6, 7), (6, 7, 8)),
    "medio": ((9, 10, 11), (10, 11, 12)),
    "anular": ((13, 14, 15), (14, 15, 16)),
    "menique": ((17, 18, 19), (18, 19, 20)),
}
ESQUELETO = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
)
NOTAS = {
    "Left": {
        "menique": "C4",
        "anular": "D4",
        "medio": "E4",
        "indice": "F4",
        "pulgar": "G4",
    },
    "Right": {
        "pulgar": "A4",
        "indice": "B4",
        "medio": "C5",
        "anular": "D5",
        "menique": "E5",
    },
}
FRECUENCIAS = {
    "C4": 261.63,
    "D4": 293.66,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
    "B4": 493.88,
    "C5": 523.25,
    "D5": 587.33,
    "E5": 659.25,
}

INVERTIR_LATERALIDAD_MEDIAPIPE = True

def ajustar_lateralidad(etiqueta):
    if not INVERTIR_LATERALIDAD_MEDIAPIPE:
        return etiqueta
    return {"Left": "Right", "Right": "Left"}.get(etiqueta, etiqueta)

def angulo_articular(a, b, c):
    lado_a = a - b
    lado_c = c - b
    norma = np.linalg.norm(lado_a) * np.linalg.norm(lado_c)
    if norma < 1e-8:
        return None
    coseno = np.dot(lado_a, lado_c) / norma
    return float(np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0))))

def medir_flexiones(puntos):
    flexiones = {}
    for dedo, articulaciones in DEDOS.items():
        angulos = [
            angulo_articular(*(puntos[indice] for indice in articulacion))
            for articulacion in articulaciones
        ]
        if all(angulo is not None for angulo in angulos):
            flexiones[dedo] = sum(angulos) / len(angulos)
    return flexiones

# Global state for web streaming and note triggers
frame_actual = None
lock = threading.Lock()
notas_activas_web = set()
estado_app = {"manos": 0, "depurar": False}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Piano Virtual con Seguimiento de Manos</title>
    <style>
        body {
            background-color: #121212;
            color: #ffffff;
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
        }
        h1 {
            margin-bottom: 5px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .video-box {
            margin: 20px 0;
            border: 3px solid #333;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        img {
            display: block;
            max-width: 100%;
            height: auto;
        }
        .controls {
            margin-top: 15px;
        }
        button {
            background-color: #1f6feb;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 5px;
        }
        button:hover {
            background-color: #388bfd;
        }
        .piano {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            background: #222;
            padding: 10px;
            border-radius: 8px;
        }
        .key {
            width: 50px;
            height: 150px;
            background: #fff;
            color: #000;
            margin: 0 2px;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            padding-bottom: 10px;
            font-weight: bold;
            border-radius: 0 0 4px 4px;
            box-shadow: inset 0 -5px 10px rgba(0,0,0,0.2);
            transition: background 0.1s;
        }
        .key.active {
            background: #4ec9b0 !important;
            color: #fff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Piano Virtual con Seguimiento de Manos</h1>
        <p>Flexiona tus dedos frente a la cámara para tocar notas musicales.</p>

        <div class="video-box">
            <img src="{{ url_for('video_feed') }}" alt="Video en vivo">
        </div>

        <div class="controls">
            <button onclick="toggleDebugMode()">Alternar Ángulos (Depuración)</button>
        </div>

        <div class="piano" id="piano">
            <!-- Teclas generadas por JS -->
        </div>
    </div>

    <script>
        const notas = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5"];
        const pianoEl = document.getElementById('piano');

        notas.forEach(nota => {
            const key = document.createElement('div');
            key.className = 'key';
            key.id = 'key-' + nota;
            key.innerText = nota;
            pianoEl.appendChild(key);
        });

        function toggleDebugMode() {
            fetch('/toggle_debug')
                .then(res => res.json())
                .then(data => console.log("Modo depuración:", data.depurar));
        }

        setInterval(() => {
            fetch('/notes')
                .then(res => res.json())
                .then(data => {
                    notas.forEach(nota => {
                        const keyEl = document.getElementById('key-' + nota);
                        if (data.active.includes(nota)) {
                            keyEl.classList.add('active');
                            playTone(nota);
                        } else {
                            keyEl.classList.remove('active');
                        }
                    });
                });
        }, 50);

        // Sintetizador Web Audio API local para la interfaz web
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const frecuencias = {
            "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00,
            "A4": 440.00, "B4": 493.88, "C5": 523.25, "D5": 587.33, "E5": 659.25
        };
        const activeTones = {};

        function playTone(nota) {
            if (activeTones[nota]) return;
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            const freq = frecuencias[nota];
            if (!freq) return;

            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();

            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, audioCtx.currentTime);

            gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 1.0);

            osc.connect(gain);
            gain.connect(audioCtx.destination);

            osc.start();
            osc.stop(audioCtx.currentTime + 1.0);

            activeTones[nota] = true;
            setTimeout(() => {
                delete activeTones[nota];
            }, 300);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    def generar():
        global frame_actual
        while True:
            with lock:
                if frame_actual is None:
                    time.sleep(0.03)
                    continue
                success, encoded = cv2.imencode('.jpg', frame_actual)
                if not success:
                    continue
                frame_bytes = encoded.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generar(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/notes')
def get_notes():
    global notas_activas_web
    return {"active": list(notas_activas_web)}

@app.route('/toggle_debug')
def toggle_debug():
    estado_app["depurar"] = not estado_app["depurar"]
    return {"depurar": estado_app["depurar"]}

def bucle_vision(model_path, camera_id, confidence):
    global frame_actual, notas_activas_web
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    captura = cv2.VideoCapture(camera_id)
    if not captura.isOpened():
        print(f"Error: No se pudo abrir la cámara {camera_id}")
        return

    opciones = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    detector = vision.HandLandmarker.create_from_options(opciones)

    presionados = {}
    suavizados = {}
    ausentes = {}
    notas_activas = {}
    indice_frame = 0
    ultimo_timestamp = -1
    fps = captura.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = max(int(indice_frame * 1000 / fps), ultimo_timestamp + 1)
            ultimo_timestamp = timestamp
            indice_frame += 1
            resultado = detector.detect_for_video(imagen, timestamp)

            manos_visibles = 0
            claves_visibles = set()
            depurar = estado_app["depurar"]

            if resultado.hand_landmarks:
                for landmarks, categorias in zip(resultado.hand_landmarks, resultado.handedness):
                    mano = ajustar_lateralidad(categorias[0].category_name)
                    if mano not in NOTAS or len(landmarks) != 21:
                        continue
                    puntos = np.array(
                        [(punto.x * ancho, punto.y * alto) for punto in landmarks],
                        dtype=np.float32,
                    )
                    flexiones = medir_flexiones(puntos)
                    manos_visibles += 1
                    claves_visibles.update((mano, dedo) for dedo in flexiones)

                    # Dibujar mano
                    pixeles = [(int(x), int(y)) for x, y in puntos]
                    etiqueta = "IZQUIERDA" if mano == "Left" else "DERECHA"
                    cv2.putText(frame, etiqueta, (pixeles[0][0] + 8, pixeles[0][1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    for inicio, fin in ESQUELETO:
                        cv2.line(frame, pixeles[inicio], pixeles[fin], (255, 150, 60), 2)
                    for punto in pixeles:
                        cv2.circle(frame, punto, 4, (0, 220, 255), -1)

                    for dedo, articulaciones in DEDOS.items():
                        if dedo not in flexiones:
                            continue
                        punta = articulaciones[-1][-1]
                        px, py = pixeles[punta]
                        nota = NOTAS[mano][dedo]
                        if notas_activas.get((mano, dedo), 0) > 0:
                            cv2.circle(frame, (px, py), 16, (40, 220, 40), -1)
                            cv2.putText(frame, nota, (px - 20, py - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 40), 2)
                        if depurar:
                            cv2.putText(frame, f"{flexiones[dedo]:.0f} deg", (px - 22, py + 24),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                    for dedo, angulo in flexiones.items():
                        clave = (mano, dedo)
                        anterior = suavizados.get(clave, angulo)
                        suavizados[clave] = (
                            (1.0 - PESO_ANGULO_ACTUAL) * anterior
                            + PESO_ANGULO_ACTUAL * angulo
                        )
                        umbral_presionar, umbral_soltar = (
                            UMBRAL_PULGAR
                            if dedo == "pulgar"
                            else (UMBRAL_PRESIONAR, UMBRAL_SOLTAR)
                        )
                        if (
                            not presionados.get(clave, False)
                            and suavizados[clave] < umbral_presionar
                        ):
                            presionados[clave] = True
                            notas_activas[clave] = 12
                        elif (
                            presionados.get(clave, False)
                            and suavizados[clave] > umbral_soltar
                        ):
                            presionados[clave] = False
                        ausentes[clave] = 0

            for clave in tuple(presionados):
                if clave not in claves_visibles:
                    ausentes[clave] = ausentes.get(clave, 0) + 1
                    if ausentes[clave] >= MARCOS_PARA_REINICIAR:
                        presionados[clave] = False
                        suavizados.pop(clave, None)
                if notas_activas.get(clave, 0) > 0:
                    notas_activas[clave] -= 1

            # Panel superior
            cv2.rectangle(frame, (0, 0), (ancho, 60), (25, 25, 25), -1)
            cv2.putText(frame, f"Manos detectadas: {manos_visibles}/2", (16, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "Accede desde el navegador en http://127.0.0.1:5000", (16, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 220, 255), 1)

            with lock:
                frame_actual = frame.copy()

            notas_activas_web = {
                NOTAS[mano][dedo]
                for mano, dedo in presionados
                if presionados[(mano, dedo)]
                or notas_activas.get((mano, dedo), 0) > 0
            }

    finally:
        detector.close()
        captura.release()

def analizar_argumentos():
    parser = argparse.ArgumentParser(
        description="Piano virtual web con MediaPipe Hand Landmarker."
    )
    parser.add_argument("--model-asset", type=Path, default=MODELO_DEFAULT)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=CONFIANZA)
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()

def main():
    args = analizar_argumentos()
    if not args.model_asset.is_file():
        raise FileNotFoundError(f"No se encontró el modelo en {args.model_asset}")

    hilo_vision = threading.Thread(target=bucle_vision, args=(args.model_asset, args.camera, args.confidence), daemon=True)
    hilo_vision.start()

    print(f"Servidor web iniciado. Abre tu navegador en: http://127.0.0.1:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
