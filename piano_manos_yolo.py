"""Piano virtual con el modelo oficial MediaPipe Hand Landmarker.

Controles: Q o Esc para salir; M para mostrar u ocultar los ángulos.
Las manos se etiquetan con la clasificacion de lateralidad de MediaPipe.
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DIRECTORIO_APP = Path(__file__).resolve().parent
MODELO_DEFAULT = DIRECTORIO_APP / "models" / "hand_landmarker.task"
CONFIANZA = 0.5
UMBRAL_PRESIONAR = 140.0
UMBRAL_SOLTAR = 160.0
UMBRAL_PULGAR = (135.0, 155.0)
PESO_ANGULO_ACTUAL = 0.6
MARCOS_PARA_REINICIAR = 8

# Tripletas para medir las articulaciones PIP/DIP; el pulgar usa MCP/IP.
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
    """Corrige las etiquetas del espejo para asignarlas a la mano anatómica."""
    if not INVERTIR_LATERALIDAD_MEDIAPIPE:
        return etiqueta
    return {"Left": "Right", "Right": "Left"}.get(etiqueta, etiqueta)


def angulo_articular(a, b, c):
    """Calcula el ángulo en b, en grados, evitando errores numéricos."""
    lado_a = a - b
    lado_c = c - b
    norma = np.linalg.norm(lado_a) * np.linalg.norm(lado_c)
    if norma < 1e-8:
        return None
    coseno = np.dot(lado_a, lado_c) / norma
    return float(np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0))))


def medir_flexiones(puntos):
    """Mide los ángulos de las articulaciones con landmarks en coordenadas de píxel."""
    flexiones = {}
    for dedo, articulaciones in DEDOS.items():
        angulos = [
            angulo_articular(*(puntos[indice] for indice in articulacion))
            for articulacion in articulaciones
        ]
        if all(angulo is not None for angulo in angulos):
            flexiones[dedo] = sum(angulos) / len(angulos)
    return flexiones


def analizar_argumentos():
    parser = argparse.ArgumentParser(
        description="Toca notas de piano flexionando los dedos frente a la cámara."
    )
    parser.add_argument(
        "--model-asset",
        type=Path,
        default=MODELO_DEFAULT,
        help="Ruta al modelo .task de MediaPipe (incluido en models/).",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Índice de la cámara (por defecto: 0)."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIANZA,
        help="Confianza mínima de detección y presencia de la mano (0 a 1).",
    )
    argumentos = parser.parse_args()
    if not 0 <= argumentos.confidence <= 1:
        parser.error("--confidence debe estar entre 0 y 1.")
    return argumentos


def crear_nota(frecuencia, duracion=1.5):
    """Genera una nota sintetizada con armónicos y decaimiento tipo piano."""
    frecuencia_mixer, _, canales = pygame.mixer.get_init()
    tiempo = np.arange(int(frecuencia_mixer * duracion)) / frecuencia_mixer
    onda = np.zeros_like(tiempo)
    for armonico, amplitud in ((1, 1.0), (2, 0.48), (3, 0.22), (4, 0.10), (5, 0.04)):
        decaimiento = np.exp(-(2.8 + armonico * 0.7) * tiempo)
        onda += amplitud * np.sin(2 * np.pi * armonico * frecuencia * tiempo) * decaimiento

    ataque = np.minimum(1.0, tiempo / 0.008)
    onda *= ataque
    pico = np.max(np.abs(onda))
    if pico > 0:
        onda = onda / pico * 0.65
    datos = (onda * 32767).astype(np.int16)
    if canales == 2:
        datos = np.column_stack((datos, datos))
    return pygame.sndarray.make_sound(datos)


def dibujar_mano(frame, puntos, mano, sonidos_activos, flexiones, depurar):
    """Dibuja los puntos de la mano y el ángulo medido de cada dedo."""
    pixeles = [(int(x), int(y)) for x, y in puntos]
    etiqueta = "IZQUIERDA" if mano == "Left" else "DERECHA"
    cv2.putText(
        frame,
        etiqueta,
        (pixeles[0][0] + 8, pixeles[0][1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
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
        if sonidos_activos.get((mano, dedo), 0) > 0:
            cv2.circle(frame, (px, py), 16, (40, 220, 40), -1)
            cv2.putText(
                frame, nota, (px - 20, py - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 40), 2,
            )
        if depurar:
            cv2.putText(
                frame, f"{flexiones[dedo]:.0f} deg", (px - 22, py + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1,
            )


def dibujar_panel(frame, manos_visibles):
    _, ancho = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (ancho, 92), (25, 25, 25), -1)
    estado = f"Manos detectadas: {manos_visibles}/2"
    cv2.putText(
        frame, estado, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )
    cv2.putText(
        frame, "Q/Esc: salir   M: angulos",
        (16, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 220, 255), 1,
    )
    cv2.putText(
        frame, "IZQ: C4 D4 E4 F4 G4     DER: A4 B4 C5 D5 E5",
        (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 255, 190), 1,
    )
    cv2.putText(
        frame, "Puntos MediaPipe (amarillo); dedos doblados disparan una nota",
        (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
    )


def dibujar_teclado(frame, notas_resaltadas):
    """Dibuja las diez teclas del piano y resalta las notas recién tocadas."""
    alto, ancho = frame.shape[:2]
    notas = tuple(FRECUENCIAS)
    panel_alto = min(104, max(72, alto // 4))
    panel_y = alto - panel_alto
    cv2.rectangle(frame, (0, panel_y), (ancho, alto), (25, 25, 25), -1)

    if notas_resaltadas:
        estado = "Sonando: " + " + ".join(nota for nota in notas if nota in notas_resaltadas)
    else:
        estado = "Piano: flexiona un dedo para tocar"
    cv2.putText(
        frame, estado, (12, panel_y + 17),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 245, 255), 1,
    )

    margen = 10
    area_ancho = max(1, ancho - 2 * margen)
    area_alto = max(1, alto - panel_y - 25)
    for indice, nota in enumerate(notas):
        x1 = margen + round(indice * area_ancho / len(notas))
        x2 = margen + round((indice + 1) * area_ancho / len(notas))
        color = (80, 190, 90) if nota in notas_resaltadas else (245, 245, 245)
        cv2.rectangle(
            frame, (x1, panel_y + 23), (x2, panel_y + 23 + area_alto),
            color, -1,
        )
        cv2.rectangle(
            frame, (x1, panel_y + 23), (x2, panel_y + 23 + area_alto),
            (35, 35, 35), 1,
        )
        texto = nota
        escala = 0.48 if x2 - x1 >= 42 else 0.38
        (texto_ancho, _), _ = cv2.getTextSize(
            texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 1
        )
        texto_x = x1 + max(1, (x2 - x1 - texto_ancho) // 2)
        texto_y = panel_y + 23 + area_alto - 7
        cv2.putText(
            frame, texto, (texto_x, texto_y),
            cv2.FONT_HERSHEY_SIMPLEX, escala,
            (20, 70, 20) if nota in notas_resaltadas else (30, 30, 30), 1,
        )


def ejecutar(argumentos):
    if not argumentos.model_asset.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo: {argumentos.model_asset}. "
            "Coloca hand_landmarker.task en models/ o usa --model-asset RUTA."
        )

    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(16)
    sonidos = {nota: crear_nota(frecuencia) for nota, frecuencia in FRECUENCIAS.items()}

    captura = None
    try:
        captura = cv2.VideoCapture(argumentos.camera)
        if not captura.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara {argumentos.camera}. "
                "Comprueba que esté conectada o prueba con --camera 1."
            )

        opciones = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(argumentos.model_asset)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=argumentos.confidence,
            min_hand_presence_confidence=argumentos.confidence,
            min_tracking_confidence=argumentos.confidence,
        )
        detector = vision.HandLandmarker.create_from_options(opciones)
        try:
            presionados = {}
            suavizados = {}
            ausentes = {}
            notas_activas = {}
            depurar = False
            indice_frame = 0
            ultimo_timestamp = -1
            fps = captura.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0

            while True:
                ok, frame = captura.read()
                if not ok:
                    raise RuntimeError("La cámara dejó de entregar imágenes.")
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
                for landmarks, categorias in zip(
                    resultado.hand_landmarks, resultado.handedness
                ):
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
                    dibujar_mano(
                        frame, puntos, mano, notas_activas, flexiones, depurar
                    )

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
                            sonidos[NOTAS[mano][dedo]].play()
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

                dibujar_panel(frame, manos_visibles)
                notas_resaltadas = {
                    NOTAS[mano][dedo]
                    for mano, dedo in presionados
                    if presionados[(mano, dedo)]
                    or notas_activas.get((mano, dedo), 0) > 0
                }
                dibujar_teclado(frame, notas_resaltadas)
                cv2.imshow("Piano virtual de manos - MediaPipe", frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord("m"):
                    depurar = not depurar
        finally:
            detector.close()
    finally:
        if captura is not None:
            captura.release()
        cv2.destroyAllWindows()
        pygame.quit()


def main():
    ejecutar(analizar_argumentos())


if __name__ == "__main__":
    main()
