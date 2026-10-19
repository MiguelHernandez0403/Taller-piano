# Piano virtual con seguimiento de manos

La app utiliza el modelo preentrenado **MediaPipe Hand Landmarker** de Google,
incluido en `models/hand_landmarker.task`. Localiza 21 puntos de hasta dos manos
y corrige la etiqueta izquierda/derecha para la imagen reflejada de la cámara.
Al flexionar un dedo, mide los ángulos de sus articulaciones y reproduce la nota asignada. 

Modelo preentrenado oficial:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

## Preparación

Necesitas Python 3.10 o posterior, una cámara web y el modelo incluido en
`models/hand_landmarker.task`. Se usa `pygame-ce` para la versión de escritorio y `Flask` para la versión web local. Abre una
terminal en la carpeta del proyecto y ejecuta:

```powershell
python -m pip install -r requirements.txt
python -m pip install flask
```

## Modos de Ejecucion

### 1. Version de escritorio
```powershell
python piano_manos_yolo.py
```

### 2. Version web local
Para desplegar la aplicacion de manera local a traves de una pagina web:
```powershell
python piano_web.py
```
Luego abre tu navegador web y entra a `http://127.0.0.1:5000`.

Para cambiar la cámara, indicar otro modelo MediaPipe o ajustar la confianza:

```powershell
python piano_manos_yolo.py --camera 1
python piano_web.py --port 8080 --camera 0
```

## Notas asignadas

| Mano detectada en pantalla | Dedo | Nota |
| --- | --- | --- |
| Izquierda | Meñique, anular, medio, índice, pulgar | C4, D4, E4, F4, G4 |
| Derecha | Pulgar, índice, medio, anular, meñique | A4, B4, C5, D5, E5 |

La escala cruza ambas manos: el meñique derecho toca C4 y el meñique izquierdo
toca E5.

## Controles

- **Q** o **Esc** (en escritorio): cerrar la app.
- **M** (en escritorio) o el botón en la interfaz web: mostrar u ocultar los ángulos detectados en cada dedo; los puntos detectados se dibujan sobre la mano.
- El piano visual aparece en la parte inferior; las teclas se iluminan al tocar
  la nota y el encabezado muestra las notas que están sonando.

## Entrenamiento posterior

Primero comprueba la experiencia con el modelo preentrenado incluido. Luego se
puede comparar `best.pt` con MediaPipe sobre las mismas imágenes y movimientos,
preparar ejemplos específicos de las manos y condiciones de uso, y entrenar y
validar un modelo personalizado. No se cambia al modelo entrenado hasta medir
que mejora la detección y la activación de los dedos en ese uso.
