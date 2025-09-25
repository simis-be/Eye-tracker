# Eye-tracker 2 Final
Este es mi trabajo de fin de grado, un eye-tracker que incluye las siguientes librerias:
Kivy para la intefaz
pathlib
cv2
pyautogui
time
numpy

# Eye-Tracker Python

Este proyecto implementa un sistema de seguimiento ocular (eye-tracker) utilizando Python. Permite estimar la dirección de la mirada, detectar fijaciones y mostrar un mapa de fijaciones, con una interfaz gráfica interactiva para controlar parámetros de sensibilidad y velocidad.

## Características

- Captura de video en tiempo real desde una cámara web.
- Detección facial y ocular usando Haar Cascades.
- Estimación precisa del centro del iris.
- Calibración dinámica adaptativa para cada usuario.
- Estabilización de la mirada mediante filtrado híbrido y filtro de Kalman.
- Detección de fijaciones con registro de duración y posición.
- Interfaz gráfica con Kivy para controlar el sistema y ajustar parámetros.
- Persistencia de configuraciones entre sesiones mediante archivo JSON.

## Requisitos

- Python 3.8 o superior

Librerías necesarias:

```bash
pip install opencv-python numpy kivy

