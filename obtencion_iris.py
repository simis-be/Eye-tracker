import cv2
import numpy as np
import math
from collections import deque

# Simulación de escalas calibradas por cuadrante (esto se ajustaría en calibración real)


def set_res(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
frame_w, frame_h = 800, 600
set_res(cap, frame_w, frame_h)

import time
time.sleep(1)

# Warm-up de la cámara
for _ in range(10):
    cap.read()

print("Resolución real:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Error: No se pudo cargar el archivo Haar Cascade.")
    exit()

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

fixation_buffer = deque(maxlen=6)  # Últimas 6 posiciones de la mirada
fixation_map = []  # Posiciones fijas detectadas







def retinal_filter(gray):
    compressed = np.log1p(gray.astype(np.float32)) * (255 / np.log1p(255))
    blur = cv2.GaussianBlur(compressed, (15, 15), 2)
    contrast = cv2.addWeighted(compressed, 1.5, blur, -0.5, 0)
    return cv2.addWeighted(compressed, 0.5, contrast, 0.5, 0).astype(np.uint8)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = frame.shape[:2]

    # Preprocesamiento
    
    pre_filtered = retinal_filter(gray)

    cv2.imshow('Pre-filtered', pre_filtered)

    faces = face_cascade.detectMultiScale(pre_filtered, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        face_roi = pre_filtered[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20),
            maxSize=(80, 80)
        )

        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

        projected_points = []
        projected_points2 = []

        for (ex, ey, ew, eh) in eyes:
            iris_x = x + ex + ew // 2
            iris_y = y + ey + eh // 2
            iris_center = (iris_x, iris_y)

            cv2.circle(frame, iris_center, 6, (0, 255, 255), 2)  # Círculo vacío

            

           

        


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



