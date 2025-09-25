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

import time

def show_calibration_point(win_name, screen_pos, duration=1):
    """Muestra un punto en la ventana durante `duration` segundos en `screen_pos`"""
    start = time.time()
    while time.time() - start < duration:
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        x = int(screen_pos[0] * frame_w / SCREEN_WIDTH)
        y = int(screen_pos[1] * frame_h / SCREEN_HEIGHT)
        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def calibrate_gaze(cap):
    print("Iniciando calibracion... Mire cada punto cuando aparezca.")
    points = {
        "center": (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
        "top_left": (0, 0),
        "top_right": (SCREEN_WIDTH, 0),
        "bottom_left": (0, SCREEN_HEIGHT),
        "bottom_right": (SCREEN_WIDTH, SCREEN_HEIGHT)
    }

    iris_data = {key: [] for key in points}

    for key, screen_pos in points.items():
        print(f"Mire al punto: {key}")
        show_calibration_point('Calibración', screen_pos)

        start = time.time()
        while time.time() - start < 1:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_center = (x + w // 2, y + h // 2)
                face_roi = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_roi)
                for (ex, ey, ew, eh) in eyes:
                    iris_x = x + ex + ew // 2
                    iris_y = y + ey + eh // 2
                    iris_data[key].append((iris_x, iris_y))
                    break
                break

    # Calcular escala por cuadrante simulando diferencia entre iris_center y face_center
    def avg_pos(pos_list):
        x_vals = [p[0] for p in pos_list]
        y_vals = [p[1] for p in pos_list]
        return np.mean(x_vals), np.mean(y_vals)

    center_iris = avg_pos(iris_data['center'])

    scales = {
        'Q1': (1.0, 1.0),
        'Q2': (1.0, 1.0),
        'Q3': (1.0, 1.0),
        'Q4': (1.0, 1.0),
    }

    # Escala relativa = (iris_movimiento_en_x / pantalla_movimiento_en_x), mismo para Y
    for quadrant, point_key in zip(['Q1', 'Q2', 'Q3', 'Q4'], ['top_left', 'top_right', 'bottom_left', 'bottom_right']):
        q_iris = avg_pos(iris_data[point_key])
        dx = abs(q_iris[0] - center_iris[0])
        dy = abs(q_iris[1] - center_iris[1])
        dx_screen = abs(points[point_key][0] - points['center'][0])
        dy_screen = abs(points[point_key][1] - points['center'][1])
        scale_x = dx_screen / dx if dx != 0 else 1.0
        scale_y = dy_screen / dy if dy != 0 else 1.0
        scales[quadrant] = (scale_x, scale_y)

    print("Calibracion completada. Escalas por cuadrante:")
    for q, s in scales.items():
        print(f"{q}: X={s[0]:.2f}, Y={s[1]:.2f}")

    return scales


def luminance_compression(frame):
    return np.log1p(frame.astype(np.float32)) * (255 / np.log1p(255))

def compute_gaze_position_linear(iris_center, eye_reference, screen_size, scale_x=5.0, scale_y=5.0):
    dx = iris_center[0] - eye_reference[0]
    dy = iris_center[1] - eye_reference[1]

    screen_center = (screen_size[0] // 2, screen_size[1] // 2)

    gaze_x = screen_center[0] + dx * scale_x
    gaze_y = screen_center[1] + dy * scale_y

    return int(np.clip(gaze_x, 0, screen_size[0])), int(np.clip(gaze_y, 0, screen_size[1]))


def compute_gaze_position_calibrated(iris_center, face_center, screen_size,scales):
    dx = iris_center[0] - face_center[0]
    dy = iris_center[1] - face_center[1]

    screen_width, screen_height = screen_size
    screen_center = (screen_width // 2, screen_height // 2)

    # Determinar cuadrante del iris
    if iris_center[0] < face_center[0] and iris_center[1] < face_center[1]:
        scale_x, scale_y = scales['Q1']
    elif iris_center[0] >= face_center[0] and iris_center[1] < face_center[1]:
        scale_x, scale_y = scales['Q2']
    elif iris_center[0] < face_center[0] and iris_center[1] >= face_center[1]:
        scale_x, scale_y = scales['Q3']
    else:
        scale_x, scale_y = scales['Q4']

    gaze_x = screen_center[0] + scale_x * dx
    gaze_y = screen_center[1] + scale_y * dy

    return int(np.clip(gaze_x, 0, screen_width)), int(np.clip(gaze_y, 0, screen_height))

# función para actualizar el mapa de fijación
# esta función se llama cada vez que se detecta una nueva posición de mirada
def update_fixation_map(gaze_position):
    fixation_buffer.append(gaze_position)
    if len(fixation_buffer) == fixation_buffer.maxlen:
        xs, ys = zip(*fixation_buffer)
        if np.std(xs) < 10 and np.std(ys) < 10:
            fixation_map.append(gaze_position)

def compute_gaze_position(iris_center, face_center, screen_size, ch=30.0):
    screen_width, screen_height = screen_size
    dx = iris_center[0] - face_center[0]
    dy = iris_center[1] - face_center[1]

    alpha = math.atan2(dy, dx)  # Ángulo de desplazamiento del iris
    theta_x = 0.15  # Ajuste angular experimental (puede ser calibrado)
    theta_y = 0.12

    hx = ch * math.tan(alpha + theta_x)
    hy = ch * math.tan(alpha + theta_y)

    gaze_x = (screen_width // 2) + hx
    gaze_y = (screen_height // 2) + hy

    return int(np.clip(gaze_x, 0, screen_width)), int(np.clip(gaze_y, 0, screen_height))

def compute_gaze_position_hybrid(iris_center, face_center, screen_size, scales, ch=30.0):
    screen_width, screen_height = screen_size
    screen_center = (screen_width // 2, screen_height // 2)

    dx = iris_center[0] - face_center[0]
    dy = iris_center[1] - face_center[1]

    # Escalas calibradas por cuadrante
    if iris_center[0] < face_center[0] and iris_center[1] < face_center[1]:
        scale_x, scale_y = scales['Q1']
    elif iris_center[0] >= face_center[0] and iris_center[1] < face_center[1]:
        scale_x, scale_y = scales['Q2']
    elif iris_center[0] < face_center[0] and iris_center[1] >= face_center[1]:
        scale_x, scale_y = scales['Q3']
    else:
        scale_x, scale_y = scales['Q4']

    # Cálculo angular
    alpha = math.atan2(dy, dx)
    theta_x = 0.15  # Ajuste angular experimental (puede ser calibrado)
    theta_y = 0.12

    hx = ch * math.tan(alpha + theta_x)
    hy = ch * math.tan(alpha + theta_y)

    # Combinar desplazamiento lineal con escala calibrada
    gaze_x = screen_center[0] + scale_x * (dx + hx)
    gaze_y = screen_center[1] + scale_y * (dy + hy)

    return int(np.clip(gaze_x, 0, screen_width)), int(np.clip(gaze_y, 0, screen_height))

#
scales = calibrate_gaze(cap)  # Calibrar la mirada al inicio


gaze_history = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = frame.shape[:2]

    # Preprocesamiento
    gray_luminance = luminance_compression(gray).astype(np.uint8)
    gray_blur = cv2.GaussianBlur(gray_luminance, (15, 15), 2)
    contrast_enhanced = cv2.addWeighted(gray_luminance, 1.5, gray_blur, -0.5, 0)
    pre_filtered = cv2.addWeighted(gray_luminance, 0.5, contrast_enhanced, 0.5, 0)

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

            # Calcular posición de mirada
            gaze_position = compute_gaze_position_hybrid(iris_center, face_center, (SCREEN_WIDTH, SCREEN_HEIGHT), scales)
            # Suavizar con media móvil
            gaze_history.append(gaze_position)
            avg_x = int(sum(p[0] for p in gaze_history) / len(gaze_history))
            avg_y = int(sum(p[1] for p in gaze_history) / len(gaze_history))
            smoothed_gaze = (avg_x, avg_y)

            # Dibujar punto de mirada
            frame_gaze_x = int(avg_x * frame_w / SCREEN_WIDTH)
            frame_gaze_y = int(avg_y * frame_h / SCREEN_HEIGHT)
            cv2.circle(frame, (frame_gaze_x, frame_gaze_y), 8, (255, 140, 0), -1)

            # Actualizar mapa de fijación con posición suavizada
            update_fixation_map(smoothed_gaze)

           

        

        # Dibujar regiones de fijación
        for pos in fixation_map:
            fx = int(pos[0] * frame_w / SCREEN_WIDTH)
            fy = int(pos[1] * frame_h / SCREEN_HEIGHT)
            cv2.circle(frame, (fx, fy), 3, (0, 100, 255), -1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
