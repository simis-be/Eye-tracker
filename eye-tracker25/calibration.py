import time
import numpy as np
import cv2
from collections import Counter

def show_calibration_point(screen_res, frame_res, screen_pos, duration=0.5):
    frame_w, frame_h = frame_res
    screen_w, screen_h = screen_res
    start = time.time()
    while time.time() - start < duration:
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        x = int(screen_pos[0] * frame_w / screen_w)
        y = int(screen_pos[1] * frame_h / screen_h)
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        cv2.namedWindow("Calibración", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibración", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Calibración", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def calibrate_gaze(cap, face_cascade, eye_cascade):
    screen_res = (1920, 1080)
    frame_w, frame_h = 800, 600
    
    # Nuevos puntos de calibración en una cuadrícula de 3x3
    points = {
        # Fila 1 (Superior)
        "P1_1": (96, 54), "P1_2": (480, 54), "P1_3": (960, 54), "P1_4": (1440, 54), "P1_5": (1824, 54),
        # Fila 2
        "P2_1": (96, 270), "P2_2": (480, 270), "P2_3": (960, 270), "P2_4": (1440, 270), "P2_5": (1824, 270),
        # Fila 3 (Central)
        "P3_1": (96, 540), "P3_2": (480, 540), "P3_3": (960, 540), "P3_4": (1440, 540), "P3_5": (1824, 540),
        # Fila 4
        "P4_1": (96, 810), "P4_2": (480, 810), "P4_3": (960, 810), "P4_4": (1440, 810), "P4_5": (1824, 810),
        # Fila 5 (Inferior)
        "P5_1": (96, 1026), "P5_2": (480, 1026), "P5_3": (960, 1026), "P5_4": (1440, 1026), "P5_5": (1824, 1026),
    }

    iris_data = {key: [] for key in points}

    for key, pos in points.items():
        print(f"Mirar: {key}")
        show_calibration_point(screen_res, (frame_w, frame_h), pos)

        start = time.time()
        while time.time() - start < 1:
            ret, frame = cap.read()
            if not ret: continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi)
                if len(eyes) > 0:
                    eye = min(eyes, key=lambda e: e[0])
                    ex, ey, ew, eh = eye
                    iris_x = x + ex + ew // 2
                    iris_y = y + ey + eh // 2
                    iris_data[key].append((iris_x, iris_y))
                break

    def avg(pts): return np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts]) # Average of points
    
    def get_mode(points):
        if not points:
            return (0, 0)
        rounded = [(int(x), int(y)) for x, y in points]
        counter = Counter(rounded)
        most_common_coords = [coord for coord, _ in counter.most_common(3)]
        xs = [p[0] for p in most_common_coords]
        ys = [p[1] for p in most_common_coords]
        return (np.mean(xs), np.mean(ys))
    center_iris= get_mode(iris_data["P3_3"])

    scales = {}
    # Ya no usas los cuadrantes Q1, Q2, etc., sino los nombres de los puntos
    for key, pos in points.items():
        qx, qy = get_mode(iris_data[key])
        dx = abs(qx - center_iris[0]) or 1e-5
        dy = abs(qy - center_iris[1]) or 1e-5
        sx = abs(pos[0] - points["P3_3"][0]) / dx if dx else 1
        sy = abs(pos[1] - points["P3_3"][1]) / dy if dy else 1
        
        scale_boost_x = 1.3  # Menos boost en X (más estable)
        scale_boost_y = 1.6  # Más boost en Y (los ojos se mueven menos verticalmente)
        scales[key] = (sx * scale_boost_x, sy * scale_boost_y)

    print("Calibración completada.")
    print(center_iris)

    return scales, center_iris
