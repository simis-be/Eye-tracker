import time
import numpy as np
import cv2
from collections import Counter

def show_calibration_point(screen_res, frame_res, screen_pos, duration=1):
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
    points = {
        "center": (960, 540),
        "top_left": (0, 0),
        "top_right": (1920, 0),
        "bottom_left": (0, 1080),
        "bottom_right": (1920, 1080),
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
    center_iris= get_mode(iris_data["center"])

    scales = {}
    for quadrant, key in zip(["Q1", "Q2", "Q3", "Q4"], ["top_left", "top_right", "bottom_left", "bottom_right"]):
        qx, qy = get_mode(iris_data[key])
        dx = abs(qx - center_iris[0]) or 1e-5 
        dy = abs(qy - center_iris[1]) or 1e-5  # evita división por cero, pero no lo falsea
        sx = abs(points[key][0] - points["center"][0]) / dx if dx else 1
        sy = abs(points[key][1] - points["center"][1]) / dy if dy else 1

        scale_boost = 1.8
        scales[quadrant] = (sx * scale_boost, sy * scale_boost)

    print("[✔] Calibración completada.")
    print(f"[{key}] dx={dx:.2f}, dy={dy:.2f} | sx={sx:.2f}, sy={sy:.2f}")
    print(f"iris_y={iris_y}, center_y={center_iris[1]}, dy={dy}")

    print(center_iris)

    return scales, center_iris
