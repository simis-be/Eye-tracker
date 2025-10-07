import math
import numpy as np


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