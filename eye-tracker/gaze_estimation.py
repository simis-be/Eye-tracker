import math
import numpy as np

def project_gaze_trig(iris, face, screen, CH=50.0, alpha_deg=0.0, theta_deg=10.0, sensitivity=1000.0):
    dx = (iris[0] - face[0]) * sensitivity
    dy = (iris[1] - face[1]) * sensitivity

    alpha = math.radians(alpha_deg)
    theta_x = math.radians(theta_deg)
    theta_y = math.radians(theta_deg - 2)

    hx = CH * math.tan(alpha + theta_x)
    hy = CH * math.tan(alpha + theta_y)

    cx, cy = screen[0] // 2, screen[1] // 2
    gx = int(np.clip(cx + hx + dx, 0, screen[0]))
    gy = int(np.clip(cy + hy + dy, 0, screen[1]))
    return gx, gy

def project_gaze_trig2(iris, face, screen, CH=100.0, alpha_deg=5.0, sensitivity=1.0, scales=None, center_iris=None):
    dx = iris[0] - face[0]
    dy = iris[1] - face[1]

    # Determinar cuadrante si scales y center_iris están disponibles
    if scales is not None and center_iris is not None:
        if iris[0] < center_iris[0] and iris[1] < center_iris[1]:
            quadrant = 'Q1'
        elif iris[0] >= center_iris[0] and iris[1] < center_iris[1]:
            quadrant = 'Q2'
        elif iris[0] < center_iris[0] and iris[1] >= center_iris[1]:
            quadrant = 'Q3'
        else:
            quadrant = 'Q4'

        sx, sy = scales.get(quadrant, (1, 1))
    else:
        sx, sy = 1, 1  # Escala por defecto si no se proporciona

    # Aplicar sensibilidad y escalado
    dx_scaled = dx * sensitivity * sx
    dy_scaled = dy * sensitivity * sy

    # Proyección trigonométrica
    alpha = math.radians(alpha_deg)
    theta_x = math.atan(dx_scaled / CH)
    theta_y = math.atan(dy_scaled / CH)

    hx = CH * math.tan(alpha + theta_x)
    hy = CH * math.tan(alpha + theta_y)

    cx, cy = screen[0] // 2, screen[1] // 2
    gx = int(np.clip(cx + hx , 0, screen[0]))
    gy = int(np.clip(cy + hy , 0, screen[1]))

    return gx, gy



def project_gaze_trig3(iris, face, screen, CH=60.0, alpha_deg=10.0, 
                               pixel_to_mm=0.25, sensitivity=1.0):
   
    dx_mm = (iris[0] - face[0]) * pixel_to_mm * sensitivity 
    dy_mm = (iris[1] - face[1]) * pixel_to_mm * sensitivity 
    
    # Calcular ángulos
    alpha = math.radians(alpha_deg)
    theta_x = math.atan(dx_mm / CH)
    theta_y = math.atan(dy_mm / CH)
    
    # Proyección geométrica
    hx = CH * math.tan(alpha + theta_x)
    hy = CH * math.tan(alpha + theta_y)
    
    # Convertir de vuelta a píxeles de pantalla
    cx, cy = screen[0] // 2, screen[1] // 2
    gx = int(np.clip(cx + hx/pixel_to_mm, 0, screen[0]))
    gy = int(np.clip(cy + hy/pixel_to_mm, 0, screen[1]))
    
    return gx, gy


def smooth_gaze(history):
    return (
        int(sum(x for x, _ in history) / len(history)),
        int(sum(y for _, y in history) / len(history))
    )
