import numpy as np
import cv2

def set_res(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def retinal_filter(gray):
    compressed = np.log1p(gray.astype(np.float32)) * (255 / np.log1p(255))
    blur = cv2.GaussianBlur(compressed, (15, 15), 2)
    contrast = cv2.addWeighted(compressed, 1.5, blur, -0.5, 0)
    return cv2.addWeighted(compressed, 0.5, contrast, 0.5, 0).astype(np.uint8)

def refine_iris_center_improved(eye_img):
    """Detección mejorada del centro del iris con múltiples métodos"""
    eye_blur = retinal_filter(eye_img)
    eye_blur = cv2.medianBlur(eye_blur, 5)
    
    # Método 1: HoughCircles (más preciso)
    circles = cv2.HoughCircles(
        eye_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=eye_img.shape[0]//4,
        param1=80,
        param2=15,
        minRadius=3,
        maxRadius=20
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return (circles[0][0][0], circles[0][0][1])
    
    # Método 2: Detección de pupila (área más oscura)
    # Aplicar threshold para encontrar las áreas más oscuras
    _, thresh = cv2.threshold(eye_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)  # Invertir para que pupila sea blanca
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filtrar contornos por área y circularidad
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 400:  # Área razonable para pupila
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Suficientemente circular
                        valid_contours.append((contour, area))
        
        if valid_contours:
            # Tomar el contorno con área más apropiada (no necesariamente el más grande)
            valid_contours.sort(key=lambda x: abs(x[1] - 50))  # Preferir área cercana a 50px
            best_contour = valid_contours[0][0]
            
            # Calcular centroide
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    
    return None

def detect_individual_eyes(frame, face_cascade, eye_cascade):
    """
    Detecta cada ojo por separado y retorna la información individual
    incluyendo el estado de parpadeo.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocessed_frame = retinal_filter(gray)
    
    is_blinking = False
    iris_data = {}
    face_center = None
    
    faces = face_cascade.detectMultiScale(preprocessed_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_center = (x + w // 2, y + h // 2)
        face_roi = preprocessed_frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20),
            maxSize=(80, 80)
        )
        
        if len(eyes) < 2:
            is_blinking = True # No se detectan ambos ojos, podría ser un parpadeo
        
        eyes = sorted(eyes, key=lambda e: e[0])
        
        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
            eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
            
            # Aproximación simple de detección de parpadeo
            if eh / (ew + 1e-5) < 0.3: # Si la altura del ojo es muy pequeña en relación a su anchura
                is_blinking = True
            
            refined = refine_iris_center_improved(eye_roi)
            if refined:
                iris_x = x + ex + refined[0]
                iris_y = y + ey + refined[1]
                eye_label = "left" if i == 0 else "right"
                iris_data[eye_label] = (iris_x, iris_y)
    
    return face_center, iris_data, is_blinking

def calculate_gaze_with_individual_eyes(eye_data, face_center, screen_size):
    """
    Calcula la dirección de la mirada usando información individual de cada ojo
    para mejorar la precisión vertical
    """
    if len(eye_data) < 2:
        return None
    
    left_eye = eye_data.get('left')
    right_eye = eye_data.get('right')
    
    if not left_eye or not right_eye:
        return None
    
    # Calcular movimiento relativo de cada ojo dentro de su socket
    left_rel = left_eye['relative_pos']
    right_rel = right_eye['relative_pos']
    left_rect = left_eye['eye_rect']
    right_rect = right_eye['eye_rect']
    
    # Normalizar posiciones relativas (-1 a 1)
    left_norm_x = (left_rel[0] - left_rect[2]/2) / (left_rect[2]/2)
    left_norm_y = (left_rel[1] - left_rect[3]/2) / (left_rect[3]/2)
    
    right_norm_x = (right_rel[0] - right_rect[2]/2) / (right_rect[2]/2)
    right_norm_y = (right_rel[1] - right_rect[3]/2) / (right_rect[3]/2)
    
    # Promedio ponderado (dar más peso al ojo con detección más confiable)
    # Por ahora, promedio simple, pero se puede mejorar
    avg_norm_x = (left_norm_x + right_norm_x) / 2
    avg_norm_y = (left_norm_y + right_norm_y) / 2
    
    # Convertir a coordenadas de pantalla
    screen_w, screen_h = screen_size
    gaze_x = int(screen_w/2 + avg_norm_x * screen_w/2)
    gaze_y = int(screen_h/2 + avg_norm_y * screen_h/2)
    
    # Aplicar límites
    gaze_x = max(0, min(screen_w-1, gaze_x))
    gaze_y = max(0, min(screen_h-1, gaze_y))
    
    return (gaze_x, gaze_y)

def weighted_average_iris(eye_data, weights=None):
    """
    Calcula un promedio ponderado de los iris basado en confianza de detección
    """
    if len(eye_data) < 2:
        return None
    
    if weights is None:
        weights = {'left': 0.5, 'right': 0.5}
    
    total_x = 0
    total_y = 0
    total_weight = 0
    
    for eye_label, data in eye_data.items():
        if eye_label in weights:
            iris_center = data['iris_center']
            weight = weights[eye_label]
            
            total_x += iris_center[0] * weight
            total_y += iris_center[1] * weight
            total_weight += weight
    
    if total_weight > 0:
        return (int(total_x / total_weight), int(total_y / total_weight))
    
    return None

# Función de compatibilidad con tu código existente
def detect_iris_only_improved(frame, face_cascade, eye_cascade):
    """
    Versión mejorada que mantiene compatibilidad con tu código actual
    pero mejora la detección vertical
    """
    eye_data, face_center = detect_individual_eyes(frame, face_cascade, eye_cascade)
    
    if not eye_data:
        return [], face_center
    
    # Extraer solo los centros de iris para compatibilidad
    iris_centers = []
    for eye_label in ['left', 'right']:
        if eye_label in eye_data:
            iris_centers.append(eye_data[eye_label]['iris_center'])
    
    return iris_centers, face_center

def detect_iris_only(frame, face_cascade, eye_cascade):
    """
    Detecta los centros del iris en el frame dado, sin dibujar ni realizar calibración.
    Retorna los centros de los iris detectados y el centro de la cara.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocessed_frame = retinal_filter(gray)

    faces = face_cascade.detectMultiScale(preprocessed_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        face_roi = preprocessed_frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20),
            maxSize=(80, 80)
        )

        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        iris_centers = []

        for (ex, ey, ew, eh) in eyes:
            # Estimación directa del centro del ojo
            iris_x = x + ex + ew // 2
            iris_y = y + ey + eh // 2
            iris_centers.append((iris_x, iris_y))
            cv2.circle(frame, (iris_x, iris_y), 6, (0, 255, 255), 2)
            



        return iris_centers, face_center

    return [], None
