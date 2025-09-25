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

def refine_iris_center(eye_img):
    eye_blur = retinal_filter(eye_img)
    eye_blur =cv2.medianBlur(eye_blur,5)
    circles = cv2.HoughCircles(
        eye_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=eye_img.shape[0]/4,
        param1=100,
        param2=13,
        minRadius=5,
        maxRadius=15
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return (circles[0][0][0], circles[0][0][1])  # x, y del círculo
    else:
        # Fallback con momentos
        _, thresh = cv2.threshold(eye_blur, 50, 255, cv2.THRESH_BINARY_INV)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def detect_iris_centers(frame, face_cascade, eye_cascade, preprocessed_frame=None):
    if preprocessed_frame is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preprocessed_frame = cv2.equalizeHist(gray)

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
            eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
            filtered_eye = retinal_filter(eye_roi)

            refined = refine_iris_center(filtered_eye)
            if refined:
                iris_x = x + ex + refined[0]
                iris_y = y + ey + refined[1]
                iris_centers.append((iris_x, iris_y))

                # Dibujo para depuración
                cv2.circle(frame, (iris_x, iris_y), 6, (0, 255, 255), 2)

        return iris_centers, face_center

    return [], None

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
