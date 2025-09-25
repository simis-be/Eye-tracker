import cv2
import numpy as np
import math


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

def compute_gaze_position(iris_center, face_center, screen_size):
    """
    Maps the detected iris center to screen coordinates.
    
    Args:
    - iris_center: (x, y) coordinates of the detected iris in the image.
    - face_center: (x, y) coordinates of the detected face center.
    - screen_size: (width, height) of the screen.
    
    Returns:
    - (gaze_x, gaze_y): The estimated screen position where the user is looking.
    """

    screen_width, screen_height = screen_size
    screen_center = (screen_width // 2, screen_height // 2)

    # Compute relative position of iris with respect to face center
    dx = iris_center[0] - face_center[0]
    dy = iris_center[1] - face_center[1]

    # Angle α: User’s head position relative to the screen center
    alpha = math.atan2(dy, dx)  # Angle in radians

    # Compute scaling factors for different screen quadrants
    scale_x = 1.5 if iris_center[0] > screen_center[0] else 1.0
    scale_y = 1.2 if iris_center[1] > screen_center[1] else 1.0

    # Compute gaze coordinates on the screen using projection
    gaze_x = screen_center[0] + scale_x * dx * np.tan(alpha)
    gaze_y = screen_center[1] + scale_y * dy * np.tan(alpha)

    # Ensure gaze coordinates stay within screen bounds
    gaze_x = np.clip(gaze_x, 0, screen_width)
    gaze_y = np.clip(gaze_y, 0, screen_height)

    return int(gaze_x), int(gaze_y)




def set_res(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))

# Asegúrate de cambiar el número de dispositivo de la cámara si no es 0
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

frame_w = 640
frame_h = 480
set_res(cap, frame_w, frame_h)

# Verificar si la cámara está disponible
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Verificar la resolución actual
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolución actual: {actual_width}x{actual_height}")

# Cargar el Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Error: No se pudo cargar el archivo Haar Cascade.")
    exit()

def luminance_compression(frame):
    return np.log1p(frame.astype(np.float32)) * (255 / np.log1p(255))

def compute_nflg(eye_region):
    gradient_x = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    normal_x, normal_y = -gradient_y, gradient_x  # Normal a la frontera del iris
    nflg = np.sum(gradient_magnitude * (normal_x + normal_y)) / (eye_region.shape[0] * eye_region.shape[1])
    return nflg

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break

    # Voltear el frame
    frame = cv2.flip(frame, 1)

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar compresión de luminancia
    gray_luminance = luminance_compression(gray).astype(np.uint8)
    
    # Aplicar filtro gaussiano
    gray_blur = cv2.GaussianBlur(gray_luminance, (15, 15), 2)
    
    # Contraste mejorado
    contrast_enhanced = cv2.addWeighted(gray_luminance, 1.5, gray_blur, -0.5, 0)
    
    # Pre-filtrado final
    pre_filtered = cv2.addWeighted(gray_luminance, 0.5, contrast_enhanced, 0.5, 0)
    
    cv2.imshow('Pre-filtered', pre_filtered)

    # Detectar caras
    faces = face_cascade.detectMultiScale(
        pre_filtered,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Dibujar un rectángulo alrededor de las caras detectadas y detectar ojos
    for (x, y, w, h) in faces:
        face_roi = pre_filtered[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi)
        
        for (ex, ey, ew, eh) in eyes:
            eye_region = face_roi[ey:ey+eh, ex:ex+ew]
            nflg_value = compute_nflg(eye_region)
            print(f"NFLG para el ojo detectado: {nflg_value}")
            
            # Dibujar rectángulo del ojo
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
            
            # Dibujar círculo del iris (asumiendo que el iris está centrado)
            iris_x = x + ex + ew // 2
            iris_y = y + ey + eh // 2
            iris_radius = ew // 4  # Tamaño aproximado del iris
            cv2.circle(frame, (iris_x, iris_y), iris_radius, (0, 255, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame resultante
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura cuando todo esté hecho
cap.release()
cv2.destroyAllWindows()




# Screen resolution (example values, update accordingly)
print('hola')