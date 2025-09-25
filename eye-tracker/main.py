import cv2
from calibration import calibrate_gaze
from gaze_estimation import project_gaze_trig2, smooth_gaze
from fixation_map import update_fixation_map, fixation_map
from utils import set_res, retinal_filter, detect_iris_centers, detect_iris_only
from dynamic_calibration import DynamicCalibrator
from collections import deque
import pyautogui


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
set_res(cap, 800, 600)

for _ in range(10): cap.read()  # warm-up

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

scales, center_iris = calibrate_gaze(cap, face_cascade, eye_cascade)


gaze_history = deque(maxlen=5)
calibrator = DynamicCalibrator(screen_size=(1920, 1080))
calibrator.load_calibration(scales, center_iris)

last_cursor_pos = None
cursor_move_threshold = 25  # píxeles


cv2.namedWindow("Gaze Tracker", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar frame.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocessed = retinal_filter(gray)
    iris_centers, face_center = detect_iris_only(frame, face_cascade, eye_cascade) #detect_iris_centers(frame, face_cascade, eye_cascade, preprocessed)
    

    if len(iris_centers) >= 2 and face_center is not None:
        # Ordena y promedia los dos ojos más a la izquierda
        iris_centers = sorted(iris_centers, key=lambda pt: pt[0])[:2]
        avg_x = int((iris_centers[0][0] + iris_centers[1][0]) / 2)
        avg_y = int((iris_centers[0][1] + iris_centers[1][1]) / 2)
        iris_center = (avg_x, avg_y)
       
        cv2.circle(frame, iris_center, 6, (0, 100, 255), -1)

        # Calibración (solo si activa)

        calibrator.update(iris_center, face_center)
        gaze_pos=calibrator.estimate_gaze(iris_center, face_center)
        #gaze_pos=calibrator.estimate_gaze2(iris_center, face_center) 

       
        gaze_history.append(gaze_pos)
        smoothed = smooth_gaze(gaze_history)
        update_fixation_map(smoothed)

        fx = int(smoothed[0] * frame.shape[1] / 1920)
        fy = int(smoothed[1] * frame.shape[0] / 1080)
        cv2.circle(frame, (fx, fy), 8, (255, 140, 0), -1)

        for pos in fixation_map:
            px = int(pos[0] * frame.shape[1] / 1920)
            py = int(pos[1] * frame.shape[0] / 1080)
            cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)





    cv2.imshow("Gaze Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
