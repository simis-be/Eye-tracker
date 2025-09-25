# -*- coding: utf-8 -*-

import json
from pathlib import Path
import cv2
from calibration import calibrate_gaze
from gaze_estimation import project_gaze_trig2, smooth_gaze
from fixation_map import update_fixation_map, fixation_map
from utils import set_res, retinal_filter, detect_iris_only_improved, detect_iris_only, detect_individual_eyes
from dynamic_calibration import ImprovedDynamicCalibrator
from fixation_map_noise import GazeStabilizer  # Importar tu clase avanzada
from collections import deque
import pyautogui
import time
import numpy as np

pyautogui.FAILSAFE = False 
ACTIVE_REGION = None  # None = pantalla completa; o {'x':0, 'y':0, 'w':960, 'h':1080}
REGION_TIMEOUT = 1.5  # segundos de fijación para cambiar de región


def initialize_system(cap, face_cascade, eye_cascade):
    """Función para inicializar/reinicializar el sistema de calibración"""
    print("\n=== INICIANDO CALIBRACIÓN ===")

    # Leer la configuración de sensibilidad desde el archivo
    sensibilidad_h = 50  # Valor predeterminado
    sensibilidad_v = 50  # Valor predeterminado
    
    velocidad_h = 50 
    velocidad_v = 50

    ruta_tracker = Path("..")/ "tf" / "eye_tracker_config.json"
    ruta_tracker_absoluta = ruta_tracker.resolve()
    
    if ruta_tracker_absoluta.exists():
        try:
            with open(ruta_tracker_absoluta, 'r', encoding='utf-8') as f:
                config = json.load(f)
                sensibilidad_h = config.get("sensibilidad_h", 50)
                sensibilidad_v = config.get("sensibilidad_v", 50)

                velocidad_h = config.get("velocidad_h", 50) 
                velocidad_v = config.get("velocidad_v", 50)
            print(f" Configuración de sensibilidad cargada.")
        except Exception as e:
            print(f" Error leyendo config: {e}")



    scales, center_iris = calibrate_gaze(cap, face_cascade, eye_cascade)
    
    # Reinicializar el sistema de estabilización
    gaze_stabilizer = GazeStabilizer(screen_size=(1920, 1080), sensibilidad_h=sensibilidad_h, sensibilidad_v=sensibilidad_v)
    
    # Reinicializar el calibrador dinámico
    calibrator = ImprovedDynamicCalibrator(screen_size=(1920, 1080))
    calibrator.load_calibration(scales, center_iris)
    
    print("Calibración completada")
    return gaze_stabilizer, calibrator, velocidad_h, velocidad_v

# Inicialización inicial
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
set_res(cap, 800, 600)

for _ in range(10): cap.read()  # warm-up

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Calibración inicial
gaze_stabilizer, calibrator, velocidad_h, velocidad_v = initialize_system(cap, face_cascade, eye_cascade)

last_cursor_pos = None
cursor_move_threshold = 25  # píxeles

# Variables para FPS
frame_count = 0
start_time = time.time()

# Estado de calibración
calibrating = False

cv2.namedWindow("Gaze Tracker", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n=== CONTROLES ===")
print("Q: Salir")
print("C: Limpiar fijaciones")
print("R: Recalibrar sistema")
print("==================\n")


# En tu eye-tracker principal, añade al inicio del loop:
if Path("tracker_command.json").exists():
    try:
        cmd = json.loads(Path("tracker_command.json").read_text())
        if cmd.get("comando") == "recalibrar":
            initialize_system
            calibrating = True
        Path("tracker_command.json").unlink()
    except:
        pass
    

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar frame.")
        break

    frame = cv2.flip(frame, 1)
    
    # Mostrar mensaje de calibración si está en proceso
    if calibrating:
        cv2.putText(frame, "CALIBRANDO... Presiona ESC para cancelar", 
                   (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Gaze Tracker", frame)
        
        # Verificar si se presiona ESC durante calibración
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            calibrating = False
            print("Calibración cancelada")
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocessed = retinal_filter(gray)
    iris_centers, face_center = detect_iris_only(frame, face_cascade, eye_cascade)
    face_center, iris_data, is_blinking = detect_individual_eyes(frame, face_cascade, eye_cascade)
    if not is_blinking and len(iris_data) >= 2 and face_center is not None:
        iris_center = (
        (iris_data['left'][0] + iris_data['right'][0]) // 2,
        (iris_data['left'][1] + iris_data['right'][1]) // 2
        )




    if len(iris_centers) >= 2 and face_center is not None:
        # Ordena y promedia los dos ojos más a la izquierda
        iris_centers = sorted(iris_centers, key=lambda pt: pt[0])[:2]
        avg_x = int((iris_centers[0][0] + iris_centers[1][0]) / 2)
        avg_y = int((iris_centers[0][1] + iris_centers[1][1]) / 2)
        iris_center = (avg_x, avg_y)
       
        cv2.circle(frame, iris_center, 6, (0, 100, 255), -1)

        # Calibración dinámica
        calibrator.update(iris_center, face_center)
        raw_gaze_pos = calibrator.estimate_gaze(iris_center, face_center)
    
        # Procesar con el estabilizador (reduce ruido y detecta fijaciones)
        stabilized_gaze, current_fixation = gaze_stabilizer.process_gaze(raw_gaze_pos)

        calibrator.update(iris_center, face_center, current_fixation=current_fixation)
        
        if stabilized_gaze is not None:
            # Usar la posición estabilizada en lugar de smooth_gaze
            smoothed = stabilized_gaze
            
            
            # Dibujar posición actual estabilizada
            fx = int(smoothed[0] * frame.shape[1] / 1920)
            fy = int(smoothed[1] * frame.shape[0] / 1080)
            cv2.circle(frame, (fx, fy), 8, (255, 140, 0), -1)


            FIXATION_CLICK_DURATION = 1.5  # segundos

            # Dibujar fijación actual (si existe)
            if current_fixation is not None:
                
                fix_x = int(current_fixation[0] * frame.shape[1] / 1920)
                fix_y = int(current_fixation[1] * frame.shape[0] / 1080)
                cv2.circle(frame, (fix_x, fix_y), 15, (0, 255, 0), 3)  # Verde para fijación actual
                cv2.putText(frame, "FIXATION", (fix_x + 20, fix_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                current_fix_duration = gaze_stabilizer.fixation_detector.current_fixation_frames / 25.0  # 25 FPS
    
                if current_fix_duration >= FIXATION_CLICK_DURATION:
                 
        # Hacer clic
                 pyautogui.click()
                 print("Clic por fijación!")
        # Opcional: resetear fijación para evitar clics múltiples
                 gaze_stabilizer.fixation_detector.current_fixation_frames = 0

            # Dibujar todas las fijaciones detectadas
            fixation_map = gaze_stabilizer.get_fixation_map()
            for pos in fixation_map:
                px = int(pos[0] * frame.shape[1] / 1920)
                py = int(pos[1] * frame.shape[0] / 1080)
                cv2.circle(frame, (px, py), 8, (0, 255, 255), -1)  # Amarillo para fijaciones pasadas

            # Mostrar estadísticas
            stats = gaze_stabilizer.get_stats()
            cv2.putText(frame, f"Fixations: {stats['total_fixations']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Current Fix: {stats['current_fixation']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Calcular y mostrar FPS
            frame_count += 1
            if frame_count % 25 == 0:  # Cada 25 frames (1 segundo a 25fps)
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if current_fixation:
        fx, fy = current_fixation
        duration = gaze_stabilizer.fixation_detector.current_fixation_frames / 25.0
    
    # Definir zonas: izquierda y derecha
        if fx < 960 and duration >= REGION_TIMEOUT and ACTIVE_REGION != "left":
            print("Cambiando a zona izquierda")
            ACTIVE_REGION = "left"
        # Reinicia fijación para evitar clic
            gaze_stabilizer.fixation_detector.current_fixation_frames = 0
        
        elif fx > 960 and duration >= REGION_TIMEOUT and ACTIVE_REGION != "right":
            print("Cambiando a zona derecha")
            ACTIVE_REGION = "right"
            gaze_stabilizer.fixation_detector.current_fixation_frames = 0
    if ACTIVE_REGION == "left":
    # Mapear mirada del 0-100% de X al 0-50% de la pantalla
        target_x = np.clip(stabilized_gaze[0], 0, 1920) * 0.5
        target_y = stabilized_gaze[1]
    elif ACTIVE_REGION == "right":
        target_x = 960 + np.clip(stabilized_gaze[0], 0, 1920) * 0.5
        target_y = stabilized_gaze[1]
    else:
        target_x, target_y = stabilized_gaze  # pantalla completa  


    if stabilized_gaze is not None:
        current_cursor_pos = pyautogui.position()
        # Obtener la posición actual del cursor
        # Obtener la posición objetivo del puntero desde el sistema de estabilización
        target_x = stabilized_gaze[0]
        target_y = stabilized_gaze[1]

        # 1. Aplicar la limitación de velocidad
        gain_x = velocidad_h / 50.0  
        gain_y = velocidad_v / 50.0

        move_x = (target_x - current_cursor_pos[0]) * gain_x
        move_y = (target_y - current_cursor_pos[1]) * gain_y

        max_move_per_frame = 150 
        final_move_x = np.clip(move_x, -max_move_per_frame, max_move_per_frame)
        final_move_y = np.clip(move_y, -max_move_per_frame, max_move_per_frame) 

        # Aplicar movimiento
        new_x = current_cursor_pos[0] + final_move_x
        new_y = current_cursor_pos[1] + final_move_y


        # 2. Restringir la posición a los límites de la pantalla
        screen_w, screen_h = pyautogui.size()
        final_x = int(max(0, min(screen_w - 1, new_x)))
        final_y = int(max(0, min(screen_h - 1, new_y)))

        # 3. Mover el puntero
        pyautogui.moveTo(final_x, final_y, duration=0.01, _pause=False)

    # Mostrar controles en pantalla
    cv2.putText(frame, "Q: Quit | C: Clear Fixations | R: Recalibrate", 
               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Gaze Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Limpiar fijaciones
        gaze_stabilizer.clear_fixations()
        print(" Fijaciones limpiadas")
    elif key == ord('r') or key == ord('R'):
        # Iniciar recalibración
        calibrating = True
        print("\n Iniciando recalibración...")
        try:
            # Reinicializar el sistema completo
            gaze_stabilizer, calibrator = initialize_system(cap, face_cascade, eye_cascade)
            
            # Reiniciar contadores de FPS
            frame_count = 0
            start_time = time.time()
            
            calibrating = False
            print("Recalibración exitosa")
            
        except Exception as e:
            calibrating = False
            print(f"Error en recalibración: {e}")
            print("Continuando con calibración anterior...")

    if stabilized_gaze and abs(stabilized_gaze[0] - 960) < 100 and abs(stabilized_gaze[1] - 540) < 100:

        error_x = abs(stabilized_gaze[0] - 960)
        error_y = abs(stabilized_gaze[1] - 540)
        cv2.putText(frame, f"Error: {error_x:.0f}x{error_y:.0f}px", 
                 (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

cap.release()
cv2.destroyAllWindows()

# Mostrar estadísticas finales
final_stats = gaze_stabilizer.get_stats()
print(f"\n=== ESTADÍSTICAS FINALES ===")
print(f"Total de fijaciones detectadas: {final_stats['total_fixations']}")

    