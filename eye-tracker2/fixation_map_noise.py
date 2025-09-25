
from collections import deque
import time
import numpy as np
import cv2
from collections import deque



class FixationDetector:
    def __init__(self, fixation_threshold=50, min_fixation_duration=6, fps=25):
        """
        Args:
            fixation_threshold: Distancia máxima en píxeles para considerar una fijación
            min_fixation_duration: Mínimo número de frames para considerar una fijación válida
            fps: Frames por segundo del sistema
        """
        self.fixation_threshold = fixation_threshold
        self.min_fixation_duration = min_fixation_duration
        self.fps = fps
        
        # Historial de posiciones de mirada
        self.gaze_history = deque(maxlen=min_fixation_duration * 2)
        self.timestamps = deque(maxlen=min_fixation_duration * 2)
        
        # Fijaciones detectadas
        self.fixations = []
        self.current_fixation = None
        self.current_fixation_start = None
        self.current_fixation_frames = 0
        
        print(f"FixationDetector inicializado: umbral={fixation_threshold}px, duración={min_fixation_duration} frames")

    def add_gaze_point(self, gaze_pos):
        """Añade un nuevo punto de mirada y detecta fijaciones"""
        current_time = time.time()
        
        self.gaze_history.append(gaze_pos)
        self.timestamps.append(current_time)
        
        if len(self.gaze_history) < 2:
            return None
        
        # Calcular si estamos en una fijación
        if self._is_fixating(gaze_pos):
            if self.current_fixation is None:
                # Iniciar nueva fijación
                self.current_fixation = gaze_pos
                self.current_fixation_start = current_time
                self.current_fixation_frames = 1
            else:
                # Continuar fijación actual
                self.current_fixation_frames += 1
                # Actualizar centro de fijación (promedio ponderado)
                weight = 0.1
                self.current_fixation = (
                    int(self.current_fixation[0] * (1-weight) + gaze_pos[0] * weight),
                    int(self.current_fixation[1] * (1-weight) + gaze_pos[1] * weight)
                )
        else:
            # Finalizar fijación si es válida
            if (self.current_fixation is not None and 
                self.current_fixation_frames >= self.min_fixation_duration):
                
                fixation_duration = current_time - self.current_fixation_start
                self.fixations.append({
                    'position': self.current_fixation,
                    'duration': fixation_duration,
                    'frames': self.current_fixation_frames,
                    'timestamp': self.current_fixation_start
                })
                
                print(f"Fijación detectada: {self.current_fixation}, duración: {self.current_fixation_frames} frames")
            
            # Reset fijación actual
            self.current_fixation = None
            self.current_fixation_frames = 0

        return self.current_fixation

    def _is_fixating(self, gaze_pos):
        """Determina si la posición actual está dentro del umbral de fijación"""
        if not self.gaze_history or len(self.gaze_history) < 3:
            return True
        
        # Calcular centro de las últimas posiciones
        recent_positions = list(self.gaze_history)[-self.min_fixation_duration:]
        if len(recent_positions) < 2:
            return True
        
        center_x = sum(pos[0] for pos in recent_positions) / len(recent_positions)
        center_y = sum(pos[1] for pos in recent_positions) / len(recent_positions)
        center = (center_x, center_y)
        
        # Calcular distancia al centro
        distance = np.sqrt((gaze_pos[0] - center[0])**2 + (gaze_pos[1] - center[1])**2)
        
        return distance <= self.fixation_threshold

    def get_fixation_map(self):
        """Retorna el mapa de fijaciones actual"""
        return [fix['position'] for fix in self.fixations]

    def get_current_fixation(self):
        """Retorna la fijación actual si existe"""
        return self.current_fixation

    def clear_fixations(self):
        """Limpia todas las fijaciones"""
        self.fixations.clear()
        self.current_fixation = None
        self.current_fixation_frames = 0


class NoiseReducer:
    def __init__(self, window_size=5, outlier_threshold=3.0):
        """
        Sistema de reducción de ruido para posiciones de mirada
        
        Args:
            window_size: Tamaño de la ventana para suavizado
            outlier_threshold: Umbral para detectar outliers (en desviaciones estándar)
        """
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
        self.position_history = deque(maxlen=window_size * 2)
        self.velocity_history = deque(maxlen=window_size)
        
    def smooth_position(self, raw_position):
        """
        Aplica suavizado y filtrado de ruido a la posición
        """
        if raw_position is None:
            return None
        
        self.position_history.append(raw_position)
        
        if len(self.position_history) < 3:
            return raw_position
        
        # 1. Detección de outliers
        filtered_position = self._filter_outliers(raw_position)
        
        # 2. Suavizado temporal
        smoothed_position = self._temporal_smoothing(filtered_position)
        
        # 3. Filtro de velocidad
        velocity_filtered = self._velocity_filter(smoothed_position)
        
        return velocity_filtered
    
    def _filter_outliers(self, position):
        """Filtra outliers basado en la desviación estándar"""
        if len(self.position_history) < 5:
            return position
        
        recent_positions = list(self.position_history)[-5:]
        
        # Calcular estadísticas
        x_values = [pos[0] for pos in recent_positions]
        y_values = [pos[1] for pos in recent_positions]
        
        mean_x, std_x = np.mean(x_values), np.std(x_values)
        mean_y, std_y = np.mean(y_values), np.std(y_values)
        
        # Verificar si es outlier
        x_z_score = abs(position[0] - mean_x) / (std_x + 1e-5)
        y_z_score = abs(position[1] - mean_y) / (std_y + 1e-5)
        
        if x_z_score > self.outlier_threshold or y_z_score > self.outlier_threshold:
            # Retornar la mediana de las posiciones recientes
            median_x = np.median(x_values)
            median_y = np.median(y_values)
            return (int(median_x), int(median_y))
        
        return position
    
    def _temporal_smoothing(self, position):
        """Aplica suavizado temporal usando media móvil ponderada"""
        if len(self.position_history) < 3:
            return position
        
        recent_positions = list(self.position_history)[-self.window_size:]
        
        # Pesos decrecientes (más peso a posiciones más recientes)
        weights = np.exp(np.linspace(-2, 0, len(recent_positions)))
        weights = weights / np.sum(weights)
            # Aplicar más suavizado en X (horizontal)
        x_values = [pos[0] for pos in recent_positions]
        y_values = [pos[1] for pos in recent_positions]

        smoothed_x = sum(x * w for x, w in zip(x_values, weights)) * 0.95  # +10% suavizado en X
        smoothed_y = sum(y * w for y, w in zip(y_values, weights))
        
        return (int(smoothed_x), int(smoothed_y))
    
    def _velocity_filter(self, position):
        """Filtra cambios de velocidad excesivos"""
        if len(self.position_history) < 2:
            return position
        
        prev_position = self.position_history[-2]
        
        # Calcular velocidad actual
        velocity = (
            position[0] - prev_position[0],
            position[1] - prev_position[1]
        )
        
        self.velocity_history.append(velocity)
        
        if len(self.velocity_history) < 3:
            return position
        
        # Calcular velocidad promedio
        recent_velocities = list(self.velocity_history)[-3:]
        avg_velocity = (
            sum(v[0] for v in recent_velocities) / len(recent_velocities),
            sum(v[1] for v in recent_velocities) / len(recent_velocities)
        )
        
        # Limitar velocidad excesiva
        max_velocity = 150  # píxeles por frame
        
        velocity_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
        if velocity_magnitude > max_velocity:
            # Reducir la velocidad
            scale_factor = max_velocity / velocity_magnitude
            limited_velocity = (
                velocity[0] * scale_factor,
                velocity[1] * scale_factor
            )
            
            return (
                int(prev_position[0] + limited_velocity[0]),
                int(prev_position[1] + limited_velocity[1])
            )
        
        return position





# En tu archivo fixation_map_noise.py

class GazeStabilizer:
    def __init__(self, screen_size=(1920, 1080), sensibilidad_h=50, sensibilidad_v=50):
        self.screen_w, self.screen_h = screen_size
        
        # Guardar los valores de sensibilidad para su uso
        self.sensibilidad_h = sensibilidad_h
        self.sensibilidad_v = sensibilidad_v
        
        # 🎯 Usa la sensibilidad para ajustar el umbral de NoiseReducer
        # Una sensibilidad de 100 (máxima) dará un umbral alto, menos filtrado agresivo.
        # Una sensibilidad de 10 (mínima) dará un umbral bajo, más filtrado agresivo.
        noise_reducer_threshold = (self.sensibilidad_h + self.sensibilidad_v) / 100 * 4.0 + 1.0 # Escalar de 10-100 a ~1.4-5.0
        
        # Ajusta el umbral de detección de fijación
        # Una sensibilidad alta (100) dará un umbral alto (más fácil de fijar).
        # Una sensibilidad baja (10) dará un umbral bajo (más difícil de fijar).
        fixation_threshold = (self.sensibilidad_h + self.sensibilidad_v) / 100 * 60 + 20 # Escalar de 10-100 a ~26-80

        # Pasa los parámetros ajustados a tus clases
        self.noise_reducer = NoiseReducer(window_size=5, outlier_threshold=noise_reducer_threshold)
        self.fixation_detector = FixationDetector(
            fixation_threshold=fixation_threshold, 
            min_fixation_duration=6, 
            fps=25
        )
        
        # Mantienes el resto de tu constructor sin cambios
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.01 
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 2.0 
        
        self.speed_correction = 0.65
        self.center_position = (screen_size[0] // 2, screen_size[1] // 2)
        self.last_stabilized_gaze = None
        
        print("GazeStabilizer con sistema de filtrado híbrido inicializado.")
        print(f"Umbral de NoiseReducer: {noise_reducer_threshold:.2f}, Umbral de Fijación: {fixation_threshold:.2f}")

    def process_gaze(self, raw_gaze_position):
        # Este método no necesita cambios, ya que ahora usa los atributos de la clase que ya están ajustados
        # por los sliders. La lógica interna de NoiseReducer y FixationDetector hace el trabajo.
        if raw_gaze_position is None:
            return None, None
        
        pre_filtered_position = self.noise_reducer.smooth_position(raw_gaze_position)
        if pre_filtered_position is None:
            return None, None

        measurement = np.array([[np.float32(pre_filtered_position[0])], [np.float32(pre_filtered_position[1])]])
        self.kf.predict()
        estimated = self.kf.correct(measurement)
        
        stabilized_gaze = (int(estimated[0,0]), int(estimated[1,0]))
        
        current_fixation = self.fixation_detector.add_gaze_point(stabilized_gaze)
        
        return stabilized_gaze, current_fixation

    def get_fixation_map(self):
        return self.fixation_detector.get_fixation_map()
    
    def get_current_fixation(self):
        return self.fixation_detector.get_current_fixation()
    
    def clear_fixations(self):
        self.fixation_detector.clear_fixations()
    
    def get_stats(self):
        return {
            'total_fixations': len(self.fixation_detector.fixations),
            'current_fixation': self.get_current_fixation() is not None,
            'stabilization_method': 'Hybrid Filter'
        }