import numpy as np
from gaze_estimation import project_gaze_trig2
from collections import deque

class ImprovedDynamicCalibrator:
    def __init__(self, screen_size=(1920, 1080), history_len=60):
        self.screen_w, self.screen_h = screen_size
        self.history_len = history_len
        
        # 💡 VALORES DE ESCALA INICIAL FIJOS (Se sobrescribirán por load_calibration)
        self.scale_x = 8.0
        self.scale_y = 12.0
        self.alpha = 0.0 # ❌ Deshabilitado
        
        
        self.min_scale_x = 2.5
        self.max_scale_x = 80.0
        self.min_scale_y = 4.0
        self.max_scale_y = 120.0
        
        # Factor de corrección de velocidad global
        self.global_speed_factor = 1.0  # Sin freno. Usar 0.7 si es demasiado rápido.
        
       
        self.min_movement_x = 0.2
        self.min_movement_y = 0.1
        self.max_std_x = 50.0
        self.max_std_y = 70.0
        
        # Historial para análisis de movimiento (Mantenido, pero no usado para adaptar)
        self.iris_history = deque(maxlen=history_len) # Usar deque para eficiencia
        self.face_history = deque(maxlen=history_len) # Usar deque para eficiencia
        self.movement_history_x = []
        self.movement_history_y = []
        
        # Variables de calibración
        self.calibrated_scales = None
        self.center_iris = None
        
        # Detección de movimientos verticales específicos
        self.vertical_movement_boost = 1.4
        self.last_significant_movement = None
        
        self.gaze_center_offset = (0, 0) # Sesgo aprendido
        self.gaze_offset_history = deque(maxlen=60)


        print(f"Calibrador inicializado en modo ESTÁTICO FIJO.")
        
    # ----------------------------------------------------------------------------------
    ## MÉTODOS DE CÁLCULO ESTÁTICO
    # ----------------------------------------------------------------------------------

    def update(self, iris_center, face_center, current_fixation=None):
        """
        ¡MODIFICADO! Deshabilita la adaptación dinámica. Solo recolecta datos 
        y aprende el sesgo central (offset) si hay fijación.
        """
        if self.center_iris is None:
            return

        # 1. Almacenar historial (uso limitado en este modo)
        self.iris_history.append(iris_center)
        self.face_history.append(face_center)

        # 2. Lógica de APRENDIZAJE DE SESGO (ÚNICA PARTE ADAPTABLE)
        # Esto es vital para centrar la mirada, incluso en modo estático.
        if current_fixation is not None:
            fx, fy = current_fixation
            # Si el punto estimado (fx, fy) está cerca del centro de la pantalla
            if abs(fx - 960) < 100 and abs(fy - 540) < 100:
                # Calcular movimiento relativo al centro del iris
                dx = iris_center[0] - self.center_iris[0]
                dy = iris_center[1] - self.center_iris[1]
                
                # Ajuste LENTO del offset de centro
                alpha = 0.01  
                self.gaze_center_offset = (
                    self.gaze_center_offset[0] * (1 - alpha) + dx * alpha,
                    self.gaze_center_offset[1] * (1 - alpha) + dy * alpha
                )
        
        # ❌ Eliminada toda la lógica de cálculo de range, std y actualización de self.scale_x/y
        return

    def _estimate_gaze_quadrant_based(self, iris_center, face_center):
        """Estimación de mirada usando la calibración de 25 puntos (estática)."""
        
        if self.center_iris is None:
            return 0, 0
            
        dx = iris_center[0] - self.center_iris[0]
        dy = iris_center[1] - self.center_iris[1]

        # 1. Corregir la posición con el sesgo aprendido (Única corrección dinámica)
        corrected_dx = dx - self.gaze_center_offset[0]
        corrected_dy = dy - self.gaze_center_offset[1]
    
        cx, cy = self.screen_w // 2, self.screen_h // 2
        
        # 2. Calcular la posición RAW para encontrar el punto de calibración más cercano
        # Usamos una escala de referencia neutral (ej. 10.0)
        raw_gx = int(cx + corrected_dx * 10.0) 
        raw_gy = int(cy + corrected_dy * 10.0)
    
        # 3. Encontrar el punto de calibración más cercano (Vecino más cercano)
        closest_point_key = self._find_closest_calibration_point(raw_gx, raw_gy)

        # 4. Usar las ESCALAS FIJAS del punto más cercano
        if closest_point_key in self.calibrated_scales:
            scale_x, scale_y = self.calibrated_scales[closest_point_key]
        else:
            # Fallback a escalas promedio cargadas al inicio (son self.scale_x/y)
            scale_x, scale_y = self.scale_x, self.scale_y
    
        # 5. Aplicar velocidad y boost vertical
        adjusted_scale_x = scale_x * self.global_speed_factor
        adjusted_scale_y = scale_y * self.global_speed_factor * self.vertical_movement_boost 
    
        # 6. Calcular posición final en pantalla
        gx = int(np.clip(cx + corrected_dx * adjusted_scale_x, 0, self.screen_w - 1))
        gy = int(np.clip(cy + corrected_dy * adjusted_scale_y, 0, self.screen_h - 1))
        
        return (gx, gy)

    # ----------------------------------------------------------------------------------
    ## MÉTODOS AUXILIARES (Limpieza y consolidación)
    # ----------------------------------------------------------------------------------
    
    def _find_closest_calibration_point(self, gaze_x, gaze_y):
        """Busca el punto de calibración más cercano (25 puntos)."""
        min_distance = float('inf')
        closest_point_key = None
    
        # 💡 CONSOLIDACIÓN: Diccionario de 25 puntos
        calibration_points = {
            "P1_1": (96, 54), "P1_2": (480, 54), "P1_3": (960, 54), "P1_4": (1440, 54), "P1_5": (1824, 54),
            "P2_1": (96, 270), "P2_2": (480, 270), "P2_3": (960, 270), "P2_4": (1440, 270), "P2_5": (1824, 270),
            "P3_1": (96, 540), "P3_2": (480, 540), "P3_3": (960, 540), "P3_4": (1440, 540), "P3_5": (1824, 540),
            "P4_1": (96, 810), "P4_2": (480, 810), "P4_3": (960, 810), "P4_4": (1440, 810), "P4_5": (1824, 810),
            "P5_1": (96, 1026), "P5_2": (480, 1026), "P5_3": (960, 1026), "P5_4": (1440, 1026), "P5_5": (1824, 1026),
        }

        for key, pos in calibration_points.items():
            distance = np.sqrt((pos[0] - gaze_x)**2 + (pos[1] - gaze_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_point_key = key
                
        return closest_point_key

    # ❌ Métodos eliminados (innecesarios en modo estático):
    # _update_horizontal_scale, _update_vertical_scale, _analyze_movement_patterns,
    # _determine_quadrant, _apply_dynamic_correction, _estimate_gaze_dynamic, get_movement_stats

    # ----------------------------------------------------------------------------------
    ## MÉTODOS DE CARGA Y ESTADO (Mantenidos)
    # ----------------------------------------------------------------------------------
    
    def load_calibration(self, scales, center_iris):
        """Carga calibración de 25 puntos y calcula escalas promedio fijas."""
        self.calibrated_scales = scales
        self.center_iris = center_iris
        
        if isinstance(scales, dict) and len(scales) > 0:
            total_scale_x = sum(s[0] for s in scales.values())
            total_scale_y = sum(s[1] for s in scales.values())
            count = len(scales)
            
            if count > 0:
                # 💡 Escalas promedio Fijas usadas como FALLBACK
                self.scale_x = total_scale_x / count
                self.scale_y = (total_scale_y / count) * 1.2
                
            print(f"Calibración cargada y fijada: X={self.scale_x:.2f}, Y={self.scale_y:.2f}")


    def get_movement_stats(self):
        """
        Retorna estadísticas de movimiento y estabilidad del iris/cara.
        Útil para diagnosticar la calidad de la señal en modo estático.
        """
        history_len = 30 # Usar un segmento reciente de la historia
        
        # Asegúrate de tener suficientes datos en los deques
        if len(self.iris_history) < history_len or len(self.face_history) < history_len:
            return {'std_x': 0.0, 'std_y': 0.0, 'range_x': 0.0, 'range_y': 0.0}

        # Calcular los movimientos relativos (Iris - Centro_Iris)
        recent_dx = [(i[0] - self.center_iris[0]) for i in list(self.iris_history)[-history_len:]]
        recent_dy = [(i[1] - self.center_iris[1]) for i in list(self.iris_history)[-history_len:]]
        
        # Calcular Desviación Estándar (STD) y Rango
        std_x = np.std(recent_dx)
        std_y = np.std(recent_dy)
        range_x = np.percentile(recent_dx, 95) - np.percentile(recent_dx, 5)
        range_y = np.percentile(recent_dy, 95) - np.percentile(recent_dy, 5)
        
        return {
            'std_x': float(std_x),
            'std_y': float(std_y),
            'range_x': float(range_x),
            'range_y': float(range_y)
        }
    def estimate_gaze(self, iris_center, face_center):
        """Método de entrada principal que usa la estimación estática."""
        if self.calibrated_scales and self.center_iris:
            return self._estimate_gaze_quadrant_based(iris_center, face_center)
        else:
            print("Error: Calibración no cargada. Retornando (0, 0).")
            return (0, 0)
        
    def get_calibration_status(self):
        """Retorna el estado actual de la calibración"""
        return {
            'mode': 'STATIC_FIXED',
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
            'offset_x': self.gaze_center_offset[0],
            'offset_y': self.gaze_center_offset[1],
            'samples': len(self.iris_history)
        }

    def reset(self):
        """Reset del calibrador"""
        print("Reset de calibrador ejecutado.")
        self.iris_history.clear()
        self.face_history.clear()
        self.gaze_offset_history.clear()
        self.gaze_center_offset = (0, 0)
        self.calibrated_scales = None
        self.center_iris = None
        # Recargar valores iniciales estáticos
        self.scale_x = 8.0
        self.scale_y = 12.0