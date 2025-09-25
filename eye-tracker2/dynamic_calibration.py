import numpy as np
from gaze_estimation import project_gaze_trig2
from collections import deque

class ImprovedDynamicCalibrator:
    def __init__(self, screen_size=(1920, 1080), history_len=60):
        self.screen_w, self.screen_h = screen_size
        self.history_len = history_len
        
        # Separar escalas para movimientos horizontales y verticales
        # Reducir escalas iniciales para compensar el exceso de velocidad
        self.scale_x = 6.0  # Reducido desde 8.0
        self.scale_y = 9.0  # Reducido desde 12.0
        self.alpha = 0.12   # Reducir velocidad de adaptación
        
        # Rangos de escala ajustados
        self.min_scale_x = 2.5  # Reducido
        self.max_scale_x = 40.0 # Reducido
        self.min_scale_y = 4.0  # Reducido
        self.max_scale_y = 60.0 # Reducido
        
        # Factor de corrección de velocidad global
        self.global_speed_factor = 0.7  # Reducir velocidad general en 20%
        
        # Umbrales específicos para cada eje
        self.min_movement_x = 2.0
        self.min_movement_y = 1.0
        self.max_std_x = 20.0
        self.max_std_y = 30.0
        
        # Historial para análisis de movimiento
        self.iris_history = []
        self.face_history = []
        self.movement_history_x = []
        self.movement_history_y = []
        
        # Variables de calibración
        self.calibrated_scales = None
        self.center_iris = None
        
        # Detección de movimientos verticales específicos
        self.vertical_movement_boost = 1.2  # Reducido desde 1.3
        self.last_significant_movement = None
        
        self.gaze_center_offset = (0, 0) # Nuevo: para almacenar el sesgo del usuario
        self.gaze_offset_history = deque(maxlen=30)


        print(f"Calibrador inicializado con escalas corregidas: X={self.scale_x}, Y={self.scale_y}")
        print(f"Factor de velocidad global: {self.global_speed_factor}")

    def update(self, iris_center, face_center, current_fixation=None):
        """Actualización mejorada con análisis separado de movimientos X e Y"""
        
        self.iris_history.append(iris_center)
        self.face_history.append(face_center)
        
        # Mantener historial limitado
        if len(self.iris_history) > self.history_len:
            self.iris_history.pop(0)
            self.face_history.pop(0)
        
        if current_fixation is not None:
            fx, fy = current_fixation
        # Si está fijando cerca del centro de la pantalla (ajusta umbrales según tu resolución)
            if abs(fx - 960) < 100 and abs(fy - 540) < 100:
            # Actualizar el sesgo de forma gradual
                dx = iris_center[0] - face_center[0]
                dy = iris_center[1] - face_center[1]
            
            # Ajuste lento del offset de centro (evita cambios bruscos)
                alpha = 0.05  # muy lento
                self.gaze_center_offset = (
                    self.gaze_center_offset[0] * (1 - alpha) + dx * alpha,
                    self.gaze_center_offset[1] * (1 - alpha) + dy * alpha
                )
        # Necesitamos suficientes datos para análisis
        if len(self.iris_history) < 20:
            return
            
        
        # Calcular los movimientos relativos
        iris_dx = [i[0] - f[0] for i, f in zip(self.iris_history, self.face_history)]
        iris_dy = [i[1] - f[1] for i, f in zip(self.iris_history, self.face_history)]

    # Lógica para detectar sesgo (cuando el usuario está en el centro de la pantalla)
        if len(self.iris_history) > 30:
            recent_dx = iris_dx[-30:]
            recent_dy = iris_dy[-30:]

        # Si el movimiento es bajo, el usuario probablemente está mirando al centro
            if np.std(recent_dx) < 2.0 and np.std(recent_dy) < 2.0:
                mean_dx = np.mean(recent_dx)
                mean_dy = np.mean(recent_dy)

            # Actualizar el sesgo de forma gradual
                self.gaze_center_offset = (
                 self.gaze_center_offset[0] * 0.9 + mean_dx * 0.1,
                    self.gaze_center_offset[1] * 0.9 + mean_dy * 0.1
                )

        # Calcular movimientos relativos
        iris_dx = [i[0] - f[0] for i, f in zip(self.iris_history, self.face_history)]
        iris_dy = [i[1] - f[1] for i, f in zip(self.iris_history, self.face_history)]
        # Lógica para detectar sesgo (cuando el usuario está en el centro de la pantalla)
        if len(self.iris_history) > 30:
            recent_dx = iris_dx[-30:]
            recent_dy = iris_dy[-30:]
            
        
            # Si el movimiento es bajo, el usuario probablemente está mirando al centro
        # Análisis de movimiento horizontal
            if np.std(recent_dx) < 2.0 and np.std(recent_dy) < 2.0:
                mean_dx = np.mean(recent_dx)
                mean_dy = np.mean(recent_dy)
                
                self.gaze_offset_history.append((mean_dx, mean_dy))
                
                # Actualizar el sesgo de forma gradual
                if len(self.gaze_offset_history) >= 20:
                     self.gaze_center_offset = (
                        np.mean([o[0] for o in self.gaze_offset_history]),
                        np.mean([o[1] for o in self.gaze_offset_history])
                     )
            
        # Análisis de movimiento horizontal
        self._update_horizontal_scale(iris_dx)
        
        # Análisis de movimiento vertical (con lógica especial)
        self._update_vertical_scale(iris_dy)
        
        # Detección de patrones de movimiento
        self._analyze_movement_patterns(iris_dx, iris_dy)
        
        # Limitar escalas
        self.scale_x = np.clip(self.scale_x, self.min_scale_x, self.max_scale_x)
        self.scale_y = np.clip(self.scale_y, self.min_scale_y, self.max_scale_y)
        
        
        print(f"Escalas actualizadas: X={self.scale_x:.2f}, Y={self.scale_y:.2f}")

    def _update_horizontal_scale(self, iris_dx):
        """Actualización específica para movimientos horizontales"""
        recent_dx = iris_dx[-30:]  # Últimos 30 frames
        
        range_dx = np.percentile(recent_dx, 95) - np.percentile(recent_dx, 5)
        std_dx = np.std(recent_dx)
        
        if range_dx > self.min_movement_x and std_dx < self.max_std_x:
            target_scale_x = (self.screen_w * 0.8) / max(range_dx, 1e-5)
            self.scale_x = (1 - self.alpha) * self.scale_x + self.alpha * target_scale_x
            print(f"Escala X actualizada: range={range_dx:.2f}, std={std_dx:.2f}, nueva_escala={self.scale_x:.2f}")

    def _update_vertical_scale(self, iris_dy):
        """Actualización específica para movimientos verticales con lógica mejorada"""
        recent_dy = iris_dy[-30:]
        
        range_dy = np.percentile(recent_dy, 95) - np.percentile(recent_dy, 5)
        std_dy = np.std(recent_dy)
        
        # Criterios más permisivos para movimientos verticales
        if range_dy > self.min_movement_y and std_dy < self.max_std_y:
            # Aplicar boost para movimientos verticales
            target_scale_y = (self.screen_h * 0.9) / max(range_dy, 1e-5)
            target_scale_y *= self.vertical_movement_boost
            
            # Actualización más agresiva para movimientos verticales
            alpha_y = min(self.alpha * 1.5, 0.3)
            self.scale_y = (1 - alpha_y) * self.scale_y + alpha_y * target_scale_y
            
            print(f"Escala Y actualizada: range={range_dy:.2f}, std={std_dy:.2f}, nueva_escala={self.scale_y:.2f}")
        else:
            # Si no hay suficiente movimiento vertical, incrementar gradualmente la sensibilidad
            if range_dy < self.min_movement_y:
                self.scale_y = min(self.scale_y * 1.02, self.max_scale_y)
                print(f"Incrementando sensibilidad vertical: {self.scale_y:.2f}")

    def _analyze_movement_patterns(self, iris_dx, iris_dy):
        """Analiza patrones de movimiento para detectar tendencias"""
        if len(iris_dx) < 10:
            return
        
        # Detectar movimientos verticales significativos
        recent_dy = iris_dy[-10:]
        vertical_movement = max(recent_dy) - min(recent_dy)
        
        if vertical_movement > 3.0:  # Movimiento vertical significativo detectado
            self.last_significant_movement = 'vertical'
            # Boost temporal para movimientos verticales
            self.scale_y = min(self.scale_y * 1.1, self.max_scale_y)
            print(f"Movimiento vertical significativo detectado: {vertical_movement:.2f}")

    def load_calibration(self, scales, center_iris):
        """Carga calibración externa"""
        self.calibrated_scales = scales
        self.center_iris = center_iris
        
        # Si tenemos escalas calibradas, usarlas como punto de partida
        if scales:
            if isinstance(scales, dict):
                # Tu formato de calibración por cuadrantes
                # Promediar las escalas de todos los cuadrantes
                total_scale_x = sum(s[0] for s in scales.values())
                total_scale_y = sum(s[1] for s in scales.values())
                count = len(scales)
                
                
                
                if count > 0:
                    self.scale_x = total_scale_x / count
                    self.scale_y = (total_scale_y / count) * 1.2  # Boost inicial para vertical
                    
                print(f"Calibración cargada desde cuadrantes: X={self.scale_x:.2f}, Y={self.scale_y:.2f}")
                print(f"Escalas por cuadrante: {scales}")
                
            elif isinstance(scales, (list, tuple)) and len(scales) >= 2:
                # Formato lista/tupla
                self.scale_x = scales[0]
                self.scale_y = scales[1] * 1.2  # Boost inicial para vertical
                print(f"Calibración cargada desde lista: X={self.scale_x:.2f}, Y={self.scale_y:.2f}")

    def estimate_gaze(self, iris_center, face_center):
        """Estimación de mirada con mejoras para movimientos verticales"""
        
        if self.calibrated_scales and self.center_iris:
            # Si tenemos calibración por cuadrantes, usar el método mejorado
            if isinstance(self.calibrated_scales, dict):
                return self._estimate_gaze_quadrant_based(iris_center, face_center)
            else:
                # Usar función trigonométrica calibrada original
                gaze_pos = project_gaze_trig2(
                    iris=iris_center,
                    face=face_center,
                    screen=(self.screen_w, self.screen_h),
                    CH=50.0,
                    alpha_deg=10.0,
                    sensitivity=1.0,
                    scales=self.calibrated_scales,
                    center_iris=self.center_iris
                )
                
                # Aplicar corrección dinámica
                corrected_gaze = self._apply_dynamic_correction(iris_center, face_center, gaze_pos)
                return corrected_gaze
        else:
            # Usar estimación dinámica pura
            return self._estimate_gaze_dynamic(iris_center, face_center)

    def _estimate_gaze_quadrant_based(self, iris_center, face_center):
        dx = iris_center[0] - self.center_iris[0]
        dy = iris_center[1] - self.center_iris[1]

        # Corregir la posición con el sesgo aprendido
        corrected_dx = dx - self.gaze_center_offset[0]
        corrected_dy = dy - self.gaze_center_offset[1]
    
    # Calcular la posición de la mirada en pantalla
        cx, cy = self.screen_w // 2, self.screen_h // 2
        raw_gx = int(cx + corrected_dx * self.scale_x)
        raw_gy = int(cy + corrected_dy * self.scale_y)
    
    # Encontrar el punto de calibración más cercano a la mirada
        closest_point_key = self._find_closest_calibration_point(raw_gx, raw_gy)

    # Usar las escalas del punto de calibración más cercano
        if closest_point_key in self.calibrated_scales:

            scale_x, scale_y = self.calibrated_scales[closest_point_key]
        
        # Aplicar corrección dinámica y de velocidad
            adjusted_scale_x = scale_x * self.global_speed_factor
            adjusted_scale_y = scale_y * self.global_speed_factor * 1.1 # Ligero boost vertical
        
        else:
        # Fallback a escalas promedio
            adjusted_scale_x = self.scale_x * self.global_speed_factor
            adjusted_scale_y = self.scale_y * self.global_speed_factor
    
    # Calcular posición final en pantalla
        gx = int(np.clip(cx + corrected_dx * adjusted_scale_x, 0, self.screen_w - 1))
        gy = int(np.clip(cy + corrected_dy * adjusted_scale_y, 0, self.screen_h - 1))
    
        return (gx, gy)

    def _find_closest_calibration_point(self, gaze_x, gaze_y):
    
        min_distance = float('inf')
        closest_point_key = None
    
        calibration_points = {
        "top_left": (96, 54), "top_center": (960, 54), "top_right": (1824, 54),
        "center_left": (96, 540), "center": (960, 540), "center_right": (1824, 540),
        "bottom_left": (96, 1026), "bottom_center": (960, 1026), "bottom_right": (1824, 1026),
        }

        for key, pos in calibration_points.items():
            distance = np.sqrt((pos[0] - gaze_x)**2 + (pos[1] - gaze_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_point_key = key
            
        return closest_point_key
    
    def _determine_quadrant(self, dx, dy):
        """Determina el cuadrante basado en la dirección del movimiento"""
        if dx <= 0 and dy <= 0:
            return "Q1"  # top_left
        elif dx > 0 and dy <= 0:
            return "Q2"  # top_right
        elif dx <= 0 and dy > 0:
            return "Q3"  # bottom_left
        else:
            return "Q4"  # bottom_right
    def _apply_dynamic_correction(self, iris_center, face_center, base_gaze):
        """Aplica corrección dinámica sobre la estimación base"""
        dx = iris_center[0] - face_center[0]
        dy = iris_center[1] - face_center[1]
        
        # Calcular corrección basada en escalas dinámicas
        if isinstance(self.calibrated_scales, dict):
            # Para calibración por cuadrantes, usar escalas promedio como referencia
            avg_scale_x = sum(s[0] for s in self.calibrated_scales.values()) / len(self.calibrated_scales)
            avg_scale_y = sum(s[1] for s in self.calibrated_scales.values()) / len(self.calibrated_scales)
            correction_x = dx * (self.scale_x - avg_scale_x) * 0.3
            correction_y = dy * (self.scale_y - avg_scale_y) * 0.4
        else:
            correction_x = dx * (self.scale_x - (self.calibrated_scales[0] if self.calibrated_scales else 10)) * 0.3
            correction_y = dy * (self.scale_y - (self.calibrated_scales[1] if self.calibrated_scales else 10)) * 0.4
        
        corrected_x = int(np.clip(base_gaze[0] + correction_x, 0, self.screen_w - 1))
        corrected_y = int(np.clip(base_gaze[1] + correction_y, 0, self.screen_h - 1))
        
        return (corrected_x, corrected_y)

    def _estimate_gaze_dynamic(self, iris_center, face_center):
        """Estimación dinámica mejorada con corrección de velocidad"""
        dx = iris_center[0] - face_center[0]
        dy = iris_center[1] - face_center[1]

        corrected_dx = dx - self.gaze_center_offset[0]
        corrected_dy = dy - self.gaze_center_offset[1]
        
        cx, cy = self.screen_w // 2, self.screen_h // 2
        
        # Aplicar escalas dinámicas con factor de corrección de velocidad
        adjusted_scale_x = self.scale_x * self.global_speed_factor
        adjusted_scale_y = self.scale_y * self.global_speed_factor
        
        gx = int(np.clip(cx + corrected_dx * adjusted_scale_x, 0, self.screen_w - 1))
        gy = int(np.clip(cy + corrected_dy * adjusted_scale_y, 0, self.screen_h - 1))
        
        return (gx, gy)

    def get_calibration_status(self):
        """Retorna el estado actual de la calibración"""
        return {
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
            'samples': len(self.iris_history),
            'vertical_boost': self.vertical_movement_boost,
            'last_movement': self.last_significant_movement
        }

    def reset(self):
        """Reset del calibrador"""
        print("Reset de calibrador ejecutado.")
        self.iris_history.clear()
        self.face_history.clear()
        self.movement_history_x.clear()
        self.movement_history_y.clear()
        self.scale_x = 6.0  # Valores corregidos
        self.scale_y = 9.0
        self.last_significant_movement = None

    # Dentro de la clase ImprovedDynamicCalibrator
    def get_movement_stats(self):
        """Retorna las estadísticas de movimiento para la adaptación del filtro."""
    # Asegúrate de que los historiales tengan suficientes datos
        if len(self.iris_history) < 20:
            return {'std_x': 0, 'std_y': 0}

        recent_dx = [i[0] - f[0] for i, f in zip(self.iris_history[-20:], self.face_history[-20:])]
        recent_dy = [i[1] - f[1] for i, f in zip(self.iris_history[-20:], self.face_history[-20:])]
    
        std_x = np.std(recent_dx)
        std_y = np.std(recent_dy)
    
        return {
        'std_x': std_x,
        'std_y': std_y
        }