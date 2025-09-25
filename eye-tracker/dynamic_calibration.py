import numpy as np
from gaze_estimation import project_gaze_trig2, project_gaze_trig3





class DynamicCalibrator:
    def __init__(self, screen_size=(1920, 1080), history_len=60):
        self.screen_w, self.screen_h = screen_size
        self.iris_history = []
        self.face_history = []
        self.history_len = history_len

        self.scale_x = 10.0
        self.scale_y = 10.0
        self.alpha = 0.1

        self.min_scale = 5.0
        self.max_scale = 30.0

        self.no_movement_counter = 0
        self.stuck_threshold = 60  # frames sin movimiento = reseteo

        self.last_gaze = None
        self.calibrated_scales = None
        self.center_iris = None
        self.quadrants = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        self.quadrant_data = {
            q: {
                'iris_history': [],
                'face_history': [],
                'scale_x': 10.0,
                'scale_y': 10.0
            }
            for q in self.quadrants
        }

    def update(self, iris_center, face_center):
            
        # Determinar cuadrante actual
            qx, qy = iris_center[0], iris_center[1]
            quadrant = self._get_quadrant(qx, qy)

            qdata = self.quadrant_data[quadrant]
            qdata['iris_history'].append(iris_center)
            qdata['face_history'].append(face_center)

            target_scale_x = None
            target_scale_y = None

            if len(qdata['iris_history']) < 20:
                return

            iris_dx = [i[0] - f[0] for i, f in zip(qdata['iris_history'], qdata['face_history'])]
            iris_dy = [i[1] - f[1] for i, f in zip(qdata['iris_history'], qdata['face_history'])]

            range_dx = np.percentile(iris_dx, 95) - np.percentile(iris_dx, 5)
            range_dy = np.percentile(iris_dy, 95) - np.percentile(iris_dy, 5)

            std_dx = np.std(iris_dx)
            std_dy = np.std(iris_dy)

            if range_dx > 3 and std_dx < 15:
                target_scale_x = self.screen_w / max(range_dx, 1e-5)
                qdata['scale_x'] = (1 - self.alpha) * qdata['scale_x'] + self.alpha * target_scale_x
            else:
                print(f"[{quadrant.upper()}] No se actualiza scale_x (range_dx={range_dx:.2f}, std_dx={std_dx:.2f})")

            if range_dy > 1.5 and std_dy < 25:
                target_scale_y = self.screen_h / max(range_dy, 1e-5)
                qdata['scale_y'] = (1 - self.alpha) * qdata['scale_y'] + self.alpha * target_scale_y
            else:
                print(f"[{quadrant.upper()}] No se actualiza scale_y (range_dy={range_dy:.2f}, std_dy={std_dy:.2f})")

        # Limitar la relación y valores extremos
            max_ratio = 3.0
            if qdata['scale_y'] > qdata['scale_x'] * max_ratio:
                qdata['scale_y'] = qdata['scale_x'] * max_ratio

            qdata['scale_x'] = np.clip(qdata['scale_x'], 10, 1000)
            qdata['scale_y'] = np.clip(qdata['scale_y'], 10, 1000)

        # Limitar historial
            qdata['iris_history'] = qdata['iris_history'][-self.history_len:]
            qdata['face_history'] = qdata['face_history'][-self.history_len:]

            print(f"[{quadrant.upper()}] scale_x={qdata['scale_x']:.2f}, scale_y={qdata['scale_y']:.2f}, range=({range_dx:.1f}, {range_dy:.1f})")



    def load_calibration(self, scales, center_iris):
        self.calibrated_scales = scales
        self.center_iris = center_iris

    def estimate_gaze(self, iris_center, face_center):
       

    # Proyección calibrada
        return project_gaze_trig2(
        iris=iris_center,
        face=face_center,
        screen=(self.screen_w, self.screen_h),
        CH=50.0,
        alpha_deg=10.0,
        sensitivity=1.0,
        scales=self.calibrated_scales,
        center_iris=self.center_iris
        )
    
    def estimate_gaze2(self, iris_center, face_center):

        return project_gaze_trig3 (iris=iris_center,
        face=face_center,
        screen=(self.screen_w, self.screen_h),
        CH=50.0,
        alpha_deg=5.0,
        sensitivity=1.0,
        pixel_to_mm=0.25,
        )
    

    def _estimate_gaze_dynamic(self, iris_center, face_center):
    # Tu método original basado en escalas dinámicas
        dx = iris_center[0] - face_center[0]
        dy = iris_center[1] - face_center[1]
        cx, cy = self.screen_w // 2, self.screen_h // 2

        quadrant = self._get_quadrant(iris_center[0], iris_center[1])
        qdata = self.quadrant_data[quadrant]

        gx = int(np.clip(cx + dx * qdata['scale_x'], 0, self.screen_w))
        gy = int(np.clip(cy + dy * qdata['scale_y'], 0, self.screen_h))
        return gx, gy


    def _get_quadrant(self, x, y):
        cx, cy = self.screen_w // 2, self.screen_h // 2
        if x < cx and y < cy:
            return 'top_left'
        elif x >= cx and y < cy:
            return 'top_right'
        elif x < cx and y >= cy:
            return 'bottom_left'
        else:
            return 'bottom_right'


    def reset(self):
        print("🔄 Reset de calibrador ejecutado.")
        self.iris_history.clear()
        self.face_history.clear()
        self.scale_x = 10.0
        self.scale_y = 10.0
        self.no_movement_counter = 0
        self.last_gaze = None
