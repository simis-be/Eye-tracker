import csv
import time
import os

class CalibrationLogger:
    """Clase para registrar las posiciones reales y estimadas durante la calibración o pruebas."""

    def __init__(self, filename="gaze_log.csv"):
        # Define la ruta del archivo
        self.file_path = os.path.join(os.getcwd(), filename)
        self.headers = [
            'timestamp', 
            'target_x', 'target_y', 
            'cursor_x', 'cursor_y', 
            'scale_x', 'scale_y', 
            'std_x', 'std_y'
        ]
        self.initialize_file()
        print(f"Logger inicializado. Guardando datos en: {self.file_path}")

    def initialize_file(self):
        """Crea el archivo CSV y escribe los encabezados."""
        # Sobrescribe el archivo si ya existe, para iniciar una nueva prueba
        with open(self.file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)
    
    def log_data(self, target_pos, cursor_pos, scale_stats):
        """
        Registra un único punto de datos.
        :param target_pos: Posición real del objetivo (x, y).
        :param cursor_pos: Posición estimada/cursor (x, y).
        :param scale_stats: Diccionario con estadísticas de la calibración dinámica.
        """
        # Asegúrate de que los datos tengan el formato correcto
        row = [
            time.time(),
            target_pos[0], target_pos[1],
            cursor_pos[0], cursor_pos[1],
            scale_stats.get('scale_x', 0),
            scale_stats.get('scale_y', 0),
            scale_stats.get('std_x', 0),
            scale_stats.get('std_y', 0)
        ]
        
        # Escribe la fila en el archivo
        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    def get_file_path(self):
        """Retorna la ruta del archivo para su posterior análisis."""
        return self.file_path