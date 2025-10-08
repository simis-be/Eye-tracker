from collections import deque, Counter
import numpy as np

# Ejemplo: buffer de los últimos 15 puntos
fixation_buffer = deque(maxlen=15)
fixation_map = []

def update_fixation_map(gaze_pos):
    fixation_buffer.append(gaze_pos)

    if len(fixation_buffer) == fixation_buffer.maxlen:
        xs, ys = zip(*fixation_buffer)

        if np.std(xs) < 10 and np.std(ys) < 10:
            # Punto más común (modo)
            common = Counter(fixation_buffer).most_common(1)[0][0]
            fixation_map.append(common)

            # 🪵 Logging en consola
            print(f"[FIXATION] Añadido punto estable: {common}")

            # 📝 Opcional: guardar en un archivo
            with open("fixation_log.txt", "a") as f:
                f.write(f"{common[0]},{common[1]}\n")
