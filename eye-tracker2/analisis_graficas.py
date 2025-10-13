import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_and_plot_gaze_data(file_path="precision_test.csv"):
    """
    Lee el archivo CSV generado por el CalibrationLogger, calcula el error de precisión
    y genera dos gráficas para la memoria, excluyendo los primeros 5 segundos.
    """
    if not os.path.exists(file_path):
        print(f"Error: Archivo de registro no encontrado en la ruta: {file_path}")
        print("Asegúrate de ejecutar el eye-tracker en modo 'TESTING' primero.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return

    # --- 1. CÁLCULO DE MÉTRICAS CLAVE Y FILTRADO (CAMBIOS AQUÍ) ---

    # Error de distancia (Euclidiana)
    df['error_x'] = df['target_x'] - df['cursor_x']
    df['error_y'] = df['target_y'] - df['cursor_y']
    df['precision_error'] = np.sqrt(df['error_x']**2 + df['error_y']**2)

    # Normalizar el tiempo (útil para referencia, aunque se use el timestamp absoluto)
    df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]

    # 💡 LÓGICA DE CORTE: Ignorar los primeros 5 segundos
    CUTOFF_TIME = 5.0 # segundos

    if len(df) == 0:
        print("El archivo CSV no contiene datos.")
        return

    # Calcular la marca de tiempo absoluta a partir de la cual empezar a contar
    start_timestamp = df['timestamp'].iloc[0]
    cutoff_absolute_time = start_timestamp + CUTOFF_TIME

    # Filtrar el DataFrame
    df_filtered = df[df['timestamp'] >= cutoff_absolute_time].copy()

    if len(df_filtered) == 0:
        print(f"Error: No hay datos después de los primeros {CUTOFF_TIME} segundos.")
        return

    # Recalcular el tiempo relativo para que el primer punto filtrado sea t=0 en la gráfica
    df_filtered['relative_time_filtered'] = df_filtered['timestamp'] - df_filtered['timestamp'].iloc[0]

    # Recalcular métricas resumen con el dataset filtrado
    mean_error = df_filtered['precision_error'].mean()
    std_error = df_filtered['precision_error'].std()

    print(f"\n--- Resultados del Análisis (Filtrado a partir de {CUTOFF_TIME}s) ---")
    print(f"Muestras analizadas: {len(df_filtered)} (Ignoradas: {len(df) - len(df_filtered)})")
    print(f"Error de Precisión Medio (en píxeles): {mean_error:.2f} px")
    print(f"Desviación Estándar del Error (Jitter): {std_error:.2f} px")

    # ----------------------------------------------------------------------
    
    # --- 2. GRÁFICA 1: ERROR DE PRECISIÓN VS. TIEMPO (USANDO DATOS FILTRADOS) ---
    
    plt.figure(figsize=(12, 6))
    
    # Usar 'relative_time_filtered' para que el eje X inicie en 0
    plt.plot(df_filtered['relative_time_filtered'], df_filtered['precision_error'], 
             label='Error de Precisión (px)', linewidth=1.5, alpha=0.7)
    
    # Añadir línea del error medio para referencia
    plt.axhline(mean_error, color='r', linestyle='--', label=f'Error Medio ({mean_error:.0f}px)')
    
    plt.title(f'Precisión del Eye-Tracker a lo Largo del Tiempo (Excluidos los primeros {CUTOFF_TIME}s)', fontsize=14)
    plt.xlabel('Tiempo Transcurrido (segundos, donde t=0 es el 5s real)', fontsize=12)
    plt.ylabel('Error de Distancia (Píxeles)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.show()

    # --- 3. GRÁFICA 2: DISPERSIÓN DE PUNTOS DE MIRADA (ESTABILIDAD) ---
    
    # El punto objetivo se toma del dataset original (asumiendo que no cambia)
    target_x = df['target_x'].iloc[0]
    target_y = df['target_y'].iloc[0]
    
    plt.figure(figsize=(8, 8))
    
    # Usar el DataFrame filtrado para mostrar solo la dispersión después de 5s
    plt.scatter(df_filtered['cursor_x'], df_filtered['cursor_y'], s=5, alpha=0.4, 
                label=f'Posiciones Estimadas del Cursor (t > {CUTOFF_TIME}s)')
    
    # Marcar el punto objetivo real
    plt.scatter(target_x, target_y, marker='X', color='red', s=250, label='Punto Objetivo Real (Target)')
    
    plt.title(f'Dispersión de Puntos de Mirada (Estabilidad, t > {CUTOFF_TIME}s)', fontsize=14)
    plt.xlabel('Coordenada X del Cursor (Píxeles)', fontsize=12)
    plt.ylabel('Coordenada Y del Cursor (Píxeles)', fontsize=12)
    plt.legend()
    
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.grid(True, alpha=0.3)
    plt.show()

# --- EJECUTAR EL ANÁLISIS ---
if __name__ == "__main__":
    analyze_and_plot_gaze_data("precision_test.csv")