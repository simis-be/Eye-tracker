import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_and_plot_gaze_data(file_path="precision_test.csv"):
    """
    Lee el archivo CSV generado por el CalibrationLogger, calcula el error de precisión
    y genera dos gráficas para la memoria:
    1. Error de Precisión vs. Tiempo.
    2. Dispersión de Puntos de Mirada (Estabilidad).
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

    # --- 1. CÁLCULO DE MÉTRICAS CLAVE ---
    
    # Error de distancia (Euclidiana) entre el objetivo real y la posición del cursor
    # Error = sqrt((Target X - Cursor X)^2 + (Target Y - Cursor Y)^2)
    df['error_x'] = df['target_x'] - df['cursor_x']
    df['error_y'] = df['target_y'] - df['cursor_y']
    df['precision_error'] = np.sqrt(df['error_x']**2 + df['error_y']**2)
    
    # Calcular métricas resumen
    mean_error = df['precision_error'].mean()
    std_error = df['precision_error'].std()
    
    print(f"\n--- Resultados del Análisis del Archivo '{file_path}' ---")
    print(f"Número total de muestras: {len(df)}")
    print(f"Error de Precisión Medio (en píxeles): {mean_error:.2f} px")
    print(f"Desviación Estándar del Error (Jitter): {std_error:.2f} px")
    
    # --- 2. GRÁFICA 1: ERROR DE PRECISIÓN VS. TIEMPO ---
    
    # Normalizar el tiempo para que el eje X empiece en cero
    df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['relative_time'], df['precision_error'], label='Error de Precisión (px)', linewidth=1.5, alpha=0.7)
    
    # Añadir línea del error medio para referencia
    plt.axhline(mean_error, color='r', linestyle='--', label=f'Error Medio ({mean_error:.0f}px)')
    
    plt.title('Precisión del Eye-Tracker a lo Largo del Tiempo (Error Euclidiano)', fontsize=14)
    plt.xlabel('Tiempo Transcurrido (segundos)', fontsize=12)
    plt.ylabel('Error de Distancia (Píxeles)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.show()

    # --- 3. GRÁFICA 2: DISPERSIÓN DE PUNTOS DE MIRADA (ESTABILIDAD) ---
    
    # El punto objetivo es constante durante la prueba (tomamos la primera fila)
    target_x = df['target_x'].iloc[0]
    target_y = df['target_y'].iloc[0]
    
    plt.figure(figsize=(8, 8))
    
    # Dibujar la nube de puntos del cursor (Estabilidad/Jitter)
    plt.scatter(df['cursor_x'], df['cursor_y'], s=5, alpha=0.4, label='Posiciones Estimadas del Cursor')
    
    # Marcar el punto objetivo real
    plt.scatter(target_x, target_y, marker='X', color='red', s=250, label='Punto Objetivo Real (Target)')
    
    plt.title('Dispersión de Puntos de Mirada (Estabilidad)', fontsize=14)
    plt.xlabel('Coordenada X del Cursor (Píxeles)', fontsize=12)
    plt.ylabel('Coordenada Y del Cursor (Píxeles)', fontsize=12)
    plt.legend()
    
    # Es crucial establecer el aspecto igual para que la dispersión se vea fiel a la realidad
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.grid(True, alpha=0.3)
    plt.show()

# --- EJECUTAR EL ANÁLISIS ---
# Llamar a la función con el nombre de tu archivo de registro
if __name__ == "__main__":
    analyze_and_plot_gaze_data("precision_test.csv")