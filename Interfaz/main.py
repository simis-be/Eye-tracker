# -*- coding: utf-8 -*-

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.properties import StringProperty
from kivy.lang import Builder
import subprocess
import json
from pathlib import Path

# Archivo de configuración para guardar las sensibilidades
CONFIG_FILE = Path("eye_tracker_config.json")
tracker_process = None

Builder.load_file("menu.kv")

class MenuPrincipal(BoxLayout):
    estado_tracker = StringProperty("Tracker: Detenido")
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Sincronizar los sliders con la configuración cargada
        Clock.schedule_once(self.sincronizar_ui_con_config, 0.1)

        # Iniciar el tracker automáticamente
       # Clock.schedule_once(self.iniciar_tracker_automatico, 2)
        Clock.schedule_interval(self.actualizar_estado_ui, 0.5)

    def sincronizar_ui_con_config(self, dt):
        """Asegura que los sliders reflejen los valores del archivo de configuración."""
        self.ids.slider_h.value = self.config.get('sensibilidad_h', 50)
        self.ids.slider_v.value = self.config.get('sensibilidad_v', 50) 

        self.ids.slider_speed_h.value = self.config.get('velocidad_h', 50)
        self.ids.slider_speed_v.value = self.config.get('velocidad_v', 50)

    def cargar_config(self):
        """Carga la configuración o crea una por defecto."""
        if CONFIG_FILE.exists():
            try:
                config = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
                # Usar valores predeterminados si faltan en el archivo
                return {
                    "sensibilidad_h": config.get("sensibilidad_h", 50),
                    "sensibilidad_v": config.get("sensibilidad_v", 50),
                   
                    "velocidad_h": config.get("velocidad_h", 50),
                    "velocidad_v": config.get("velocidad_v", 50)



                }
            except Exception as e:
                print(f"❌ Error leyendo config: {e}")
        return {"sensibilidad_h": 50, "sensibilidad_v": 50, "velocidad_h": 50, "velocidad_v": 50}

    def guardar_config(self):
        try:
            CONFIG_FILE.write_text(json.dumps(self.config, indent=4), encoding='utf-8')
            print("✅ Configuración guardada.")
        except Exception as e:
            print(f"❌ Error guardando config: {e}")

    


    def controlar_tracker(self, accion):
        """
        Inicia, detiene o reinicia el proceso del eye-tracker.
        Acciones: 'iniciar', 'detener', 'recalibrar'.
        """
        global tracker_process

        if accion == 'iniciar':
            if tracker_process is None or tracker_process.poll() is not None:
            # 1. Resolver y verificar la ruta del archivo
                ruta_tracker = Path("..")/ "tf" / "eye-tracker2" / "main.py"
                ruta_tracker_absoluta = ruta_tracker.resolve()

                print(f"Buscando el script en: {ruta_tracker_absoluta}")

                if not ruta_tracker_absoluta.exists():
                    print(f"❌ Error: El archivo {ruta_tracker_absoluta} no se encontró.")
                    return # Salir si el archivo no existe

            # 2. Ejecutar el subproceso con la ruta absoluta
                try:
                # Usa `subprocess.run` para ver la salida de error
                # El argumento `capture_output=True` capturará los errores
                    resultado = subprocess.run(["python", str(ruta_tracker_absoluta)], check=True, capture_output=True, text=True)
                    print("✅ Tracker iniciado correctamente.")
                    print("Salida del tracker:", resultado.stdout)

                except FileNotFoundError:
                    print("❌ Error: No se pudo encontrar el comando 'python'. Asegúrate de que Python está en tu PATH.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error al ejecutar el tracker. Código de salida: {e.returncode}")
                    print(f"Salida de error del tracker: {e.stderr}")
                except Exception as e:
                    print(f"❌ Ocurrió un error inesperado: {e}")

        elif accion == 'iniciar':
            if tracker_process is None or tracker_process.poll() is not None:
                ruta_tracker = Path("..") /"tf"/ "eye-tracker2" / "main.py"
                ruta_tracker = ruta_tracker.resolve()
                if ruta_tracker.exists():
                    tracker_process = subprocess.Popen(["python", str(ruta_tracker)])
                    print("✅ Tracker iniciado")
        elif accion == 'recalibrar':
            print("🔄 Recalibrando...")
            self.controlar_tracker('detener')
            # Esperar un momento antes de reiniciar
            Clock.schedule_once(lambda dt: self.controlar_tracker('iniciar'), 1)
            
    def actualizar_estado_ui(self, dt):
        """Actualiza el estado de la UI y el texto del botón de control."""
        global tracker_process
        tracker_activo = tracker_process is not None and tracker_process.poll() is None
        self.estado_tracker = "Tracker: Activo ✅" if tracker_activo else "Tracker: Detenido ❌"
        self.ids.estado_label.text = self.estado_tracker
        
        # Cambiar el texto del botón principal
        if tracker_activo:
            self.ids.btn_principal.text = "Detener"
            self.ids.btn_principal.background_color = [0.8, 0.2, 0.2, 1] # Rojo
        else:
            self.ids.btn_principal.text = "Iniciar"
            self.ids.btn_principal.background_color = [0.2, 0.8, 0.2, 1] # Verde
    
    def on_sensibilidad_h_change(self, value):
        self.config["sensibilidad_h"] = int(value)
        self.guardar_config()
        print(f"🎚 Sensibilidad Horizontal: {int(value)}")

    def on_sensibilidad_v_change(self, value):
        self.config["sensibilidad_v"] = int(value)
        self.guardar_config()
        print(f"🎚 Sensibilidad Vertical: {int(value)}")

    def on_velocidad_h_change(self, value):
        self.config["velocidad_h"] = int(value)
        self.guardar_config()
        print(f"🚀 Velocidad Horizontal: {int(value)}")
        
    def on_velocidad_v_change(self, value):
        self.config["velocidad_v"] = int(value)
        self.guardar_config()
        print(f"🚀 Velocidad Vertical: {int(value)}")


    def iniciar_tracker(self):
        """Inicia el proceso del eye-tracker."""
        global tracker_process
        
    
    # Define la ruta relativa a la carpeta 'tf'
    # Esto asegura que el path sea correcto sin importar donde se ejecuta el script.
        ruta_base = Path(__file__).parent.parent
    
    # Ahora, construye la ruta al eye-tracker desde la ruta base
        ruta_tracker = ruta_base/ "tf" / "eye-tracker2" / "main.py"
    
        print(f"Buscando el script en: {ruta_tracker.resolve()}")

        if not ruta_tracker.exists():
            print(f"❌ Error: El archivo {ruta_tracker.resolve()} no se encontró.")
            return # Salir si el archivo no existe

        try:
            tracker_process = subprocess.Popen(["python", str(ruta_tracker)])
            print("✅ Tracker iniciado correctamente.")
        
        except FileNotFoundError:
            print("❌ Error: No se pudo encontrar el comando 'python'. Asegúrate de que Python está en tu PATH.")
        except Exception as e:
            print(f"❌ Ocurrió un error inesperado al iniciar el tracker: {e}")

       
        
    def detener_tracker(self):
        """Detiene el proceso del eye-tracker."""
        global tracker_process
        if tracker_process is not None and tracker_process.poll() is None:
            tracker_process.terminate()
            tracker_process.wait()
            tracker_process = None
            print("Tracker detenido")

    def actualizar_estado(self, dt):
        """Actualiza el estado de la UI."""
        global tracker_process
        tracker_activo = tracker_process is not None and tracker_process.poll() is None
        self.estado_tracker = "Tracker: Activo ✅" if tracker_activo else "Tracker: Detenido"
        self.ids.estado_label.text = self.estado_tracker
        
    def recalibrar(self):
        print("🎯 Acción ejecutada: Recalibrar")
        self.detener_tracker()
        self.iniciar_tracker()

    def on_sensibilidad_h_change(self, value):
        self.config["sensibilidad_h"] = int(value)
        self.guardar_config()
        print(f"🎚 Sensibilidad Horizontal: {int(value)}")
        
    def on_sensibilidad_v_change(self, value):
        self.config["sensibilidad_v"] = int(value)
        self.guardar_config()
        print(f"🎚 Sensibilidad Vertical: {int(value)}")

class EyeTrackerApp(App):
    def cargar_config(self):
        """Loads configuration from file or returns defaults."""
        if CONFIG_FILE.exists():
            try:
                config = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
                return {
                    "sensibilidad_h": config.get("sensibilidad_h", 50),
                    "sensibilidad_v": config.get("sensibilidad_v", 50)
                }
            except Exception as e:
                print(f"❌ Error leyendo config: {e}")
        return {"sensibilidad_h": 50, "sensibilidad_v": 50}
    
    def build(self):
        config = self.cargar_config()
        return MenuPrincipal(config=config)

if __name__ == "__main__":
    EyeTrackerApp().run()