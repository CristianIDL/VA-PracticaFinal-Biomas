'''
src.audio.py: Módulo para manejar la reproducción de audio según el bioma detectado.
'''

import pygame
import os
from pathlib import Path
from src.prints import crear_headline
from time import time

def inicializar_audio():
    """Inicializa el sistema de audio."""
    try:
        pygame.mixer.init()
        print(f"Sistema de audio inicializado correctamente.")
        return True
    except Exception as e:
        print(f"Error al inicializar el sistema de audio: {e}")
        return False
    
def reproducir_audio(bioma):
    """Reproduce el archivo de audio correspondiente al bioma dado."""
    
    # Normalizamos la entrada del bioma
    bioma = bioma.lower().strip()

    # Mapeo de biomas a archivos de audio
    mapeo_biomas = {
        "desconocido": "desconocido.mp3",
        "montana": "montana.mp3",
        "pradera": "pradera.mp3",
        "playa": "playa.mp3"
    }

    # Seleccionamos el archivo de audio correspondiente
    nombre_archivo = mapeo_biomas.get(bioma, "desconocido.mp3") # desconocido por defecto

    ruta_audio = os.path.join('audio', nombre_archivo)

    # Verificar si el archivo existe
    if not Path(ruta_audio).is_file():
        print(f"El archivo de audio no existe: {ruta_audio}")
        print(f"Por favor, revisa que el archivo existe en la carpeta 'audio/'")
        return
    
    try:
        print(crear_headline(f"Reproduciendo audio para el bioma: {bioma}"))
        # Cargar el archivo
        pygame.mixer.music.load(ruta_audio)
        # Reproducir el audio
        pygame.mixer.music.play()
        
        # Esperar a que termine la reproducción
        while pygame.mixer.music.get_busy():
            time.sleep(2) # Esperar 2 segundos antes de verificar nuevamente
        
        return True
    
    except Exception as e:
        print(f"Error al reproducir el audio: {e}")
        return False    

def detener_audio():
    """Detiene la reproducción de audio."""
    pygame.mixer.music.stop()

def finalizar_audio():
    """Finaliza el sistema de audio."""
    pygame.mixer.quit()