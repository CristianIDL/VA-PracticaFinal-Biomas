'''
main.py: Programa para clasificación de biomas usando visión artificial.
'''
import cv2
import os
import time

from tkinter import Tk, filedialog
from src.segmentar_kmeans import cargar_imagen, segmentar_kmeans, visualizar_comparacion
from src.audio import inicializar_audio, reproducir_audio, detener_audio, finalizar_audio
from src.prints import crear_headline

def seleccionar_imagen():
    """Cuadro de diálogo para seleccionar una imagen."""
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal
    root.attributes('-topmost', True)  # Mantener el cuadro hasta adelanta

    archivo = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )

    root.destroy() # Cerrar la ventana al finalizar
    return archivo

def cargar_imagen(ruta, max_dimension=600):
    """Carga una imagen y redimensionarla a un tamaño manejable."""

    imagen = cv2.imread(ruta)
    
    if imagen is None:  
        raise ValueError(f"No se pudo cargar la imagen desde la ruta: {ruta}")
    
    # Redimensionar si es necesario
    altura, ancho = imagen.shape[:2]

    if max(altura, ancho) > max_dimension:
        escala = max_dimension / float(max(altura, ancho))
        nuevo_ancho = int(ancho * escala)
        nueva_altura = int(altura * escala)
        imagen = cv2.resize(imagen,(nuevo_ancho, nueva_altura), interpolation=cv2.INTER_AREA)
        print(f"Imagen redimensionada de [{ancho}x{altura}] a [{nuevo_ancho}x{nueva_altura}] para procesamiento.")

    # Convertir a espacio de color RGB
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    return imagen_rgb

def main():
    """Main"""

    print(crear_headline("Práctica Final - CLASIFICACIÓN DE BIOMAS MEDIANTE VISIÓN ARTIFICIAL"))

    # Inicializar audio
    audio = inicializar_audio()

    inEternum = True

    while inEternum:
        # 1. Selección de imagen
        print("\n[1/6] Selección de imagen")
        ruta_imagen = seleccionar_imagen()

        if not ruta_imagen:
            print("No se seleccionó ninguna imagen. Saliendo...")
            break

        print(f"Imagen seleccionada: {ruta_imagen}")

        # 2. Solicitar número de clases (K)
        print("\n[2/6] Configuración de segmentación")
        try:
            k = int(input("Ingresa el número de clases (k) para la segmentación: "))
            if k < 2:
                print("Debe haber al menos 2 clases. Se usará k = 3 por defecto.")  
                k = 3
        except:
            print("Valor inválido. Se usará k = 3 por defecto.")
            k = 3

        # 3. Solicitar número de puntos muestreados
        print("\n[3/6] Configuración de muestreo")
        try:
            n_puntos = int(input("Ingresa el número de puntos de muestreo: "))
            if n_puntos <= 0:
                print("El número de puntos debe ser positivo. Se usarán 100 puntos por defecto.")
                n_puntos = 100
        except:
            print("Valor inválido. Se usarán 100 puntos por defecto.")
            n_puntos = 100

        # 4. Detección de imagen

        print("\n[4/6] Procesando imagen...")
        t_inicio = time.time()

        try:
            imagen_cargada = cargar_imagen(ruta_imagen)
            imagen_segmentada, _, _, t_procesamiento = segmentar_kmeans(imagen_cargada, k=k)
            visualizar_comparacion(imagen_cargada, imagen_segmentada, k, t_procesamiento)

            # TODO: Terminar la lógica de extracción de características y clasificación

            bioma = "desconocido"  # Placeholder para el bioma detectado
            confianza = 0.85  # Placeholder para el nivel de confianza

        except Exception as e:
            print(f"Error durante el procesamiento de la imagen: {e}")
            continue

        t_procesamiento = time.time() - t_inicio
        print(f"Procesamiento completado en {t_procesamiento:.2f} segundos.")

        # 5. Reproducir audio correspondiente al bioma

        print("\n[5/6] Reproducir audio del bioma detectado")
        
        if audio:
            reproducir_audio(bioma)

        # 6. Mostrar resultados
        print(f"\n[6/6] Resultados:")
        print(f"Bioma detectado: {bioma}")
        print(f"Nivel de confianza: {confianza:.2f}")
        print("- - " * 30)

        # 7. Preguntar si desea procesar otra imagen
        print("\n" + "= = " * 30)
        continuar = input("¿Deseas procesar otra imagen? (s/n): ").lower().strip()

        if continuar not in ['s', 'si', 'sí', 'y', 'yes']:
            print("\nFinalizando el programa. Gracias por usarlo :D")
            inEternum = False

    # Una vez termina el while, finalizar audio
    if audio:
        finalizar_audio()

if __name__ == "__main__":
    main()