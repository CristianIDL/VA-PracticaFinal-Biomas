'''
segmentar_kmeans.py: Implementación de K-Means para clustering de datos.
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from src.prints import crear_headline

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


def segmentar_kmeans(imagen, k=3, intentos=3):
    """Segmenta una imagen usando K-Means clustering por HSV"""

    print(crear_headline(f"Segmentación K-Means con {k} clusters (HSV) ..."))

    inicio = time.time()

    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)

    # Convertimos la imagen a una matriz de píxeles
    altura, ancho, canales = imagen_hsv.shape
    pixeles = imagen_hsv.reshape((-1, 3))  # Ahora son valores HSV

    # Convertimos a float32
    pixeles = np.float32(pixeles)

    print(f"Total de pixeles: {len(pixeles):,}")

    # Definimos los criterios de parada y aplicamos K-Means
    critario = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # - - - EJECUTAR K-MEANS - - -

    _, etiquetas, centros = cv2.kmeans(
        pixeles, # Datos de entrada
        k, # Número de clusters
        None, # No le damos un array de etiquetas inicial
        critario, # Saber cuando acá terminamos el algoritmo
        intentos, # Número de intentos    
        cv2.KMEANS_RANDOM_CENTERS # Centros iniciales aleatorios
    )

    # Pasamos los centros a uint8
    centros = np.uint8(centros)

    # Creamos la imagen segmentada en HSV
    imagen_segmentada_hsv = centros[etiquetas.flatten()]
    imagen_segmentada_hsv = imagen_segmentada_hsv.reshape((imagen_hsv.shape))

    # Regresamos la imagen segmentada a RGB para su visualización
    imagen_segmentada = cv2.cvtColor(imagen_segmentada_hsv, cv2.COLOR_HSV2RGB)

    t_procesamiento = time.time() - inicio
    print(f"Segmentación completada en {t_procesamiento:.2f} segundos.") 

    print(f"\nCentros de color (HSV y RGB): {centros}")
    for i, centro_hsv in enumerate(centros):
        # Convertimos el centro de HSV a RGB para mostrarlo
        centro_rgb = cv2.cvtColor(np.uint8([[centro_hsv]]), cv2.COLOR_HSV2RGB)[0][0]
        print(f"+ Cluster {i+1}:")
        print(f"  HSV: H={centro_hsv[0]:3d}° S={centro_hsv[1]:3d} V={centro_hsv[2]:3d}")
        print(f"  RGB: {tuple(centro_rgb)}, Hex: #{centro_rgb[0]:02x}{centro_rgb[1]:02x}{centro_rgb[2]:02x}")

    # Calculamos las estadísticas de los clusters
    print(f"\nEstadísticas de los {k} clusters:")
    for i in range(k):
        conteo = np.sum(etiquetas == i)
        porcentaje = (conteo / len(pixeles)) * 100
        print(f"+ Cluster {i+1}: {conteo:,} píxeles ({porcentaje:.2f}%)")

    return imagen_segmentada, etiquetas, centros, t_procesamiento

def visualizar_comparacion(imagen_original, imagen_segmentada, k, t_procesamiento):
    """Muestra la imagen original y la segmentada lado a lado."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Imagen original
    axes[0].imshow(imagen_original)
    axes[0].set_title("Imagen Original", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Imagen segmentada
    axes[1].imshow(imagen_segmentada)
    axes[1].set_title(f"Imagen Segmentada K-Means (k={k})\nTiempo: {t_procesamiento:.2f} seg"
                      , fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# TODO: Agregar funcion para probar diferentes K si es necesario.