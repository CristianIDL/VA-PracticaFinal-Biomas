'''
src/feature_extraction.py: Extracción de características de imágenes segmentadas.
'''

import numpy as np
import cv2


def extraer_features(imagen_rgb, etiquetas, centros_hsv, k):
    """
    Extrae características de una imagen segmentada con K-means en espacio HSV.
    
    Args:
        imagen_rgb: Imagen original en RGB
        etiquetas: Etiquetas de cluster por píxel (de K-means)
        centros_hsv: Centros de clusters en espacio HSV
        k: Número de clusters
    
    Returns:
        numpy array con features extraídas
    """
    
    altura, ancho = imagen_rgb.shape[:2]
    total_pixeles = altura * ancho
    etiquetas_planas = etiquetas.flatten()
    
    features = []
    
    # ========== FEATURES 1-3K: COLORES DOMINANTES HSV ==========
    # Para cada cluster: H, S, V (3 valores × k clusters)
    for i in range(k):
        h, s, v = centros_hsv[i]
        features.extend([h, s, v])
    
    # ========== FEATURES (3K+1) - (4K): PROPORCIONES ==========
    # Porcentaje de píxeles en cada cluster
    for i in range(k):
        proporcion = np.sum(etiquetas_planas == i) / total_pixeles
        features.append(proporcion)
    
    # ========== FEATURES (4K+1) - (5K): POSICIÓN VERTICAL ==========
    # Posición vertical promedio de cada cluster (0=arriba, 1=abajo)
    # Útil para detectar: cielo arriba, agua/tierra abajo
    y_coords, x_coords = np.meshgrid(np.arange(altura), np.arange(ancho), indexing='ij')
    y_coords_flat = y_coords.flatten()
    
    for i in range(k):
        mascara = etiquetas_planas == i
        if np.sum(mascara) > 0:
            pos_vertical_promedio = np.mean(y_coords_flat[mascara]) / altura
        else:
            pos_vertical_promedio = 0.5  # Fallback al centro
        features.append(pos_vertical_promedio)
    
    # ========== FEATURES (5K+1) - (6K): POSICIÓN HORIZONTAL ==========
    # Posición horizontal promedio de cada cluster (0=izquierda, 1=derecha)
    for i in range(k):
        mascara = etiquetas_planas == i
        if np.sum(mascara) > 0:
            pos_horizontal_promedio = np.mean(x_coords.flatten()[mascara]) / ancho
        else:
            pos_horizontal_promedio = 0.5  # Fallback al centro
        features.append(pos_horizontal_promedio)
    
    # ========== FEATURES ADICIONALES: ESTADÍSTICAS GLOBALES ==========
    
    # Feature: Dominancia del cluster más grande
    proporciones = [np.sum(etiquetas_planas == i) / total_pixeles for i in range(k)]
    features.append(max(proporciones))  # Proporción del cluster dominante
    
    # Feature: Índice de diversidad (entropía simplificada)
    # Valores altos = muchos clusters con proporciones similares
    # Valores bajos = un cluster domina
    diversidad = -sum([p * np.log(p + 1e-10) for p in proporciones if p > 0])
    features.append(diversidad)
    
    # Feature: Compactación vertical
    # Mide qué tan agrupados verticalmente están los clusters
    # Útil para diferenciar paisajes horizontales (playa) vs complejos (bosque)
    pos_verticales = []
    for i in range(k):
        mascara = etiquetas_planas == i
        if np.sum(mascara) > 0:
            pos_verticales.append(np.mean(y_coords_flat[mascara]) / altura)
    
    if len(pos_verticales) > 1:
        compactacion_vertical = np.std(pos_verticales)
    else:
        compactacion_vertical = 0.0
    features.append(compactacion_vertical)
    
    # Feature: Saturación promedio (indicador de viveza de colores)
    saturacion_promedio = np.mean([centros_hsv[i][1] for i in range(k)])
    features.append(saturacion_promedio)
    
    # Feature: Valor (brillo) promedio
    valor_promedio = np.mean([centros_hsv[i][2] for i in range(k)])
    features.append(valor_promedio)
    
    # Convertir a numpy array
    return np.array(features, dtype=np.float32)


def obtener_nombres_features(k):
    """
    Retorna los nombres de las features para documentación.
    
    Args:
        k: Número de clusters usado
    
    Returns:
        Lista de nombres de features
    """
    nombres = []
    
    # Colores HSV de cada cluster
    for i in range(k):
        nombres.extend([f'cluster_{i+1}_H', f'cluster_{i+1}_S', f'cluster_{i+1}_V'])
    
    # Proporciones
    for i in range(k):
        nombres.append(f'cluster_{i+1}_proporcion')
    
    # Posiciones verticales
    for i in range(k):
        nombres.append(f'cluster_{i+1}_pos_vertical')
    
    # Posiciones horizontales
    for i in range(k):
        nombres.append(f'cluster_{i+1}_pos_horizontal')
    
    # Features globales
    nombres.extend([
        'proporcion_dominante',
        'diversidad',
        'compactacion_vertical',
        'saturacion_promedio',
        'valor_promedio'
    ])
    
    return nombres


def calcular_dimensiones_features(k):
    """
    Calcula cuántas features se generan para un valor de k dado.
    
    Args:
        k: Número de clusters
    
    Returns:
        Número total de features
    """
    # 3k (HSV) + k (proporción) + k (pos_vert) + k (pos_horiz) + 5 (globales)
    return 6 * k + 5


if __name__ == "__main__":
    # Test de dimensiones
    for k_test in [3, 5, 7]:
        dim = calcular_dimensiones_features(k_test)
        print(f"K={k_test} → {dim} features")
        print(f"  Nombres: {obtener_nombres_features(k_test)[:5]}... (primeras 5)")
        print()