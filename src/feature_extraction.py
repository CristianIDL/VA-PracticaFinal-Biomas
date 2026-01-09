'''
src/feature_extraction.py: Extracción de características de imágenes segmentadas.
Versión 2: Incluye ordenamiento de clusters y features de textura (LBP).
'''

import numpy as np
import cv2
from skimage.feature import local_binary_pattern


def ordenar_clusters_por_brillo(centros_hsv, etiquetas):
    """
    Ordena los clusters por brillo (canal V) para hacerlos comparables entre imágenes.
    
    CRÍTICO: Sin esto, cluster 0 en imagen A ≠ cluster 0 en imagen B
    
    Args:
        centros_hsv: Array de centros HSV shape (k, 3)
        etiquetas: Array de etiquetas shape (altura, ancho)
    
    Returns:
        centros_ordenados: Centros ordenados por V (de oscuro a brillante)
        etiquetas_reordenadas: Etiquetas actualizadas según el nuevo orden
    """
    k = len(centros_hsv)
    
    # Obtener índices ordenados por canal V (índice 2)
    orden = np.argsort(centros_hsv[:, 2])  # De menor a mayor brillo
    
    # Reordenar centros
    centros_ordenados = centros_hsv[orden]
    
    # Crear mapeo de etiquetas antiguas → nuevas
    mapeo = np.zeros(k, dtype=int)
    for nuevo_idx, viejo_idx in enumerate(orden):
        mapeo[viejo_idx] = nuevo_idx
    
    # Reordenar etiquetas
    etiquetas_reordenadas = mapeo[etiquetas]
    
    return centros_ordenados, etiquetas_reordenadas


def extraer_lbp_features(imagen_rgb, n_bins=10):
    """
    Extrae features de textura usando Local Binary Patterns (LBP).
    
    LBP mide la rugosidad/textura de la imagen:
    - Playa: valores bajos (textura suave)
    - Pradera: valores medios (textura moderada)
    - Montaña: valores altos (textura rugosa)
    
    Args:
        imagen_rgb: Imagen en RGB
        n_bins: Número de bins para el histograma LBP
    
    Returns:
        Array de features LBP (histograma normalizado)
    """
    # Convertir a escala de grises
    imagen_gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
    
    # Calcular LBP
    # P=8 puntos, R=1 radio (estándar para textura local)
    lbp = local_binary_pattern(imagen_gris, P=8, R=1, method='uniform')
    
    # Crear histograma
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalizar histograma (suma = 1)
    hist = hist.astype(float)
    hist = hist / (hist.sum() + 1e-10)
    
    return hist


def extraer_features(imagen_rgb, etiquetas, centros_hsv, k):
    """
    Extrae características de una imagen segmentada con K-means en espacio HSV.
    
    MEJORAS V2:
    - Ordena clusters por brillo (V) para consistencia
    - Agrega features de textura (LBP)
    
    Args:
        imagen_rgb: Imagen original en RGB
        etiquetas: Etiquetas de cluster por píxel (de K-means)
        centros_hsv: Centros de clusters en espacio HSV
        k: Número de clusters
    
    Returns:
        numpy array con features extraídas
    """
    
    # ========== PASO CRÍTICO: ORDENAR CLUSTERS ==========
    centros_hsv, etiquetas = ordenar_clusters_por_brillo(centros_hsv, etiquetas)
    
    altura, ancho = imagen_rgb.shape[:2]
    total_pixeles = altura * ancho
    etiquetas_planas = etiquetas.flatten()
    
    features = []
    
    # ========== FEATURES 1-3K: COLORES DOMINANTES HSV (ORDENADOS) ==========
    for i in range(k):
        h, s, v = centros_hsv[i]
        features.extend([h, s, v])
    
    # ========== FEATURES (3K+1) - (4K): PROPORCIONES ==========
    for i in range(k):
        proporcion = np.sum(etiquetas_planas == i) / total_pixeles
        features.append(proporcion)
    
    # ========== FEATURES (4K+1) - (5K): POSICIÓN VERTICAL ==========
    y_coords, x_coords = np.meshgrid(np.arange(altura), np.arange(ancho), indexing='ij')
    y_coords_flat = y_coords.flatten()
    
    for i in range(k):
        mascara = etiquetas_planas == i
        if np.sum(mascara) > 0:
            pos_vertical_promedio = np.mean(y_coords_flat[mascara]) / altura
        else:
            pos_vertical_promedio = 0.5
        features.append(pos_vertical_promedio)
    
    # ========== FEATURES (5K+1) - (6K): POSICIÓN HORIZONTAL ==========
    for i in range(k):
        mascara = etiquetas_planas == i
        if np.sum(mascara) > 0:
            pos_horizontal_promedio = np.mean(x_coords.flatten()[mascara]) / ancho
        else:
            pos_horizontal_promedio = 0.5
        features.append(pos_horizontal_promedio)
    
    # ========== FEATURES ADICIONALES: ESTADÍSTICAS GLOBALES ==========
    
    proporciones = [np.sum(etiquetas_planas == i) / total_pixeles for i in range(k)]
    features.append(max(proporciones))
    
    diversidad = -sum([p * np.log(p + 1e-10) for p in proporciones if p > 0])
    features.append(diversidad)
    
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
    
    saturacion_promedio = np.mean([centros_hsv[i][1] for i in range(k)])
    features.append(saturacion_promedio)
    
    valor_promedio = np.mean([centros_hsv[i][2] for i in range(k)])
    features.append(valor_promedio)
    
    # ========== NUEVO: FEATURES DE TEXTURA (LBP) ==========
    lbp_features = extraer_lbp_features(imagen_rgb, n_bins=10)
    features.extend(lbp_features)
    
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
    
    # Colores HSV de cada cluster (ORDENADOS por V)
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
    
    # Features de textura LBP (10 bins)
    for i in range(10):
        nombres.append(f'lbp_bin_{i+1}')
    
    return nombres


def calcular_dimensiones_features(k):
    """
    Calcula cuántas features se generan para un valor de k dado.
    
    Args:
        k: Número de clusters
    
    Returns:
        Número total de features
    """
    # 3k (HSV) + k (proporción) + k (pos_vert) + k (pos_horiz) + 5 (globales) + 10 (LBP)
    return 6 * k + 15


if __name__ == "__main__":
    # Test de dimensiones
    print("Nueva configuración de features (con ordenamiento y LBP):\n")
    for k_test in [3, 5, 7]:
        dim = calcular_dimensiones_features(k_test)
        print(f"K={k_test} → {dim} features")
        nombres = obtener_nombres_features(k_test)
        print(f"  Primeras 5: {nombres[:5]}")
        print(f"  Últimas 5 (LBP): {nombres[-5:]}")
        print()