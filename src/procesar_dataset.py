'''
procesar_dataset.py: Procesa todo el dataset y extrae features para entrenamiento.
'''

import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from src.segmentar_kmeans import cargar_imagen, segmentar_kmeans
from src.feature_extraction import extraer_features, obtener_nombres_features, calcular_dimensiones_features
from src.prints import crear_headline


def procesar_dataset(k=5, max_dimension=600):
    """
    Procesa todas las imágenes del dataset y extrae features.
    
    Args:
        k: Número de clusters para K-means
        max_dimension: Dimensión máxima de imagen para procesamiento
    
    Returns:
        Diccionario con features, labels y metadata
    """
    
    print(crear_headline(f"PROCESAMIENTO DEL DATASET (K={k})"))
    
    # Configuración
    dataset_path = Path('data/raw')
    biomas = ['playa', 'montana', 'pradera', 'no_identificado']
    
    # Contenedores para datos
    todas_features = []
    todas_labels = []
    todos_filenames = []
    errores = []
    
    tiempo_inicio_total = time.time()
    total_imagenes = 0
    
    # Calcular total de imágenes
    for bioma in biomas:
        bioma_path = dataset_path / bioma
        if bioma_path.exists():
            total_imagenes += len(list(bioma_path.glob('*.jpg'))) + \
                            len(list(bioma_path.glob('*.png'))) + \
                            len(list(bioma_path.glob('*.jpeg')))
    
    print(f"\n📊 Total de imágenes a procesar: {total_imagenes}")
    print(f"🔢 Clusters K-means: {k}")
    print(f"📏 Features por imagen: {calcular_dimensiones_features(k)}")
    print(f"{'='*60}\n")
    
    contador_global = 0
    
    # Procesar cada bioma
    for bioma in biomas:
        bioma_path = dataset_path / bioma
        
        if not bioma_path.exists():
            print(f"⚠️  Saltando {bioma}/ (no existe)\n")
            continue
        
        print(f"📁 Procesando: {bioma.upper()}")
        print(f"{'-'*60}")
        
        # Obtener todas las imágenes
        extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        archivos = []
        for ext in extensiones:
            archivos.extend(bioma_path.glob(ext))
        
        archivos = sorted(archivos)  # Ordenar para consistencia
        
        if len(archivos) == 0:
            print(f"⚠️  No hay imágenes en {bioma}/\n")
            continue
        
        # Procesar cada imagen
        for idx, archivo in enumerate(archivos, 1):
            contador_global += 1
            
            try:
                # Cargar imagen
                imagen_rgb = cargar_imagen(str(archivo), max_dimension=max_dimension)
                
                # Convertir a HSV para K-means
                imagen_hsv = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2HSV)
                
                # Aplicar K-means (sin prints)
                altura, ancho, _ = imagen_hsv.shape
                pixeles = imagen_hsv.reshape((-1, 3))
                pixeles = np.float32(pixeles)
                
                criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, etiquetas, centros = cv2.kmeans(
                    pixeles, k, None, criterio, 3, cv2.KMEANS_RANDOM_CENTERS
                )
                
                centros_hsv = np.uint8(centros)
                etiquetas = etiquetas.reshape((altura, ancho))
                
                # Extraer features
                features = extraer_features(imagen_rgb, etiquetas, centros_hsv, k)
                
                # Guardar datos
                todas_features.append(features)
                todas_labels.append(bioma)
                todos_filenames.append(str(archivo.name))
                
                # Progreso
                porcentaje = (contador_global / total_imagenes) * 100
                print(f"[{contador_global}/{total_imagenes}] ({porcentaje:5.1f}%) "
                      f"{archivo.name:30s} → ✓ Features extraídas")
                
            except Exception as e:
                error_msg = f"{archivo.name}: {str(e)}"
                errores.append(error_msg)
                print(f"[{contador_global}/{total_imagenes}] "
                      f"{archivo.name:30s} → ❌ ERROR: {str(e)}")
        
        print()  # Espacio entre biomas
    
    # Convertir a arrays numpy
    print(f"\n{'='*60}")
    print("📦 Consolidando datos...")
    
    X = np.array(todas_features, dtype=np.float32)
    y = np.array(todas_labels)
    filenames = np.array(todos_filenames)
    
    tiempo_total = time.time() - tiempo_inicio_total
    
    print(f"✓ Procesamiento completado en {tiempo_total:.2f} segundos")
    print(f"\n📊 Estadísticas:")
    print(f"  • Imágenes procesadas: {len(X)}")
    print(f"  • Errores: {len(errores)}")
    print(f"  • Shape de features: {X.shape}")
    print(f"  • Clases únicas: {np.unique(y)}")
    
    # Distribución por clase
    print(f"\n📈 Distribución por clase:")
    for bioma in biomas:
        count = np.sum(y == bioma)
        porcentaje = (count / len(y)) * 100 if len(y) > 0 else 0
        print(f"  • {bioma:20s}: {count:3d} ({porcentaje:5.1f}%)")
    
    # Mostrar errores si los hay
    if errores:
        print(f"\n⚠️  ERRORES ENCONTRADOS ({len(errores)}):")
        for error in errores[:10]:  # Mostrar máximo 10
            print(f"  • {error}")
        if len(errores) > 10:
            print(f"  ... y {len(errores) - 10} errores más")
    
    # Crear diccionario de datos
    datos = {
        'features': X,
        'labels': y,
        'filenames': filenames,
        'feature_names': obtener_nombres_features(k),
        'k': k,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'biomas': biomas,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Guardar datos
    output_path = Path('data/processed')
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / 'features.pkl'
    
    print(f"\n💾 Guardando datos en: {output_file}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(datos, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    tamano_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ Archivo guardado ({tamano_mb:.2f} MB)")
    
    print(f"\n{'='*60}")
    print("✓ PROCESAMIENTO COMPLETADO EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"\nPróximo paso: python train.py")
    
    return datos


def cargar_features():
    """Carga el archivo de features procesado."""
    features_path = Path('data/processed/features.pkl')
    
    if not features_path.exists():
        raise FileNotFoundError(
            f"No se encontró {features_path}. "
            "Ejecuta primero: python procesar_dataset.py"
        )
    
    with open(features_path, 'rb') as f:
        datos = pickle.load(f)
    
    return datos


if __name__ == "__main__":
    # Procesar dataset con K=5
    datos = procesar_dataset(k=5, max_dimension=600)
    
    # Verificar que se puede cargar
    print("\n🧪 Verificando que se puede cargar el archivo...")
    datos_cargados = cargar_features()
    print(f"✓ Archivo cargado correctamente")
    print(f"  Shape de features: {datos_cargados['features'].shape}")
    print(f"  Timestamp: {datos_cargados['timestamp']}")