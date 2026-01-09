'''
predict.py: Predicción de biomas usando el modelo entrenado.
'''

import numpy as np
import pickle
import cv2
from pathlib import Path
from src.feature_extraction import extraer_features


class ClasificadorBiomas:
    """Clasificador de biomas que encapsula modelo, scaler y label encoder."""
    
    def __init__(self, model_path='models/clasificador_biomas.pkl'):
        """
        Inicializa el clasificador cargando el modelo entrenado.
        
        Args:
            model_path: Ruta al archivo del modelo
        """
        self.model_path = Path(model_path)
        self.modelo_data = None
        self.modelo = None
        self.scaler = None
        self.label_encoder = None
        self.k = None
        
        self._cargar_modelo()
    
    def _cargar_modelo(self):
        """Carga el modelo y sus componentes desde el archivo pickle."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"❌ No se encontró el modelo en: {self.model_path}\n"
                f"   Ejecuta primero: python train.py"
            )
        
        print(f"📂 Cargando modelo desde: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.modelo_data = pickle.load(f)
        
        # Extraer componentes
        self.modelo = self.modelo_data['modelo']
        self.scaler = self.modelo_data['scaler']
        self.label_encoder = self.modelo_data['label_encoder']
        self.k = self.modelo_data['k']
        
        print(f"✓ Modelo cargado exitosamente")
        print(f"  • Tipo: {self.modelo_data['modelo_tipo']}")
        print(f"  • Accuracy (test): {self.modelo_data['test_accuracy']:.4f}")
        print(f"  • K-means clusters: {self.k}")
        print(f"  • Clases: {self.label_encoder.classes_}")
    
    def predecir(self, imagen_rgb, etiquetas, centros_hsv):
        """
        Predice el bioma de una imagen ya segmentada.
        
        Args:
            imagen_rgb: Imagen original en RGB
            etiquetas: Etiquetas de K-means (array 2D)
            centros_hsv: Centros de K-means en HSV
        
        Returns:
            tuple: (bioma_predicho, confianza, probabilidades_dict)
        """
        
        # 1. Extraer features
        features = extraer_features(imagen_rgb, etiquetas, centros_hsv, self.k)
        features = features.reshape(1, -1)  # Reshape para predicción
        
        # 2. Normalizar features
        features_scaled = self.scaler.transform(features)
        
        # 3. Predecir
        prediccion_encoded = self.modelo.predict(features_scaled)[0]
        bioma_predicho = self.label_encoder.inverse_transform([prediccion_encoded])[0]
        
        # 4. Obtener probabilidades (si el modelo las soporta)
        if hasattr(self.modelo, 'predict_proba'):
            probabilidades = self.modelo.predict_proba(features_scaled)[0]
            confianza = probabilidades[prediccion_encoded]
            
            # Crear diccionario de probabilidades por clase
            prob_dict = {}
            for idx, clase in enumerate(self.label_encoder.classes_):
                prob_dict[clase] = probabilidades[idx]
        else:
            # Para modelos sin predict_proba (algunos SVM)
            confianza = 1.0  # Placeholder
            prob_dict = {bioma_predicho: 1.0}
        
        return bioma_predicho, confianza, prob_dict
    
    def predecir_desde_ruta(self, ruta_imagen, k=None, max_dimension=600):
        """
        Predice el bioma directamente desde una ruta de imagen.
        Realiza todo el pipeline: carga, segmentación, extracción, predicción.
        
        Args:
            ruta_imagen: Ruta a la imagen
            k: Número de clusters (usa el del modelo si no se especifica)
            max_dimension: Dimensión máxima para redimensionar
        
        Returns:
            tuple: (bioma_predicho, confianza, probabilidades_dict, imagen_rgb, imagen_segmentada)
        """
        from src.segmentar_kmeans import cargar_imagen
        
        if k is None:
            k = self.k
        
        # Cargar imagen
        imagen_rgb = cargar_imagen(ruta_imagen, max_dimension=max_dimension)
        
        # Convertir a HSV y segmentar con K-means
        imagen_hsv = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2HSV)
        altura, ancho, _ = imagen_hsv.shape
        pixeles = imagen_hsv.reshape((-1, 3))
        pixeles = np.float32(pixeles)
        
        criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, etiquetas, centros = cv2.kmeans(
            pixeles, k, None, criterio, 3, cv2.KMEANS_RANDOM_CENTERS
        )
        
        centros_hsv = np.uint8(centros)
        etiquetas = etiquetas.reshape((altura, ancho))
        
        # Crear imagen segmentada para visualización
        imagen_segmentada_hsv = centros_hsv[etiquetas.flatten()]
        imagen_segmentada_hsv = imagen_segmentada_hsv.reshape(imagen_hsv.shape)
        imagen_segmentada = cv2.cvtColor(imagen_segmentada_hsv, cv2.COLOR_HSV2RGB)
        
        # Predecir
        bioma, confianza, prob_dict = self.predecir(imagen_rgb, etiquetas, centros_hsv)
        
        return bioma, confianza, prob_dict, imagen_rgb, imagen_segmentada
    
    def get_info(self):
        """Retorna información del modelo cargado."""
        return {
            'modelo_tipo': self.modelo_data['modelo_tipo'],
            'test_accuracy': self.modelo_data['test_accuracy'],
            'k': self.k,
            'clases': list(self.label_encoder.classes_),
            'n_features': self.modelo_data['n_features'],
            'timestamp': self.modelo_data['timestamp']
        }


def cargar_clasificador(model_path='models/clasificador_biomas.pkl'):
    """
    Función helper para cargar el clasificador.
    
    Args:
        model_path: Ruta al modelo
    
    Returns:
        ClasificadorBiomas instanciado
    """
    return ClasificadorBiomas(model_path)


if __name__ == "__main__":
    # Test del clasificador
    import sys
    
    print("=" * 70)
    print("🧪 TEST DEL CLASIFICADOR")
    print("=" * 70)
    
    try:
        # Cargar clasificador
        clasificador = cargar_clasificador()
        
        # Mostrar info
        print("\n📋 Información del modelo:")
        info = clasificador.get_info()
        for key, value in info.items():
            print(f"  • {key}: {value}")
        
        # Si se proporciona una imagen de prueba
        if len(sys.argv) > 1:
            ruta_test = sys.argv[1]
            print(f"\n🖼️  Probando con imagen: {ruta_test}")
            
            bioma, confianza, probs, _, _ = clasificador.predecir_desde_ruta(ruta_test)
            
            print(f"\n🎯 RESULTADO:")
            print(f"  • Bioma predicho: {bioma}")
            print(f"  • Confianza: {confianza:.2%}")
            
            print(f"\n📊 Probabilidades por clase:")
            for clase, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                barra = "█" * int(prob * 30)
                print(f"  {clase:20s}: {prob:.2%} {barra}")
        else:
            print("\n💡 Para probar con una imagen: python predict.py <ruta_imagen>")
        
        print("\n✓ Test completado exitosamente")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()