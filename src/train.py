'''
train.py: Entrenamiento del clasificador de biomas.
'''

import numpy as np
import pickle
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.prints import crear_headline


def cargar_features():
    """Carga el archivo de features procesado."""
    features_path = Path('data/processed/features.pkl')
    
    if not features_path.exists():
        raise FileNotFoundError(
            f"No se encontró {features_path}. "
            "Ejecuta primero: python procesar_dataset.py"
        )
    
    print(f"📂 Cargando features desde: {features_path}")
    with open(features_path, 'rb') as f:
        datos = pickle.load(f)
    
    print(f"✓ Features cargadas exitosamente")
    print(f"  • Muestras: {datos['n_samples']}")
    print(f"  • Features: {datos['n_features']}")
    print(f"  • Clases: {datos['biomas']}")
    
    return datos


def preparar_datos(datos, test_size=0.2, random_state=42):
    """
    Prepara los datos para entrenamiento: split y normalización.
    
    Args:
        datos: Diccionario con features y labels
        test_size: Proporción del conjunto de prueba
        random_state: Semilla para reproducibilidad
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    
    print("\n" + crear_headline("PREPARACIÓN DE DATOS"))
    
    X = datos['features']
    y = datos['labels']
    
    # Codificar labels a números
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n📊 Mapeo de clases:")
    for i, clase in enumerate(label_encoder.classes_):
        count = np.sum(y == clase)
        print(f"  {i} → {clase:20s} ({count} muestras)")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_encoded  # Mantener proporciones por clase
    )
    
    print(f"\n✂️  División de datos:")
    print(f"  • Entrenamiento: {len(X_train)} muestras ({(1-test_size)*100:.0f}%)")
    print(f"  • Prueba: {len(X_test)} muestras ({test_size*100:.0f}%)")
    
    # Normalizar features
    print(f"\n🔧 Normalizando features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Features normalizadas")
    print(f"  • Media: ~0.0, Desviación estándar: ~1.0")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder


def entrenar_modelo(X_train, y_train, modelo_tipo='random_forest', optimizar=False):
    """
    Entrena un modelo clasificador.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento
        modelo_tipo: 'random_forest', 'svm', o 'knn'
        optimizar: Si True, usa GridSearchCV para hiperparámetros
    
    Returns:
        Modelo entrenado
    """
    
    print("\n" + crear_headline(f"ENTRENAMIENTO - {modelo_tipo.upper()}"))
    
    inicio = time.time()
    
    if modelo_tipo == 'random_forest':
        if optimizar:
            print("🔍 Buscando mejores hiperparámetros con GridSearchCV...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            modelo_base = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=5,
                random_state=42
            )
            modelo = GridSearchCV(
                modelo_base, 
                param_grid, 
                cv=5,               
                scoring='f1_macro', 
                n_jobs=-1, 
                verbose=1)
            modelo.fit(X_train, y_train)
            
            print(f"\n✓ Mejores parámetros encontrados:")
            for param, value in modelo.best_params_.items():
                print(f"  • {param}: {value}")
            
            modelo = modelo.best_estimator_
        else:
            print("🌲 Entrenando Random Forest con parámetros por defecto...")
            modelo = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            modelo.fit(X_train, y_train)
    
    elif modelo_tipo == 'svm':
        if optimizar:
            print("🔍 Buscando mejores hiperparámetros con GridSearchCV...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            }
            modelo_base = SVC(random_state=42)
            modelo = GridSearchCV(modelo_base, param_grid, cv=5,
                                 scoring='accuracy', n_jobs=-1, verbose=1)
            modelo.fit(X_train, y_train)
            
            print(f"\n✓ Mejores parámetros encontrados:")
            for param, value in modelo.best_params_.items():
                print(f"  • {param}: {value}")
            
            modelo = modelo.best_estimator_
        else:
            print("🎯 Entrenando SVM con parámetros por defecto...")
            modelo = SVC(
                C=10,
                gamma='scale',
                kernel='rbf',
                random_state=42
            )
            modelo.fit(X_train, y_train)
    
    elif modelo_tipo == 'knn':
        if optimizar:
            print("🔍 Buscando mejores hiperparámetros con GridSearchCV...")
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            modelo_base = KNeighborsClassifier()
            modelo = GridSearchCV(modelo_base, param_grid, cv=5,
                                 scoring='accuracy', n_jobs=-1, verbose=1)
            modelo.fit(X_train, y_train)
            
            print(f"\n✓ Mejores parámetros encontrados:")
            for param, value in modelo.best_params_.items():
                print(f"  • {param}: {value}")
            
            modelo = modelo.best_estimator_
        else:
            print("👥 Entrenando K-NN con parámetros por defecto...")
            modelo = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean'
            )
            modelo.fit(X_train, y_train)
    
    else:
        raise ValueError(f"Modelo '{modelo_tipo}' no reconocido. Usa: random_forest, svm, o knn")
    
    tiempo_entrenamiento = time.time() - inicio
    print(f"\n✓ Entrenamiento completado en {tiempo_entrenamiento:.2f} segundos")
    
    return modelo


def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, label_encoder):
    """
    Evalúa el modelo con métricas y visualizaciones.
    
    Args:
        modelo: Modelo entrenado
        X_train, X_test: Features de entrenamiento y prueba
        y_train, y_test: Labels de entrenamiento y prueba
        label_encoder: LabelEncoder para nombres de clases
    """
    
    print("\n" + crear_headline("EVALUACIÓN DEL MODELO"))
    
    # Predicciones
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    
    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n📊 Accuracy:")
    print(f"  • Entrenamiento: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  • Prueba:        {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    if train_acc - test_acc > 0.1:
        print(f"  ⚠️  Posible overfitting (diferencia: {(train_acc-test_acc)*100:.2f}%)")
    
    # Reporte de clasificación
    print(f"\n📈 Reporte de clasificación (conjunto de prueba):")
    print("=" * 70)
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.tight_layout()
    
    # Guardar figura
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    print(f"\n💾 Matriz de confusión guardada en: results/confusion_matrix.png")
    
    plt.show()
    
    return test_acc


def guardar_modelo(modelo, scaler, label_encoder, datos_features, test_accuracy, modelo_tipo):
    """
    Guarda el modelo entrenado y sus componentes.
    
    Args:
        modelo: Modelo entrenado
        scaler: StandardScaler usado
        label_encoder: LabelEncoder usado
        datos_features: Diccionario original de features
        test_accuracy: Accuracy en conjunto de prueba
        modelo_tipo: Tipo de modelo usado
    """
    
    print("\n" + crear_headline("GUARDANDO MODELO"))
    
    # Crear directorio de modelos
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Crear diccionario con todo lo necesario para predicción
    modelo_completo = {
        'modelo': modelo,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'k': datos_features['k'],
        'feature_names': datos_features['feature_names'],
        'n_features': datos_features['n_features'],
        'biomas': datos_features['biomas'],
        'test_accuracy': test_accuracy,
        'modelo_tipo': modelo_tipo,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Guardar
    model_path = models_dir / 'clasificador_biomas.pkl'
    
    print(f"💾 Guardando modelo en: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(modelo_completo, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    tamano_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Modelo guardado exitosamente ({tamano_mb:.2f} MB)")
    
    print(f"\n📋 Información del modelo guardado:")
    print(f"  • Tipo: {modelo_tipo}")
    print(f"  • Accuracy (test): {test_accuracy:.4f}")
    print(f"  • K-means clusters: {datos_features['k']}")
    print(f"  • Features: {datos_features['n_features']}")
    print(f"  • Clases: {len(datos_features['biomas'])}")
    print(f"  • Timestamp: {modelo_completo['timestamp']}")
    
    return model_path


def main():
    """Pipeline completo de entrenamiento."""
    
    print(crear_headline("ENTRENAMIENTO DEL CLASIFICADOR DE BIOMAS"))
    
    try:
        # 1. Cargar features
        datos = cargar_features()
        
        # 2. Preparar datos
        X_train, X_test, y_train, y_test, scaler, label_encoder = preparar_datos(
            datos, test_size=0.2, random_state=42
        )
        
        # 3. Seleccionar tipo de modelo
        print("\n" + "="*70)
        print("🤖 Modelos disponibles:")
        print("  1. Random Forest (recomendado)")
        print("  2. SVM (Support Vector Machine)")
        print("  3. K-Nearest Neighbors")
        print("="*70)
        
        try:
            opcion = input("\nSelecciona el modelo (1-3) [default: 1]: ").strip()
            if not opcion:
                opcion = '1'
            
            modelo_map = {
                '1': 'random_forest',
                '2': 'svm',
                '3': 'knn'
            }
            modelo_tipo = modelo_map.get(opcion, 'random_forest')
        except:
            print("Opción inválida. Usando Random Forest por defecto.")
            modelo_tipo = 'random_forest'
        
        # 4. Preguntar si optimizar hiperparámetros
        try:
            optimizar = input("\n¿Optimizar hiperparámetros con GridSearchCV? (s/n) [default: n]: ").strip().lower()
            optimizar = optimizar in ['s', 'si', 'sí', 'y', 'yes']
        except:
            optimizar = False
        
        if optimizar:
            print("⚠️  La optimización puede tardar varios minutos...")
        
        # 5. Entrenar modelo
        modelo = entrenar_modelo(X_train, y_train, modelo_tipo, optimizar)
        
        # 6. Evaluar modelo
        test_accuracy = evaluar_modelo(modelo, X_train, X_test, y_train, y_test, label_encoder)
        
        # 7. Guardar modelo
        model_path = guardar_modelo(modelo, scaler, label_encoder, datos, test_accuracy, modelo_tipo)
        
        # 8. Mensaje final
        print("\n" + "="*70)
        print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\n🎯 Próximo paso: python main.py")
        print(f"   El modelo será cargado automáticamente desde: {model_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Asegúrate de ejecutar primero: python procesar_dataset.py")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()