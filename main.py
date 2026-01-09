'''
main.py: Programa para clasificación de biomas usando visión artificial.
'''
import cv2
import os
import time
import numpy as np

from tkinter import Tk, filedialog
from src.segmentar_kmeans import cargar_imagen, segmentar_kmeans, visualizar_comparacion
from src.audio import inicializar_audio, reproducir_audio, detener_audio, finalizar_audio
from src.prints import crear_headline
from src.predict import cargar_clasificador


def seleccionar_imagen():
    """Cuadro de diálogo para seleccionar una imagen."""
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal
    root.attributes('-topmost', True)  # Mantener el cuadro hasta adelante

    archivo = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )

    root.destroy()  # Cerrar la ventana al finalizar
    return archivo


def mostrar_resultados(bioma, confianza, probabilidades, k, t_procesamiento):
    """
    Muestra los resultados de la clasificación de forma visual.
    
    Args:
        bioma: Bioma predicho
        confianza: Nivel de confianza de la predicción
        probabilidades: Diccionario con probabilidades por clase
        k: Número de clusters usados
        t_procesamiento: Tiempo total de procesamiento
    """
    print("\n" + "="*70)
    print("🎯 RESULTADOS DE LA CLASIFICACIÓN")
    print("="*70)
    
    # Resultado principal
    print(f"\n🌍 Bioma detectado: {bioma.upper()}")
    print(f"📊 Confianza: {confianza:.2%}")
    
    # Barra visual de confianza
    barra_longitud = int(confianza * 40)
    barra = "█" * barra_longitud + "░" * (40 - barra_longitud)
    print(f"   [{barra}]")
    
    # Interpretación de confianza
    if confianza >= 0.8:
        interpretacion = "✓ Alta confianza"
    elif confianza >= 0.6:
        interpretacion = "⚠️  Confianza media"
    else:
        interpretacion = "⚠️  Baja confianza - resultado incierto"
    print(f"   {interpretacion}")
    
    # Distribución de probabilidades
    print(f"\n📈 Distribución de probabilidades:")
    print("-" * 70)
    
    # Ordenar por probabilidad descendente
    probs_ordenadas = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)
    
    for clase, prob in probs_ordenadas:
        barra = "█" * int(prob * 30)
        emoji = "👉" if clase == bioma else "  "
        print(f"{emoji} {clase:20s}: {prob:6.2%} {barra}")
    
    # Información técnica
    print(f"\n⚙️  Información técnica:")
    print(f"  • Clusters K-Means: {k}")
    print(f"  • Tiempo de procesamiento: {t_procesamiento:.2f} seg")
    
    print("="*70)


def main():
    """Función principal del programa."""
    
    print(crear_headline("CLASIFICACIÓN DE BIOMAS MEDIANTE VISIÓN ARTIFICIAL"))
    
    # Cargar el clasificador una sola vez
    print("\n🤖 Inicializando clasificador...")
    try:
        clasificador = cargar_clasificador()
        print("✓ Clasificador listo\n")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Pasos para solucionar:")
        print("   1. python procesar_dataset.py")
        print("   2. python train.py")
        print("   3. python main.py")
        return
    except Exception as e:
        print(f"\n❌ Error al cargar el clasificador: {e}")
        return
    
    # Inicializar audio
    audio = inicializar_audio()
    
    inEternum = True
    
    while inEternum:
        print("\n" + "="*70)
        
        # 1. Selección de imagen
        print("\n[1/6] 📁 Selección de imagen")
        ruta_imagen = seleccionar_imagen()
        
        if not ruta_imagen:
            print("⚠️  No se seleccionó ninguna imagen. Saliendo...")
            break
        
        print(f"✓ Imagen seleccionada: {os.path.basename(ruta_imagen)}")
        
        # 2. Solicitar número de clases (K)
        print(f"\n[2/6] ⚙️  Configuración de segmentación")
        print(f"💡 El modelo fue entrenado con K={clasificador.k}")
        
        try:
            k_input = input(f"Ingresa el número de clases (k) [default: {clasificador.k}]: ").strip()
            if k_input == "":
                k = clasificador.k
            else:
                k = int(k_input)
                if k < 2:
                    print(f"⚠️  Debe haber al menos 2 clases. Usando k={clasificador.k}")
                    k = clasificador.k
                elif k != clasificador.k:
                    print(f"⚠️  ADVERTENCIA: El modelo fue entrenado con k={clasificador.k}")
                    print(f"   Usar k={k} afectará la precisión.")
        except:
            print(f"⚠️  Valor inválido. Usando k={clasificador.k}")
            k = clasificador.k
        
        print(f"✓ Usando K={k} clusters")
        
        # 3. Solicitar número de puntos muestreados
        print(f"\n[3/6] 📊 Configuración de muestreo")
        try:
            n_puntos = int(input("Ingresa el número de puntos de muestreo [default: 100]: "))
            if n_puntos <= 0:
                print("⚠️  El número de puntos debe ser positivo. Se usarán 100 puntos por defecto.")
                n_puntos = 100
        except:
            print("⚠️  Valor inválido. Se usarán 100 puntos por defecto.")
            n_puntos = 100
        
        print(f"✓ Usando {n_puntos} puntos de muestreo")
        
        # 4. Procesamiento de imagen
        print(f"\n[4/6] 🔄 Procesando imagen...")
        t_inicio = time.time()
        
        try:
            # Cargar imagen
            imagen_cargada = cargar_imagen(ruta_imagen)
            
            # Segmentar con K-means
            imagen_segmentada, etiquetas, centros_hsv, t_kmeans = segmentar_kmeans(
                imagen_cargada, k=k
            )
            
            # Visualizar comparación
            visualizar_comparacion(imagen_cargada, imagen_segmentada, k, t_kmeans)
            
            print(f"✓ Segmentación completada")
            
        except Exception as e:
            print(f"❌ Error durante la segmentación: {e}")
            continue
        
        # 5. Clasificación
        print(f"\n[5/6] 🧠 Clasificando bioma...")
        
        try:
            # Predecir usando el clasificador
            bioma, confianza, probabilidades = clasificador.predecir(
                imagen_cargada, etiquetas, centros_hsv
            )
            
            t_procesamiento = time.time() - t_inicio
            
            print(f"✓ Clasificación completada")
            
        except Exception as e:
            print(f"❌ Error durante la clasificación: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 6. Mostrar resultados
        mostrar_resultados(bioma, confianza, probabilidades, k, t_procesamiento)
        
        # 7. Reproducir audio correspondiente al bioma
        print(f"\n[6/6] 🔊 Reproduciendo audio del bioma...")
        
        if audio:
            try:
                reproducir_audio(bioma)
                print(f"✓ Audio reproducido: {bioma}")
            except Exception as e:
                print(f"⚠️  No se pudo reproducir audio: {e}")
        else:
            print("⚠️  Sistema de audio no disponible")
        
        # 8. Preguntar si desea procesar otra imagen
        print("\n" + "="*70)
        continuar = input("¿Deseas procesar otra imagen? (s/n): ").lower().strip()
        
        if continuar not in ['s', 'si', 'sí', 'y', 'yes']:
            print("\n" + "="*70)
            print("✓ Finalizando el programa. ¡Gracias por usarlo! 🌍")
            print("="*70)
            inEternum = False
    
    # Finalizar audio
    if audio:
        finalizar_audio()


if __name__ == "__main__":
    main()