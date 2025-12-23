'''
Programa para clasificación de biomas usando visión artificial.
'''
import os
import time
from tkinter import Tk, filedialog

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

def main():
    """Main"""
    print("= = " * 30)
    print("Práctica Final - CLASIFICACIÓN DE BIOMAS MEDIANTE VISIÓN ARTIFICIAL")
    print("= = " * 30)

    inEternum = True

    while inEternum:
        # 1. Selección de imagen}
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

        # TODO: Implementar la lógica de clasificación de biomas aquí

        bioma = "Playa"  # Placeholder para el bioma detectado
        confianza = 0.85  # Placeholder para el nivel de confianza

        t_procesamiento = time.time() - t_inicio

        print(f"Procesamiento completado en {t_procesamiento:.2f} segundos.")

        # 5. Reproducir audio correspondiente al bioma

        print("\n[5/6] Reproducir audio del bioma detectado")
        # TODO: Implementar reproducción de audio

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

if __name__ == "__main__":
    main()