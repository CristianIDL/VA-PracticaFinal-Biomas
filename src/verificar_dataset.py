'''
verificar_dataset.py: Verifica la integridad y estadísticas del dataset de biomas.
'''

import cv2
import os
from pathlib import Path
from collections import defaultdict
from src.prints import crear_headline

def verificar_dataset():
    """Verifica el dataset completo y muestra estadísticas."""
    
    print(crear_headline("VERIFICACIÓN DEL DATASET"))
    
    # Configuración
    dataset_path = Path('data/raw')
    biomas_esperados = ['playa', 'montana', 'pradera', 'no_identificado']
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Estadísticas
    stats = defaultdict(lambda: {
        'count': 0,
        'errores': [],
        'resoluciones': [],
        'tamanos': []
    })
    
    total_imagenes = 0
    total_errores = 0
    
    # Verificar estructura de carpetas
    print("\n[1/4] Verificando estructura de carpetas...")
    print("-" * 60)
    
    if not dataset_path.exists():
        print(f"! ERROR: No existe la carpeta {dataset_path}")
        print("   Por favor, crea la estructura: data/raw/")
        return False
    
    for bioma in biomas_esperados:
        bioma_path = dataset_path / bioma
        if not bioma_path.exists():
            print(f"! Falta carpeta: {bioma}/")
        else:
            print(f"✓ Carpeta encontrada: {bioma}/")
    
    # Analizar cada bioma
    print(f"\n[2/4] Analizando imágenes por bioma...")
    print("-" * 60)
    
    for bioma in biomas_esperados:
        bioma_path = dataset_path / bioma
        
        if not bioma_path.exists():
            print(f"\n! Saltando {bioma}/ (no existe)")
            continue
        
        print(f"\n📁 Procesando: {bioma.upper()}")
        
        # Listar archivos
        archivos = list(bioma_path.iterdir())
        archivos_imagenes = [f for f in archivos if f.suffix.lower() in extensiones_validas]
        
        stats[bioma]['count'] = len(archivos_imagenes)
        total_imagenes += len(archivos_imagenes)
        
        if len(archivos_imagenes) == 0:
            print(f"   ! No hay imágenes en esta carpeta")
            continue
        
        # Verificar cada imagen
        for idx, archivo in enumerate(archivos_imagenes, 1):
            try:
                # Intentar cargar la imagen
                imagen = cv2.imread(str(archivo))
                
                if imagen is None:
                    error_msg = f"{archivo.name}: No se pudo cargar"
                    stats[bioma]['errores'].append(error_msg)
                    total_errores += 1
                    print(f"   [{idx}/{len(archivos_imagenes)}] ! {error_msg}")
                    continue
                
                # Obtener estadísticas
                altura, ancho = imagen.shape[:2]
                tamano_kb = archivo.stat().st_size / 1024
                
                stats[bioma]['resoluciones'].append((ancho, altura))
                stats[bioma]['tamanos'].append(tamano_kb)
                
                # Verificación silenciosa (solo mostrar errores)
                if ancho < 200 or altura < 200:
                    error_msg = f"{archivo.name}: Resolución muy baja ({ancho}x{altura})"
                    stats[bioma]['errores'].append(error_msg)
                    print(f"   [{idx}/{len(archivos_imagenes)}] ! {error_msg}")
                
            except Exception as e:
                error_msg = f"{archivo.name}: {str(e)}"
                stats[bioma]['errores'].append(error_msg)
                total_errores += 1
                print(f"   [{idx}/{len(archivos_imagenes)}] ! {error_msg}")
        
        # Resumen del bioma
        if stats[bioma]['resoluciones']:
            anchos = [r[0] for r in stats[bioma]['resoluciones']]
            alturas = [r[1] for r in stats[bioma]['resoluciones']]
            tamanos = stats[bioma]['tamanos']
            
            print(f"   ✓ Imágenes válidas: {len(archivos_imagenes) - len(stats[bioma]['errores'])}/{len(archivos_imagenes)}")
            print(f"   📏 Resolución promedio: {int(sum(anchos)/len(anchos))}x{int(sum(alturas)/len(alturas))} px")
            print(f"   💾 Tamaño promedio: {sum(tamanos)/len(tamanos):.1f} KB")
    
    # Resumen general
    print(f"\n[3/4] Resumen general")
    print("=" * 60)
    
    for bioma in biomas_esperados:
        if bioma in stats:
            count = stats[bioma]['count']
            errores = len(stats[bioma]['errores'])
            estado = "✓" if errores == 0 else "⚠️"
            print(f"{estado} {bioma:20s}: {count:3d} imágenes ({errores} errores)")
    
    print("-" * 60)
    print(f"📊 TOTAL: {total_imagenes} imágenes")
    print(f"{'✓ Sin errores' if total_errores == 0 else f'! {total_errores} errores encontrados'}")
    
    # Verificar balance de clases
    print(f"\n[4/4] Balance de clases")
    print("-" * 60)
    
    if total_imagenes > 0:
        for bioma in biomas_esperados:
            if bioma in stats:
                count = stats[bioma]['count']
                porcentaje = (count / total_imagenes) * 100
                barra = "█" * int(porcentaje / 2)
                print(f"{bioma:20s}: {barra} {porcentaje:.1f}%")
        
        # Recomendación
        counts = [stats[b]['count'] for b in biomas_esperados if b in stats]
        if counts:
            min_count = min(counts)
            max_count = max(counts)
            desbalance = (max_count - min_count) / max_count * 100
            
            print(f"\n{'✓' if desbalance < 20 else '⚠️'} Desbalance: {desbalance:.1f}%")
            if desbalance > 20:
                print(f"   Recomendación: Equilibrar las clases (diferencia de {max_count - min_count} imágenes)")
            else:
                print(f"   Las clases están bien balanceadas")
    
    # Reporte de errores detallado
    if total_errores > 0:
        print(f"\n! ERRORES DETALLADOS:")
        print("-" * 60)
        for bioma in biomas_esperados:
            if bioma in stats and stats[bioma]['errores']:
                print(f"\n{bioma.upper()}:")
                for error in stats[bioma]['errores']:
                    print(f"  • {error}")
    
    # Conclusión
    print("\n" + "=" * 60)
    if total_errores == 0 and total_imagenes >= 80:
        print("✓ DATASET LISTO PARA PROCESAMIENTO")
        print("  Puedes continuar con: python procesar_dataset.py")
        return True
    elif total_imagenes < 80:
        print(f"! Dataset incompleto ({total_imagenes}/80 mínimo)")
        print(f"  Necesitas al menos {80 - total_imagenes} imágenes más")
        return False
    else:
        print(f"! Dataset con errores ({total_errores} errores)")
        print("  Corrige los errores antes de continuar")
        return False


if __name__ == "__main__":
    verificar_dataset()