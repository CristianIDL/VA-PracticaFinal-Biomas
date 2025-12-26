''' 
src.prints.py: Módulo para funciones de impresión y formateo de texto.
'''

def crear_headline(titulo, caracter="= = ", ancho=30):
    """
    Crea un headline y lo devuelve como string.
    
    Returns:
        String formateado con el headline
    """
    separador = caracter * ancho
    return f"{separador}\n{titulo}\n{separador}"