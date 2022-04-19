"""
    Módulo minimalista de Python para la lectura de datos
    de ficheros GKD.
    Hecho por Luis Miguel Guirado Bautista
"""

import numpy as np
from os import listdir

# Extraer los datos de un fichero con formato GKD
def extraer_datos(path: str) -> tuple([str, np.array, int, int]):
    """
    Extrae los datos de un fichero GKD

    Argumentos:
        ``path (str)``: Ruta del fichero

    Raises:
        ``Exception``: Si el archivo no puede leerse

    Devuelve:
        ``tuple([str, np.array, int, int])``:
        Una tupla con el nombre del fichero, la matriz
        de distancias de los puntos del caso y
        los tamaños m y n del problema
    """

    # Inicializacion de variables
    file = open(path, 'r')
    linenum = 0 # Numero de linea que lee en el archivo
    n, m = 0, 0 # Tamaño del conjunto y del subconjunto respectivamente
    data = np.array([])

    # Si el archivo no se puede leer...
    if not file.readable():
        raise Exception(f'El archivo {file.name} no puede leerse.')

    # Leemos el archivo por lineas
    for line in file:

        # Separamos cada uno de los números en las líneas
        element = [float(x) if x.count('.') != 0 else int(x) for x in line.split()]

        # Extraemos n y m de la primera linea e inicializamos la matriz de distancias
        if linenum == 0:
            n, m = element
            data = np.zeros((n,n), float)
        else:
            data[element[0],element[1]] = data[element[1],element[0]] = element[2]
        
        linenum += 1

    file.close()
    return file.name, data, n, m

def es_gkd(name: str) -> bool:
    """Comprueba si es fichero GKD solo por el nombre del fichero

    Argumentos:
        ``name (str)``: Nombre del fichero

    Devuelve:
        ``bool``: Si es fichero GKD o no
    """

    condiciones = [
        'GKD-b_' in name,
        '_n' in name,
        '_m' in name,
        '.txt' in name
    ]

    return all(condiciones)

def extraer_datos_dir(dirpath: str) -> list:
    """
    Extrae los datos de un directorio con ficheros de formato GKD.
    La extracción de los datos está ordenada según ``b``

    Argumentos:
        ``dirpath (str)``: Ruta del directorio (p.e ``"carpeta/datos/"``)

    Raises:
        ``Exception``: Si se ha detectado un fichero que no es GKD

    Devuelve:
        ``list``: Lista con los datos generados por ``extraer_datos``
        para cada uno de los ficheros del directorio
    """
    # Ordena los ficheros segun el orden de b (en el nombre del fichero)
    # Ejemplo:
    # x = "GKD-b_8_n25_m7.txt"
    # ["GKD-b", "8", "n35", "m7"]
    #           ^^^
    data = []

    ficheros = sorted(
        listdir(dirpath), key=lambda x: int(x.split(sep='_')[1]))

    for f in ficheros:

        if not es_gkd(f):
            raise Exception("Se ha detectado un fichero que NO es GKD")

        to_append = extraer_datos(dirpath+f)
        data.append(to_append)

    return data