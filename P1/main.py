"""
    Práctica 1a - Metaheurísticas
    Alumno: Luis Miguel Guirado Bautista
    Curso 2021/22
"""

# -*- encoding: utf-8 -*-
import time # Medir tiempos de ejecución
from os import listdir # Listar archivos de un directorio
import numpy as np # array
import random as rnd # random
import pandas # tablas en python

# Extraer los datos de un fichero con formato GKD
def extraer_datos(path: str) -> tuple([np.ndarray, int, int]):

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
    return data, n, m

# ***** GREEDY-MDD ********************************************************************

# Heuristica empleada: argmin g(u)
# Devuelve el punto escogido y su dispersion (fitness)
def min_grad(distancia: np.ndarray, s: np.ndarray) -> tuple([int, float]):

    n = np.shape(distancia)[0]
    dist_actual = distancia_sel(distancia, s)
    ret = None
    min_g = 10e10
                
    for u in [_ for _ in range(n) if _ not in s]:

        # Distancias acumuladas
        acumuladas = np.zeros(n, dtype=float)
        acumuladas[u] = np.sum(distancia[u, s])
        for v in s:
            acumuladas[v] = dist_actual + distancia[u, v]
            
        dmax = np.max([acumuladas[u], np.max(acumuladas[s])])
        dmin = np.min([acumuladas[u], np.max(acumuladas[s])])
        g_u = dmax - dmin
        if g_u < min_g:
            min_g = g_u
            ret = u
        # print(f'{dmax} - {dmin} = {g_u}')
     
    return ret, min_g

# Greedy-MDD
def greedy(data: np.ndarray, m: int) -> np.ndarray:

    # Inicializamos contador
    inicio = time.time()

    # Numero de filas de la matriz de distancias -> N
    n = data.shape[0]

    # El primer elemento es aleatorio (indice)
    actual = rnd.choice(range(n))
    disp = 0
    s = np.array([actual],dtype=int)

    # Vamos escogiendo los menores segun g(u)
    while s.size < m:
        actual, disp = min_grad(data, s)
        s = np.append(s, actual)

    fin = time.time()
    tiempo = fin - inicio
    # Solución, dispersión (fitness) y tiempo empleado
    return s, disp, tiempo

# ***** BÚSQUEDA LOCAL ********************************************************************

# Genera una seleccion aleatoria de tamaño m de los [0,n-1] posibles elementos
def generar_sel(n: int, m: int) -> list[int]:
    return list(np.random.randint(0, n, m))
    
# Calcula la distancia del recorrido cerrado 'sel'
def distancia_sel(data: np.ndarray, sel: list) -> float:
    # sel[(i+1)%len(sel)] es el elemento siguiente a u
    distancia = np.sum([ data[ u,sel[ (i+1) % len(sel) ] ] for (i, u) in enumerate(sel) ])
    return distancia

# Intercambia un elemento i con un elemento j
# i debe estar dentro de s
# j no debe estar dentro de s
def intercambio(sel: list, i: int, j: int) -> tuple([int,int]):
    # i debe estar dentro de s y j no debe estar dentro de s
    if j in sel:
        raise Exception('j ya se encuentra dentro de sel')

    if i not in sel:
        raise Exception('i no se encuentra dentro de sel')

    ret = sel.copy()
    indice = ret.index(i) # Indice del elemento intercambiado
    ret[indice] = j

    return ret, indice


def escoger_vecino(distancia: np.ndarray, actual: list):

    n = distancia.shape[0] # Posibles destinos
    coste_actual = distancia_sel(distancia, actual)

    # Para cada elemento de la solucion
    for u in actual:
        # Para cada elemento que no este en la solucion
        for v in [_ for _ in range(n) if _ not in actual]:

            # Generamos una solucion vecina
            prima, indice = intercambio(actual, u, v)
            acumuladas = np.zeros(n, dtype=float)
            
            # Refactorizacion de la dispersion
            acumuladas[v] = 0
            for w in prima:
                acumuladas[v] += distancia[v,w]
                acumuladas[w] = coste_actual - distancia[w,u] + distancia[w,v]

            dmax = np.max([acumuladas[v], np.max(acumuladas[prima])])
            dmin = np.min([acumuladas[v], np.min(acumuladas[prima])])
            coste_prima = dmax - dmin
            coste_intercambio = coste_prima - coste_actual
            # print(f'{coste_prima} - {coste_actual} = {coste_intercambio}')

            # Si S' es favorable, terminamos
            if coste_intercambio < 0:
                return prima, coste_prima

    # Se ha generado todo el entorno de S_act, paramos
    return prima, coste_prima


def busqueda_local(data: np.array, m: int):

    inicio = time.time()

    MAX_ITERS = 100000

    n = data.shape[0]
    actual = generar_sel(n, m)
    dist_actual = distancia_sel(data, actual)
    iters = 0

    while True:

        iters += 1

        prima, dist_prima = escoger_vecino(data, actual)

        if dist_prima < dist_actual:
            actual = prima
            dist_actual = dist_prima
            continue

        if iters >= MAX_ITERS or dist_prima >= dist_actual:
            fin = time.time()
            tiempo = fin - inicio
            return actual, dist_actual, tiempo

# ************************************************************************

def main():
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ! https://mh.danimolina.net/mdd/MDD_test.html !
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Directorio padre de los casos a evaluar
    data_dir = 'datos/p1a/'
    ficheros = listdir(data_dir)

    # Ordena los ficheros segun el orden de b (en el nombre del fichero)
    # Ejemplo:
    # x = "GKD-b_8_n25_m7.txt"
    # ["GKD-b", "8", "n35", "m7"]
    #           ^^^
    ficheros = sorted(ficheros, key=lambda x: int(x.split(sep='_')[1]))

    # Tablas donde almacenaremos los resultados
    tabla_greedy = {
        'Coste': [],
        'Tiempo': []
    }
    
    tabla_bl = {
        'Coste': [],
        'Tiempo': []
    }

    disp_media = 0
    tiempo_medio = 0
    semilla = 0
    for f in ficheros:

        print(f"\t*** --- Caso \"{f}\" --- ***")
        
        semilla = int(np.random.randint(low=0, high=10e6, size=1))
        np.random.seed(semilla)
        print(f'Semilla: {semilla}')

        print("--- Algoritmo Greedy-MDD (media de 5 casos) --- ")
        datos, n, m = extraer_datos(data_dir+f)
        # print(f'n = {n}\tm = {m}')
        for _ in range(5):
            s, disp, tiempo = greedy(datos, m)
            disp_media += disp
            tiempo_medio += tiempo
            # print(s, disp)
        disp_media /= 5
        tiempo_medio = (tiempo_medio*1000)/5
        tabla_greedy['Coste'] += [np.round(disp_media,4)]
        tabla_greedy['Tiempo'] += [np.round(tiempo_medio,4)]
        # print(f'Dispersión media: {disp_media}\nTiempo medio (ms): {tiempo_medio}')
        
        disp_media = tiempo_medio = 0
        print("--- Algoritmo Búsqueda Local (media de 5 casos) --- ")
        for _ in range(5):
            s, disp, tiempo = busqueda_local(datos, m)
            disp_media += disp
            tiempo_medio += tiempo
            # print(s, disp)
        disp_media /= 5
        tiempo_medio = (tiempo_medio*1000)/5
        tabla_bl['Coste'] += [np.round(disp_media,4)]
        tabla_bl['Tiempo'] += [np.round(tiempo_medio,4)]
        # print(f'Dispersion media: {disp_media}\nTiempo medio (ms): {tiempo_medio}')
        
    # Generamos las tablas
    tabla_final = {
        'Coste': [sum(tabla_greedy['Coste'])/len(ficheros), sum(tabla_bl['Coste'])/len(ficheros)],
        'Tiempo': [sum(tabla_greedy['Tiempo'])/len(ficheros), sum(tabla_bl['Tiempo'])/len(ficheros)]
    }
    tabla_greedy = pandas.DataFrame(tabla_greedy, index=ficheros)
    tabla_bl = pandas.DataFrame(tabla_bl, index=ficheros)
    tabla_final = pandas.DataFrame(tabla_final, index=['Greedy', 'BL'])
    print(tabla_greedy)
    print(tabla_bl)
    print(tabla_final)

if __name__ == '__main__':
    main()