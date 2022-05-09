"""
    Módulo minimalista con los algoritmos necesarios para la práctica 2
    - Generador de soluciones aleatorias
    - Comprobación de restricciones
    - Operadores de cruce uniformes y de posicion
    - Algoritmos genéticos
    - Algoritmos meméticos
    Hecho por Luis Miguel Guirado Bautista
"""
import numpy as np

# Generador de soluciones aleatorias de N tamaño
# con M elementos seleccionados
def solucion_aleatoria(n: int, m: int):
    if m > n:
        raise ValueError(f'm > n ({m} > {n})')
    x = np.append(np.zeros(n-m, int), np.ones(m, int))
    np.random.shuffle(x)
    return x


# Comprueba si la solución S cumple las restricciones
def factible(s: np.ndarray, n: int, m: int):
    return all([
        s.size == n,
        np.count_nonzero(s) == m,
        not any(s < 0) or any(s > 1)
    ])

# BÚSQUEDA LOCAL
def busqueda_local(d: np.ndarray, n: int, m: int):

    def coste():
        # TODO
        pass

    # Seria el algoritmo genetico / memetico
    def escoger_vecino(): pass # BORRAR (PLACEHOLDER)

    MAX_ITERS = 100_000

    actual = solucion_aleatoria(n, m)
    coste_actual = coste()
    coste_vecino = 0
    iters = 0

    while iters < MAX_ITERS or coste_vecino >= coste_actual:

        iters += 1

        vecino, coste_vecino = escoger_vecino()

        if coste_vecino < coste_actual:
            actual = vecino
            coste_actual = coste_vecino
            continue

    return actual


# ******************************* OPERADORES DE CRUCE *******************************
# CRUCE UNIFORME
def cruce_uniforme(p1: np.ndarray, p2: np.ndarray, n: int, m: int, d: np.ndarray):

    # Implementación del reparador **************************************************
    def reparar(h: np.ndarray, n: int, m: int, d: np.ndarray):
        v = m - np.count_nonzero(h) # Unos deseados - Unos de h
        avrg = np.mean(d) # Peso medio de los arcos (media de todas las distancias)

        if v != 0:
            # Sobran elementos
            while v < 0:
                escogido = np.random.choice(h == 1, 1)

                # Eliminamos los elementos que supongan mayor coste
                for j in range(n):
                    if h[j] == 1:
                        max = -np.inf
                        suma = 0
                        for i in range(n):
                            if i != j: suma += abs(d[i,j] - avrg)
                        suma = abs(suma)
                        if suma > max:
                            max = suma
                            escogido = j

                h[escogido] = 0
                avrg = np.mean(d)
                v += 1
            # Faltan elementos
            while v > 0:
                escogido = np.random.choice(h == 0, 1)
                
                # Añadimos los elementos que supongan menor coste
                for j in range(n):
                    if h[j] == 0:
                        min = np.inf
                        suma = 0
                        for i in range(n):
                            if i != j:
                                suma += abs(d[i, j] - avrg)
                        suma = abs(suma)
                        if suma < min:
                            min = suma
                            escogido = j

                h[j] = 1
                avrg = np.mean(d)
                v -= 1
    # -******************************************************************************

    if not (factible(p1,n,m) and factible(p2,n,m)):
        raise Exception(f'No todos los padres son factibles')
    elif p1 in p2: # p1 == p2
        return p1 # Devolvemos directamente, no tiene sentido hacer cómputo"""

    h = np.empty(n)
    
    for idx in range(n):
        if p1[idx] == p2[idx]:
            h[idx] = p1[idx]
        else:
            h[idx] = np.random.choice([p1[idx], p2[idx]], 1)

    reparar(h, n, m, d)

    #! A veces, la solucion reparada no es factible (1s de h < m)
    #! Esto es una solucion posiblemente temporal que
    #! sacrifica rendimiento por el cumplimiento obligatorio de la
    #! restricción
    if not factible(h, n, m):
        h = cruce_uniforme(p1,p2,n,m,d)

    return h.astype(int)


# CRUCE DE POSICION
def cruce_posicion(p1: np.ndarray, p2: np.ndarray, n: int, m: int, num_hijos=1):
    if not (factible(p1, n, m) and factible(p2, n, m)):
        raise Exception(f'No todos los padres son factibles')
    elif p1 in p2: # p1 == p2
        return p1 # Devolvemos directamente, no tiene sentido hacer cómputo

    if num_hijos < 1:
        raise ValueError(f'num_hijos debe ser >1, cuando num_hijos es {num_hijos}')

    # Inicializamos el hijo, los restos y el padre del cual va a heredar sus
    # caracteristicas de forma aleatoria
    h = np.empty(n)
    restos = []
    primer_padre = np.random.random_integers(0,1,1) % 2 == 0

    # Encontramos coincidencias entre los dos padres
    for idx in range(n):
        if p1[idx] == p2[idx]:
            h[idx] = p1[idx]
        else:
            restos.append((idx, p1[idx] if primer_padre else p2[idx]))

    # Generamos los hijos
    hijos = []
    for i in range(num_hijos):
        np.random.shuffle(restos)
        for idx, ele in restos:
            h[idx] = ele
        hijos.append(h.astype(int))

    # Si la lista tiene solo un elemento, solamente devolvemos el elemento
    return hijos if num_hijos > 1 else hijos[0]

# TODO
# ALGORITMO GENÉTICO GENERACIONAL
def AGG_uniforme():
    pass

# TODO
def AGG_posicion():
    pass

# TODO
# ALGORITMO GENÉTICO ESTACIONARIO
def AGE_uniforme():
    pass

# TODO
def AGE_posicion():
    pass

def AM():
    pass

