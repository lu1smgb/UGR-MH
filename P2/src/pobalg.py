"""
    Módulo minimalista con los algoritmos necesarios para la práctica 2
    - Generador de soluciones aleatorias
    - Comprobación de restricciones
    - Operadores de cruce uniformes y de posicion
    - Algoritmos genéticos
    - Algoritmos meméticos
    - Optimizado con Numba

    Hecho por Luis Miguel Guirado Bautista
"""

import numpy as np
from numba import njit

# Ignoramos las advertencias de 'deprecation' de Numba **************************************************************
# https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# -******************************************************************************************************************

TAM_POB_GEN = 50        # Tamaño de las poblaciones en los algoritmos genéticos
TAM_POB_MEM = 10        # Tamaño de la poblacion del algoritmo memético
MAX_EVALS = 100_000     # Máximo de evaluaciones/iteraciones de los algoritmos genéticos
STOP_BL_MEM = 400       # Iteraciones máximas de la BL dentro del algoritmo memético
PROB_CRUCE_AGG = 0.7    # Probabilidad de cruce en el generacional
PROB_CRUCE_AGE = 1      # Probabilidad de cruce en el estacionario
PROB_CRUCE_AM = 0.7     # Probabilidad de cruce en el genetico
PROB_MUTACION = 0.1     # Probabilidad de mutacion de los algoritmos geneticos y memetico
SEED = 1                # Semilla aleatoria
np.random.seed(SEED)

# Generador de soluciones aleatorias de N tamaño
# con M elementos seleccionados
@njit
def solucion_aleatoria(n: int, m: int) -> np.ndarray:
    assert n >= m
    ceros = np.zeros(n-m, dtype=np.int64)
    unos = np.ones(m, dtype=np.int64)
    x = np.append(ceros, unos)
    np.random.shuffle(x)
    return x

# Comprueba si es un vector binario (solo 0 y 1s)
@njit
def es_binario(s: np.ndarray) -> bool:
    for i in s:
        if i != 0 and i != 1:
            return False
    return True

# Comprueba si la solución S cumple las restricciones
@njit
def factible(s: np.ndarray, n: int, m: int) -> bool:
    return s.size == n and np.count_nonzero(s) == m and es_binario(s)

# Genera una población inicial de 'hab' habitantes
@njit
def poblacion_aleatoria(n: int, m: int, hab: int) -> np.ndarray:
    ret = []
    for _ in range(hab):
        ret.append(solucion_aleatoria(n,m))
    return ret

@njit
def evaluar(s: np.ndarray, d: np.ndarray) -> float:
    sumas = []
    for i in range(len(s)):
        if s[i] == 1:
            suma = 0
            for j in range(len(s)):
                if s[j] == 1 and i != j: suma += d[i,j]
            sumas.append(suma)
    return max(sumas) - min(sumas)

# Operador de mutación: intercambio de un gen con otro contrario
@njit
def mutar(s: np.ndarray) -> None:
    assert es_binario(s)

    escoger = lambda: np.random.choice(s.size)

    i = escoger()
    j = escoger()
    # Nos aseguramos de que x_i != x_j
    while i == j and s[i] == s[j]:
        j = escoger()

    s[i], s[j] = s[j], s[i]

@njit
def torneo_binario(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    p1 = np.random.choice(len(p))
    p2 = np.random.choice(len(p))
    if evaluar(p[p1], d) <= evaluar(p[p2], d):
        return p[p1]
    else:
        return p[p2]
        
#? Según teoría, una funcion para encontrar el peor podría acelerar la
#? convergencia del AGE
@njit
def encontrar_peor(p: np.ndarray, d: np.ndarray) -> float:
    resultados: list[float] = [evaluar(s, d) for s in p]
    return resultados.index(max(resultados))

@njit
def encontrar_mejor(p: np.ndarray, d: np.ndarray) -> tuple[int, float]:
    resultados: list[float] = [evaluar(s, d) for s in p]
    idx: int = resultados.index(min(resultados))
    return idx, resultados[idx] # Indice del mejor y coste

@njit
def hab_iguales(h1: np.ndarray, h2: np.ndarray) -> bool:
    assert h1.size == h2.size
    for idx in range(h1.size):
        if h1[idx] != h2[idx]:
            return False
    return True

@njit
def average(s: np.ndarray, d: np.ndarray) -> float:
    suma: float = 0
    for i in range(len(s)):
        for j in range(len(s)):
            if s[i] == 1 and s[j] == 1 and i != j:
                suma += d[i,j]
    return suma


# Implementacion del reparador para el cruce uniforme
@njit
def reparar(h: np.ndarray, n: int, m: int, d: np.ndarray) -> None:
    v: int = m - np.count_nonzero(h)  # Unos deseados menos los unos de h
    # Peso medio de los arcos (media de todas las distancias)
    avrg: float = average(h, d)

    # La solucion tiene m elementos seleccionados?
    if v != 0:
        # Sobran elementos
        while v < 0:
            # Eliminamos los elementos que supongan mayor coste
            for j in range(n):
                if h[j] == 1:
                    max: float = -np.inf
                    suma: float = 0
                    for i in range(n):
                        if i != j:
                            suma += abs(d[i, j] - avrg)
                    if suma > max:
                        max = suma
                        escogido = j

            h[escogido] = 0
            avrg = average(h, d)
            v += 1
        # Faltan elementos
        while v > 0:
            # Añadimos los elementos que supongan menor coste
            for j in range(n):
                if h[j] == 0:
                    min: float = np.inf
                    suma: float = 0
                    for i in range(n):
                        if i != j:
                            suma += abs(d[i, j] - avrg)
                    if suma < min:
                        min = suma
                        escogido = j

            h[escogido] = 1
            avrg = average(h, d)
            v -= 1

# ******************************* OPERADORES DE CRUCE *******************************
# CRUCE UNIFORME
@njit
def cruce_uniforme(p1: np.ndarray, p2: np.ndarray, n: int, m: int, d: np.ndarray) -> np.ndarray:
    assert factible(p1, n, m) and factible(p2, n, m)

    # Comprobamos si los padres son iguales
    if hab_iguales(p1, p2):
        return p1.astype(np.int64)

    # Empezamos a generar el hijo
    h: np.ndarray = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        if p1[idx] == p2[idx]:
            h[idx] = p1[idx]
        else:
            primer_padre = bool(np.random.randint(0,2))
            h[idx] = p1[idx] if primer_padre else p2[idx]

    reparar(h, n, m, d)

    return h

# CRUCE DE POSICION
@njit
def cruce_posicion(p1: np.ndarray, p2: np.ndarray, n: int, m: int) -> np.ndarray:
    assert factible(p1, n, m) and factible(p2, n, m)

    if hab_iguales(p1, p2):
        return p1

    # Inicializamos el hijo, los restos y el padre del cual va a heredar sus
    # caracteristicas de forma aleatoria
    h: np.ndarray = np.zeros(n, dtype=np.int64)
    restos: list[int] = []
    primer_padre: bool = np.random.randint(0, 2) > 0

    # Encontramos coincidencias entre los dos padres (los restos)
    for idx in range(n):
        if p1[idx] == p2[idx]:
            h[idx] = p1[idx]
        else:
            # Posición del gen en el cromosoma y el valor del gen, respectivamente
            restos.append((idx, p1[idx] if primer_padre else p2[idx]))
    restos = np.array(restos)

    # Mezclamos los genes de las posiciones de 'restos'
    np.random.shuffle(restos[:,1])
    # Insertamos los genes desordenados en el hijo
    for idx, ele in restos:
        h[idx] = ele

    return h

# ***********************************************************************************
# *********************************** GREEDY Y BL ***********************************

@njit
def intercambio(i: int, j: int, s: np.ndarray) -> np.ndarray:
    ret: np.ndarray = s
    assert ret[i] == 1 and ret[j] == 0
    ret[i], ret[j] = ret[j], ret[i]
    return ret

@njit
def escoger_BL(d: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    
    coste_actual: float = evaluar(s, d)
    # Para cada elemento de u/i dentro de S
    for u in range(len(s)):
        # Para cada elemento de v/j fuera de S
        for v in range(len(s)):
            if s[u] == 1 and s[v] == 0:
                sumas: np.ndarray = np.zeros(len(s))
                suma_anterior: float = 0
                # Generamos un VECINO
                prima: np.ndarray = intercambio(u, v, s)
                # Para cada elemento de w fuera de S
                for w in range(len(prima)):
                    if prima[w] == 1:
                        sumas[v] += d[v,w]
                        sumas[w] += sumas[w] - d[w,u] + d[w,v]
                max_w_prima: float = np.max(sumas[prima == 1]) # Maximo de todos los d(v)
                min_w_prima: float = np.min(sumas[prima == 1]) # Minimo de todos los d(v)
                max_w_actual: float = np.max(sumas[s == 1])
                min_w_actual: float = np.min(sumas[s == 1])
                dmax_prima: float = np.max(np.array([sumas[v], max_w_prima]))
                dmin_prima: float = np.min(np.array([sumas[v], min_w_prima]))
                dmax_actual: float = np.max(np.array([sumas[v], max_w_actual]))
                dmin_actual: float = np.min(np.array([sumas[v], min_w_actual]))
                diff_prima: float = dmax_prima - dmin_prima
                diff_actual: float = dmax_actual - dmin_actual
                if diff_prima - diff_actual < 0:
                    coste_prima = evaluar(prima, d)
                    return prima, coste_prima
    return s, coste_actual

@njit
def BL(n: int, m: int, d: np.ndarray) -> tuple[np.ndarray, float]:

    actual: np.ndarray = solucion_aleatoria(n, m)
    coste_actual: float = evaluar(actual, d)
    coste_vecino = np.inf
    iters: int = 0

    while iters < MAX_EVALS and coste_vecino >= coste_actual:

        iters += 1

        vecino, coste_vecino = escoger_BL(d, actual)

        if coste_vecino < coste_actual:
            actual, coste_actual = vecino, coste_vecino

    return actual, coste_actual

@njit
def escoger_greedy(d: np.ndarray, s: np.ndarray) -> tuple[int, float]:
    sumas: np.ndarray = np.zeros(len(s), dtype=np.float64)
    suma_anterior: float = 0
    mejor_disp = np.inf
    mejor: int = np.random.choice(len(s))
    for u in range(len(s)):
        suma_actual: float = 0
        # Para cada elemento u no escogido
        # Suma desde u hasta cada uno de los v
        if s[u] == 0:
            for v in range(len(s)):
                if s[v] == 1:
                    suma_actual += d[u,v]
        # Luego para cada elemento v existente
        elif s[u] == 1:
            suma_actual = d[u, v] + suma_anterior
            suma_anterior = suma_actual
        sumas[u] = suma_actual
    
    # Para cada u, calculamos d_max(u) y d_min(u)
    max_v: float = np.max(sumas[s == 1])
    min_v: float = np.min(sumas[s == 1])
    for u in range(len(s)):
        if s[u] == 0:
            dmax: float = np.max(np.array([sumas[u], max_v]))
            dmin: float = np.min(np.array([sumas[u], min_v]))
            disp: float = dmax - dmin
            # Y escogemos el de menor dispersión
            if disp < mejor_disp:
                mejor_disp = disp
                mejor = u
    return mejor

@njit
def Greedy(n: int, m: int, d: np.ndarray) -> tuple[np.ndarray, float]:

    # El primer elemento es aleatorio (indice)
    s: np.ndarray = np.zeros(n, dtype=np.int64)
    s[np.random.choice(n)] = 1

    # Vamos escogiendo los menores segun g(u)
    while len(s[s == 1]) < m:
        idx = escoger_greedy(d, s)
        s[idx] = 1

    costo = evaluar(s, d)
    return s, costo

# ***********************************************************************************
# ****************************** ALGORITMOS GENETICOS *******************************

# ALGORITMO GENÉTICO GENERACIONAL UNIFORME
@njit
def AGG_uniforme(n: int, m: int, d: np.ndarray) -> tuple[np.ndarray, float]:

    # Generaciones
    t = 0

    # Esperanza matemática
    cruces_esperados = int(TAM_POB_GEN//2 * PROB_CRUCE_AGG)
    mutaciones = int(TAM_POB_GEN * PROB_MUTACION)

    pob = poblacion_aleatoria(n, m, TAM_POB_GEN) # Población inicial

    while t < MAX_EVALS:

        t += 1

        # Variable para el reemplazamiento
        mejor_anterior = encontrar_mejor(pob, d)[0]

        #* SELECCION
        # P' <- P
        # P' será la población durante una iteración del AG
        pob_prima = pob

        #* CRUCE
        # Cruzamos las 'cruces_esperados' primeras parejas de P
        i = 0
        cruces_realizados = 0
        while cruces_realizados < cruces_esperados and i < n-1:
            reemplazado = np.random.choice(2)
            pob_prima[i+reemplazado] = cruce_uniforme(
                pob_prima[i],pob_prima[i+1],n,m,d)
            cruces_realizados += cruces_realizados + 1
            i += 2

        #* MUTACION
        # Mutamos 'mutaciones' veces a nuestra población
        #? Un hijo puede mutar varias veces? Si -> replace=True
        #? Si replace=False, daria errores en la mayoria de los casos
        #? por que el nº mutaciones podria ser mayor que el tamaño de
        #? la poblacion y la funcion no escogeria elementos repetidos -> Exception
        idxs = np.random.choice(round(len(pob)), mutaciones)
        for i in idxs:
            mutar(pob_prima[i])
        
        #* REEMPLAZAMIENTO
        # Si la mejor solucion de P anterior no sobrevive
        # esta pasa a reemplazar la peor solucion de P'
        for hab in pob_prima:
            if hab_iguales(pob[mejor_anterior], hab):
                peor = encontrar_peor(pob_prima, d)
                pob_prima[peor] = pob[mejor_anterior]
                break
        pob = pob_prima
        
    #* EVALUACION
    mejor, coste = encontrar_mejor(pob, d)
    return pob[mejor], coste

# ALGORITMO GENÉTICO GENERACIONAL
@njit
def AGG_posicion(n: int, m: int, d: np.ndarray) -> tuple[np.ndarray, float]:

    # Generaciones
    t = 0

    # Esperanza matemática
    cruces_esperados = int(TAM_POB_GEN//2 * PROB_CRUCE_AGG)
    mutaciones = int(TAM_POB_GEN * PROB_MUTACION)

    pob = poblacion_aleatoria(n, m, TAM_POB_GEN)

    while t < MAX_EVALS:

        t += 1

        mejor_anterior = encontrar_mejor(pob, d)[0]

        #* SELECCION
        pob_prima = pob

        #* CRUCE
        i = 0
        cruces_realizados = 0
        while cruces_realizados < cruces_esperados and i < n-1:
            reemplazado = np.random.choice(2)
            pob_prima[i+reemplazado] = cruce_posicion(
                pob_prima[i], pob_prima[i+1], n, m)
            cruces_realizados += cruces_realizados + 1
            i += 2

        #* MUTACION
        idxs = np.random.choice(len(pob_prima), mutaciones)
        for i in idxs:
            mutar(pob_prima[i])

        #* REEMPLAZAMIENTO
        for hab in pob_prima:
            if hab_iguales(pob[mejor_anterior], hab):
                peor = encontrar_peor(pob_prima, d)
                pob_prima[peor] = pob[mejor_anterior]
                break
        pob = pob_prima

    #* EVALUACION
    mejor, coste = encontrar_mejor(pob, d)
    return pob[mejor], coste

# ALGORITMO GENÉTICO ESTACIONARIO UNIFORME
@njit
def AGE_uniforme(n: int, m: int, d: np.ndarray) -> tuple[np.ndarray, float]:

    # Generaciones
    t = 0

    pob = poblacion_aleatoria(n, m, TAM_POB_GEN) # Población inicial

    while t < MAX_EVALS:

        t += 1

        #* SELECCION
        # 2 padres aleatorios mediante torneo binario
        pob_prima = [torneo_binario(pob, d) for _ in range(2)]

        #* CRUCE Y MUTACION
        # recombinar pob_prima (cruzar con prob 1)
        for p in pob_prima:
            hijo = cruce_uniforme(pob_prima[0], pob_prima[1], n, m, d)
            if np.random.rand() < PROB_MUTACION:
                mutar(hijo)
            p = hijo

        #* REEMPLAZAMIENTO
        # Reemplazamos a los dos peores de P por los hijos de P'
        # siempre que el coste de los hijos sea mejor
        for hab_prima in pob_prima:
            peor = encontrar_peor(pob, d)
            if evaluar(pob[peor], d) >= evaluar(hab_prima, d):
                pob[peor] = hab_prima
                break

    #* EVALUACION
    mejor, coste = encontrar_mejor(pob, d)
    return pob[mejor], coste

# ALGORITMO GENÉTICO ESTACIONARIO DE POSICION
@njit
def AGE_posicion(n: int, m: int, d: np.ndarray) -> tuple[np.ndarray, float]:

    t = 0

    pob = poblacion_aleatoria(n, m, TAM_POB_GEN)

    while t < MAX_EVALS:

        t += 1

        #* SELECCION
        pob_prima = [torneo_binario(pob, d) for _ in range(2)]

        #* CRUCE Y MUTACION
        # recombinar pob_prima (cruzar con prob 1)
        for p in pob_prima:
            hijo = cruce_uniforme(pob_prima[0], pob_prima[1], n, m, d)
            if np.random.rand() < PROB_MUTACION:
                mutar(hijo)
            p = hijo
        
        #* REEMPLAZAMIENTO
        for hab_prima in pob_prima:
            peor = encontrar_peor(pob, d)
            if evaluar(pob[peor], d) >= evaluar(hab_prima, d):
                pob[peor] = hab_prima
                break

    #* EVALUACION
    mejor, coste = encontrar_mejor(pob, d)
    return pob[mejor], coste

# ***********************************************************************************
# ****************************** ALGORITMO MEMETICO *******************************

# @njit
# def pasar_a_enteros(s: np.ndarray) -> np.ndarray:
#     ret: list[int] = []
#     for idx in range(len(s)):
#         if s[idx] == 1:
#             ret.append(idx)
#     return np.array(ret)

# @njit
# def pasar_a_binario(s: np.ndarray) -> np.ndarray:
#     ret = np.zeros(len(s), dtype=np.int64)
#     ret[s] = 1
#     return np.array(ret)

# @njit
# def BL_AM():
#     pass

# # ALGORITMO MEMETICO
# @njit
# def AM(n: int, m: int, d: np.ndarray, p: float, mejores: bool = False) -> tuple[np.ndarray, float]:

#     assert p > 0 and p <= 1 and g > 0
    
#     t: int = 0

#     pob: np.ndarray = poblacion_aleatoria(n, m, TAM_POB_MEM)
#     pareja_a_cruzar: int = 0

#     for hab in pob:
#         hab: np.ndarray = pasar_a_enteros(hab)

#     while t < MAX_EVALS:

#         t += 1

#         if t > 0 and t % TAM_POB_MEM == 0 and np.random.rand() < p: # BL

#             if mejores:
#                 pass
#             else:
#                 pass

#         else: # AGG-Uniforme

#             # Variable para el reemplazamiento
#             mejor_anterior = encontrar_mejor(pob, d)[0]

#             #* SELECCION
#             # P' <- P
#             # P' será la población durante una iteración del AG
#             pob_prima = pob

#             #* CRUCE
#             # Cruzamos las 'cruces_esperados' primeras parejas de P
#             cruces_realizados = 0
#             if cruces_realizados < cruces_esperados and pareja_a_cruzar < n-1:
#                 reemplazado = np.random.choice(2)
#                 pob_prima[i+reemplazado] = cruce_uniforme(
#                     pob_prima[i], pob_prima[i+1], n, m, d)
#                 cruces_realizados += cruces_realizados + 1
#                 i += 2

#             #* MUTACION
#             # Mutamos 'mutaciones' veces a nuestra población
#             #? Un hijo puede mutar varias veces? Si -> replace=True
#             #? Si replace=False, daria errores en la mayoria de los casos
#             #? por que el nº mutaciones podria ser mayor que el tamaño de
#             #? la poblacion y la funcion no escogeria elementos repetidos -> Exception
#             idxs = np.random.choice(round(len(pob)), mutaciones)
#             for i in idxs:
#                 mutar(pob_prima[i])

#             #* REEMPLAZAMIENTO
#             # Si la mejor solucion de P anterior no sobrevive
#             # esta pasa a reemplazar la peor solucion de P'
#             for hab in pob_prima:
#                 if hab_iguales(pob[mejor_anterior], hab):
#                     peor = encontrar_peor(pob_prima, d)
#                     pob_prima[peor] = pob[mejor_anterior]
#                     break
#             pob = pob_prima

#             #* EVALUACION
#             mejor, coste = encontrar_mejor(pob_prima, d)


# ***********************************************************************************
