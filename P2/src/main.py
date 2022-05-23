import gkdmh
import pobalg
import time

def main():

    datapath = 'datos/'
    algoritmos = [
        pobalg.Greedy,
        pobalg.BL,
        pobalg.AGG_uniforme,
        pobalg.AGG_posicion,
        pobalg.AGE_uniforme,
        pobalg.AGE_posicion,
    ]
    print(f'Extrayendo datos desde {datapath}...')
    datos = gkdmh.extraer_datos_dir(datapath)
    primero = True

    for caso in datos:
        # Extraemos los datos del caso
        nombre_caso, d, n, m = caso

        # Realizamos una compilación de todas las funciones para
        # que los tiempos resultantes del primer caso sean más
        # acordes al resto (ejecutamos el primer caso 2 veces en total)
        if primero:
            print('Preparando (compilando) los algoritmos, espere...')
            for algoritmo in algoritmos:
                algoritmo(n, m, d)
                print(f'{algoritmo.__name__} cargado')
            primero = False
        
        # Ahora ejecutamos cada uno de los algoritmos para cada uno de los casos
        print(f'\t--- Caso {nombre_caso} ---')
        print(f'\t n = {n} \t\t\t m = {m}')
        # *********** Codigo para probar los cruces **********
        # p1 = pobalg.solucion_aleatoria(n, m)
        # p2 = pobalg.solucion_aleatoria(n, m)
        # h1 = pobalg.cruce_uniforme(p1, p2, n, m, d)
        # h2 = pobalg.cruce_posicion(p1, p2, n, m)
        # unos1 = np.count_nonzero(h1)
        # unos2 = np.count_nonzero(h2)
        # if unos1 != m or unos2 != m:
        #     raise Exception(
        #         f'unos1,unos2 = {unos1,unos2} y m = {m} desiguales'
        #     )
        # print(f'\n{p1} Padre 1')
        # print(f'{p2} Padre 2')
        # print(f'{h1} Hijo uniforme')
        # print(f'{h2} Hijo posicion \n')
        # ****************************************************
        # Algoritmos Greedy, BL y geneticos
        for algoritmo in algoritmos:
            print(f'{algoritmo.__name__}')
            inicio = time.time()
            coste = round(algoritmo(n,m,d)[1], 4)
            fin = time.time()
            tiempo = round(fin - inicio, 4)
            print(f'Coste: {coste}')
            if tiempo < 0.1:
                print(f'Tiempo {tiempo*1000} ms\n')
            else:
                print(f'Tiempo: {tiempo} segundos\n')
        
        # Algoritmos meméticos
        print(f'{pobalg.AM.__name__}')
        for i in range(3):
            if i == 0:
                print(f'Probabilidad de BL: 1')
            elif i == 1:
                print(f'Probabilidad de BL: 0.1')
            else:
                print(f'Probabilidad de BL: 0.1 * mejor')

if __name__ == "__main__":
    main()