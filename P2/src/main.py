import gkdmh

def main():

    datapath = 'datos/'
    print(f'Extrayendo datos desde {datapath}...')
    datos = gkdmh.extraer_datos_dir(datapath)

    for caso in datos:
        nombre_caso = caso[0]
        print(f'\t--- Caso {nombre_caso} ---')
        # ...


if __name__ == "__main__":
    main()