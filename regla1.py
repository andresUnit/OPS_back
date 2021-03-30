import pandas as pd
import numpy as np
from datetime import datetime

def obtener_fecha(fecha):
    """
    Obtener fecha en formato datatime año normal
    fecha: Month - FYXX
    """
    #diccionario meses
    dic_meses = {'Jan':'01', 'Feb':'02', 'Mar': '03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08',
             'Sep':'09','Oct':'10', 'Nov': '11', 'Dec':'12'} 
    
    
    mes_fy = fecha.split('-')
    mes = mes_fy[0][:3]
    fy = mes_fy[1][-2:]
    if int(dic_meses[mes]) >= 7: # En Julio comienza nuevo FY
        fy = int(fy)-1
        
    date = str(dic_meses[mes])+'-'+str(fy)
    date = datetime.strptime(date, '%m-%y')
    return date


def regla1(df,mes_actual):
    """
    Copia textual a si mismo
    Se considera como copia textual de si mismo, registros de una persona
    que se repiten en tiempo
    Entradas:
            df: DataFrame batch con columnas Llave1, Llave 2, similitud normalizada, fecha fiscal OPS, Empresa,
                Operación. 
            mes: mes para el cual se realiza la consulta (ej. FEB-FY19)
    Salidas:
            df: DataFrame con copia anterior y copia mes
    """
    mes_actual_fecha = obtener_fecha(mes_actual) # mes actual formato fecha
    llaves1 = df['Llave'].unique()
    df["regla 1"] = 1
    for llave in llaves1:
        # Seleccion OPS escritas por la misma persona
        operacion = df[(df['Llave']==llave)]['Operacion'].unique()[0] # operaciones persona
        empresa = df[(df['Llave']==llave)]['EmpresaContratista'].unique()[0] # una persona solo esta dentro de una empresa
        nombre = df[(df['Llave']==llave)]['ObservadorPrincipal'].unique()[0]
        email = df[(df['Llave']==llave)]['Email'].unique()[0]
        
        # Llave 1 = Llave 2
        conjunto = df[(df['Llave']==llave)&(df['Llave']==llave)]
        # nombre operacion de Llave 1
        conjunto_copia = conjunto[conjunto['SimilitudNormalizada']==1]
        # solo nos interesa saber si la ops es copiada, no cuantas
        # veces es copiada. Por esos se quitan  todos los duplicados
        conjunto_copia = conjunto_copia.drop_duplicates(subset ="IdActividad", 
                     keep = 'first') 
        # copia mes actual
        copias_mes = len(conjunto_copia[conjunto_copia['FechaFiscalOPS'] == mes_actual_fecha])
        # copia con meses anteriores
        copias_pasado = len(conjunto_copia[conjunto_copia['FechaFiscalOPS'] < mes_actual_fecha])
        # copias totales
        copias_totales = copias_mes + copias_pasado
        if copias_totales > 0:
            df[df["Llave"]==llave]["regla 1"]=0



    return df


def regla2(df,mes_actual):
    """
    Copia textual a otra persona en la misma operacion
    Se considera como copia textual a otras persona, si similititud normalizada
    Entradas:
            df: DataFrame batch con columnas Llave1, Llave 2, similitud normalizada, fecha fiscal OPS, Empresa,
                Operación. 
            mes: mes para el cual se realiza la consulta (ej. FEB-FY19)
    Salidas:
            df: DataFrame con copia anterior ,copia mes y copias totales
    """
    mes_actual_fecha = obtener_fecha(mes_actual) # mes actual formato fecha
    llaves1 = df['Llave'].unique()
    df["regla 2"] = 1
    for llave in llaves1:
        # Seleccion OPS escritas por una persona diferente
        operacion = df[(df['Llave']==llave)]['Operacion'].unique()[0]
        empresa = df[(df['Llave']==llave)]['EmpresaContratista'].unique()[0] # una persona solo esta dentro de una empresa
        nombre = df[(df['Llave']==llave)]['ObservadorPrincipal'].unique()[0]
        email = df[(df['Llave']==llave)]['Email'].unique()[0]
        
        # Llave 1 diferente a Llave 2 (copia a otra persona)
        conjunto = df[(df['Llave']==llave)&(df['Llave']!=llave)]
        # nombre operacion de Llave     
        # copia textual similitud normalizada = 1        
        conjunto_copia = conjunto[conjunto['SimilitudNormalizada']==1]
        # solo nos interesa saber si la ops es copiada, no cuantas
        # veces es copiada. Por esos se quitan  todos los duplicados
        conjunto_copia = conjunto_copia.drop_duplicates(subset ="IdActividad1", 
                     keep = 'first') 
        # copia mes actual
        copias_mes = len(conjunto_copia[conjunto_copia['FechaFiscalOPS'] == mes_actual_fecha])
        # copia con meses anteriores
        copias_pasado = len(conjunto_copia[conjunto_copia['FechaFiscalOPS'] < mes_actual_fecha])
        # copias totales
        copias_totales = copias_mes + copias_pasado
        # crear nuevo DataFrame
        # nombre copias mes
        if copias_totales > 0:
            df[df["Llave"]==llave]["regla 2"]=0

    return df

def regla3(df,mes_actual,umbral=0.73):
    """
    Similitud a si mismo
    Se considera como copia textual a si mismo, si similititud normalizada < 0.7
    Entradas:
            df: DataFrame batch con columnas Llave1, Llave 2, similitud normalizada, fecha fiscal OPS, Empresa,
                Operación. 
            mes: mes para el cual se realiza la consulta (ej. FEB-FY19)
    Salidas:
            df: DataFrame con copia anterior ,copia mes y copias totales
    """
    mes_actual_fecha = obtener_fecha(mes_actual)
    # mes actual formato fecha
    llaves1 = df['Llave'].unique()
    df["regla 3"] = 1
    for llave in llaves1:
        # Seleccion OPS escritas por una persona diferente
        operaciones = df[(df['Llave'] == llave)]['Operacion'].unique()
        if len(operaciones) > 1:
            print('Hay mas de una operacion')
        empresa = df[(df['Llave'] == llave)]['EmpresaContratista'].unique()[0]
        # una persona solo esta dentro de una empresa
        nombre = df[(df['Llave'] == llave)]['ObservadorPrincipal'].unique()[0]
        email = df[(df['Llave']==llave)]['Email'].unique()[0]
        for operacion in operaciones:
            # Llave 1 igual a Llave 2 
            conjunto = df[(df['Llave']==llave)&(df['Llave']==llave)]
            # nombre operacion de Llave     
            # copia textual similitud normalizada = 1        
            conjunto_copia = conjunto[(conjunto['SimilitudNormalizada']<1)&(conjunto['SimilitudNormalizada']>umbral)]
            # solo nos interesa saber si la ops es copiada, no cuantas
            # veces es copiada. Por esos se quitan  todos los duplicados
            conjunto_copia = conjunto_copia.drop_duplicates(subset ="IdActividad",keep = 'first') 
            # copia mes actual
            copias_mes = len(conjunto_copia[conjunto_copia['FechaFiscalOPS'] == mes_actual_fecha])
            # copia con meses anteriores
            copias_pasado = len(conjunto_copia[conjunto_copia['FechaFiscalOPS'] < mes_actual_fecha])
            # copias totales
            copias_totales = copias_mes + copias_pasado
            # crear nuevo DataFrame
            # nombre copias mes
            if copias_totales > 0:
                df[df["Llave"]==llave]["regla 3"]=0


    return df

data = "resultadoSimi.xlsx"
df = pd.read_excel(data)

df_regla1 = regla1(df,'Feb - FY21')
print(len(df_regla1))


df_regla2 = regla2(df_regla1,'Feb - FY21')
print(len(df_regla2))
print(df_regla2.keys())

df_regla3 = regla3(df_regla2,'Feb- FY21')
print(len(df_regla3))
print(df_regla3.keys())

df_regla3.to_excel("dataProcesada.xlsx", index = False)

