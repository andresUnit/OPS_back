import json
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


def Regla5(df, mes_actual = "Dec - FY21"):
    """
    Registro de OPS vacio si numero de caracteres menor a 5 y mayor a 0
    
    """
    # columna largo de observacion
    mes_actual_fecha = obtener_fecha(mes_actual)
    print(mes_actual_fecha)
    #df = df[df['FechaFiscalOPS']==mes_actual_fecha]
    df["regla5"] = 1
    for index, row in df.iterrows():
        largo_obs = len(str(row['OPS']))
        if (largo_obs>5) & (largo_obs<20):
            df.loc[index, "regla5"] = 0
            
    
    return df



data = "salidaregla4Ene.xlsx"
batch = pd.read_excel(data)
#batch.set_index("index", inplace = True)
print(batch)
df_regla5 = Regla5(batch)
print(len(df_regla5))
df_regla5.to_excel("salidaregla5Ene.xlsx", index = False)

