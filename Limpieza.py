
from datetime import datetime
import unicodedata
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

import os
files = []
for file in os.listdir("D:\GIT\OPS\data"):
    if file.startswith("results"):
        files.append(file)
        
data = pd.DataFrame()
for i in tqdm(files):
    aux = pd.read_csv(i)
    data=pd.concat([aux,data], axis = 0)
    data.drop_duplicates(["title"], inplace=True, keep="first")

data.to_excel("testFinalF.xlsx", index = False)

data = pd.read_excel("dataCanada.xlsx")
df = pd.read_excel("dataTotalF.xlsx")


data = pd.read_excel("dataSantiagoCorp.xlsx")

data=data[data['observationstatus']=="Completed"]

data.to_excel("dataSantiagoCorp.xlsx", index= False)


def transforma_fiscal(fecha):
    """
    Se transforma fecha en formato datatime
    a formato fiscal year
    
    """
    
    dic_meses = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug',
             'Sep','Oct', 'Nov', 'Dec']
    mes = fecha.month
    ano = fecha.year
    if mes >= 7:
        ano = ano + 1
        
    fecha_fiscal = dic_meses[mes-1]+' - FY'+ str(ano)[-2:]
    return fecha_fiscal


def crear_observacion_final(df):
    """
    Se juntan las columnas otras observaciones y
    observaciones positivas.
    """
    df= df.fillna(value=np.nan)
    df['ObservacionFinal'] = np.nan
    for i in range(len(df)):
        obs = df['Observacion'].iloc[i]
        obs_pos = df['ObservacionPositiva'].iloc[i]
        if (type(obs)==str) or (type(obs_pos)==str): # si alguno de los 2 es string 
            if type(obs) is not str: # si observacion es NaN
                obs = '' # observacion es vacio           
            if type(obs_pos) is not str:
                obs_pos = '' # observacion es vacio
            df.loc[i,'ObservacionFinal'] = obs+obs_pos
        
    return df


def renombrar_columnas(df):
    """
    Se renombran columnas
    """
    
    #df['observationdate'] = df['observationdate'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    #df['observationcreatedontimestam1'] = df['observationcreatedontimestam1'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))



    columnas_rename = {'title':'IdActividad',
                    'observationstatus':'Estado',
                    'leadobservername':'ObservadorPrincipal',
                    'leadobserverdepartmentname':'AreaObservadorPrincipal',
                    'otherobservations':'Observacion',
                    'ptoonlypositiveobservations':'ObservacionPositiva',
                    'leadobservercontractorcompan':'EmpresaContratistaPersonaLiderActividad',
                    'leadobserveroperatinggroupfy':'Operacion',
                    'grcasset':'Asset',
                    'leadobserverorganisationleve1':'OrganizationLevel1',
                    'leadobserverorganisationleve2':'OrganizationLevel2',
                    'leadobserverorganisationleve3':'OrganizationLevel3',
                    'leadobserverorganisationleve4':'OrganizationLevel4',
                    'leadobserverorganisationleve5':'OrganizationLevel5',
                    'leadobserverorganisationleve6':'OrganizationLevel6',
                    'observationcreatedontimestam1':'Creado',
                    'locationvisitedname':'LugarVisitado',
                    'observationdate':'FechaObservacion',
                    'leadobserveremail':'Email'   
    }
    
    columnas = ['IdActividad','Estado', 'ObservadorPrincipal', 'AreaObservadorPrincipal', 'Observacion',
                'ObservacionPositiva',
                'EmpresaContratistaPersonaLiderActividad', 'Operacion', 'Asset', 'Creado', 'LugarVisitado',
                'OrganizationLevel1','OrganizationLevel2','OrganizationLevel3',
                'OrganizationLevel4','OrganizationLevel5','OrganizationLevel6',
                'FechaObservacion', 'Email']
    
    df = df.rename(columns = columnas_rename)
    df = df[columnas]
    
    df = df.reset_index()
    # calcular fecha solo para archivos completados
    df['FechaObservacion'] = (df["FechaObservacion"]/1000).apply(lambda x: datetime.utcfromtimestamp(x))
    df['FiscalDate'] = df[df['Estado']=='Completed']['FechaObservacion'].apply(transforma_fiscal)
    df = crear_observacion_final(df)
    return df
    
    df = df.rename(columns = columnas_rename)
    df = df[columnas]
    # calcular fecha solo para archivos completados
    df = df.reset_index()
    df['FiscalDate'] = df[df['Estado']=='Completed']['FechaObservacion'].apply(transforma_fiscal)
    df = crear_observacion_final(df)
    return df


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



assert(obtener_fecha('Feb - FY19') > obtener_fecha('Dec - FY19'))
assert(obtener_fecha('Apr - FY20') > obtener_fecha('Oct - FY20'))



def limpieza_rrhh(x):
    """
    Limpiar campos de base de datos
    """
    x= x.strip()
    x= x.lower()
    x=x.replace(' ','')
    x= unicodedata.normalize("NFKD", x).encode("ascii","ignore").\
        decode("ascii").replace("#", "ñ").replace("%", "Ñ")
    # limpiar espacios vaciones
    x=x.replace('  ','')
    x=x.replace('   ','')
    x=x.replace('    ','')
    x=x.replace('     ','')
    x=x.replace('      ','')
    return x



def clean_ops(df):
    """
    Data cleaning de OPS filtrada
    df: OPS (df)
    """
    
    df = df[df['Estado']=="Completed"]
    print(len(df))
    
    # creacion nueva variable
    df=df[pd.notnull(df["FiscalDate"])]
    df['FechaFiscalOPS'] = df['FiscalDate'].apply(obtener_fecha)
    #====
    #Limpieza de datos
    #====
    
    # Eliminar aquellas con estado == Borrador
    
    # Se crea nueva columna Nombre
    # se reemplza por NN aquellos sin Nombre
    df['Nombre'] = df['ObservadorPrincipal'].replace(np.nan,'NN')
    
    
    # Limpieza en observador principal
    # se cambian los nan por 'nan' para poder procesar sobre los campos vacios
    df['AreaObservadorPrincipal'] =\
    df['AreaObservadorPrincipal'].replace(np.nan,'nan')
    
    df['AreaObservador']=\
    df['AreaObservadorPrincipal'].\
    apply(lambda x: limpieza_rrhh(x))

    # Se crea columna OPS
    # OPS: Observacion con algoritmos de limpieza
    df["OPS"]=np.nan

    df["OPS"]=\
    df['ObservacionFinal'].\
    apply(lambda x: limpieza_rrhh(str(x)))

    df['ObservacionFinal'] = \
    df['ObservacionFinal'].apply(lambda x: str(x).replace('\n', ' '))
    
    
    # Se reemplaza en empresa contratista.... nan por propio
    df['EmpresaContratistaPersonaLiderActividad'] =\
    df['EmpresaContratistaPersonaLiderActividad'].replace(np.nan,'propio')

    # limpieza de columna Empresa contratista.....
    df['EmpresaContratista']=\
    df['EmpresaContratistaPersonaLiderActividad'].\
    apply(lambda x: limpieza_rrhh(str(x)))
    
    

    
    #columna tipi
    # Contratista si el valor es diferente a propio.
    # Si hay otro valor se asigna BHP
    df["Tipo"]=\
        df['EmpresaContratista'].\
        apply(lambda x: "Contratista" if x!='propio' else "BHP" )
    
    # Se crea una llave unica
    # Si empresa contratista de la persona lider de la actividad == propio 
    # Llave = Nombre####Apellido####AreaDelObservadorPrincipal####Email
    # Si no (persona es de empresa contratista)
    # Llave = Nombre###Apellido####EmpresaContratistaDeLaPersonaLiderDeLaActividad####Email
    df["Llave"]=df.apply(lambda x: str(x["Nombre"]).strip().replace(' ','#').lower()+"####"\
                    +str(x['AreaObservador'])+"####"+\
                    str(x["Email"]) if
                    x['EmpresaContratista']=='propio'\
                    else str(x["Nombre"]).strip().replace(' ','#').lower()+"####"+\
                    str(x['EmpresaContratista'])+"####"\
                    +str(x["Email"]), axis=1)
    
    # se cambian todos los 'nan' por Nan
    #df = df.replace('nan',np.nan)
    
    
    return df
    
def quitar_duplicado(df,batch):
    """
    Quitar elementos duplicados del batch
    df: base de datos
    batch: muestra de datos
    """
    conteo = 0
    new_batch = pd.DataFrame()
    for i in range(len(batch)):
        ID = batch['IdActividad'].iloc[i]
        muestra = df[df['IdActividad'] == ID]
        if len(muestra)==0: # si es unico
            conteo += 1
            dato = batch[batch['IdActividad']==ID]
            new_batch = pd.concat([new_batch,dato])

    
    return new_batch, conteo


data=data[data['observationstatus']=="Completed"]
batch = renombrar_columnas(data)
print('renombrar_columnas:',len(batch))
print('renombrar_columnas:',batch.keys())
batch1 = clean_ops(batch)
print('Limpieza:', len(batch1))
print(batch1.keys())

batch1.drop("index", inplace=True, axis = 1)
batch1.to_excel("salidalimEne.xlsx", index = False)
