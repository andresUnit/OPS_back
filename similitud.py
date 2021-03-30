import os
import json
import sys
import shutil
import uuid

import datetime
import uuid
from urllib.parse import unquote_plus
import pandas as pd
import torch
from botocore.exceptions import ClientError
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertModel, BertTokenizer, AutoTokenizer
import numpy as np
from numpy import dot
from numpy.linalg import norm
from transformers import DistilBertTokenizer
from tqdm import tqdm





###### importar libreria transformer y modelo sentence BERT ######
import sentence_transformers as st





###################################################################

import procesamiento_similitud as procesamiento


model = st.SentenceTransformer('sentence-model/')

data = pd.read_excel("salidaregla7Ene.xlsx")
data = data[pd.notnull(data["ObservacionFinal"])]
# calcular vector de de caracteristicas
df_similitud = procesamiento.calcular_vector_caracteristicas(data,model)

# luego este batch debe ser utilizado para calcular las similitudes
# con la lambda CalcularSimilitudes
# 
print(len(df_similitud))


df_similitud.to_excel("resultadoSimiEne.xlsx", index = False)

def similitud_coseno_numpy(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    result = cos_sim
    return result


def similitud(data):
    
    """
    Se calculan las similitudes entre los comentarios del nuevo batch(df1) y
    la base de datos(df) y se obtiene como salida un df con las similitudes.
    
    df: Base de datos total
    df1: Nuevo batch
    
    return:
        df: Nueva base de datos = df + df1
        df_similitud: DataFrame de similitudes entre comentarios   
    
                
    """
    
    df = data.copy()
    n_columna1 = 'ObservacionFinal'
    delta = datetime.timedelta(days = 30)
    df["regla1"] = 1
    df["regla2"] = 1
    df["regla3"] = 1
    df["Semejanza_Persona"] = 0
    df["Semejanza_Difer"] = 0
    df["Semejanza_Persona"] = df["Semejanza_Persona"].astype(object)
    df["Semejanza_Difer"] = df["Semejanza_Difer"].astype(object)
    # se agrega df1 a df para comparar cada elemento de df1 con todos los elementos existentes.


    

  
    
    # loop en cada un de los comentarios del batch
    for i in tqdm(range(len(df))):
        obs_i = df[n_columna1].iloc[i] #observacion i OPS
        fecha_i = df['Creado'].iloc[i] # fecha creacion obs i
        operacion_i = df['Operacion'].iloc[i]
        empresa_i = df['EmpresaContratistaPersonaLiderActividad'].iloc[i]
        llave_i = df["Llave"].iloc[i]
        # Se selecciona a partir de la base de datos el conjunto de elementos que tienen la misma operacion y
        # fueron creados antes que la obs_i
        conjunto = df[(df['Operacion']==operacion_i)&(df['Creado']<fecha_i)]
        conjunto_name = conjunto[(conjunto["Creado"]> fecha_i-delta)&(conjunto["Llave"]==llave_i)]
        conjunto_Noname = conjunto[(conjunto["Creado"]> fecha_i-delta)&(conjunto["Llave"]!=llave_i)]
        aux = []
        aux1 = []
        #print("frase------------")
        #print(obs_i)
        #print("conjunto ---------------------------")
        #print(conjunto_name)
        for j in range(len(conjunto_Noname)):                
            obs_j = conjunto_Noname[n_columna1].iloc[j] # observacion j 
            ponderador = np.min([len(obs_i),len(obs_j)])/np.max([len(obs_i),len(obs_j)])
            similitud = similitud_coseno_numpy(np.array(df["VectorCaracteristicas"].iloc[i]),np.array(conjunto_Noname["VectorCaracteristicas"].iloc[j]))
            similitudPon = similitud*ponderador
            
            if similitudPon == 1.0:
                aux.append(obs_j)  
                df["regla2"].iloc[i] = 0
        for h in range(len(conjunto_name)):                
            obs_h = conjunto_name[n_columna1].iloc[h] # observacion j 
            ponderador = np.min([len(obs_i),len(obs_h)])/np.max([len(obs_i),len(obs_h)])
            similitud = similitud_coseno_numpy(np.array(df["VectorCaracteristicas"].iloc[i]),np.array(conjunto_name["VectorCaracteristicas"].iloc[h]))
            similitudPon = similitud*ponderador
            #print(similitudPon)
            if similitudPon == 1.0:
                df["regla1"].iloc[i] = 0
            elif (similitudPon <= 1.0) & (similitudPon > 0.7):
                aux1.append(obs_h)
                df["regla3"].iloc[i] = 0

        df["Semejanza_Persona"].iloc[i] = aux1
        df["Semejanza_Difer"].iloc[i] = aux
    return df
    
aux = similitud(df_similitud)