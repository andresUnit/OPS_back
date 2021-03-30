import numpy as np
import pandas as pd


def calcular_fv_sentence_transformer(comentario,model):
    """
    Calcular vector de caracteristicas utilizando SentenceTransformer
    comentario: str
    modelo: Sentence transfromer model
    """
    sentence_embeddings = model.encode(comentario)
    return sentence_embeddings    

def calcular_vector_caracteristicas(df,model):
    """
    Calcular vector de caracterisitcas para DataFrame
    df: DataFrame
    modelo: Sentence transformer model
    """
    df["VectorCaracteristicas"]=df['ObservacionFinal'].apply(lambda x:calcular_fv_sentence_transformer(str(x),model))
    return df