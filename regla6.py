

import json
import datetime
import os
import shutil
import sys
from urllib.parse import unquote_plus



import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from tqdm import tqdm





MODEL_NAME = "bert_model_conducta.pth"

def Regla6(df):

    # cargar modelo
    model = load_model()

    # cargar data

    # labels
    label_cols = ["1", "2.2", "3.4", "4.6", "5.8", "7"]

    # tokenizador
    max_length = 100
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    test_comments = list(df["ObservacionFinal"].values)

    # Encoding input data
    test_encodings = tokenizer.batch_encode_plus(
        test_comments, max_length=max_length, pad_to_max_length=True
    )
    test_input_ids = test_encodings['input_ids']
    test_token_type_ids = test_encodings['token_type_ids']
    test_attention_masks = test_encodings['attention_mask']

    # Tensores
    test_inputs = torch.tensor(test_input_ids)
    test_masks = torch.tensor(test_attention_masks)
    test_token_types = torch.tensor(test_token_type_ids)

    # Create test dataloader

    batch_size = 8
    test_data = TensorDataset(test_inputs, test_masks,
                              test_token_types)  # test_labels,
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=batch_size)

    # Test
    model.eval()

    # track variables
    logit_preds, pred_labels, tokenized_texts = [], [], []

    # Predict
    for i, batch in tqdm(enumerate(test_dataloader)):
        batch = tuple(t for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_token_types = batch

        with torch.no_grad():
            # Forward pass
            outs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)
            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            #b_labels = b_labels.to('cpu').numpy()

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        pred_labels.append(pred_label)

    # Flatten outputs
    tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]
    # Converting flattened binary values to boolean values

    # boolean output after thresholding
    pred_bools = [pl > 0.6 for pl in pred_labels]
    df_bool = pd.DataFrame(data=pred_bools, columns=label_cols)
    df_int = df_bool.astype(int)
    df_nota_aux = df_int.idxmax(axis=1)
    df_nota = pd.DataFrame(columns = ["regla6"])
    df_nota["regla6"] = df_nota_aux.values
    df_concat = pd.concat([df, df_nota], axis=1)

    return df_concat


def get_date():
    return datetime.datetime.today().strftime("%Y/%m/%d")


def load_model():
    config = BertConfig()
    config.num_labels = 6
    config.vocab_size = 105879
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(MODEL_NAME,map_location=torch.device('cpu')), strict = False)
    return model


data = "salidaregla7Ene.xlsx"
batch = pd.read_excel(data)
batch = aux
pd.isnull(batch["ObservacionFinal"]).sum()
batch[pd.isnull(batch["ObservacionFinal"])]
batch = batch[pd.notnull(batch["ObservacionFinal"])]
batch.reset_index(inplace = True, drop= True)
print(batch)
batch = batch.reset_index(drop = True)
df_regla6 = Regla6(batch)
print(len(df_regla6))
df_regla6.to_excel("salidaSpence.xlsx", index=False)
