
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, DistilBertConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from .. import SentenceTransformer
#from ..evaluation import SentenceEvaluator


class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, device:str = None):
        """
        A CrossEncoder takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
        on a scale of 0 ... 1.

        It does not yield a sentence embedding and does not work for individually sentences.

        :param model_name: Any model name from Huggingface Models Repository that can be loaded with AutoModel. We provide several pre-trained CrossEncoder models that can be used for common tasks
        :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that outputs a continous score 0...1. If > 1, it output several scores that can be soft-maxed to get probability scores for the different classes.
        :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max length of the model will be used
        :param device: Device that should be used for the model. If None, it will use CUDA if available.
        """

        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt")

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized

   


    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores


    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logging.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)

