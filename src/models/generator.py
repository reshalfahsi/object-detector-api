from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from . import train
from . import predict


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PoemGenerator(nn.Module):
    """
    Deep Learning Model for Generating Poem
    """

    def __init__(self, input_size, embedding_size=200, nheads=8, hidden_dim=256, num_encoder_layers=6, num_decoder_layers=6):
        super(PoemGenerator, self).__init__()

        self.__network_parameters = {}

        self.__network_parameters['is_cuda'] = torch.cuda.is_available()
        self.__network_parameters['device'] = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.__network_parameters['learning_rate'] = 1e-3
        self.__network_parameters['optimizer'] = torch.optim.Adam(
            self.parameters(), lr=self.__network_parameters['learning_rate'], weight_decay=0.0)

        self.__network_parameters['loss_function'] = nn.CrossEntropyLoss()

        if self.__network_parameters['is_cuda']:
            self.cuda()
            self.__network_parameters['loss_function'].cuda()

        self.__network_parameters['best_loss'] = 9.9999999999e9
        self.__network_parameters['start_epoch'] = 0
        self.__network_parameters['batch_size'] = 768

        self.__network_parameters['train_dataset'] = None
        self.__network_parameters['train_loader'] = None

        self.__network_parameters['input_size'] = input_size
        self.pos_encoder = PositionalEncoding(embedding_size)
        encoder_layers = TransformerEncoderLayer(
            embedding_size, nheads, hidden_dim)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_encoder_layers)
        self.encoder = nn.Embedding(input_size, embedding_size)
        self.decoder = nn.Linear(embedding_size, input_size)

    def forward(self, x):

        return None

    def get_network_parameters(self, key=''):
        if key == '':
            print('Failed to get network parameters')
            return None

        return self.__network_parameters[key]

    def set_network_parameters(self, key='', value=None):
        if key == '' or value is None:
            print("Please insert the key or the value")
            return None

        self.__network_parameters[key] = value

    def compile(self, dataset=None, loss_fn=None, optimizer=None) -> None:

        if loss_fn is not None:
            self.__network_parameters['loss_function'] = loss_fn
            if self.__network_parameters['is_cuda']:
                self.__network_parameters['loss_function'].cuda()

        if optimizer is not None:
            self.__network_parameters['optimizer'] = optimizer

        if dataset is None:
            print("Please insert a valid dataset format.")
            return

        self.__network_parameters['train_dataset'] = dataset
        self.__network_parameters['train_loader'] = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.__network_parameters['batch_size'], shuffle=True, num_workers=0)

        return None

    def fit(self, path: str = '') -> bool:

        if path == '':
            print("Please insert the valid path")
            return False

        return train.process(self, path)

    def predict(self, weight_path='', path=''):

        if path == '' and weight_path == '':
            print("Please Insert Path!")
            return None
        
        return predict.predict(self, weight_path, path)