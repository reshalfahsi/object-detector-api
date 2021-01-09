"""
TextLibrary Dataset handler.

Mostly copy-paste from https://github.com/domschl/torch-poet
"""
from src.models.train import train
from typing import Union
import torch
from ..utils import TextLibrary

class TextLibraryDataset(torch.utils.data.Dataset):
    def __init__(self, textlib: TextLibrary, sample_length: int, device: Union[str, torch.device], text_quanta: int = 10) -> None:
        self.textlib = textlib
        self.device = device
        self.vocab_size = len(textlib.i2c)
        self.text_quanta = text_quanta
        self.sample_length = sample_length
        self.length = int((len(self.textlib.data)-sample_length-1)/text_quanta)
        # self.gpu_data=torch.cuda.LongTensor(textlib.encode(textlib.data[:-1]))
        self.__encoding = (textlib.c2i, textlib.i2c)
        self.data = textlib.encode(textlib.data)
        self.data = torch.LongTensor(self.data).to(self.device)

    def __len__(self):
        return self.length
    
    def get_encoding(self):
        return self.__encoding
    
    def get_data(self):
        return self.data
    
    def set_data(self, _data_):
        self.data = _data_

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.length:
            return None
        X = self.data[idx*self.text_quanta:idx *
                      self.text_quanta+self.sample_length].to(self.device)
        y = self.data[idx*self.text_quanta+1:idx *
                      self.text_quanta+self.sample_length+1].to(self.device)
        return X, y

def split_train_test(dataset : TextLibraryDataset) -> tuple:
    
    data_length_split = int(0.8 * len(dataset))
    data = dataset.get_data()
    data_train = data[:data_length_split]
    data_test = data[data_length_split:]
    train = dataset
    test = dataset
    train.set_data(data_train)
    test.set_data(data_test)

    return (train, test)
