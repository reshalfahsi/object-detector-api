"""
TextLibrary Dataset handler.

Mostly copy-paste from https://github.com/domschl/torch-poet
"""

import torch
from ..utils import TextLibrary


class TextLibraryDataset(torch.utils.data.Dataset):
    def __init__(self, textlib: TextLibrary, sample_length: int, device: str, text_quanta: int = 10) -> None:
        self.textlib = textlib
        self.device = device
        self.vocab_size = len(textlib.i2c)
        self.text_quanta = text_quanta
        self.sample_length = sample_length
        self.length = int((len(self.textlib.data)-sample_length-1)/text_quanta)
        # self.gpu_data=torch.cuda.LongTensor(textlib.encode(textlib.data[:-1]))
        self.data = torch.LongTensor(
            textlib.encode(textlib.data)).to(self.device)

    def __len__(self):
        return self.length

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
