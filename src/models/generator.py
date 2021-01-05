import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import train
from . import predict

class PoemGenerator(nn.Module):
    """
    Deep Learning Model for Generating Poem
    """

    def __init__(self):
        super(PoemGenerator, self).__init__()
        
        self.__network_parameters = {}

        self.transformer = nn.Transformer()
        self.query_pos = nn.Parameter()

    def forward(self):
        
        return None
    
    def train(self):
        return None
    
    def predict(self):
        return None
