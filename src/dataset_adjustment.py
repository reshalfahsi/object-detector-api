from .utils import build_dataset
from .datasets import TextLibraryDataset
import torch

def adjust_dataset(params:dict) -> TextLibraryDataset:
    textlib = build_dataset(params)
    device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
    textlib_dataset = TextLibraryDataset(textlib, params['sample_length'], device)
    return textlib_dataset