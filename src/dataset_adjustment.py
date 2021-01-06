from .utils import build_dataset, TextLibrary

def adjust_dataset(params:dict) -> TextLibrary:
    return build_dataset(params)