import os

model = None

def predict(_model_, weight_path, path):
    global model
    model = _model_
    
    return None