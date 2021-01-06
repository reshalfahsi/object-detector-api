import torch
from torch.utils.data import DataLoader

import os

model = None

def eval():
    return None

def save_checkpoint(state, filename=''):
    """Save checkpoint if a new best is achieved"""
    if (filename == ''):
        print('Path Empty!')
        return None
    torch.save(state, filename)  # save checkpoint
    print("=> Saving a new best")

def process(_model_, path):
    global model

    model = _model_
    success = False
    checkpoint = {}

    if path == '':
        print("Please Insert Path!")
        return success
    if os.path.isfile(path):
        try:
            print("=> loading checkpoint '{}' ...".format(path))
            if model.get_network_parameters('is_cuda'):
                checkpoint = torch.load(path)
            else:
                # Load GPU model on CPU
                checkpoint = torch.load(
                    path, map_location=lambda storage, loc: storage)

            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']

            model.set_network_parameters('start_epoch', start_epoch)
            model.set_network_parameters('best_loss', best_loss)

            model.load_state_dict(checkpoint['state_dict'])
            model.get_network_parameters('optimizer').load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
                path, checkpoint['epoch']))
        except:
            print("Training Failed!")
            return success

def train():
    return None