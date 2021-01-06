"""
Training Mechanism Part of Deep Learning Model for Generating Poem.

Mostly copy-paste from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import torch
from torch.utils.data import DataLoader

import os
import time
import math

model = None


def eval(test_loader, input_size, device, batch_size):
    global model

    model.eval()  # Turn on the evaluation mode
    total_loss = 0.

    src_mask = model.generate_square_subsequent_mask(batch_size).to(device)

    with torch.no_grad():
        for (data, target) in test_loader:
            if data.size(0) != batch_size:
                src_mask = model.generate_square_subsequent_mask(
                    data.size(0)).to(device)
            output = model(data, src_mask)
            output_flat = output.view(-1, input_size)
            total_loss += len(data) * model.get_network_parameters(
                'loss_function')(output_flat, target).item()
    return total_loss / (len(test_loader) - 1)


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
            model.get_network_parameters('optimizer').load_state_dict(
                checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
                path, checkpoint['epoch']))
        except:
            print("Training Failed!")
            return success

        for epoch in range(model.get_network_parameters('num_epochs')):

            loss_now = train(epoch, path)

            model.set_network_parameters('loss_now', loss_now)
            model.set_network_parameters('epoch_now', epoch)

            success = True

        return success


def train(epoch, path):
    global model

    model.train()

    batch_size = model.get_network_parameters('batch_size')
    start_epoch = model.get_network_parameters('start_epoch')
    best_loss = model.get_network_parameters('best_loss')

    train_dataset = model.get_network_parameters('train_dataset')
    train_loader = model.get_network_parameters('train_loader')
    test_loader = model.get_network_parameters('test_loader')

    cuda = model.get_network_parameters('is_cuda')
    device = model.get_network_parameters('device')

    c2i_encoding = model.get_network_parameters('c2i_encoding')
    i2c_encoding = model.get_network_parameters('i2c_encoding')

    input_size = model.get_network('input_size')
    mask = model.generate_square_subsequent_mask(batch_size).to(device)
    total_loss = 0.
    scheduler = model.get_network_parameters('scheduler')

    epoch_start_time = time.time()

    for idx, (data, target) in enumerate(train_loader):
        batch_time = time.time()

        if cuda:
            data, target = data.cuda(), target.cuda()

        model.get_network_parameters('optimizer').zero_grad()

        if data.size(0) != batch_size:
            mask = model.generate_square_subsequent_mask(
                data.size(0)).to(device)

        output = model(target, mask)
        loss = model.get_network_parameters('loss_function')(
            output.view(-1, input_size), target)

        if cuda:
            loss.cpu()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        model.get_network_parameters('optimizer').step()

        total_loss += loss.item()
        log_interval = 200
        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - batch_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, idx, len(
                          train_dataset) // batch_size, scheduler.get_lr()[0],
                      elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            batch_time = time.time()

    val_loss = eval(test_loader, input_size, device, batch_size)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_loss:
        best_loss = val_loss
        model.set_network_parameters('best_loss', best_loss)
        save_checkpoint({'c2i_encoding': c2i_encoding, 'c2i_encoding': i2c_encoding, 'epoch': start_epoch + epoch + 1, 'state_dict': model.state_dict(),
                         'optimizer_state_dict': model.get_network_parameters('optimizer').state_dict(), 'best_loss': best_loss}, path)
    return val_loss
