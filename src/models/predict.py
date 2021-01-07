import os
import torch

model = None

def predict(_model_, weight_path, text):
    global model
    model = _model_
    
    checkpoint = {}

    if os.path.isfile(weight_path):
        try:
            print("=> loading checkpoint '{}' ...".format(weight_path))
            if model.get_network_parameters('is_cuda'):
                checkpoint = torch.load(weight_path)
            else:
                # Load GPU model on CPU
                checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage)

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (trained for {} epochs)".format(weight_path, checkpoint['epoch']))
        except:
            print("Please Train your Network First!")
            return     
    else:
        print("Please Train your Network First!")
        return None

    model.eval()

    device = model.get_network_parameters('device')
    mask =  model.generate_square_subsequent_mask(1).to(device)

    text = [checkpoint['c2i_encoding'][c] for c in text]
    text = torch.LongTensor(text).to(device)
    text = text.unsqueeze(0)
    output = model(text, mask)
    output = output.cpu().detach().numpy()
    output = [checkpoint['i2c_encoding'][i] for i in output]

    return output