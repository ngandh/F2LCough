from tqdm import tqdm
from statistics import mean

from protonets.utils import filter_opt
from protonets.models import get_model

def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)

def evaluate(model, data_loader, meters, desc=None):
    model.eval()

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    list_loss_val = []
    list_acc_val = []
    for sample in data_loader:
        _, output = model.loss(sample)
        # print("TAD = ", output)
        list_loss_val.append(output['loss'])
        list_acc_val.append(output['acc'])
        for field, meter in meters.items():
            meter.add(output[field])
    
    
    # print(mean(list_acc_val))
    return meters, mean(list_acc_val), mean(list_loss_val)
