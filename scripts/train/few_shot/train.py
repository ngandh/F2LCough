import os
import json
from functools import partial
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
import torchvision
import torchnet as tnt

from protonets.engine import Engine

import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils
from datetime import datetime, timedelta
import time
import dill 

def update_weights(opt, model, train_loader, val_loader):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    opt['log.exp_dir'] = os.path.join('./fewshotspeech/results', opt['log.exp_dir'], timestamp)
    opt['log.exp_dir'] = "/home/pi/Documents/F2LCough/cough_classification/F2LCough/results/" + timestamp
    os.makedirs(opt['log.exp_dir'])
    
    # save opts
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')
    
    if opt['data.cuda']:
        model.cuda()

    engine = Engine()

    meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }

    if val_loader is not None:
        meters['val'] = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] }

    def on_start(state):
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['train.decay_every'], gamma=0.5)
    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state['scheduler'].step()
    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        state['acc_train'] = state['output']['acc']
        state['loss_train'] = state['output']['loss']

        for field, meter in meters['train'].items():
            meter.add(state['output'][field])
    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        state['acc_val'] = 0
        state['loss_val'] = 0
        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

        if val_loader is not None:
            _, state['acc_val'], state['loss_val']  = model_utils.evaluate(state['model'],
                                val_loader,
                                meters['val'],
                                desc="Epoch {:d} valid".format(state['epoch']))

        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))

        meter_vals['epoch'] = state['epoch']
        # with open(trace_file, 'a') as f:
        #     json.dump(meter_vals, f)
        #     f.write('\n')

        if val_loader is not None:
            if state['loss_val'] < hook_state['best_loss']:
                hook_state['best_loss'] = state['loss_val']
                print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))
                print("update: ", opt['log.exp_dir'])
                state['model'].cpu()
                torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'), pickle_module=dill)
                if opt['data.cuda']:
                    state['model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > opt['train.patience']:
                    print("==> patience {:d} exceeded".format(opt['train.patience']))
                    state['stop'] = True
        else:
            state['model'].cpu()
            torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'), pickle_module=dill)
            if opt['data.cuda']:
                state['model'].cuda()
        # return acc_val, loss_val
      


    engine.hooks['on_end_epoch'] = partial(on_end_epoch, { })

    start = time.time()
    w, acc_train, loss, acc_val, loss_val = engine.train(
        model = model,
        loader = train_loader,
        optim_method = getattr(optim, opt['train.optim_method']),
        optim_config = { 'lr': opt['train.learning_rate'],
                        'weight_decay': opt['train.weight_decay'] },
        max_epoch = opt['train.epochs']
    )
    
    end = time.time()
    elapsed = str(timedelta(seconds= end-start))
    # print('grad_train: ', grad_train)
    return w, acc_train, loss, acc_val, loss_val

def main(opt):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    opt['log.exp_dir'] = os.path.join('./results', opt['log.exp_dir'], timestamp)
    opt['log.exp_dir'] = "/home/pi/Documents/F2LCough/cough_classification/F2LCough/results/" + timestamp
    os.makedirs(opt['log.exp_dir'])
    
    # save opts
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')

    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    torch.manual_seed(1234)
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    if opt['data.trainval']:
        data = data_utils.load(opt, ['trainval'])
        train_loader = data['trainval']
        val_loader = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']

    model = model_utils.load(opt)

    if opt['data.cuda']:
        model.cuda()

    engine = Engine()

    meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }

    if val_loader is not None:
        meters['val'] = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] }

    def on_start(state):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['train.decay_every'], gamma=0.5)
    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state['scheduler'].step()
    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        state['acc_train'] = state['output']['acc']
        state['loss_train'] = state['output']['loss']
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])
    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        state['acc_val'] = 0
        state['loss_val'] = 0
        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

        if val_loader is not None:
            _, state['acc_val'], state['loss_val']  = model_utils.evaluate(state['model'],
                                 val_loader,
                                 meters['val'],
                                 desc="Epoch {:d} valid".format(state['epoch']))

        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))

        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        if val_loader is not None:
            if meter_vals['val']['loss'] < hook_state['best_loss']:
                hook_state['best_loss'] = meter_vals['val']['loss']
                print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))
                print("opt: ", opt['log.exp_dir'])
                state['model'].cpu()
                torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'), pickle_module=dill)
                if opt['data.cuda']:
                    state['model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > opt['train.patience']:
                    print("==> patience {:d} exceeded".format(opt['train.patience']))
                    state['stop'] = True
        else:
            state['model'].cpu()("==> patience {:d} exceeded".format(opt['train.patience']))      
            torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'), pickle_module=dill)
            if opt['data.cuda']:
                state['model'].cuda()
        # return acc_val, loss_val

    engine.hooks['on_end_epoch'] = partial(on_end_epoch, { })

    start = time.time()
    w, acc_train, loss, acc_val, loss_val = engine.train(
        model = model,
        loader = train_loader,
        optim_method = getattr(optim, opt['train.optim_method']),
        optim_config = { 'lr': opt['train.learning_rate'],
                         'weight_decay': opt['train.weight_decay'] },
        max_epoch = opt['train.epochs']
    )
    end = time.time()
    elapsed = str(timedelta(seconds= end-start))
    print("Total Time: {}".format(elapsed))
    print("acc_train = {}".format(acc_train))
    print("loss = {}".format(loss))
    print("acc_val = {}".format(acc_val))
    print("loss_val = {}".format(loss_val))
