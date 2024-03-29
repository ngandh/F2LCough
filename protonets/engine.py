from tqdm import tqdm
import torch 
from statistics import mean

class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end', 
                      'loss_val', 'acc_val']

        self.hooks = { }
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None
        self.hooks['loss_val'] = None
        self.hooks['acc_val'] = None

    def train(self, **kwargs):
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0, # epochs done so far
            't': 0, # samples seen so far
            'batch': 0, # samples seen in current epoch
            'stop': False,
            'acc_val': 0,
            'loss_val':0,
            'acc_train': 0,
            'loss_train':0
        }

        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()
            epoch_loss = []
            epoch_grad = []

            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                state['sample'] = sample
                self.hooks['on_sample'](state)

                state['optimizer'].zero_grad()
                loss, state['output'] = state['model'].loss(state['sample'])
                self.hooks['on_forward'](state)

                loss.backward()
                self.hooks['on_backward'](state)

                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)
                
                #add to calculate grad_train: start
                param_grad = []
                for tag, param in state['model'].named_parameters():
                  # print(tag + '_grad: {}\n'.format(torch.mean(param.grad))) -- của 1 param
                  param_grad.append(torch.mean(param.grad).item())

                # print('param_grad: ', param_grad)
                epoch_grad.append(mean(param_grad))
                #add to calculate grad_train: end
                epoch_loss.append(loss.item())
            print('grad_per_epoch: ', sum(epoch_grad) / len(epoch_grad))
            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)
        #add to return grad_train
        return state['model'].state_dict(), state['output']['acc'], sum(epoch_loss) / len(epoch_loss), state['acc_val'], state['loss_val']
      
