import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import time
import torch
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
import json
import argparse
import sys
sys.argv=['']
del sys
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import scripts.train.few_shot.train as model_train
# model_train.reload(functions.readfunctions)
import tensorflow as tf

import torch
from tensorboardX import SummaryWriter



def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args(args=['--model=cnn', '--dataset=cifar', '--gpu=0', '--iid=1', '--epochs=10'])
    return args

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


if __name__ == '__main__':
    #---------setup------------------------------
    parser = argparse.ArgumentParser(description='Train prototypical networks')
    # data args
    default_dataset = 'googlespeech'
    parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                        help="data set name (default: {:s})".format(default_dataset))
    default_split = 'vinyals'
    parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                        help="split name (default: {:s})".format(default_split))
    parser.add_argument('--data.way', type=int, default=1, metavar='WAY',
                        help="number of classes per episode (default: 60)")
    parser.add_argument('--data.shot', type=int, default=2, metavar='SHOT',
                        help="number of support examples per class (default: 5)")
    parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                        help="number of query examples per class (default: 5)")
    parser.add_argument('--data.test_way', type=int, default=1, metavar='TESTWAY',
                        help="number of classes per episode in test. 0 means same as data.way (default: 5)")
    parser.add_argument('--data.test_shot', type=int, default=2, metavar='TESTSHOT',
                        help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
    parser.add_argument('--data.test_query', type=int, default=13, metavar='TESTQUERY',
                        help="number of query examples per class in test. 0 means same as data.query (default: 15)")
    parser.add_argument('--data.train_episodes', type=int, default=100, metavar='NTRAIN',
                        help="number of train episodes per epoch (default: 100)")
    parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                        help="number of test episodes per epoch (default: 100)")
    parser.add_argument('--data.trainval', action='store_true', help="run in train+validation mode (default: False)")
    parser.add_argument('--data.sequential', action='store_true', help="use sequential sampler instead of episodic (default: False)")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")

    # model args
    default_model_name = 'protonet_conv'
    default_encoding = 'C64'
    parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                        help="model name (default: {:s})".format(default_model_name))
    parser.add_argument('--model.x_dim', type=str, default="1,51,40", metavar='XDIM',
                        help="dimensionality of input images (default: '1,28,28')")
    parser.add_argument('--model.hid_dim', type=int, default=64, metavar='HIDDIM',
                        help="dimensionality of hidden layers (default: 64)")
    parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                        help="dimensionality of input images (default: 64)")
    parser.add_argument('--model.encoding', type=str, default=default_encoding, metavar='MODELENC',
                        help="model encoding (default: {:s})".format(default_encoding))
    # train args
    parser.add_argument('--train.epochs', type=int, default=2, metavar='NEPOCHS', #default = 20
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                        help='optimization method (default: Adam)')
    parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                        help='number of epochs after which to decay the learning rate')
    default_weight_decay = 0.0
    parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                        help="weight decay (default: {:f})".format(default_weight_decay))
    parser.add_argument('--train.patience', type=int, default=200, metavar='PATIENCE',
                        help='number of epochs to wait before validation improvement (default: 1000)')

    # log args
    default_fields = 'loss,acc'
    parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                        help="fields to monitor during training (default: {:s})".format(default_fields))
    default_exp_dir = 'fewshotspeech/results'
    parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                        help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))

    # speech data args
    parser.add_argument('--speech.include_background', action='store_true', help="mix background noise with samples (default: False)")
    parser.add_argument('--speech.include_silence', action='store_true', help="one of the classes out of n should be silence (default: False)")
    parser.add_argument('--speech.include_unknown', action='store_true', help="one of the classes out of n should be unknown (default: False)")
    parser.add_argument('--speech.sample_rate', type=int, default=16000, help='desired sampling rate of the input')
    parser.add_argument('--speech.clip_duration', type=int, default=1000, help='clip duration in milliseconds')
    parser.add_argument('--speech.window_size', type=int, default=40)
    parser.add_argument('--speech.window_stride', type=int,default=20)
    parser.add_argument('--speech.num_features', type=int, default=40, help='Number of mfcc features to extract')
    parser.add_argument('--speech.time_shift', type=int, default=100, help='time shift the audio in milliseconds')
    parser.add_argument('--speech.bg_volume', type=float, default=0.1, help='background volumen to mix in between 0 and 1')
    parser.add_argument('--speech.bg_frequency', type=float, default=1.0, help='Amount of samples that should be mixed with background noise (between 0 and 1)')
    parser.add_argument('--speech.num_silence', type=int, default=1000, help='Number of silence samples to generate')
    parser.add_argument('--speech.foreground_volume', type=float, default=1)

    #triplet
    parser.add_argument('--tripletLoss', default=False)

    args = vars(parser.parse_args(args=[]))

    opt = args

    # opt['data.dataset'] = 'coughspeech'

    print(opt['data.dataset'])
    #-------------------start-----------------------------------------
    start_time = time.time()

    # define paths
    print('data: heavy, night')
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    # opt['data.cuda'] =  True
    # opt['data.way'] =  2
    # opt['data.shot'] =  1
    # opt['data.query'] =  12
    # opt['data.test_way'] =  2
    # opt['data.test_shot'] =  1
    # opt['data.test_query'] =  12
    # opt['data.train_episodes'] =  200
    # opt['data.test_episodes'] =  100
    # opt['model.hid_dim'] =  64
    # opt['model.z_dim'] =  64
    # opt['train.optim_method']='Adam'
    # opt['train.learning_rate']=0.001
    # opt['train.decay_every']=20
    # opt['train.weight_decay']=0.0
    opt['data.cuda'] =  False # True if using GPU
    opt['data.way'] =  2
    opt['data.shot'] =  1 #2
    opt['data.test_way'] =  2
    opt['data.test_shot'] =  1 #2
    opt['data.query'] = 1 # 12
    opt['data.test_query'] =  1 #13
    opt['data.train_episodes'] =  2 #200
    opt['data.test_episodes'] =  2 #100
    opt['model.hid_dim'] =  64
    opt['model.z_dim'] =  64
    opt['train.optim_method']='Adam'
    opt['train.learning_rate']=0.001
    opt['train.decay_every']=20
    opt['train.weight_decay']=1e-5
    # opt['train.patience']=200
    # opt['log.exp_dir']=4
    # opt['model.encoding']='TCResNet8Dilated'
    # opt['model.encoding']='CNN_BiLSTM'
    # opt['model.encoding']='Triplet'
    opt['model.encoding']='Attention' #mine
    # opt['model.encoding']='Attention_kernel_v2'
    # opt['model.encoding']='Attention_dilation_v2'
    # opt['model.encoding']='Attention_kernel_dilation_v2'
    opt['speech.include_unknown']= False
    opt['speech.include_background']= False
    # for ws in range(40, 88, 2):
    opt['speech.window_size'] = 128 #-> 500
    opt['speech.window_stride'] = 128/2 #-> 400
    # print(opt['speech.window_size'])
    # opt['tripletLoss'] = True
    # print(opt['tripletLoss'])
    print('encoder: ', opt['model.encoding'])
    print('way: {}, shot: {}'.format(opt['data.way'], opt['data.shot']))
    print('query: {}, test query: {}'.format(opt['data.query'], opt['data.test_query']))

    args = args_parser()
    # exp_details(args)
    args.gpu = 0 # 1 if using gpu
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    opt['data.trainval'] = False
    

    # load dataset and user groups
    if opt['data.trainval']:
        data = data_utils.load(opt, ['trainval'])
        train_dataset = data['trainval']
        test_dataset = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        # print("data", data)
        train_dataset = data['train']
        test_dataset = data['val']
        # print(test_dataset)

    # i = (test_dataset[1])
    # print(i)

    # BUILD MODEL
    opt['model.x_dim'] = 1,51,40 #mfcc FIRST #40ms -20ms
    opt['model.x_dim'] = 1,16,40 # ws = 128ms - 64ms
    # opt['model.x_dim'] = 1,3,40 # ws = 500ms - 400ms
    # opt['model.x_dim'] = 1, 63, 40 #mfcc second ws = 64ms - 32ms
    # # opt['model.x_dim'] = 1,13,98 #gfcc
    # opt['model.x_dim'] = 1,81,40 #lfcc
    # opt['model.x_dim'] = 1,16,256 #CNN-Bi 1CONV
    # opt['model.x_dim'] = 1,81,40 #CNN-BI 2CONV
    global_model = model_utils.load(opt)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    if opt['data.cuda']:
        global_model.cuda()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    opt['log.fields'] = ['acc','loss']
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    print(args.epochs)
    print(args.num_users)

    print(type(global_model))
    print(type(train_dataset))
    print(type(test_dataset))
    start_time = time.time()
    # for epoch in tqdm(range(args.epochs)):
    for epoch in range(0,1):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = [0]
        idxs_users = [0, 1, 2, 3, 4]
        print('idxs_users', idxs_users)
        # print(idxs_users)

        for idx in idxs_users:
            print(idx)

            # local_model = LocalUpdate()
            w, acc_train, loss, acc_val, loss_val = model_train.update_weights(
                opt, 
                model=copy.deepcopy(global_model), 
                train_loader=train_dataset[idx], 
                val_loader=test_dataset[idx])
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            break

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        print(loss_avg)
    end_time = time.time()
    total_time = (end_time - start_time) * 1000 # (ms)
    print(total_time)