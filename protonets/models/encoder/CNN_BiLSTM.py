import torch
import torch.nn as nn
import torch.nn.functional as F
from protonets.models.encoder.baseUtil import Flatten, get_padding
from collections import OrderedDict
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from protonets.models.encoder.TCResNet import TCResNet8, TCResNet8Dilated



class CNN_BiLSTM(nn.Module): #in_c, in_h, in_w, width_multiplier=1.0):
    def __init__(self, in_c, in_h, in_w, width_multiplier=1.0):
        super(CNN_BiLSTM, self).__init__()
        sequence_length = 28
        input_size = in_c*in_h*in_w
        input_size = 3402 #lfcc
        input_size = 672
        hidden_size = 128
        num_layers = 2
        num_classes = 48
        batch_size = 100
        num_epochs = 2
        learning_rate = 0.003
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv_k_size = (3,1)
        out_channels = 64
        self.conv1 = nn.Conv2d(in_c, 64, self.conv_k_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_c, 16, self.conv_k_size, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels, out_channels, self.k_size, stride=self.stride1, 
        #                 bias=False, padding=get_padding(in_height, in_width, self.k_size[0], 
        #                 self.k_size[1], self.stride1,d_h=dilation), dilation=dilation)
        # self.conv4 = nn.Conv2d(in_channels, out_channels, self.k_size, stride=self.stride1, 
        #                 bias=False, padding=get_padding(in_height, in_width, self.k_size[0], 
        #                 self.k_size[1], self.stride1,d_h=dilation), dilation=dilation)
        # self.conv5 = nn.Conv2d(in_channels, out_channels, self.k_size, stride=self.stride1, 
        #                 bias=False, padding=get_padding(in_height, in_width, self.k_size[0], 
                        # self.k_size[1], self.stride1,d_h=dilation), dilation=dilation)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        x = x.cuda()
        # print("x shape: ", x.shape) #x shape:  torch.Size([28, 1, 81, 40])
        out = self.conv1(x)
        h0 = torch.zeros(self.num_layers*2, out.size(0), self.hidden_size).cuda()# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, out.size(0), self.hidden_size).cuda()
        # print("outt shape: ", out.shape) #outt shape:  torch.Size([28, 64, 81, 42])
        # Forward propagate LSTM
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2]*out.shape[3])
        out, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        #print(out.shape) #out.shape = 28, 64, 256
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        # print(out.shape) #out.shape = 28, 48
        # out = TCResNet8Dilated(out[0], out[1], out[2])
        return out
