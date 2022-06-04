import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import numpy as np
import os

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Conv_emb(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv_emb, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, 1, stride=stride, padding=0)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()
        # print("oc: ", output_channels)

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        
        return out


class Res_2d_mp(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Res_2d_mp, self).__init__()
        # output_channels = 64
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        # print("x: ", x.shape)
        # print("out: ", out.shape)
        out = x + out
        
        out = self.mp(self.relu(out))
        
        return out

class MyModel(nn.Module):
    def __init__(self, ndim=64, edim=64, cdim=1):
        super(MyModel, self).__init__()
        # audio embedding
        ndim=16
        edim=16
        self.spec_bn = nn.BatchNorm2d(1).cuda()
        self.layer1 = Conv_2d(1, ndim, pooling=2).cuda()
        self.layer2 = Res_2d_mp(ndim, ndim, pooling=2).cuda()
        self.layer3 = Conv_2d(ndim, ndim*2, pooling=1).cuda()
        self.layer4 = Res_2d_mp(ndim*2, ndim*2, pooling=1).cuda()
        self.layer5 = Res_2d_mp(ndim*2, ndim*2, pooling=1).cuda()
        self.layer6 = Res_2d_mp(ndim*2, ndim*2, pooling=(1)).cuda()
        self.layer7 = Conv_2d(ndim*2, ndim*4, pooling=(2,3)).cuda()
        self.layer8 = Conv_emb(ndim*4, ndim*4).cuda()
        self.audio_fc1 = nn.Linear(ndim*4, ndim*2).cuda()
        self.audio_bn = nn.BatchNorm1d(ndim*2).cuda()
        self.audio_fc2 = nn.Linear(ndim*2, edim).cuda()

        # others
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def spec_to_embedding(self, spec):
        # input normalization
        # spec = spec.reshape(spec.shape[0], spec.shape[2],spec.shape[1])
        # print(spec.shape)
        out = spec.unsqueeze(1).cuda()
        # print(out.shape)
        out = self.spec_bn(out).cuda()
        # print(out.shape)

        # CNN
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.squeeze(2)
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2])
        # print(out.shape)
        out = nn.MaxPool1d(out.size(-1))(out)
        out = out.view(out.size(0), -1)

        # projection
        out = self.audio_fc1(out)
        out = self.audio_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.audio_fc2(out)
        return out

    def forward(self, spec):
        
        # print(spec.shape)
        audio_emb = self.spec_to_embedding(spec)
        return audio_emb
        
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("MLP in: ", x.shape) #torch.Size([28, 1, 16])
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        # x = x.view(-1, x.shape[0])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        # x = x.squeeze()
        # return self.softmax(x)
        return x

