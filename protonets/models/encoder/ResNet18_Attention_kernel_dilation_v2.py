from torch import nn
import torch.nn.functional as F
import torch
import math
from dill import dumps, loads


#NOTE: add norm l2 to loss
#update dilation: 1, 1, 2, 4, kernel: (3,3) -> (3,1)'; (7,7) -.(\7,1)
class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg = nn.AvgPool2d(2)
        self.max = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio , kernel_size=1, stride=1, padding='same',
                                   bias=True)# activation=tf.nn.relu, ban dau: out: 
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding='same', bias=True)
        #ban dau in = 64
    def forward(self, inputs):
        # print(self.in_planes)
        # print("in avg:", inputs.shape)
        avg_out = F.adaptive_avg_pool2d(inputs, (1,1))
        max_out = F.adaptive_max_pool2d(inputs, (1,1))
        avg_out = avg_out.view(avg_out.shape[0], avg_out.shape[1])
        max_out = max_out.view(max_out.shape[0], max_out.shape[1])
        
        # print("avg", avg_out.shape)
        # print("max", max_out.shape)
        
        avg = avg_out.view(avg_out.shape[0], avg_out.shape[1], 1, 1)
        max = max_out.view(max_out.shape[0], max_out.shape[1], 1, 1 )
        # print("avg2", avg.shape)
        # print("max2", max.shape)
        # avg = layers.Reshape((1, 1, avg.shape[1]))(avg)  # shape (None, 1, 1 feature)
        # max = layers.Reshape((1, 1, max.shape[1]))(max)  # shape (None, 1, 1 feature)
        # print("convavg: ", self.conv1(avg).shape )
        avg_out = self.conv2(self.conv1(avg))
        avg_out = F.relu(avg_out)
        max_out = self.conv2(self.conv1(max))
        out = avg_out + max_out
        out = F.sigmoid(out)
        # print("channel: ", out.shape)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=(kernel_size,1), padding='same', stride=1) # activation=tf.nn.sigmoid
        
    def forward(self, inputs):
        avg_out = torch.mean(inputs, axis=1) #axis ban dau = 3
        max_out = torch.max(inputs, axis=1)[0] #axis ban dau = 3
        # print("mean, max: ", avg_out.shape, max_out.shape)
        out = torch.stack([avg_out, max_out], axis=1)
        # print("stack: ", out.shape)
        out = self.conv1(out)
        out = F.sigmoid(out)
        # print("spatial: ", out.shape)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, training=False):
        super(BasicBlock, self).__init__()
        self.training = training
        self.stride = stride
        # print('di: ', dilation)
        # self.conv1 = nn.Conv2d(in_channels, out_channels, padding='same', kernel_size=3, stride=2)
        self.conv1 = Conv2dSame(in_channels=in_channels, out_channels=out_channels, kernel_size=(7, 1), stride=(stride, stride), groups=1, bias=True, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=self.training)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), padding='same', stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=self.training)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, self.expansion * out_channels,
                                                                kernel_size=1, stride=stride, dilation=dilation),
                                          nn.BatchNorm2d(self.expansion * out_channels, affine=self.training))
        else:
            # print("zoo")
            # self.shortcut = lambda x,_: x
            self.shortcut = lambda x: x

    def forward(self, inputs):
        # print("block: ", inputs.shape)
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # print("out, bn2: ", out.shape)
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        # print("outbefore: ", out.shape)
        # print("stride: ", self.stride)
        # print("outshort: ", self.shortcut(inputs).shape)
        out = out + self.shortcut(inputs)
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(out_channels, 1, 1, padding='same')
        self.bn1 = nn.BatchNorm2d()

        self.conv2 = nn.Conv2d(out_channels, (3, 1), strides, padding='same')
        self.bn2 = nn.BatchNorm2d()
        self.conv3 = nn.Conv2d(out_channels * self.expansion, 1, 1, padding='same')
        self.bn3 = nn.BatchNorm2d()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(self.expansion * out_channels, kernel_size=1,
                                                                stride=strides, padding='same'),
                                        nn.BatchNorm2d())
        else:
            self.shortcut = lambda x, _: x

    def forward(self, x, training=False):
        out = F.relu(self.bn1(self.conv1(x), training))
        out = F.relu(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)

        out = out + self.shortcut(x, training)
        out = F.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, blocks, layer_dims, num_classes=9, training = False):
        super(ResNet, self).__init__()
        self.in_channels = 64 #ban dau 64
        num_classes=48

        self.stem = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(3,1), padding='same', stride=1),
                                nn.BatchNorm2d(64, affine = training)) #ban dau in: 64

        self.layer1 = self.build_resblock(blocks, 64, layer_dims[0], stride=1, dilation = 1 )
        self.layer2 = self.build_resblock(blocks, 128, layer_dims[1], stride=2, dilation = 2)
        self.layer3 = self.build_resblock(blocks, 256, layer_dims[2], stride=2, dilation = 4)
        self.layer4 = self.build_resblock(blocks, 512, layer_dims[3], stride=2, dilation = 1)
        self.final_bn  = nn.BatchNorm2d(512)

        self.avgpool = nn.AvgPool2d(2)
        # self.fc = nn.Flatten() #, activation='softmax')
        self.fc = nn.Linear(512, num_classes)

    # 2. ResBlock
    def build_resblock(self, blocks, out_channels, num_blocks, stride, dilation):
        strides = [stride] + [1] * (num_blocks - 1)  # [1]*3 = [1, 1, 1]
        res_blocks = nn.Sequential()
        res = []

        for stride in strides:
            res.append(blocks(self.in_channels, out_channels, stride, dilation))
            self.in_channels = out_channels
        res_blocks = nn.Sequential(*res)
        # print(res_blocks)

        return res_blocks

    def forward(self, inputs, training=False):
        # print("in: ", inputs.shape)
        out = self.stem(inputs)
        # out = inputs
        out = F.relu(out)

        out = self.layer1(out) #, training=training)
        out = self.layer2(out) #, training=training)
        out = self.layer3(out) #, training=training)
        out = self.layer4(out)#, training=training)
        out = self.final_bn(out)#, training=training)
        out = F.relu(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.shape[0], out.shape[1])
        # print("avgRes: ", out.shape)
        out = self.fc(out)
        out = F.softmax(out)

        return out

def ResNet18_kernel_dilation_v2():
    return ResNet(BasicBlock, [2, 2, 2, 2])