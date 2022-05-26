# base model for blind SR, input LR, output kernel + SR
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import autocast as autocast

class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()

        self.ResBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, padding=3),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.ResBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel,kernel_size=5, padding=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.ResBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(in_channels=3*channel, out_channels=channel, kernel_size=3, padding=1) 
        
    def forward(self, x):
        
        residual = x
        x1 = self.ResBlock1(x)
        x1 += residual
        
        x2 = self.ResBlock2(x)
        x2 += residual

        x3 = self.ResBlock3(x)
        x3 += residual

        
        out = torch.cat([x1,x2,x3], axis=1)
        
        out = self.conv(out)

        return out

class SplitModule(nn.Module):
    def __init__(self, channel, split_num):
        super(SplitModule, self).__init__()

        self.channel = channel
        self.split_num = split_num
        self.share = int(self.channel / self.split_num)
        self.mod = int(self.channel % self.split_num)

        self.Match = nn.Conv2d(in_channels=self.channel, out_channels=(self.channel-self.mod), kernel_size=3, padding=1)
        
        self.AffineModule = nn.Sequential(
            nn.Conv2d(in_channels=(int(self.channel/self.split_num)), out_channels=(int(self.channel/self.split_num)), kernel_size=1),
            nn.BatchNorm2d(int(self.channel/self.split_num)),
            nn.ReLU(),
        )

        self.split_layers = []
        for i in range(self.split_num):
            self.split_layers.append(self.AffineModule)
        # self.split_layers = nn.Sequentail(self.split_layers)

    def forward(self, x):
        x = self.Match(x)

        tmp = 0
        kernel_list = []
        for i in range(self.share, self.channel+self.share, self.share):
            kernel_list.append(x[:,tmp:i,:,:])
            tmp = i

        for j, kernels in enumerate(kernel_list):
            kernel_list[j] = self.split_layers[j](kernels)

        out = torch.cat(kernel_list, axis=1)

        return out



class KernelEstimation(nn.Module):
    ''' Network of KernelEstimation'''
    def __init__(self, in_nc=3, kernel_size=21, channels=[128, 256, 128, 64, 32], split_num=2):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        self.head = nn.Conv2d(in_channels=in_nc, out_channels=channels[0], kernel_size=3, padding=1, bias=True, dtype=torch.float32)
        
        self.RB1 = ResBlock(channel=channels[0])
        self.SP1 = SplitModule(channel=channels[0], split_num=8)
        
        self.conv1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=1, bias=True)
        self.RB2 = ResBlock(channel=channels[1])
        self.SP2 = SplitModule(channel=channels[1], split_num=8)
        
        self.conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1, bias=True)
        self.RB3 = ResBlock(channel=channels[2])
        self.SP3 = SplitModule(channel=channels[2], split_num=8)

        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[3], out_channels=169, kernel_size=3, padding=1),
            nn.BatchNorm2d(169),
        )
        
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        
        x = self.head(x)
        x = self.RB1(x)
        multiple = self.SP1(x)
        
        # multiple = multiple.permute(0,1,3,2)
        # x = torch.matmul(x, multiple)
        x = x * multiple
        
        x = self.conv1(x)
        x = self.RB2(x)
        multiple2 = self.SP2(x)
        
        # x = torch.matmul(x, multiple2)
        x = x * multiple2
        
        x = self.conv2(x)
        x = self.RB1(x)
        multiple3 = self.SP1(x)
        x = x * multiple3
        
        # x = torch.matmul(x, multiple3)
        x = self.tail(x)
        out = self.softmax(x)        

        return out