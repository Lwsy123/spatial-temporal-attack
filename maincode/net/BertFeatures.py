import math
import torch
from torch import nn
import numpy as np
from d2l import  torch as d2l
from torch.nn import Linear, Conv1d, MaxPool1d, BatchNorm1d
from torch.nn import functional as F

filter_num = [32,64,128, 256,512]
kernel_size = [7,7,7,7]
pool_kernel_size = [8,8,8,8,8]
conv_stride_size = [1,1,1,1, 1]
pool_stride_size = [4,4,4,4,4]
max_distance_size = 16
num_buckets_size = 8

filters_num = [32, 64, 128]
kernels_size = [5, 5, 5]
pool_size = [8, 8, 8]

class CNNfeatureextract(nn.Module):

    def __init__(self, num_input, dropout, *args, **kwargs):

        super(CNNfeatureextract, self).__init__(*args, **kwargs)

        self.seq1 = nn.Sequential(
            nn.Conv1d(1, out_channels=filter_num[0], kernel_size=kernel_size[0],padding="same"),

            nn.BatchNorm1d(filter_num[0]),
            nn.ELU(alpha=1.0),

            nn.Conv1d(in_channels=filter_num[0], out_channels=filter_num[0], kernel_size=kernel_size[0],padding="same"),
            
           

            nn.BatchNorm1d(filter_num[0]),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=pool_kernel_size[0],stride=pool_stride_size[0],padding=4),
            
            nn.Dropout(0.2)
        )
        self.SE1 = SE_Block(filter_num[0], 8)

        self.seq2 = nn.Sequential(
            nn.Conv1d(filter_num[0], out_channels=filter_num[1], kernel_size=kernel_size[1],padding="same"),

            nn.BatchNorm1d(filter_num[1]),
            nn.ReLU(),

            nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1],padding="same"),
             
            nn.BatchNorm1d(filter_num[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size[1],stride=pool_stride_size[1],padding=4),
            
            nn.Dropout(0.2)
        )
        self.SE2 = SE_Block(filter_num[1], 16)

        self.seq3 =  nn.Sequential(
            nn.Conv1d(filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2],padding="same"),

            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(),

            nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2],padding="same"),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size[2],stride=pool_stride_size[2],padding=4),
            
            nn.Dropout(0.2)
        )

        self.SE3 = SE_Block(filter_num[2], 8)

        self.seq4 =  nn.Sequential(
            nn.Conv1d(filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3],padding="same"),

            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(),

            nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3],padding="same"),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size[3],stride=pool_stride_size[3],padding=4),
            
            nn.Dropout(0.2)
        )

        self.SE4 = SE_Block(filter_num[3], 8)
        self.Spa4 = SpatialAttention()

    def forward(self, X):
        X = self.seq1(X)
        X = self.SE1(X)
        X = self.seq2(X)
        X = self.SE2(X)
        X = self.seq3(X)
        X = self.SE3(X)
        X = self.seq4(X)
        X = self.SE4(X)
        
        return X

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
            b, c, h = x.size()
            y = self.gap(x).view(b, c)
            y = self.fc(y).view(b, c, 1)
            return x * y.expand_as(x)
