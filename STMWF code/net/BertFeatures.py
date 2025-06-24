import math
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from torch import nn
import numpy as np
from d2l import  torch as d2l
from torch.nn import Linear, Conv1d, MaxPool1d, BatchNorm1d

from Util.FeatureBlock import CNN_Block
from Util.ACResidualBlock import  ACResidualBlock
from Util.ACmix import ACmix
from Util.ResidualBlock import ResidualBlock
from torch.nn import functional as F
from Util.dicL.DLearing import DicLearning
from Util.utils import transpose_output, transpose_qkv
from .regression import cnnreg
from .selectlayer import allget
from .Globalpool import GlobalPool
from .multilayer.topMultiAttetion import TopMultiAttention

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

        # self.seq1 = ResidualBlock(num_input, filter_num[0],num_residuals=1, kernel_size=kernel_size[0], use_ELU=True)
        # self.seq2 = ResidualBlock(filter_num[0], filter_num[1],num_residuals=1, kernel_size=kernel_size[1], use_ELU=False)
        # self.seq3 = ResidualBlock(filter_num[1], filter_num[2],num_residuals=1, kernel_size=kernel_size[1], use_ELU=False)
        # self.seq4 = ResidualBlock(filter_num[2], filter_num[3],num_residuals=1, kernel_size=kernel_size[1], use_ELU=False)
        
        # self.triple1 = TripletAttention()
        # self.triple2 = TripletAttention()
        # self.triple3 = TripletAttention()
        # self.triple4 = TripletAttention()
        # # # self.norm1 = nn.LayerNorm(128)
        # self.ATT = TopMultiAttention(128, 128, 128, 256, 8, 0.1)
        # self.seq4 = ACResidualBlock(filter_num[2], filter_num[3],num_residuals=1, head_num=4, kernel_size=7, dropout=0.2)
        # # self.seq5 = ResidualBlock(filter_num[3], 512, num_residuals=3, kernel_size=kernel_size[1], use_ELU=False)
        # self.seq5 = ResidualBlock(filter_num[3], 512,num_residuals=3, kernel_size=kernel_size[1], use_ELU=False)
        # self.block5 = ACmix(256, 256, kernel_conv=8, head=4, stride=1, max_distance=max_distance_size,
        # self.block5 = ACmix(256, 256, kernel_conv=8, head=4, stride=1, max_distance=max_distance_size,
        #                     num_buckets=num_buckets_size)
        # self.drop = nn.Dropout(dropout)
        # self.bn1 = nn.BatchNorm1d(256)
        # self.block4 = ACResidualBlock(filter_num[2], filter_num[3],kernel_size=kernel_size[3], strides=conv_stride_size[3],
        #                             padding_num=3,dropout=dropout, max_distance=max_distance_size
        #                             , num_buckets=num_buckets_size, num_residuals=1)
        # self.reg = cnnreg.regression()

        # self.dividing = nn.Sequential(
        #     Rearrange('b c (n p) -> (b n) c p', n=4),
        # )
        # self.combination = nn.Sequential(
        #     Rearrange('(b n) c p -> b c (n p)', n=4),
        # )
        # self.norm1 = nn.LayerNorm(128)
        # self.att3 = TopMultiAttention(128, 128, 128, 128, 8, 0.1)
        # self.att2 = TopMultiAttention(64, 64, 64, 32, 8, 0.1)
        # self.att1 = TopMultiAttention(32, 32, 32, 32, 8, 0.1)

        self.seq1 = nn.Sequential(
            nn.Conv1d(1, out_channels=filter_num[0], kernel_size=kernel_size[0],padding="same"),

            nn.BatchNorm1d(filter_num[0]),
            nn.ELU(alpha=1.0),

            nn.Conv1d(in_channels=filter_num[0], out_channels=filter_num[0], kernel_size=kernel_size[0],padding="same"),
            
           
            # SpatialAttention(7),

            nn.BatchNorm1d(filter_num[0]),
            nn.ELU(alpha=1.0),
            # SE_Block(filter_num[0], 8),
            # SpatialAttention(),
            # # nn.ZeroPad1d(7502),
            nn.MaxPool1d(kernel_size=pool_kernel_size[0],stride=pool_stride_size[0],padding=4),
            
            nn.Dropout(0.2)
        )
        self.SE1 = SE_Block(filter_num[0], 8)
        # self.PAtt1 = pointWiseAtt(filter_num[0],ratio=16)
        # self.Spa1 = SpatialAttention()

        self.seq2 = nn.Sequential(
            nn.Conv1d(filter_num[0], out_channels=filter_num[1], kernel_size=kernel_size[1],padding="same"),

            nn.BatchNorm1d(filter_num[1]),
            nn.ReLU(),

            nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1],padding="same"),
            
            # SpatialAttention(7),    
            nn.BatchNorm1d(filter_num[1]),
            nn.ReLU(),
            # SE_Block(filter_num[1], 8),
            # SpatialAttention(),
            #nn.ZeroPad1d(7502),
            nn.MaxPool1d(kernel_size=pool_kernel_size[1],stride=pool_stride_size[1],padding=4),
            
            nn.Dropout(0.2)
        )
        self.SE2 = SE_Block(filter_num[1], 16)
        # self.PAtt2 = pointWiseAtt(filter_num[1],ratio=8)
        # self.Spa2 = SpatialAttention()

        self.seq3 =  nn.Sequential(
            nn.Conv1d(filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2],padding="same"),

            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(),

            nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2],padding="same"),
            
            # SpatialAttention(7),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(),
            # SE_Block(filter_num[2], 8),
            # SpatialAttention(),
            #nn.ZeroPad1d(7502),
            nn.MaxPool1d(kernel_size=pool_kernel_size[2],stride=pool_stride_size[2],padding=4),
            
            nn.Dropout(0.2)
        )

        self.SE3 = SE_Block(filter_num[2], 8)
        # self.Spa3 = SpatialAttention()
        # self.PAtt3 = pointWiseAtt(filter_num[2],ratio=4)

        self.seq4 =  nn.Sequential(
            nn.Conv1d(filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3],padding="same"),

            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(),

            nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3],padding="same"),
            # SE_Block(filter_num[3], 16),
            # SpatialAttention(7),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(),
            # SE_Block(filter_num[3], 8),
            # SpatialAttention(),
            #nn.ZeroPad1d(7502),
            nn.MaxPool1d(kernel_size=pool_kernel_size[3],stride=pool_stride_size[3],padding=4),
            
            nn.Dropout(0.2)
        )

        self.SE4 = SE_Block(filter_num[3], 8)
        self.Spa4 = SpatialAttention()
        # self.PAtt4 = pointWiseAtt(filter_num[3],ratio=2)

        # self.net = nn.Sequential(
        #     ConvBlock1d(in_channels=1, out_channels=32, kernel_size=7),
        #     nn.MaxPool1d(kernel_size=8, stride=4),
        #     nn.Dropout(p=0.1),
        #     ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
        #     nn.MaxPool1d(kernel_size=8, stride=4),
        #     nn.Dropout(p=0.1),
        #     ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
        #     nn.MaxPool1d(kernel_size=8, stride=4),
        #     nn.Dropout(p=0.1),
        #     ConvBlock1d(in_channels=128, out_channels=256, kernel_size=7),
        #     nn.MaxPool1d(kernel_size=8, stride=4),
        #     nn.Dropout(p=0.1),
        # )

        # self.net = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=filters_num[0], kernel_size=kernel_size[0]),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(filters_num[0]),
        #     nn.MaxPool1d(kernel_size=pool_size[0], padding=0),
        #     nn.Dropout(p=0.2),

        #     nn.Conv1d(in_channels=filters_num[0], out_channels=filters_num[1], kernel_size=kernels_size[1]),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(filters_num[1]),
        #     nn.MaxPool1d(kernel_size=pool_size[1], padding=0),
        #     nn.Dropout(p=0.2),

        #     nn.Conv1d(in_channels=filters_num[1], out_channels=filters_num[2], kernel_size=kernels_size[2]),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(filters_num[2]),
        #     nn.MaxPool1d(kernel_size=pool_size[2], padding=0),
        #     # nn.Flatten(start_dim=1),
        #     nn.Dropout(p=0.2),
        # )

        # # self.seq5 = nn.Sequential(
        # #     nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[4], kernel_size=kernel_size[0],padding="same"),

        # #     nn.BatchNorm1d(filter_num[4]),
        # #     nn.ReLU(),

        # #     nn.Conv1d(in_channels=filter_num[4], out_channels=filter_num[4], kernel_size=kernel_size[0],padding="same"),
            
           
        #     # SpatialAttention(7),

        #     nn.BatchNorm1d(filter_num[4]),
        #     nn.ReLU(),

        #     # nn.ZeroPad1d(7502),
        #     nn.MaxPool1d(kernel_size=pool_kernel_size[0],stride=pool_stride_size[0],padding=4),
        #     nn.Dropout(0.2)
        # )
        # self.SE5 = SE_Block(filter_num[4], 8)

        
        # self.dividing = nn.Sequential(
        #     Rearrange('b c (n p) -> (b n) c p', n=4),
        # )
        # self.combination = nn.Sequential(
        #     Rearrange('(b n) c p -> b c (n p)', n=4),
        # )


        # self.attention = TopMultiAttention(20, 20, 20, 100, 10, 0.3)



        # self.globalpool = GlobalPool.GlobalMaxPool1d()


    def forward(self, X):
        # X = self.net(X)
        # X = X.view(X.shape[0], 32, -1)
        # print(X.shape)
        # sliding_size = np.random.randint(0, 1 + 2500)
        # X = torch.roll(X, shifts=sliding_size, dims=-1)
        # # print(x.shape)
        # X = self.dividing(X)
        # sliding_size = np.random.randint(0, 1 + 2500)
        # X = torch.roll(X, shifts=sliding_size, dims=-1)
        # print(X.shape)
        # X= self.dividing(X)
        # print(X.shape)
        X = self.seq1(X)
        X = self.SE1(X)
        # X = self.PAtt1(X)
        # X = self.Spa1(X)
        # X = self.triple1(X)
        # Y = X.transpose(1,2)
        # X = X + self.att1(Y, Y, Y).transpose(1,2)
        #print(X.shape)
        X = self.seq2(X)
        X = self.SE2(X)
        # X = self.PAtt2(X)
        # X = self.Spa2(X)
        # X = self.triple2(X)
        # Y = X.transpose(1,2)
        # X = X + self.att2(Y, Y, Y).transpose(1,2)
        #print(X.shape)
        X = self.seq3(X)
        X = self.SE3(X)
        # X = self.PAtt3(X)
        # X = self.Spa3(X)
        # X = self.triple3(X)
        # Y = self.norm1(X.transpose(1,2))
        # X = X + self.att3(Y, Y, Y).transpose(1,2)
        X = self.seq4(X)
        X = self.SE4(X)
        # X = self.PAtt4(X)
        # X = self.Spa4(X)
        # X = self.triple4(X)
        # X = self.seq5(X)
        # X = self.SE5(X)
        # X = self.triple4()
        # X = self.combination(X)
        # X = X.permute(0, 2, 1)
        # print(X.shape)
        # print(X.shape)
        # print(X.shape)
        # print(X.shape)
        # X = self.seq5(X)
        # print(X.shape)
        # X = self.combination(X)
        # X = X.permute(0, 2, 1)
        # print(X.shape)
        
        return X

# # 定义一个基本的卷积模块，包括卷积、批归一化和ReLU激活
# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         # 定义卷积层
#         self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         # 条件性地添加批归一化层
#         self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         # 条件性地添加ReLU激活函数
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)  # 应用卷积
#         if self.bn is not None:
#             x = self.bn(x)  # 应用批归一化
#         if self.relu is not None:
#             x = self.relu(x)  # 应用ReLU
#         return x

# # 定义ZPool模块，结合最大池化和平均池化结果
# class ZPool(nn.Module):
#     def forward(self, x):
#         # 结合最大值和平均值
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

# # 定义注意力门，用于根据输入特征生成注意力权重
# class AttentionGate(nn.Module):
#     def __init__(self):
#         super(AttentionGate, self).__init__()
#         kernel_size = 7  # 设定卷积核大小
#         self.compress = ZPool()  # 使用ZPool模块
#         self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)  # 通过卷积调整通道数

#     def forward(self, x):
#         x_compress = self.compress(x)  # 应用ZPool
#         x_out = self.conv(x_compress)  # 通过卷积生成注意力权重
#         # print(x_out.shape)
#         scale = torch.sigmoid_(x_out)  # 应用Sigmoid激活
#         return x * scale  # 将注意力权重乘以原始特征


# # 定义TripletAttention模块，结合了三种不同方向的注意力门
# class TripletAttention(nn.Module):
#     def __init__(self, no_spatial=False):
#         super(TripletAttention, self).__init__()
#         self.cw = AttentionGate()  # 定义宽度方向的注意力门
#         # self.hc = AttentionGate()  # 定义高度方向的注意力门
#         self.no_spatial = no_spatial  # 是否忽略空间注意力
#         if not no_spatial:
#             self.hw = AttentionGate()  # 定义空间方向的注意力门

#     def forward(self, x):
#         # 应用注意力门并结合结果
#         x_perm1 = x.permute(0, 2, 1).contiguous()  # 转置以应用宽度方向的注意力
#         # print(x_perm1.shape)
#         x_out1 = self.cw(x_perm1)
#         x_out11 = x_out1.permute(0, 2, 1).contiguous()  # 还原转置
#         # print(x_out1.shape)
#         # x_perm2 = x.permute(0, 3, 2, 1).contiguous()  # 转置以应用高度方向的注意力
#         # x_out2 = self.hc(x_perm2)
#         # x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()  # 还原转置
#         if not self.no_spatial:
#             x_out = self.hw(x)  # 应用空间注意力
#             x_out = 1 / 2 * (x_out + x_out11)  # 结合三个方向的结果
#         else:
#             x_out =  (x_out11 )  # 结合两个方向的结果（如果no_spatial为True）
#         return x_out

class pointWiseAtt(nn.Module):
    def __init__(self, in_dim, ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pointwise_Att = SE_Block(in_dim, ratio=8)
        self.point_avg = nn.AdaptiveAvgPool1d(-1)
        self.sptial = nn.Sequential(
            nn.LazyConv1d(1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c, h = x.shape
        x = self.pointwise_Att(x)
        x = x * self.sptial(x).reshape(b,1,h)
        return x

class SpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1)
            # weight = F.sigmoid(y)
            # y = torch.where(y >= 0.7, y, 0)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)
    

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)
