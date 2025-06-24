import torch 
from torch import nn

from Util.ResidualBlock import ResidualBlock 
from .BertEmb import BertLayer
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torch.nn import functional as F

from .BertFeatures import CNNfeatureextract
# from .PisCNN import LocalProfiling
# from .extract import CNNfeatureextract
from .Globalpool.GlobalPool import GlobalMaxPool1d
from d2l import torch as d2l
from Util.ACmix import ACmix
import math
from Util.utils import transpose_output, transpose_qkv
import numpy as np
from timm.models.layers import DropPath, Mlp
from timm.models.layers import trunc_normal_
from .TransformerDecoder.Decoder import Decoder
filter_num = [32,64,128, 256, 512]
kernel_size = [8,8,7,7]
pool_kernel_size = [8,8,8,8, 8]
conv_stride_size = [1,1,1,1, 1]
pool_stride_size = [4,4,4,4, 4]
max_distance_size = 16
num_buckets_size = 8

class BertExtract(nn.Module) :
    def __init__(self, num_hidden, label_size, *args, **kwargs) -> None:
        super(BertExtract, self).__init__(*args, **kwargs)

      
        self.extract = CNNfeatureextract(1, 0.5)
       
        self.droppath = DropPath(0.1)
        # self.conv = nn.Conv1d(256, 100, kernel_size=1)
        # self.dividing = nn.Sequential(
        #     Rearrange('b c (n p) -> (b n) c p', n=4),
        # )
        # self.combination = nn.Sequential(
        #     Rearrange('(b n) c p -> b c (n p)', n=4),
        # )
        # self.extractor = LocalProfiling()
    
        self.globalpool = GlobalMaxPool1d()

        self.transfor_layer = nn.Sequential(
            nn.LayerNorm(100),        
            nn.ELU(),
            nn.Dropout(0.1)
            # ACmix(20, 20, head=4, kernel_conv=3, stride=1, padding=1),
        )
        # self.att = TopM_MHSA(256, 8, 2, 1024,0.1, 20)
        self.posemb = nn.Embedding(40, label_size).weight
        self.femb = nn.Embedding(40,256).weight
        # self.norm2 = nn.LayerNorm(256)
        # self.norm1 = nn.LayerNorm(256)
        self.project_layer = nn.Sequential(
            nn.Conv1d(256, label_size, kernel_size=1, padding="same"),
            # nn.Linear(256, 100)
            # nn.Norm1d(100),
            # nn.LayerNorm(20),        
            # nn.ELU(),
            # nn.Dropout(0.2)
        )

        self.decoder = Decoder(256, 256, 256, 512, 8, 4, 0.7)


        # self.att = d2l.MultiHeadAttention(256, 16, 0.1)
        # self.norm1 = nn.BatchNorm1d(256)
        # self.norm2 = nn.LayerNorm(256)
        # self.dropout = nn.Dropout1d(0.2)

        # self.transfor_layer = ACmix(in_planes=20, out_planes=100, head=4, kernel_conv=3, padding=1)
    
       
        # self.conv_reduction = nn.Conv1d(256, 100, kernel_size=1)
        # self.ResLayer = BertLayer.BertEncoder(2, 256)

        self.labelEmbedding = nn.Embedding(100,256).weight
        self.MLP_att = nn.Sequential(
            nn.Linear(256, label_size)
        )
        # self.MLP =  Mlp(in_features=256, hidden_features=1024, out_features=100, act_layer=nn.GELU, drop=0.1)
        self.attention1 = TopMultiAttention(label_size, label_size, label_size, 100, 4, 0.4)
        # self.attention2 = TopMultiAttention(100, 100, 100, 100, 4, 0.2)
        self.attention3 = TopMultiAttention(256, 256, 256, 256, 8, 0.4)
        # self.attention4 = TopMultiAttention(256, 256, 256, 256, 8, 0.2)

        # self.bilstm = nn.LSTM(256, 256, num_layers=2, bidirectional=False)
        # self.attention1 = TopMAttention(256, 8, 0.1, 20)
        # self.attention2 = TopMAttention(256, 8, 0.1, 20)
        # self.dic = nn.Embedding(20, 256).weight

        # self.MLP = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256 * 20, 1024),

        #     nn.GELU(),

        #     nn.Dropout(0.2),

        #     nn.Linear(1024, 256),
        #     nn.GELU(),

        #     nn.Dropout(0.1),

        #     nn.Linear(256, 100)
        # )
        # self.mlp1 = Mlp(in_features=256, hidden_features=1024, act_layer=nn.GELU, drop=0.1)
        # self.mlp2 = Mlp(in_features=256, hidden_features=1024, act_layer=nn.GELU, drop=0.1)
        # # self.mlp3 = Mlp(in_features=256, hidden_features=1024, act_layer=nn.GELU, drop=0.1)
        # self.dropout = nn.Dropout(0.1)
        # self.MLP_att = nn.Sequential(
        #     # nn.LayerNorm(emb_size),
        #     nn.Flatten(),
        #     nn.LazyLinear(512, bias= True),
        #     nn.Dropout(0.3),
        #     nn.ReLU(),

        #     nn.LazyLinear(256, bias= True),
        #     nn.Dropout(0.2),
        #       nn.ReLU(),

        #     nn.Linear(256, 100, bias= False)
        # )
        
    def forward(self, X):

        X = self.extract(X)
        # print(X.shape)
        zelta = self.project_layer(X)
        zelta1 = zelta.transpose(1,2) + self.posemb
        zelta1 = self.attention1(zelta1, zelta1, zelta1 , 0.7) # best 0.7
        X  =  X.transpose(1,2) + self.femb
       
        X = self.decoder(X, zelta1)
        X = X.mean(dim=1)
        zelta = zelta.mean(dim = -1)
        
        return  self.MLP_att(X), zelta

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
            weight = F.sigmoid(y)
            y = torch.where(weight >= 0.7, y, 0)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)
    
class TopMultiAttention(nn.Module):

    def __init__(self,query_size, key_size, value_size, num_hiddens,  num_heads, dropout = 0.2, bias = False, *args, **kwargs) -> None:
        super(TopMultiAttention, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self, X, Y, Z, rate) :
        query = transpose_qkv(self.W_q(X), self.num_heads)
        key = transpose_qkv(self.W_k(Y),self.num_heads)
        value = transpose_qkv(self.W_v(Z), self.num_heads)

        out = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(query.shape[-1])
        # out = nn.functional.sigmoid(out, dim=-1)
        out_layer = F.softmax(out, dim=-1)
        out = torch.where(F.sigmoid(out) > rate, out_layer, 0)
        # print(out.shape
        
        out = transpose_output(torch.matmul(out,self.dropout(value)), self.num_heads)
        

        return self.dropout(self.W_o(out))
        
class pointWiseAtt(nn.Module):
    def __init__(self, in_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(in_dim, in_dim, kernel_size=7, padding=3)
        self.pointwise_Att = nn.Sequential(
            nn.Linear(in_dim, in_dim*2),
            nn.GELU(),
            nn.Linear(in_dim*2, in_dim)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0,2,1)
        x = self.ln(x)
        x = self.pointwise_Att(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0,2,1)
        return x

class TopM_MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m):
        super().__init__()

        self.nets = nn.ModuleList([MHSA_Block(embed_dim, num_heads, dim_feedforward, dropout, top_m) for _ in range(num_mhsa_layers)])

    def forward(self, x, pos_embed):
        # print(pos_embed.shape)
        output = x + pos_embed
        
        for layer in self.nets:
            output = layer(output)
        return output

class TopMAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, top_m):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.top_m = top_m

        self.qkv = nn.Linear(dim , dim*3)
        self.attn_drop = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout),
        )
        self.proj_drop = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        # self.apply(self._init_weights)


    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # mask = torch.zeros(B, self.num_heads, N, N, device=q.device, requires_grad=False)
        # index = torch.topk(attn, k=self.top_m, dim=-1, largest=True)[1]
        # mask.scatter_(-1, index, 1.)
        # attn = torch.where(mask>0, attn, torch.full_like(attn, float('-inf')))

        attn = self.attn_drop(attn)
        attn = torch.where(attn > 0.5, attn, 0)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x

class MHSA_Block(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout, top_m):
        super().__init__()
        drop_path_rate = 0.1
        self.attn = TopMAttention(embed_dim, nhead, dropout, top_m)
        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # y = x * torch.where(F.sigmoid(y) > 0.5, y, 0)
        # print(self.mlp(self.norm2(x)).shape)
        x = x + self.mlp(self.norm2(x))
        return x
    

class AddAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):

        self.num_heads = num_heads

        self.W_q = nn.Linear(emb_size, emb_size, bias= False)
        self.W_k = nn.Linear(emb_size, emb_size, bias=False)
        self.W_v = nn.Linear(emb_size, emb_size, bias= False)

    def forward(self, query, key, value):
        B,N, C = query.shape
        query = transpose_qkv(self.W_q(query), self.num_heads)
        key = transpose_qkv(self.W_k(key), self.num_heads)
        value = transpose_qkv(self.W_v(value), self.num_heads)

        features = F.tanh(query + key)

        features = self.W_v(features)

        out = F.softmax(features, dim= -1)
        out = torch.matmul(self.dropout(out),value).reshape(B, N, C)
