import torch 
from torch import nn
from torch.nn import functional as F
from .BertFeatures import CNNfeatureextract
from d2l import torch as d2l
import math
from Util.utils import transpose_output, transpose_qkv
import numpy as np
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

        self.transfor_layer = nn.Sequential(
            nn.LayerNorm(100),        
            nn.ELU(),
            nn.Dropout(0.1)
        )
        self.posemb = nn.Embedding(40, label_size).weight
        self.femb = nn.Embedding(40,256).weight
        self.project_layer = nn.Sequential(
            nn.Conv1d(256, label_size, kernel_size=1, padding="same"),
        )

        self.decoder = Decoder(256, 256, 256, 512, 8, 4, 0.5)

        self.labelEmbedding = nn.Embedding(100,256).weight
        self.MLP_att = nn.Sequential(
            nn.Linear(256, label_size)
        )
        # self.MLP =  Mlp(in_features=256, hidden_features=1024, out_features=100, act_layer=nn.GELU, drop=0.1)
        self.attention1 = TopMultiAttention(label_size, label_size, label_size, 100, 4, 0.5)
        self.attention3 = TopMultiAttention(256, 256, 256, 256, 8, 0.5)
        
    def forward(self, X):

        X = self.extract(X)
        zelta = self.project_layer(X)
        zelta1 = zelta.transpose(1,2) + self.posemb
        zelta1 = self.attention1(zelta1, zelta1, zelta1 , 0.5)
        X  =  X.transpose(1,2) + self.femb
       
        X = self.decoder(X, zelta1)
        X = X.mean(dim=1)
        zelta = zelta.mean(dim = -1)
        
        return  self.MLP_att(X), zelta
    
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
        out_layer = F.softmax(out, dim=-1)
        out = torch.where(F.sigmoid(out) > rate, out_layer, 0)
        
        out = transpose_output(torch.matmul(out,self.dropout(value)), self.num_heads)
        

        return self.dropout(self.W_o(out))
