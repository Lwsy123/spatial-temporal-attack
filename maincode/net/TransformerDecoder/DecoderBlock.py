import torch
from torch import nn 
from d2l import torch as d2l
from Util.utils import transpose_qkv, transpose_output
import math
from timm.models.layers import DropPath, Mlp
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    def __init__(self,  query_size,key_size, value_size, num_hiddens,
                   num_heads, dropout, bias= False,*args, **kwargs) -> None:
        super(DecoderBlock, self).__init__(*args, **kwargs)
        # self.multiAtt = MultiHeadAttetion(key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.Att = TopMultiAttention(query_size,key_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.multiAtt = TopMulticrossAttention( query_size, 100, 100, num_hiddens, 8, dropout, bias)
        # self.multiAtt = TopExtraAttention(query_size, 100, 100, 256, num_heads)
        # self.multiAtt = TopMAttention(256, 8, 0.1, 20)
        self.norm1 = nn.LayerNorm(query_size)
        self.norm2 = nn.LayerNorm(100)
        self.dropout = nn.Dropout(0.1)
        # self.norm2 = nn.LayerNorm(256)
        # self.addnorm1 = AddNorm(256, 0.1)
        # self.addnorm2 = AddNorm(256, 0.1)
        self.ffn = Mlp(in_features=query_size, hidden_features=num_hiddens, act_layer=nn.GELU, drop=0.1)
        # self.mlp = Mlp(in_features=256, hidden_features=1024, act_layer=nn.GELU, drop=0.1)
    def forward(self, X, Zelta):
        # X = self.ln(X)
        # X = self.addnorm1(X, self.multiAtt(X, X, X))
         

        X = X + self.dropout(self.Att(self.norm1(X), self.norm1(X), self.norm1(X), 0.5)) # best 0.5
        X = X  + self.dropout(self.multiAtt(self.norm1(X),self.norm2(Zelta),self.norm2(Zelta), 0.5)) # best 0.5
        X = X + self.dropout(self.ffn(self.norm1(X)))
        # return X
        return X
        



#@save
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X )              
    

class TopMultiAttention(nn.Module):

    def __init__(self,query_size, key_size, value_size, num_hiddens,  num_heads, dropout = 0.2, bias = False, *args, **kwargs) -> None:
        super(TopMultiAttention, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, query_size, bias=bias)
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
        

class TopMulticrossAttention(nn.Module):

    def __init__(self,query_size, key_size, value_size, num_hiddens,  num_heads, dropout = 0.2, bias = False, *args, **kwargs) -> None:
        super(TopMulticrossAttention, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, 512, bias=bias)
        self.W_v = nn.Linear(value_size, 512, bias=bias)
        self.W_o = nn.Linear(num_hiddens, query_size, bias=bias)
    def forward(self, X, Y, Z, rate) :
        query = transpose_qkv(self.W_q(X), self.num_heads)
        key = transpose_qkv(self.W_k(Y),self.num_heads)
        value = transpose_qkv(self.W_v(Z), self.num_heads)

        out = torch.matmul(query, key.transpose(1,2))/math.sqrt(query.shape[-1])
        # out = nn.functional.sigmoid(out, dim=-1)
        out_layer = F.softmax(out, dim=-1)
        out = torch.where(F.sigmoid(out) > rate, out_layer, 0)
        # print(out.shape
        
        out = transpose_output(torch.matmul(out,self.dropout(value)), self.num_heads)
        

        return self.dropout(self.W_o(out))
        

class TopExtraAttention(nn.Module):

    def __init__(self,query_size, key_size, value_size, num_hiddens,  num_heads, dropout = 0.2, bias = False, *args, **kwargs) -> None:
        super(TopExtraAttention, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, 256, bias=bias)
        self.W_o = nn.Linear(num_hiddens,100, bias=bias)
    def forward(self, X, Y, Z, rate) :
        query = X
        key = self.W_k(Y).unsqueeze(0)
        value = self.W_v(Z).unsqueeze(0)
        out = torch.matmul(query, key.transpose(1,2))/math.sqrt(query.shape[-1])
        # out = nn.functional.sigmoid(out, dim=-1)
        out_layer = F.softmax(out, dim=-1)
        # out = torch.where(F.sigmoid(out) > rate, out_layer, 0)
        # print(out.shape
        
        out = torch.matmul(out_layer, self.dropout(value))
        # print(out.shape)

        return self.dropout(out)
        

        
        