import torch
from torch import nn 
from d2l import torch as d2l
from Util.utils import transpose_qkv, transpose_output
import math
from timm.models.layers import DropPath, Mlp
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                  norm_shape, ffn_num_input, ffn_num_hiddens, ffn_num_output,
                  num_heads, dropout, bias= False, use_ELU=False ,*args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        # self.multiAtt = MultiHeadAttetion(key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.multiAtt = TopMultiAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        # self.multiAtt = TopMAttention(256, 8, 0.1, 20)
        # self.norm1 = nn.LayerNorm(256)
        # self.norm2 = nn.LayerNorm(256)
        self.addnorm1 = AddNorm(norm_shape, 0.1)
        self.addnorm2 = AddNorm(norm_shape, 0.1)
        self.ffn = PositionWiseFFN(ffn_num_input,ffn_num_hiddens, ffn_num_output )
        # self.mlp = Mlp(in_features=256, hidden_features=1024, act_layer=nn.GELU, drop=0.1)
    def forward(self, X, valid_lens=None):
        # X = self.ln(X)
        # X = self.addnorm1(X, self.multiAtt(X, X, X))
        X = self.addnorm1(X ,self.multiAtt(X,X,X))
        # X = self.addnorm1( X, self.multiAtt(X))
        # print(self.mlp(self.norm2(x)).shape)
        # X = self.addnorm2(X, self.ffn(X))
        X = self.addnorm2(X, self.ffn(X))
        # return X
        return X
        


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,use_ELU=False,
    **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        if use_ELU:
            self.activation = nn.ELU()
        else :

            self.activation = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        self.dropout = nn.Dropout(0.1)
    def forward(self, X):
        X = self.dense2(self.dropout(self.activation(self.dense1(X))))
        return X


#@save
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X )              
    

class MultiHeadAttetion(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                  num_heads, dropout, bias= False, **kwargs):
        super(MultiHeadAttetion, self).__init__(**kwargs)
        
        self.num_heads = num_heads
        self.qkv = nn.Linear(256 , 256*3)
        self.attention = d2l.DotProductAttention(dropout)
        # self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        # self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        # self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self, X, valid_lens=None):
        # `queries`, `keys`, or `values` ����״:
        # (`batch_size`, ��ѯ���ߡ�����ֵ���Եĸ���, `num_hiddens`)
        # `valid_lens` ����״:
        # (`batch_size`,) or (`batch_size`, ��ѯ�ĸ���)
        # �����任������� `queries`, `keys`, or `values` ����״:
        # (`batch_size` * `num_heads`, ��ѯ���ߡ�����ֵ���Եĸ���,
        # `num_hiddens` / `num_heads`)
        B, N, C = X.shape
        qkv = self.qkv(X).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        output = self.attention(queries, keys, values, valid_lens)
        # `output_concat` ����״: (`batch_size`, ��ѯ�ĸ���, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

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
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x


class TopMultiAttention(nn.Module):

    def __init__(self,query_size, key_size, value_size, num_hiddens,  num_heads, dropout = 0.2, bias = False, *args, **kwargs) -> None:
        super(TopMultiAttention, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self, X, Y, Z) :
        query = transpose_qkv(self.W_q(X), self.num_heads)
        key = transpose_qkv(self.W_k(Y),self.num_heads)
        value = transpose_qkv(self.W_o(Z), self.num_heads)

        out = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(query.shape[-1])
        # out = nn.functional.sigmoid(out, dim=-1)

        out = torch.where(F.sigmoid(out) > 0.5, out, 0)
        out = F.softmax(out, dim=-1)
        # print(out.shape
        out = transpose_output(torch.matmul(out,self.dropout(value)), self.num_heads)
        

        return self.W_o(out)
        