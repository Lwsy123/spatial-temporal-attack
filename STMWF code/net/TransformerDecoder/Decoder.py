import torch 

from torch import nn 
from torch.nn import functional as F
from .DecoderBlock import DecoderBlock

class Decoder(nn.Module):

    def __init__(self, query_size, key_size, value_size, num_hiddens, heads, num_blocks, dropout):

        super(Decoder, self).__init__()

        self.blks = nn.Sequential()


        for i in range(num_blocks):
            self.blks.add_module("blk" + str(i), DecoderBlock(query_size, key_size, value_size, num_hiddens, heads, dropout=dropout, bias=True))
    def forward(self, X, zelta):

        for blk in self.blks:

            X = blk(X, zelta)
        
        return X 