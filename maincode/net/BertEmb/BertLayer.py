import torch
from torch import nn 
from d2l import torch as d2l
from Util.utils import transpose_qkv, transpose_output
from .Encoder import Encoder
class BertEncoder(nn.Module):
    def __init__(self, num_layers, num = 256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cls = nn.Parameter(torch.zeros(1, 1, num))
        # # self.token_embedding = nn.Embedding(100, 100)
        # self.segment_embdding = nn.Embedding(256, 100)
        self.postition_embedding = nn.Embedding(20, 256).weight

        self.blks = nn.Sequential()
        # Bert + DF
        self.blks.add_module(f"start", Encoder(num, num, num, num, num, num, 1024, num, 8, 0.2, bias=False))

        for i in range(num_layers - 1):
            self.blks.add_module(f"{i}", Encoder(num, num, num, num, num, num, 1024, num, 8, 0.2, bias=False))
        
        # Bert and DF
        # self.blks.add_module(f"start", Encoder(num, num, num, num, num, num, 256, num, 10, 0.2, bias=True, use_ELU=True))

        # for i in range(num_layers - 1):
        #     self.blks.add_module(f"{i}", Encoder(num, num, num, num, num, num, 256, num, 10, 0.2, bias=True))
        
    def forward(self,X):
        batch_size = X.shape[0]
        # print(X.shape)

        # cls = self.cls.expand(batch_size, -1, -1)

        # X = torch.cat((cls, X), dim=1)
        # print(X.shape)
        X = X +  self.postition_embedding[:, :].unsqueeze(0) 
        # new_element = torch.arange(0,256).reshape(1, 256).to(X.device)
        # dic = self.cls(new_element) + self.segment_embdding(new_element)
        # X += self.postition_embedding
        # print(X.shape)
        # print(dic.shape)
        
       # print(X.shape)
        for blk in self.blks:
            dic = blk(X)
        return dic