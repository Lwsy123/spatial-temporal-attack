import  torch 
from torch import nn 

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d,self).__init__()
    def forward(self, x):
        return torch.max_pool1d(x,kernel_size=x.shape[2])