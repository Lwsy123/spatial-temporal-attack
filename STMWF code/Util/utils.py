import  torch
from torch import nn 
from d2l import torch as d2l
import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量， valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    

def transpose_qkv(X, num_heads):
    # X的输入形状为：（batchsize, 查询 或者 "键值对"的个数， num_hiddens）
    # 输出X 的形状为：（batchsize，查询或"键值对"的个数， 
    # num_heads, num_hiddens/num_heads）
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0,2,1,3)
    # 输出X 的形状为（batch_size, num_heads, 查询或者"键值对"的个数，
    #  num_hiddens/num_heads）
    X = X.reshape(X.shape[0] * num_heads, X.shape[2], -1)
    # 最终输出的形状:(batchsize*num_heads, 查询或"键值对"的个数，
    #  num_hiddens/num_heads)
    return X

def transpose_output(X, num_heads):
    # X的输入形状为：（batch_size*num_heads, 查询或者"键值对"的个数，num_hiddens/num_heads）

    X= X.reshape(-1,  num_heads, X.shape[1], X.shape[2])
    X = X.permute(0,2,1,3)

    return X.reshape(X.shape[0], X.shape[1], -1)

def accurarcy(y_hat, y):
    y_hat = torch.argmax(y_hat,dim=-1)

    res = y_hat.type(y.dtype) == y

    return float(res.type(y.dtype).sum())

def abs_accurarcy(y_hat, y):
    nums = y.sum(dim=-1).int().reshape(-1)
    #print(nums)
    #y_hat = torch.where(y_hat > 0.5, y_hat, 0)
    if len(y_hat.shape) == 2:
        sorted, indices= torch.sort(y_hat,dim=-1,descending=True)
        #print(sorted[:,:5])
    else :
        indices = torch.argmax(y_hat, dim = -1)
        # print(indices.shape)
    res = 0
    for i,num in enumerate(nums):
        #print( y[i , indices[i, :num]])
        tol = y[i , indices[i, :num]].sum()
        res += tol

    return float(res),y.sum()

def calculate_acuracy_mode_one(model_pred, labels, top = 5):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.type(labels.dtype) == labels
    pred_one_num = torch.sum(pred_result)
    
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    # if pred_one_num == 0:
    #     return 0, 0, 
    # 模型预测的结果中有多少个是正确的
    # precision = true_predict_num / pred_one_num
    # # 模型预测正确的结果中，占所有真实标签的数量
    # recall = true_predict_num / target_one_num

    return true_predict_num, pred_one_num, target_one_num


def data_clipping(X, patch_size = 20):
    rearrange('b c (h s1) -> b (h) (s1 c)', s1 = patch_size)
    
    return X


