import torch
from torch import  nn 
import numpy as np 
from d2l import torch as d2l
from net import  BertExtract
from Util import Dataloader
from STMWFtrain import train
from loss.focalloss import focal_loss
from evaluate import evaluate
import time

for num in range(2,6):
    print(num)
    # prefix_train = "/home/siyuwang/pythoncode/WFPatch/Dataset/RatioDataset/newtrain_merged.npz"
    # prefix_eval = "/home/siyuwang/pythoncode/WFPatch/Dataset/RatioDataset/newtest_merged0.5.npz"
    prefix_train = "/home/siyuwang/pythoncode/WFPatch/Dataset/newClose/trainDataset/train{}tab.npz".format(str(num))
    prefix_eval = "/home/siyuwang/pythoncode/WFPatch/Dataset/newClose/testDataset/test{}tab.npz".format(str(num))
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    device = d2l.try_gpu()
    #for i in range(6, 100):
    train_iter = Dataloader.WFDataLoader(prefix_train,batch_size=128)
    eval_iter = Dataloader.WFDataLoader(prefix_eval, batch_size=128)
    '''
    STMWF
    '''
    net = BertExtract.BertExtract(250, 100)



    def weights_init(m):
        if type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
        elif type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    net = net.to(device)
    net.apply(weights_init)
    train(train_iter,eval_iter, net,device,num_epoches=200, lr=0.001, num=num)
    net.load_state_dict(torch.load("./best_model/STMWF/best_modelSTMWF{}tab.pth".format(str(num))))
    train(train_iter,eval_iter, net,device,num_epoches=20, lr=0.0001, num=num)
    start = time.time()
    evaluate(eval_iter, net, device, 128, num)
    end = time.time()
    print((end-start))

