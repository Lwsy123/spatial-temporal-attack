import torch
from torch import nn
import numpy as np
from d2l import torch as d2l
from  tqdm import tqdm
from Util.utils import accurarcy,calculate_acuracy_mode_one,abs_accurarcy
import os
from torch.nn import functional as F
from metric.mAP import AveragePrecisionMeter
from evaluate import evaluate

gpu = d2l.try_gpu()


def train_epoch(train_iter, net, loss_ce, loss_dic, updater, device, epochname, netname):

    accmulator = d2l.Accumulator(4)
    AverPre = AveragePrecisionMeter(False)
    loop = tqdm(train_iter,desc='Train' + epochname + netname)

    for X,Y in loop:
        X ,y = X.to(device),Y.to(device)
        y_hats, lbs = net(X)

        l = loss_dic(y_hats, y)
        l = l +  loss_ce(lbs, y)*0.5
        updater.zero_grad()
        l.backward()
        updater.step()
        res, t = abs_accurarcy(y_hats,y) 
        accmulator.add(res, t, l * y.numel(), y.numel())
        loop.set_postfix(mAP= accmulator[0]/accmulator[1], loss = accmulator[2]/accmulator[3])
    #print(AverPre.value().mean())
    return accmulator[0]/accmulator[1], accmulator[2]/accmulator[3]

def train(train_iter,eval_iter, net, device, num_epoches=20, lr=0.1, num =3 ):

    updater = torch.optim.Adam(net.parameters(), lr=lr)
    loss_ce = nn.BCEWithLogitsLoss()
    loss_dic = nn.BCEWithLogitsLoss()
    dirpath = './best_model' 
    netname = "nf"
    max_acc = 0
    trainloss = 1e6
    test_min_loss = 1e6
    best_p_tab = 0
    for epoch in range(num_epoches):
        net.train()

        epochname = "epoch" + str(epoch)
        

        mAP , l = train_epoch(train_iter, net, loss_ce, loss_dic , updater, device, epochname, netname)
        p_tab = evaluate(eval_iter, net, device, 128,num)
        if best_p_tab < p_tab:
            path = dirpath  + "/MW3F/best_modelMW3F{}tab.pth".format(str(num)) #'/TWF/best_modelAMnewnewrepre4encoderonlyextractor{}tab'.format(str(num)) + '.pth'
            torch.save(net.state_dict(), path)
            best_p_tab = p_tab




    







