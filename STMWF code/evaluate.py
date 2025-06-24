import torch 
from torch.utils import data
import numpy as np
import os 
from d2l import torch as d2l
from Util.utils import abs_accurarcy
from  tqdm import tqdm
from torch import nn 
from metric.mAP import AveragePrecisionMeter
from Util.Dataloader import WFDataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score 
from evaluation import evluation

prefix = "./dataset_alter/valid_dataset/"
dirs = os.listdir(prefix)

dirs.sort()
def load_data(batch_size):
    dataset_X = []
    dataset_y = []

    for dir in dirs:
        dataset = np.load(prefix + dir)
        X, y = dataset["X"], dataset["y"]
        
        dataset_X.append(X)
        dataset_y.append(y)

    dataset_X = np.vstack(dataset_X).reshape(-1, 5000)
    dataset_y = np.vstack(dataset_y).reshape(-1, 100)

    dataset = np.concatenate((dataset_X, dataset_y), axis= -1)

    np.random.shuffle(dataset)

    X, y = torch.tensor(dataset[:, : 5000], dtype= torch.float32).reshape(-1, 1, 5000), torch.tensor(dataset[:, 5000:], dtype= torch.float32).reshape(-1, 100)

    eval_data = data.TensorDataset(X, y)

    return data.DataLoader(eval_data, batch_size= batch_size, drop_last=True, shuffle=True)

def evaluate(eval_iter, net, device, batch_size, num):

    # eval_iter = WFDataLoader("/home/siyuwang/pythoncode/WFPatch/Dataset/newClose/testDataset/test{}tab.npz".format(str(num)), 128)
    accmulator = d2l.Accumulator(5)
    loop = tqdm(eval_iter,desc='evaluate')
    loss_ce = nn.BCEWithLogitsLoss()
    loss_dic = nn.BCEWithLogitsLoss()
    # loss_ce = TMWFLoss(num)
    net.eval()
    net.to(device)
    # metr = AveragePrecisionMeter(False)
    y_true = []
    y_pred_score = []
    for X,Y in loop:
        #print(torch.unique(Y,return_counts=True))
        #print(X.shape)
        with torch.no_grad():
            X ,y = X.to(device), Y.to(device)
            y_hats = net(X)

            y_true.append(y.cpu().numpy())
            y_pred_score.append(y_hats.cpu().numpy())
            # y_hats = net(X)
            #print(y_hats.shape)
            l = loss_ce(y_hats, y)
        res, t = abs_accurarcy(y_hats,y) 
        accmulator.add(res, t, l * y.numel(), y.numel())
        loop.set_postfix(mAP= accmulator[0]/accmulator[1], loss = accmulator[2]/accmulator[3])

    y_true = np.vstack(y_true)
    y_pred_score = np.vstack(y_pred_score)
    print(y_pred_score[0])
    print(evluation(y_pred_score, y_true, num))
    # print(precision_score(y_true, (y_pred_score>0.5), average='samples'))
    try:
        print(roc_auc_score(y_true, y_pred_score, average='macro'))
    except:
        print('it has no auc')
    # print(y_pred_score.shape)
    max_tab = 5
    tp = {}
    for tab in range(1, max_tab+1):
        tp[tab] = 0

    for idx in range(y_pred_score.shape[0]):
        cur_pred = y_pred_score[idx]
        # print(cur_pred.shape)
        
        for tab in range(1, max_tab+1):
            if len(cur_pred.shape) !=2:
                target_webs = cur_pred.argsort()[-tab:]
                # print(target_webs)
            else:
                target_webs = np.argmax(cur_pred, axis = -1)[:tab]
                target_webs = np.unique(target_webs)
            for target_web in target_webs:
                if y_true[idx,target_web] > 0:
                    tp[tab] += 1
        # else :
        #     # print(1)
           
        #     for tab, target_web in enumerate(target_webs):
        #             if y_true[idx,target_web[:tab+1]] > 0:
        #                 tp[tab + 1] += 1

    mapk=.0
    for tab in range(1, max_tab+1):
        p_tab = tp[tab] / (y_true.shape[0] * tab)
        mapk += p_tab
        if tab == num:
            print(f"p@{tab}: {round(p_tab,3)}, map@{tab}: {round(mapk/tab,3)}")
    return p_tab




