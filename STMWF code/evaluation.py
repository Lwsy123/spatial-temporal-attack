import numpy as np

def get_F1_Score(Precision, Recall):

    return 2 * (Precision * Recall) / (Precision + Recall) 

def get_Precision(TPlist, FPlist):
    
    return TPlist /(TPlist + FPlist)
    

def get_Recall(TPlist, FNlist):
    return TPlist / (TPlist + FNlist)
    

def evluation(y_pred,label,num):
    TPlist = []
    FPlist = []
    TNlist = []
    FNlist = []
    print(label[0])
    print(y_pred.shape)
    if len(label.shape) == 2:
        label_alter = np.argwhere(label == 1)[:, 1].reshape(-1)
        print(label_alter.shape)
    else :
        label_alter = label.reshape(-1)
    if len(y_pred.shape) == 2:
        y_alter = np.argsort(y_pred, axis= -1)[:, -num:]
        # print(y_alter.shape, y_alter)
        # print(y_alter)
        y_alter = np.sort(y_alter, axis=-1)
        y_alter = np.hstack(y_alter)
        print(y_alter.shape)
    else :
        y_alter = np.argmax(y_pred, axis= -1)
        # print(y_alter)
        # print(y_alter.shape)
        # print(y_alter)
        # y_alter = np.sort(y_alter)
        # y_alter = np.sort(y_alter, axis= -1)
        # y_alter = np.hstack(y_alter)
        index = np.argwhere(label == 1)[:,0].reshape(-1)
        index_selected = {}
        for i in range(index.shape[0]):
            if index[i] not in index_selected.keys():
                index_selected[index[i]] = 1
            else :
                index_selected[index[i]] += 1
        y_end = []
        for i in range(y_alter.shape[0]):
            t = y_alter[i][:index_selected[i]]
            # print(t)
            t = np.sort(t)
            y_end.append(t)
        y_alter = np.hstack(y_end)
    # print(label_alter)
    print(y_alter.shape, label_alter.shape)
    for i in range(0, np.unique(label_alter).max()):
        labelindex = np.argwhere(label_alter == i)
        yindex = np.argwhere(y_alter == i)

        TP = (y_alter[labelindex] == i).astype(np.float32).sum()
        FN = (y_alter[labelindex] != i).astype(np.float32).sum()
        FP = (label_alter[yindex] != i).astype(np.float32).sum()

        TPlist.append(TP)
        FNlist.append(FN)
        FPlist.append(FP)
    TPlist = np.nan_to_num(np.array(TPlist))
    FPlist = np.nan_to_num(np.array(FPlist))
    FNlist = np.nan_to_num(np.array(FNlist))
    # print(TPlist, FPlist, FNlist)
    Precision, Recall = np.nan_to_num(get_Precision(TPlist, FPlist)), np.nan_to_num( get_Recall(TPlist, FNlist))
    F1 = np.nan_to_num(get_F1_Score(Precision, Recall))
    return Precision.mean(), Recall.mean(), F1.mean()

