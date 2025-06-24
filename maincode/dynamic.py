import numpy as np
import os 
prefix = "/home/siyuwang/pythoncode/WFPatch/Dataset/newrepre/trainDataset/"
files = os.listdir("/home/siyuwang/pythoncode/WFPatch/Dataset/newrepre/trainDataset/")

datasetMix = []
labelMix = []
for file in files:
    dataset = np.load(prefix + file)
    data = dataset['X']
    label = dataset['y']
    print(data.shape)
    dataset = np.hstack((data, label))
    np.random.shuffle(dataset)
    index = int(len(dataset) * 0.5)

    dataset = dataset[:index]
    data, label = dataset[:, :10000], dataset[:, 10000:]

    datasetMix.append(data)
    labelMix.append(label)

datasetMix = np.vstack(datasetMix)
labelMix = np.vstack(labelMix)
print(datasetMix.shape, labelMix.shape)

np.savez(prefix + "trainmixed.npz", X= datasetMix, y = labelMix)
    
 