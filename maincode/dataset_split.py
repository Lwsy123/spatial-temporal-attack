import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

# parser = argparse.ArgumentParser(description='Split datasets')
# parser.add_argument("-i", '--infile', default="closed_2tab.npz", type=str, help='path of dataset')
# parser.add_argument("-o", '--outpath', default="processed/closed_2tab", type=str, help='path of dataset')
# args = parser.parse_args()
tb = 2
seed = 1018
infile = "/home/siyuwang/pythoncode/WFPatch/dataset_alter/close_dataset/closed_{tabnum}tab.npz".format(**{'tabnum': tb})
outpath = "/home/siyuwang/pythoncode/WFPatch/Dataset/newrepre/"
os.makedirs(outpath, exist_ok=True)

data = np.load(infile)

times = []
for timedata in data['time']:
    time = [0]
    timedata = timedata - 1
    time = timedata[1:] - timedata[:-1]
    time = np.pad(time, (1, 0), mode="constant", constant_values=0)
    # time = (time - time.min())/(time.max() - time.min())
    # print(time)
    # print(time.shape)
    # for i in range(1, 10000):
    #     if timedata[i] != 0:
    #         time.append(timedata[i] - timedata[i - 1])
    #     else : 
    #         np.pad(time, (0, 10000-len(time)), mode="constant", constant_values=0)
    #         break
    time = np.array(time) + 1
    # print(time.shape)
    # print(time)
    times.append(time)

times = np.array(times)
print(times.shape)

# IAT = data["time"][1:] - data["time"][:10000-1]
X = data["direction"] * times
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=seed)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=seed)
print(f"Train: X = {X_train.shape}, y = {y_train.shape}")
# print(f"Valid: X = {X_valid.shape}, y = {y_valid.shape}")
print(f"Test: X = {X_test.shape}, y = {y_test.shape}")

np.savez(os.path.join(outpath, "train{tabnum}tab.npz".format(**{'tabnum': tb})), X = X_train, y = y_train)
# np.savez(os.path.join(outpath, "valid.npz"), X = X_valid, y = y_valid)
np.savez(os.path.join(outpath, "test{tabnum}tab.npz".format(**{'tabnum': tb})), X = X_test, y = y_test)
print(f"split {infile} done")