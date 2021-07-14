from dataset import *
from NN import *
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

batchsz = 100
bias  = 2

PATH = "saved/CNN1statedict_1.pt"

dataset = RMLtestset()

RMLdataloader = DataLoader(dataset,  batch_size=batchsz, num_workers=0)

list = []
for i, y in enumerate(RMLdataloader):
    list.append(y)

# in list : 0-9 for label 0. 0-109 for one SNR
# [(SNR / 2) + 10] * 11 + label * 10 + 0-9

print("==Dataset ready")

CNN1 = Net()
CNN1 = CNN1.cuda()
CNN1.load_state_dict(torch.load(PATH))
CNN1.eval()

print("==Net ready")

for SNR in range(-20, 20, 2):
    acc = 0
    for labelidx in range(11):
        testidx = int(((SNR / 2) + 10) * 11 + labelidx)
        y = list[testidx]

        input = torch.reshape(y["data"], [batchsz, 1, 2, 128])
        input = input.type(dtype=torch.float32)
        input = input.cuda()
        label = y["label"]
        label = label.type(dtype=torch.long)
        label = label.cuda()

        output, h = CNN1(input)
        res = 0
        for testidy in range(100):
            res += output[testidy].argmax() == label[testidy]
        acc += res
    print(acc)

