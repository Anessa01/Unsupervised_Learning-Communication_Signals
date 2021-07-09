from dataset import *
from NN import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

testidx = 0
testidy = 1

batchsz = 200
PATH = "saved/CNN1statedict_1.pt"

dataset = RMLdataset()

RMLdataloader = DataLoader(dataset,  batch_size=batchsz, shuffle=True, num_workers=0)

list = []
for i, y in enumerate(RMLdataloader):
    list.append(y)

print("==Dataset ready")

CNN1 = Net()
CNN1 = CNN1.cuda()
CNN1.load_state_dict(torch.load(PATH))
CNN1.eval()

print("==Net ready")



y = list[testidx]

input = torch.reshape(y["data"], [batchsz, 1, 2, 128])
input = input.type(dtype = torch.float32)
input = input.cuda()
label = y["label"]
label = label.type(dtype = torch.long)
label = label.cuda()

output, h = CNN1(input)


res = 0
print(output[testidy].argmax(), label[testidy])
