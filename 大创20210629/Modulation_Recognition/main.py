from dataset import *
from NN import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batchsz = 200
PATH = "saved/CNN1statedict.pt"

dataset = RMLdataset()

RMLdataloader = DataLoader(dataset,  batch_size=batchsz, shuffle=True, num_workers=0)

print("==Dataset ready")

CNN1 = Net()
CNN1 = CNN1.cuda()
CNN1.load_state_dict(torch.load(PATH))
CNN1.eval()

print("==Net ready")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(CNN1.parameters(), lr=0.001, momentum=0.99)

print("==Optimizer ready")

for epoch in range(2000):
    running_loss = 0
    for i, y in enumerate(RMLdataloader):
        optimizer.zero_grad()

        # preprocess of inputdata
        input = torch.reshape(y["data"], [batchsz, 1, 2, 128])
        input = input.type(dtype = torch.float32)
        input = input.cuda()
        label = y["label"]
        label = label.type(dtype = torch.long)
        label = label.cuda()

        output = CNN1(input)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch % 10 == 0:  # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, loss.item()))
        running_loss = 0.0
        torch.save(CNN1.state_dict(), PATH)
