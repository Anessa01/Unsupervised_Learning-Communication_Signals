from dataset import *
from NN import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batchsz = 200
PATH = "saved/CNN1statedict_1.pt"
load = 1

dataset = RMLdataset()

RMLdataloader = DataLoader(dataset,  batch_size=batchsz, shuffle=True, num_workers=0)

testlist = []
for i, y in enumerate(RMLdataloader):
    testlist.append(y)

print("==Dataset ready")

CNN1 = Net()
CNN1 = CNN1.cuda()

if load:
    CNN1.load_state_dict(torch.load(PATH))
    CNN1.eval()

print("==Net ready")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN1.parameters(), lr=0.001)

print("==Optimizer ready")

for epoch in range(2000):
    running_loss = 0
    for i, y in enumerate(RMLdataloader):
        optimizer.zero_grad()

        # preprocess of inputdata
        input = torch.reshape(y["data"], [batchsz, 1, 2, 128])
        input = input.type(dtype=torch.float32)
        input = input.cuda()
        label = y["label"]
        label = label.type(dtype=torch.long)
        label = label.cuda()

        output, h1 = CNN1(input)

        loss = criterion(output, label)
        h1_loss = torch.sum(torch.abs(h1))
        W2_loss = 0
        for param in CNN1.conv1.parameters():
            W2_loss += torch.sum(torch.square(param))
        for param in CNN1.conv2.parameters():
            W2_loss += torch.sum(torch.square(param))


        loss = loss + 0.001 * W2_loss + 0.001 * h1_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch % 10 == 0:
        print('[epoch %d] loss: %.3f h_loss: %.3f W_loss: %.3f' % (epoch + 1, loss.item(), h1_loss, W2_loss))
        running_loss = 0.0
        torch.save(CNN1.state_dict(), PATH)

        testidx = 3
        y = testlist[testidx]

        test_input = torch.reshape(y["data"], [batchsz, 1, 2, 128])
        test_input = test_input.type(dtype = torch.float32)
        test_input = test_input.cuda()
        label = y["label"]
        label = label.type(dtype = torch.long)
        label = label.cuda()

        output, h1 = CNN1(test_input)


        res = 0
        for testidy in range(100):
            if output[testidy].argmax() == label[testidy]:
                res += 1

        print(res)
