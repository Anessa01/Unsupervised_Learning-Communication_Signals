from dataset import *
from NN import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

batchsz = 100
PATH = "saved/CNN2ss_np.pt"

PATH_conv1 = "saved/CNN2_pretrain/CNN2_conv1_statedict.pt"
PATH_conv2 = "saved/CNN2_pretrain/CNN2_conv2_statedict.pt"
PATH_fc1 = "saved/CNN2_pretrain/CNN2_fc1_statedict.pt"
PATH_fc2 = "saved/CNN2_pretrain/CNN2_fc2_statedict.pt"

dataset = RMLsmallset()

RMLdataloader = DataLoader(dataset,  batch_size=batchsz, shuffle=True, num_workers=0)

testlist = []
for i, y in enumerate(RMLdataloader):
    testlist.append(y)

print("==Dataset ready, datasz:", len(testlist))

CNN2 = Net2()
CNN2 = CNN2.cuda()
load = 1
if load:
    CNN2.conv1.load_state_dict(torch.load(PATH_conv1))
    CNN2.conv1.eval()
    CNN2.conv2.load_state_dict(torch.load(PATH_conv2))
    CNN2.conv2.eval()
    CNN2.fc1.load_state_dict(torch.load(PATH_fc1))
    CNN2.fc1.eval()
    CNN2.fc2.load_state_dict(torch.load(PATH_fc2))
    CNN2.fc2.eval()

print("==Net ready")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN2.parameters(), lr=0.001)

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

        output = CNN2(input)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch % 1 == 0:
        print('[epoch %d] loss: %.3f' %(epoch + 1, loss.item()))
        running_loss = 0.0
        torch.save(CNN2.state_dict(), PATH)

        testidx = 3
        y = testlist[testidx]

        test_input = torch.reshape(y["data"], [batchsz, 1, 2, 128])
        test_input = test_input.type(dtype=torch.float32)
        test_input = test_input.cuda()
        label = y["label"]
        label = label.type(dtype=torch.long)
        label = label.cuda()

        output = CNN2(test_input)

        res = 0
        for testidy in range(100):
            if output[testidy].argmax() == label[testidy]:
                res += 1

        print(res)
