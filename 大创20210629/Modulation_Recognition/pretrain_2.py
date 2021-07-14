from dataset import *
from NN import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

batchsz = 100
PATH = "saved/CNN2_pretrain_statedict.pt"
PATH_conv1 = "saved/CNN2_pretrain/CNN2_conv1_statedict.pt"
PATH_conv2 = "saved/CNN2_pretrain/CNN2_conv2_statedict.pt"
PATH_fc1 = "saved/CNN2_pretrain/CNN2_fc1_statedict.pt"
PATH_fc2 = "saved/CNN2_pretrain/CNN2_fc2_statedict.pt"
PATH_fc3 = "saved/CNN2_pretrain/CNN2_fc3_statedict.pt"
load = 1

dataset = RMLdataset()

RMLdataloader = DataLoader(dataset,  batch_size=batchsz, shuffle=True, num_workers=0)

testlist = []
for i, y in enumerate(RMLdataloader):
    testlist.append(y)

print("==Dataset ready")

CNN2_AE = Net2_AE()
CNN2_AE = CNN2_AE.cuda()

if load:
    CNN2_AE.conv1.load_state_dict(torch.load(PATH_conv1))
    CNN2_AE.conv1.eval()
    CNN2_AE.conv2.load_state_dict(torch.load(PATH_conv2))
    CNN2_AE.conv2.eval()
    CNN2_AE.fc1.load_state_dict(torch.load(PATH_fc1))
    CNN2_AE.fc1.eval()
    CNN2_AE.fc2.load_state_dict(torch.load(PATH_fc2))
    CNN2_AE.fc2.eval()
    CNN2_AE.fc3.load_state_dict(torch.load(PATH_fc3))
    CNN2_AE.fc3.eval()


print("==Net ready")

criterion = nn.MSELoss()
optimizer = optim.Adam(CNN2_AE.parameters(), lr=0.001)

print("==Optimizer ready")

for epoch in range(2000):
    running_loss = 0
    for i, y in enumerate(RMLdataloader):
        optimizer.zero_grad()

        # preprocess of input data
        input = torch.reshape(y["data"], [batchsz, 1, 2, 128])
        input = input.type(dtype=torch.float32)
        input = input.cuda()

        output, h = CNN2_AE(input)
        output = torch.reshape(output, [batchsz, 1, 2, 128])

        loss = criterion(output, input) * 1e6
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch % 10 == 0:
        print('[epoch %d] loss: %.3f' %(epoch + 1, loss.item()))
        running_loss = 0.0
        torch.save(CNN2_AE.state_dict(), PATH)
        torch.save(CNN2_AE.conv1.state_dict(), PATH_conv1)
        torch.save(CNN2_AE.conv2.state_dict(), PATH_conv2)
        torch.save(CNN2_AE.fc1.state_dict(), PATH_fc1)
        torch.save(CNN2_AE.fc2.state_dict(), PATH_fc2)
        torch.save(CNN2_AE.fc3.state_dict(), PATH_fc3)

        testinstance = testlist[3]["data"]
        savepic(testinstance[0], 2, 128, "source")
        input = torch.reshape(testinstance, [batchsz, 1, 2, 128])
        input = input.type(dtype=torch.float32)
        input = input.cuda()

        output, h = CNN2_AE(input)
        output = output.cpu().detach().numpy()

        savepic(output[0], 2, 128, str(epoch))

