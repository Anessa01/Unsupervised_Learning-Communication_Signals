import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 16, kernel_size=(2, 3), stride=(1, 1))
        self.fc1 = nn.Linear(1984, 128)
        self.fc2 = nn.Linear(128, 11)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(F.dropout(self.conv1(x), p=0.5))
        x = F.relu(F.dropout(self.conv2(x), p=0.5))
        x = x.view(-1, 16 * 1 * 124)
        x = F.relu(F.dropout(self.fc1(x), p=0.5))
        x = F.softmax(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
for i in range(len(params)):
    print(params[i].size())     # conv1's .weight

input = torch.randn(1, 1, 2, 128)

output = net(input)

print(output.size())
print(output)

# to make a one-hot encoding
ones = torch.sparse.torch.eye(11)
classi = torch.tensor([3])
class3 = ones.index_select(0, classi)
