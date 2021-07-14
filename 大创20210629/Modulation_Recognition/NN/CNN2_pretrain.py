import torch.nn as nn
import torch.nn.functional as F

class Net2_AE(nn.Module):
    def __init__(self):
        super(Net2_AE, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3), stride=(1, 1))
        self.fc1 = nn.Linear(9920, 256)
        self.fc2 = nn.Linear(256, 11)
        self.fc3 = nn.Linear(11, 256)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 80 * 1 * 124)
        x = F.relu(F.dropout(self.fc1(x), p=0.1))
        y = F.normalize(x)  #h1 loss
        #x = F.softmax(self.fc2(x), dim=1)
        h = F.softmax(self.fc2(y), dim=1)
        x = self.fc3(h)
        x = x.view(-1, 2, 128)
        return x, h

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features