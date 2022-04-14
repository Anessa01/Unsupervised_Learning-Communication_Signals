import torch.nn as nn
import torch.nn.functional as F

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 11)

    def forward(self, x):
        x = x.view(-1, 2 * 1 * 128)
        x = F.relu(F.dropout(self.fc1(x), p=0.2))
        x = F.relu(F.dropout(self.fc2(x), p=0.2))
        x = F.relu(F.dropout(self.fc3(x), p=0.2))
        y = F.normalize(x)  #h1 loss
        #x = F.softmax(self.fc2(x), dim=1)
        x = self.fc4(y)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
