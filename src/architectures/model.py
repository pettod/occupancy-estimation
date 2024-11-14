import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 32 * 240, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Conv + ReLu + MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
