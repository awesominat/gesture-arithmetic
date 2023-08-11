import torch
import torch.nn as nn
import torch.nn.functional as F

class NumberModel(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(kernel_size=5, in_channels=3, out_channels=32)
        self.pool1 = nn.MaxPool2d(kernel_size=5)
        self.norm1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(.8)

        self.conv2 = nn.Conv2d(kernel_size=3, in_channels=32, out_channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=5)
        self.norm2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(.8)

        self.conv3 = nn.Conv2d(kernel_size=3, in_channels=64, out_channels=128)
        self.pool3 = nn.MaxPool2d(kernel_size=5)
        self.norm3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(.8)

        self.flatten4 = nn.Flatten()
        self.linear5 = nn.Linear(1536, 1000)
        self.linear6 = nn.Linear(1000, classes)

    def forward(self, x):
        x = self.dropout1(self.norm1(F.relu(self.pool1(self.conv1(x)))))
        x = self.dropout2(self.norm2(F.relu(self.pool2(self.conv2(x)))))
        x = self.dropout3(self.norm3(F.relu(self.pool3(self.conv3(x)))))

        x = self.flatten4(x)
        x = self.linear5(x)
        x = self.linear6(x)

        return F.log_softmax(x, dim=1)