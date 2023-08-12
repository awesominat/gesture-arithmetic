import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class GestureModel(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.linear1 = nn.Linear(42, 128)
        self.norm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(.3)
        torch.nn.init.kaiming_normal_(self.linear1.weight, sqrt(5))

        self.linear2 = nn.Linear(128, 256)
        self.norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(.3)
        torch.nn.init.kaiming_uniform_(self.linear2.weight)

        self.linear3 = nn.Linear(256, 256)
        self.norm3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(.3)
        torch.nn.init.kaiming_uniform_(self.linear3.weight)

        self.linear4 = nn.Linear(256, 512)
        self.norm4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(.3)
        torch.nn.init.kaiming_normal_(self.linear4.weight, sqrt(5))

        self.linear5 = nn.Linear(512, classes)
        torch.nn.init.kaiming_normal_(self.linear5.weight, sqrt(5))

    def forward(self, x):
        x = self.dropout1(F.elu(self.norm1(self.linear1(x))))
        x = self.dropout2(F.elu(self.norm2(self.linear2(x))))
        x = self.dropout3(F.elu(self.norm3(self.linear3(x))))
        x = self.dropout4(F.elu(self.norm4(self.linear4(x))))

        x = self.linear5(x)
        # F.log_softmax(x, dim = 1)
        # no need for softmax as categorical cross entropy loss does it for us
        return x
