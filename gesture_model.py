import torch.nn as nn
import torch.nn.functional as F

class GestureModel(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.linear1 = nn.Linear(126, 3000)
        self.norm1 = nn.BatchNorm1d(3000)
        self.dropout1 = nn.Dropout(.8)

        self.linear2 = nn.Linear(3000, 2500)
        self.norm2 = nn.BatchNorm1d(2500)
        self.dropout2 = nn.Dropout(.8)

        self.linear3 = nn.Linear(2500, 2000)
        self.norm3 = nn.BatchNorm1d(2000)
        self.dropout3 = nn.Dropout(.8)

        self.linear4 = nn.Linear(2000, 1000)
        self.norm4 = nn.BatchNorm1d(1000)
        self.dropout4 = nn.Dropout(.8)

        self.linear5 = nn.Linear(1000, classes)

    def forward(self, x):
        x = self.dropout1(self.norm1(F.relu(self.linear1(x))))
        x = self.dropout2(self.norm2(F.relu(self.linear2(x))))
        x = self.dropout3(self.norm3(F.relu(self.linear3(x))))
        x = self.dropout4(self.norm4(F.relu(self.linear4(x))))

        x = self.linear5(x)
        # F.log_softmax(x, dim = 1)
        # no need for softmax as categorical cross entropy loss does it for us 
        return x 
        