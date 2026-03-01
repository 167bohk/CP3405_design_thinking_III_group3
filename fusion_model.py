import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32,1)

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1,64,batch_first=True)
        self.fc = nn.Linear(64,1)

    def forward(self,x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1])


class FusionModel(nn.Module):
    def __init__(self,cnn,lstm):
        super().__init__()
        self.cnn = cnn
        self.lstm = lstm
        self.fc = nn.Linear(2,1)

    def forward(self,img,seq):
        c = self.cnn(img)
        l = self.lstm(seq)
        x = torch.cat([c,l],dim=1)
        return torch.sigmoid(self.fc(x))