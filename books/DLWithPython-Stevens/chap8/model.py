import torch
import torch.nn.functional as F

from collections import OrderedDict
from torch import nn

class DepthNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DepthNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.fct1 = nn.Linear(16*2*2, 64) # 32/(2**4) = 2
        self.fct2 = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.pool(torch.tanh(self.conv3(x)))
        x = self.pool(torch.tanh(self.conv4(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.fct1(x)
        x = self.fct2(x)
        return x
        
class AdaptNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AdaptNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.adapt = nn.AdaptiveMaxPool2d((6,4))
        self.fct1 = nn.Linear(12 * 6 * 4, out_channels)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.adapt(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fct1(x)
        return x


class ConvNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=2, padding=1)
        self.fct1 = nn.Linear(8 * 8 * 8, 16)
        self.fct2 = nn.Linear(16, out_channels)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fct1(x)
        x = self.fct2(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d( in_channels=in_channels, out_channels=16,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d( in_channels=16, out_channels=32,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(32 * 8 * 8 , num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


def test_cnn():
    # test CNN with colored img 32x32
    t = torch.randn(3, 32, 32).unsqueeze(0) 
    model = CNN(in_channels=3, num_classes=2)
    print(t.shape)
    print(model(t))

    # test CNN with grey img 32x32
    t = torch.randn(1, 32, 32).unsqueeze(0) 
    model = CNN(in_channels=1, num_classes=10)
    print(model)
    print(t.shape)
    print(model(t))

    
def test_convnet():
    t = torch.randn(3, 32, 32).unsqueeze(0) 
    model = ConvNet(in_channels=3, out_channels=10)
    print(model)
    print(model(t))

def test_adaptnet():
    t = torch.randn(3, 32, 32).unsqueeze(0) 
    model = AdaptNet(in_channels=3, out_channels=8)
    print(model(t))

def test_depthnet():
    t = torch.randn(3, 32, 32).unsqueeze(0) 
    model = DepthNet(in_channels=3, out_channels=10)
    print(model(t))
    
if __name__ == "__main__":
    test_cnn()
    test_convnet()
    test_adaptnet()
    test_depthnet()


