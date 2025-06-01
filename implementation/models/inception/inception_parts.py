import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class InceptionBlockNaive(nn.Module):

    """
    As described in the paper "Going Deeper with convolutions a)"
    """

    def __init__(self, in_channels, red_1x1, red_3x3, red_5x5):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, red_1x1, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Conv2d(in_channels, red_3x3, kernel_size=3, stride=1, padding=1)
        self.branch3 = nn.Conv2d(in_channels, red_5x5, kernel_size=5, stride=1, padding=2)
        self.branch4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x) -> torch.Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class InceptionBlockV1(nn.Module):

    """
    As described in the paper "Going Deeper with convolutions b)"
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

