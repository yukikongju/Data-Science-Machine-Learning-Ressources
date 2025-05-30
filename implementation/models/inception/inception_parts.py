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


    def __init__(self):
        super().__init__()


    def forward(self, x):
        pass


class InceptionBlockV1(nn.Module):

    """
    As described in the paper "Going Deeper with convolutions b)"
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

