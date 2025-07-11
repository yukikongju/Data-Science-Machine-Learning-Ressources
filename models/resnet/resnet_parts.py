import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class BuildingBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvolutionBlock(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = None if in_channels == out_channels else ConvolutionBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        out += identity
        return self.relu(out)

class BottleNeckBlock(nn.Module):

    """
    Notes:
    - does applying stride on conv1 vs conv2 change something?
    """

    def __init__(self, in_channels: int, red_1x1: int, mid_3x3: int, out_1x1: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvolutionBlock(in_channels, red_1x1, kernel_size=1, padding=0)
        self.conv2 = ConvolutionBlock(red_1x1, mid_3x3, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(mid_3x3, out_1x1, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = None if in_channels == out_1x1 else ConvolutionBlock(in_channels=in_channels, out_channels=out_1x1, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        out += identity
        return self.relu(out)

