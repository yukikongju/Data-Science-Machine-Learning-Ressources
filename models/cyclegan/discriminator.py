import torch.nn as nn
from typing import List

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, padding: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                      bias=False, padding_mode='reflect'), 
            nn.InstanceNorm2d(out_channels, eps=1e-5, momentum=1e-1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):

    def __init__(self, img_channels: int, out_features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=out_features[0], 
                      kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )

        in_channel = out_features[0]
        layers = []
        for out_channel in out_features[1:]:
            stride = 1 if out_channel == out_features[-1] else 2
            layers.append(ConvBlock(in_channels=in_channel, 
                                         out_channels=out_channel, 
                                         kernel_size=4, stride=stride, padding=1))
            in_channel = out_channel
        layers.append(ConvBlock(in_channels=out_features[-1], out_channels=1, 
                                kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.initial(x)
        out = self.sigmoid(self.model(out)) # output: (B, 1, 30, 30)
        return out


