import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, padding: int, down: bool, use_act: bool = True):
        super().__init__()
        self.conv_trans = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                               padding, bias=True, padding_mode='reflect') 
            if down else 
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 
                               padding, output_padding=1, bias=True, 
                               padding_mode='zeros'),
            #  nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv_trans(x)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, 
                               out_channels=out_channels, kernel_size=3, 
                               stride=stride, padding=1, down=True, use_act=True)
        self.conv2 = ConvBlock(in_channels=in_channels, 
                               out_channels=out_channels, kernel_size=3, 
                               stride=stride, padding=1, down=True, use_act=False)
        self.identity_downsample = None if in_channels == out_channels else ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, down=True)

    def forward(self, x):
        identity = x
        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return out


class Generator(nn.Module):

    def __init__(self, img_channels: int, num_features: int = 64, num_residual_blocks: int = 6) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels=img_channels, 
                               out_channels=num_features, kernel_size=7, stride=1, 
                               padding=3, use_act=True, down=True)
        self.down_layers = nn.ModuleList([
            ConvBlock(in_channels=num_features, out_channels=num_features*2, 
                      kernel_size=3, stride=2, padding=1, down=True, use_act=True),
            ConvBlock(in_channels=num_features*2, out_channels=num_features*4, 
                      kernel_size=3, stride=2, padding=1, down=True, use_act=True),
        ])
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(in_channels=num_features*4, out_channels=num_features*4) for _ in range(num_residual_blocks)]
        )
        self.up_layers = nn.ModuleList([
            ConvBlock(in_channels=num_features*4, out_channels=num_features*2, 
                      kernel_size = 3,stride=2, padding=1, down=False, use_act=True),
            ConvBlock(in_channels=num_features*2, out_channels=num_features, 
                      kernel_size = 3,stride=2, padding=1, down=False, use_act=True),
        ])
        self.last_layer = nn.Conv2d(in_channels=num_features, 
                                    out_channels=img_channels, kernel_size=7, 
                                    stride=1, padding=3)

    def forward(self, x):
        out = self.conv1(x)
        for down_layer in self.down_layers:
            out = down_layer(out)
        out = self.res_blocks(out)
        for up_layer in self.up_layers:
            out = up_layer(out)
        out = self.last_layer(out)
        return out

