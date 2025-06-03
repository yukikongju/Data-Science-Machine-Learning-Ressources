import torch.nn as nn
from torch.nn.functional import pad

"""
Notes:
    - we can infer discriminator conv shape by looking at generator
"""


class Discriminator(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, expansion: int = 3):
        """
        Arguments
        ---------
        in_channels:
            channels of the image
        out_channels:
            dim_z
        Notes:
        - input shape: 3x64x64
        """
        super().__init__()
        blocks = [self._block(in_channels=out_channels*(2**i), out_channels=out_channels*(2**(i+1)), kernel_size=4, stride=2, padding=1) for i in range(expansion)] # 1 -> 2 -> 4 -> 8 if expansion=3

        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=4, stride=2, padding=1), # 32x32
            nn.LeakyReLU(0.2),
            *blocks,
            nn.Conv2d(in_channels=out_channels * (2**expansion), out_channels=1, 
                    kernel_size=4, stride=2, padding=0), # 1x1
            nn.Sigmoid(),
        )

    def _block(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.discriminator(x)


class Generator(nn.Module):

    def __init__(self, dim_z: int, features_g: int, img_channels: int) -> None:
        """
        Notes:
        - Input: (B, dim_z, 1, 1)
        - expansion has to be 1 more than discriminator
        """
        super().__init__()
        self.generator = nn.Sequential(
            self._block(in_channels=dim_z, out_channels=features_g*16, 
                        kernel_size=4, stride=2, padding=0), # output: (B, f_g * 16, 4, 4)
            self._block(in_channels=features_g*16, out_channels=features_g*8, 
                        kernel_size=4, stride=2, padding=0), # output: (B, f_g * 8, 8, 8)
            self._block(in_channels=features_g*8, out_channels=features_g*4, 
                        kernel_size=4, stride=2, padding=0), # output: (B, f_g * 4, 16, 16)
            self._block(in_channels=features_g*4, out_channels=features_g*2, 
                        kernel_size=4, stride=2, padding=0), # output: (B, f_g * 2, 32, 32)
            #  self._block(in_channels=features_g*2, out_channels=features_g*1, 
            #              kernel_size=4, stride=2, padding=0), # output: (B, f_g * 1, 64, 64)
            #  nn.ConvTranspose2d(in_channels=features_g*2, 
            #                     out_channels=img_channels, kernel_size=4,
            #                     stride=2, padding=1),
            #  nn.Tanh()
        )

    def _block(self, in_channels: int, out_channels: int, kernel_size: int, 
               stride: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride, padding, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


    def forward(self, x):
        return self.generator(x)


