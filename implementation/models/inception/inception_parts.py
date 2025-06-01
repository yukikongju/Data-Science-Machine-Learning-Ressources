import torch
import torch.nn as nn
from torch.nn.functional import pad

#  print(f"x1: {x1.shape}")
#  print(f"x2: {x2.shape}")
#  print(f"x3: {x3.shape}")
#  print(f"x4: {x4.shape}")



class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        #  self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class InceptionBlockNaive(nn.Module):

    """
    As described in the paper "Going Deeper with convolutions a)

    Output size at each step:
    - Branch 1: (B, in_channels, H, W) => (B, red_1x1, H, W)
    - Branch 2: (B, in_channels, H, W) => (B, red_3x3, H, W)
    - Branch 3: (B, in_channels, H, W) => (B, red_5x5, H, W)
    - Branch 4: (B, in_channels, H, W) => (B, in_channels, H, W)
    - Output: (B, red_1x1 + red_3x3 + red_5x5 + in_channels, H, W)

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

class InceptionBlockNaivePimped(nn.Module):

    """
    As described in the paper "Going Deeper with convolutions a). Using Convolution Block instead to normalize and apply relu"
    """

    def __init__(self, in_channels, red_1x1, red_3x3, red_5x5):
        super().__init__()
        self.branch1 = ConvolutionBlock(in_channels, red_1x1, kernel_size=1)
        self.branch2 = ConvolutionBlock(in_channels, red_3x3, kernel_size=3, padding=1)
        self.branch3 = ConvolutionBlock(in_channels, red_5x5, kernel_size=5, padding=2)
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

    def __init__(self, in_channels, red_1x1, red_3x3, out3x3, red_5x5, out_5x5, out_pool):
        """
        Arguments
        ---------
        red_<X>
            out channel of the "reduction step" ie 1x1 convolution
        out_<X>
            out channel of the "summary step" ie 3x3 or 5x5 convolution
        """
        super().__init__()
        self.branch1 = ConvolutionBlock(in_channels, red_1x1, 
                                        kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channels, red_3x3, kernel_size=1),
            ConvolutionBlock(red_3x3, out3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvolutionBlock(in_channels, red_5x5, kernel_size=1),
            ConvolutionBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
                ConvolutionBlock(in_channels, out_pool, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class InceptionBlockV3_F5(nn.Module):

    """
    Figure 5 described in paper "Rethinking the Inception Architecture for 
    Computer Vision"
    """

    def __init__(self, in_channels: int, red_double_3x3: int, 
                 mid_double_3x3: int, out_double_3x3: int, red_3x3: int, 
                 out_3x3: int, red_pool: int, red_1x1: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=red_double_3x3, kernel_size=1),
            ConvolutionBlock(in_channels=red_double_3x3, out_channels=mid_double_3x3, kernel_size=3, padding=1),
            ConvolutionBlock(in_channels=mid_double_3x3, out_channels=out_double_3x3, kernel_size=5, padding=2),
        )
        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=red_3x3, kernel_size=1),
            ConvolutionBlock(in_channels=red_3x3, out_channels=out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1), 
            ConvolutionBlock(in_channels=in_channels, out_channels=red_pool, kernel_size=1)
        )
        self.branch4 = ConvolutionBlock(in_channels=in_channels,
                                        out_channels=red_1x1, kernel_size=1)


    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class InceptionBlockV3_F6(nn.Module): # TODO

    """
    Figure 6 described in paper "Rethinking the Inception Architecture for 
    Computer Vision"
    """

    def __init__(self):
        super().__init__()


    def forward(self, x):
        #  x1 = self.branch1(x)
        #  print(f"x1: {x1.shape}")
        #  x2 = self.branch2(x)
        #  print(f"x2: {x2.shape}")
        #  x3 = self.branch3(x)
        #  print(f"x3: {x3.shape}")
        #  x4 = self.branch4(x)
        #  print(f"x4: {x4.shape}")
        #  return torch.cat([x1, x2, x3, x4], dim=1)
        pass


class InceptionBlockV3_F7(nn.Module):

    """
    Figure 7 described in paper "Rethinking the Inception Architecture for 
    Computer Vision"
    """

    def __init__(self, in_channels: int, red_split_3x3: int, conv_split_3x3: int, 
                 out_split_3x3_1x3: int, out_split_3x3_3x1: int, 
                 red_split_1x1: int, out_split_1x1_1x3: int, out_split_1x1_3x1: int, 
                 red_pool: int, red_1x1: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=red_split_3x3, kernel_size=1),
            ConvolutionBlock(in_channels=red_split_3x3, out_channels=conv_split_3x3, kernel_size=3, padding=1),
        )
        self.branch1_1 = ConvolutionBlock(in_channels=conv_split_3x3, 
                                          out_channels=out_split_3x3_1x3, kernel_size=(1, 3), padding=(0, 1))
        self.branch1_2 = ConvolutionBlock(in_channels=conv_split_3x3, 
                                          out_channels=out_split_3x3_3x1, kernel_size=(3, 1), padding=(1, 0))

        self.branch2 = ConvolutionBlock(in_channels=in_channels, out_channels=red_split_1x1, kernel_size=1)
        self.branch2_1 = ConvolutionBlock(in_channels=red_split_1x1, out_channels=out_split_1x1_1x3, 
                                          kernel_size=(1, 3), padding=(0, 1))
        self.branch2_2 = ConvolutionBlock(in_channels=red_split_1x1, out_channels=out_split_1x1_3x1, 
                                          kernel_size=(3, 1), padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1), 
            ConvolutionBlock(in_channels=in_channels, out_channels=red_pool, kernel_size=1)
        )
        self.branch4 = ConvolutionBlock(in_channels=in_channels,
                                        out_channels=red_1x1, kernel_size=1)


    def forward(self, x):
        x1 = self.branch1(x)
        x1_1 = self.branch1_1(x1)
        x1_2 = self.branch1_2(x1)

        x2 = self.branch2(x)
        x2_1 = self.branch2_1(x2)
        x2_2 = self.branch2_2(x2)

        x3 = self.branch3(x)

        x4 = self.branch4(x)
        return torch.cat([x1_1, x1_2, x2_1, x2_2, x3, x4], dim=1)


class InceptionBlockV3_F10(nn.Module):

    """
    Figure 10 described in paper "Rethinking the Inception Architecture for 
    Computer Vision"
    """

    def __init__(self, in_channels: int, red_B1: int, mid_B1: int, 
                 out_B1: int, red_B2: int, out_B2: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=red_B1, kernel_size=1),
            ConvolutionBlock(in_channels=red_B1, out_channels=mid_B1, kernel_size=3, padding=1),
            ConvolutionBlock(in_channels=mid_B1, out_channels=out_B1, kernel_size=3, stride=2, padding=0),
        )
        self.branch2 = nn.Sequential(
                ConvolutionBlock(in_channels=in_channels, out_channels=red_B2, kernel_size=1),
                ConvolutionBlock(in_channels=red_B2, out_channels=out_B2, kernel_size=3, stride=2, padding=0),
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return torch.cat([x1, x2, x3], dim=1)


