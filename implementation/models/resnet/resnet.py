import torch.nn as nn

from models.resnet.resnet_parts import ConvolutionBlock, BuildingBlock, BottleNeckBlock

class ResNet18(nn.Module):

    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.conv1 = ConvolutionBlock(in_channels=in_channels, 
                                      out_channels=64, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = BuildingBlock(in_channels=64, out_channels=64)
        self.conv2_2 = BuildingBlock(in_channels=64, out_channels=64)
        self.conv3_1 = BuildingBlock(in_channels=64, out_channels=128, stride=2)
        self.conv3_2 = BuildingBlock(in_channels=128, out_channels=128)
        self.conv4_1 = BuildingBlock(in_channels=128, out_channels=256, stride=2)
        self.conv4_2 = BuildingBlock(in_channels=256, out_channels=256)
        self.conv5_1 = BuildingBlock(in_channels=256, out_channels=512, stride=2)
        self.conv5_2 = BuildingBlock(in_channels=512, out_channels=512)
        self.avg_pool = nn.AdaptiveAvgPool2d(7) # FIXME ?
        self.fc = nn.Linear(512*7*7, n_classes)

    def forward(self, x):
        out = self.conv1(x) # output: (B, 64, 112, 112)
        out = self.max_pool(out) # output: (B, 64, 56, 56)
        out = self.conv2_1(out) # output: (B, 64, 56, 56)
        out = self.conv2_2(out) # output: (B, 64, 56, 56)
        out = self.conv3_1(out) # output: (B, 128, 28, 28)
        out = self.conv3_2(out) # output: (B, 128, 28, 28)
        out = self.conv4_1(out) # output: (B, 256, 14, 14)
        out = self.conv4_2(out) # output: (B, 256, 14, 14)
        out = self.conv5_1(out) # output: (B, 512, 7, 7)
        out = self.conv5_2(out) # output: (B, 512, 7, 7)
        out = self.avg_pool(out) # output: (B, 512, 7, 7)
        out = out.flatten(start_dim=1) # output: (B, 512*7*7)
        out = self.fc(out)
        return out


