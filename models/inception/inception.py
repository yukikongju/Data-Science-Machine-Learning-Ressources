import torch.nn as nn
from torch.nn.modules import padding

from models.inception.inception_parts import InceptionBlockV1, ConvolutionBlock

class InceptionNetV1(nn.Module):

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = ConvolutionBlock(in_channels=in_channels, 
                                      out_channels=64, kernel_size=7, stride=2, padding=3) 
        self.reduction_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvolutionBlock(in_channels=64, out_channels=192, kernel_size=1, padding=0)
        self.block3a = InceptionBlockV1(192, 64, 96, 128, 16, 32, 32)
        self.block3b = InceptionBlockV1(256, 128, 128, 192, 32, 96, 64)
        self.block4a = InceptionBlockV1(480, 192, 96, 208, 16, 48, 64)
        self.block4b = InceptionBlockV1(512, 160, 11, 224, 24, 64, 64)
        self.block4c = InceptionBlockV1(512, 128, 128, 256, 24, 64, 64)
        self.block4d = InceptionBlockV1(512, 112, 144, 288, 32, 64, 64)
        self.block4e = InceptionBlockV1(528, 256, 160, 320, 32, 128, 128)
        self.block5a = InceptionBlockV1(832, 256, 160, 320, 32, 128, 128)
        self.block5b = InceptionBlockV1(832, 384, 192, 384, 48, 128, 128)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout2d(p=0.4)
        self.fc = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax()



    def forward(self, x):
        out = self.conv1(x) # output: (B, 64, 112, 112)
        out = self.reduction_pool(out) # output: (B, 64, 56, 56)
        out = self.conv2(out) # output: (B, 192, 56, 56)
        out = self.reduction_pool(out) # output: (B, 192, 28, 28)
        out = self.block3a(out) # output: (B, 256, 28, 28)
        out = self.block3b(out) # output: (B, 480, 28, 28)
        out = self.reduction_pool(out) # output: (B, 480, 14, 14)
        out = self.block4a(out) # output: (B, 512, 14, 14)
        out = self.block4b(out) # output: (B, 512, 14, 14)
        out = self.block4c(out) # output: (B, 512, 14, 14)
        out = self.block4d(out) # output: (B, 528, 14, 14)
        out = self.block4e(out) # output: (B, 832, 14, 14)
        out = self.reduction_pool(out) # output: (B, 832, 7, 7)
        out = self.block5a(out) # output: (B, 832, 7, 7)
        out = self.block5b(out) # output: (B, 1024, 7, 7)
        out = self.avg_pool(out) # output: (B, 1024, 1, 1)
        out = self.dropout(out) # output: (B, 1024, 1, 1)
        out = out.flatten(start_dim=1) # output: (B, 1024 x 1 x 1)
        out = self.fc(out) # output: (B, 1000)
        out = self.softmax(out)
        return out

