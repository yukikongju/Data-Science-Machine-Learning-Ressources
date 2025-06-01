import pytest
import torch

from models.resnet.resnet_parts import BuildingBlock, BottleNeckBlock

@pytest.fixture
def tensor1():
    return torch.randint(0, 255, size=(5, 64, 224, 224)).float()

@pytest.fixture
def tensor2():
    return torch.randint(0, 255, size=(5, 256, 224, 224)).float()

def test_building_block(tensor1):
    B, C, H, W = tensor1.size()
    OUT_CHANNELS = 64
    block = BuildingBlock(C, OUT_CHANNELS)
    output = block(tensor1)
    assert output.size() == (B, OUT_CHANNELS, H, W)

def test_bottleneck_block(tensor2):
    B, C, H, W = tensor2.size()
    RED, MID, OUT = 64, 64, 256
    block = BottleNeckBlock(in_channels=C, red_1x1=RED, 
                            mid_3x3=MID, out_1x1=OUT)
    output = block(tensor2)
    assert output.size() == (B, OUT, H, W)



