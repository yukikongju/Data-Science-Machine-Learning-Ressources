import pytest
import torch

from models.inception.inception_parts import ConvolutionBlock, InceptionBlockNaive

@pytest.fixture
def image() -> torch.Tensor:
    return torch.randint(0, 255, size=(5, 192, 28, 28)).float()


def test_convblock(image):
    B, C, H, W = image.size()
    OUT_CHANNELS = 6
    block = ConvolutionBlock(C, OUT_CHANNELS, kernel_size=1)
    output = block(image)
    assert output.size() == (B, OUT_CHANNELS, H, W)

def test_inception_block_naive(image):
    B, C, H, W = image.size()
    RED_1x1, RED_3x3, RED_5x5 = 64, 128, 32
    block = InceptionBlockNaive(in_channels=C, red_1x1=RED_1x1, red_3x3=RED_3x3, red_5x5=RED_5x5)
    output = block(image)
    assert output.size() == (B, RED_1x1 + RED_3x3 + RED_5x5 + C, H, W)


