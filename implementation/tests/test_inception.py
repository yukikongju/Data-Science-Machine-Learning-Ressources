import pytest
import torch

from models.inception.inception_parts import ConvolutionBlock

@pytest.fixture
def image() -> torch.Tensor:
    return torch.randint(0, 255, size=(5, 3, 28, 28)).float()


def test_convblock(image):
    B, C, H, W = image.size()
    OUT_CHANNELS = 6
    block = ConvolutionBlock(C, OUT_CHANNELS, kernel_size=1)
    output = block(image)
    assert output.size() == (B, OUT_CHANNELS, H, W)


