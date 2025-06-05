import torch
import pytest

from models.cyclegan.discriminator import Discriminator


@pytest.fixture
def image():
    return torch.randint(0, 256, size=(5, 3, 256, 256)).float()

@pytest.fixture
def noise():
    return torch.randint(0, 256, size=(5, 1, 1, 1)).float()


def test_discriminator(image):
    B, C, H, W = image.size()
    discr = Discriminator(img_channels=C, out_features=[64, 128, 256, 512])
    output = discr(image)
    assert output.size() == (B, 1, 30, 30)


