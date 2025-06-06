import torch
import pytest

from models.cyclegan.discriminator import Discriminator
from models.cyclegan.generator import Generator


@pytest.fixture
def image():
    return torch.randint(0, 256, size=(5, 3, 256, 256)).float()

@pytest.fixture
def noise():
    return torch.randint(0, 256, size=(5, 1, 30, 30)).float()


def test_discriminator(image):
    B, C, H, W = image.size()
    discr = Discriminator(img_channels=C, out_features=[64, 128, 256, 512])
    output = discr(image)
    assert output.size() == (B, 1, 30, 30)

def test_generator(image, noise):
    B, C, H, W = image.size()
    gen = Generator(img_channels=C, num_features=64, num_residual_blocks=6)
    output = gen(image)
    assert output.size() == (B, C, H, W)



