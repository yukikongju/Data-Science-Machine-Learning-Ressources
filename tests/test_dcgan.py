import pytest
import torch

from models.dcgan.dcgan_parts import Discriminator, Generator

@pytest.fixture
def image() -> torch.Tensor:
    return torch.randint(0, 255, size=(5, 3, 64, 64)).float()

@pytest.fixture
def noise_vector() -> torch.Tensor:
    # note: alternatively called 'latent vector' or 'z'
    DIM_Z = 64
    return torch.randint(0, 255, size=(5, DIM_Z, 1, 1)).float()

def test_discriminator(image: torch.Tensor):
    B, C, H, W = image.size()
    DIM_Z = 64
    disc = Discriminator(in_channels=C, out_channels=DIM_Z, 
                         expansion=3)
    output = disc(image)
    assert output.size() == (B, 1, 1, 1)

def test_generator(noise_vector: torch.Tensor, image: torch.Tensor):
    _, DIM_Z, _, _ = noise_vector.size()
    B, C, H, W = image.size()
    FEATURES_G = 64
    gen = Generator(dim_z=DIM_Z, features_g=FEATURES_G, 
                    img_channels=C)
    output = gen(noise_vector)
    assert output.size() == (B, C, H, W)

