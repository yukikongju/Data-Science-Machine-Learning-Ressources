import pytest
import torch
import torch.optim as optim
import torch.nn as nn

from models.dcgan.dcgan_parts import Discriminator, Generator

@pytest.fixture
def image() -> torch.Tensor:
    return torch.randint(0, 255, size=(5, 3, 64, 64)).float()

@pytest.fixture
def noise_vector() -> torch.Tensor:
    # note: alternatively called 'latent vector' or 'z'
    DIM_Z = 100
    return torch.randint(0, 255, size=(5, DIM_Z, 1, 1)).float()

@pytest.fixture
def batch_images() -> torch.Tensor:
    return torch.randint(0, 255, size=(128, 3, 64, 64)).float()

def test_discriminator(image: torch.Tensor):
    B, C, H, W = image.size()
    #  DIM_Z = 64
    DIM_Z = 100
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


def test_training_step(batch_images):
    B, C, H, W = batch_images.size()
    DIM_Z, FEATURES_G = 100, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 2e-4

    discr = Discriminator(in_channels=C, out_channels=DIM_Z)
    gen = Generator(dim_z=DIM_Z, features_g=FEATURES_G, img_channels=C)

    optim_disc = optim.Adam(discr.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # train discriminator: max log(D(real)) + log (1 - D(G(z)))
    noise = torch.randint(0, 255, size=(B, DIM_Z, 1, 1)).float()
    fake = gen(noise)
    disc_fake = discr(fake.detach()).flatten(start_dim=1)
    disc_real = discr(batch_images).flatten(start_dim=1)
    lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    lossD_real = criterion(disc_real, torch.ones_like(disc_real))
    lossD = (lossD_fake + lossD_real) / 2
    discr.zero_grad()
    lossD.backward()
    optim_disc.step()

    # train generator: min log(1 - D(G(z))) <-> max D(G(z))
    output = discr(fake)
    lossG = criterion(output, torch.ones_like(output))
    gen.zero_grad()
    lossG.backward()
    optim_gen.step()


