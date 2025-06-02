import pytest
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

from models.gan.gan_parts import Discriminator, Generator


@pytest.fixture
def tensor():
    return torch.randn(5, 1, 28, 28).float()

@pytest.fixture
def tensor_fake():
    return torch.randn(5, 64).float()

@pytest.fixture
def image():
    return torch.randn(5, 1, 28, 28).float()


def test_discriminator(tensor):
    B, C, H, W = tensor.size()
    tc = tensor.flatten(start_dim=1)
    discr = Discriminator(tc.shape[1])
    output = discr(tc)
    assert output.size() == (B, 1)

def test_generator(tensor_fake):
    B, DIM_Z = tensor_fake.size()
    OUTPUT_SHAPE = 784
    gen = Generator(z_dim=DIM_Z, output_dim=OUTPUT_SHAPE)
    output = gen(tensor_fake)
    assert output.size() == (B, OUTPUT_SHAPE)

def test_train_step(image):
    B, C, H, W = image.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 64
    image_dim = 1 * 28 * 28
    batch_size = B
    num_epochs = 1

    disc = Discriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    #  transform = transforms.Compose([
    #      transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))

    #  ])

    optim_disc = optim.Adam(disc.parameters(), lr=lr)
    optim_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # -- data transform
    #  tc = transform(image)
    tc = image.flatten(start_dim=1)
    assert tc.size() == (B, 784)

    # -- train discriminator: max log(D(real)) + log(1 - D(G(z)))
    noise = torch.randn((batch_size, z_dim)).to(device)
    fake = gen(noise)
    disc_real = disc(tc).flatten(start_dim=1)
    disc_fake = disc(fake.detach()).flatten(start_dim=1)
    lossD_real = criterion(disc_real, torch.ones_like(disc_real))
    lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    lossD = (lossD_real + lossD_fake) / 2
    disc.zero_grad()
    lossD.backward()
    optim_disc.step()

    # -- train generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
    output = disc(fake).flatten(start_dim=1)
    lossG = criterion(output, torch.ones_like(output))
    gen.zero_grad()
    lossG.backward()
    optim_gen.step()


