import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST

from models.gan.gan_parts import Discriminator, Generator



device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 1 * 28 * 28
batch_size = 16
num_epochs = 5

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))

])

dataset = MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optim_disc = optim.Adam(disc.parameters(), lr=lr)
optim_gen = optim.Adam(gen.parameters(), lr=lr)

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0


