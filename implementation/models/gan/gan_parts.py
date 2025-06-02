import torch
import torch.nn as nn

class Discriminator(nn.Module):

    """
    Notes:
    - in_features: 1x28x28 = 784
    - output: binary classifier (fake vs real)
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128), 
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

class Generator(nn.Module):

    """
    Given representation of size z_dim, generate output of shape output_dim
    """

    def __init__(self, z_dim: int, output_dim: int):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, output_dim),
            nn.Tanh(), # data between (-1, 1) because we normalize dataset

        )

    def forward(self, x):
        return self.generator(x)


if __name__ == "__main__":
    # note: GAN are very sensitive to hyperparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 3e-4
    z_dim = 64
    image_dim = 28 * 28 * 1
    num_epochs = 50
    batch_size = 32


