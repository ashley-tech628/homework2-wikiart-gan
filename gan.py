"""64 x 64 deep convolutional GAN for RGB paintings."""

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, latent_dim=100, width=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, width * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(width * 8), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(width * 8, width * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width * 4), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(width * 4, width * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(width * 2, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(width, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.net(noise)


class Discriminator(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, width, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(width, width * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(width * 2, width * 4, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(width * 4, width * 8, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(width * 8, 1, 4, 1, 0)),
        )

    def forward(self, images):
        return self.net(images).reshape(-1)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(getattr(module, "weight_orig", module.weight), 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def noise_batch(count, latent_dim, device):
    return torch.randn(count, latent_dim, 1, 1, device=device)
