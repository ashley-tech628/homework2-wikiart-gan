"""Efficient style-conditional DCGAN for 64x64 or 128x128 paintings."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class ConditionalBatchNorm(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels, affine=False)
        self.affine = nn.Embedding(classes, channels * 2)
        nn.init.zeros_(self.affine.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(labels).chunk(2, dim=1)
        return self.norm(x) * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class GeneratorStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, classes: int,
                 first: bool = False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4,
                                       stride=1 if first else 2,
                                       padding=0 if first else 1, bias=False)
        self.norm = ConditionalBatchNorm(out_channels, classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x), labels))


class Generator(nn.Module):
    def __init__(self, classes: int, image_size: int = 128,
                 latent_dim: int = 128, base_channels: int = 32):
        super().__init__()
        if image_size not in {64, 128}:
            raise ValueError("image_size must be 64 or 128")
        self.classes = classes
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        channels = ([base_channels * 8, base_channels * 8, base_channels * 4,
                     base_channels * 2, base_channels]
                    if image_size == 128 else
                    [base_channels * 8, base_channels * 4, base_channels * 2,
                     base_channels])
        self.stages = nn.ModuleList()
        previous = latent_dim
        for index, width in enumerate(channels):
            self.stages.append(GeneratorStage(previous, width, classes, first=index == 0))
            previous = width
        self.to_rgb = nn.ConvTranspose2d(previous, 3, 4, stride=2, padding=1)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = noise.reshape(noise.size(0), self.latent_dim, 1, 1)
        for stage in self.stages:
            x = stage(x, labels)
        return torch.tanh(self.to_rgb(x))


class Discriminator(nn.Module):
    def __init__(self, classes: int, image_size: int = 128, base_channels: int = 32):
        super().__init__()
        if image_size not in {64, 128}:
            raise ValueError("image_size must be 64 or 128")
        channels = ([3, base_channels, base_channels * 2, base_channels * 4,
                     base_channels * 8, base_channels * 8]
                    if image_size == 128 else
                    [3, base_channels, base_channels * 2, base_channels * 4,
                     base_channels * 8])
        self.layers = nn.ModuleList(spectral_norm(nn.Conv2d(a, b, 4, stride=2, padding=1))
                                    for a, b in zip(channels[:-1], channels[1:]))
        self.linear = spectral_norm(nn.Linear(channels[-1], 1))
        self.class_embedding = nn.Embedding(classes, channels[-1])

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = images
        for layer in self.layers:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
        x = x.mean(dim=(2, 3))
        return self.linear(x).flatten() + (self.class_embedding(labels) * x).sum(dim=1)


def initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        parameter = getattr(module, "weight_orig", module.weight)
        nn.init.normal_(parameter, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        if bool(module.weight.detach().abs().sum()):
            nn.init.normal_(module.weight, 0.0, 0.02)
