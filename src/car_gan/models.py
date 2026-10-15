"""DCGAN generator and discriminator networks.

Both networks subclass ``nn.Sequential`` with the same layer order as the original
notebook, so the state-dict keys (``"0.weight"``, ``"1.running_mean"``, ...) are
unchanged and the checkpoints in ``checkpoints/`` load directly.
"""

import torch.nn as nn


class Discriminator(nn.Sequential):
    """Maps a ``3 x 64 x 64`` image to the probability that it is real, shape ``(N, 1)``."""

    def __init__(self, channels: int = 3, features: int = 64):
        f = features
        super().__init__(
            # in: 3 x 64 x 64
            nn.Conv2d(channels, f, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1
            nn.Flatten(),
            nn.Sigmoid(),  # output range: (0, 1)
        )


class Generator(nn.Sequential):
    """Maps a latent vector of shape ``(N, latent_size, 1, 1)`` to a ``3 x 64 x 64`` image in [-1, 1]."""

    def __init__(self, latent_size: int = 128, channels: int = 3, features: int = 64):
        f = features
        super().__init__(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(latent_size, f * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(True),
            # out: 512 x 4 x 4
            nn.ConvTranspose2d(f * 8, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(True),
            # out: 256 x 8 x 8
            nn.ConvTranspose2d(f * 4, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(True),
            # out: 128 x 16 x 16
            nn.ConvTranspose2d(f * 2, f, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(True),
            # out: 64 x 32 x 32
            nn.ConvTranspose2d(f, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # output range: (-1, 1)
            # out: 3 x 64 x 64
        )
        self.latent_size = latent_size


def init_weights(module: nn.Module) -> None:
    """DCGAN weight init (Radford et al., 2015): conv ~ N(0, 0.02), batch-norm ~ N(1, 0.02)."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
