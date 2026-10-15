"""Car-GAN: a DCGAN that generates 64x64 images of cars from the Stanford Cars dataset."""

from car_gan.config import Config
from car_gan.models import Discriminator, Generator, init_weights
from car_gan.utils import get_default_device, load_generator

__all__ = [
    "Config",
    "Discriminator",
    "Generator",
    "get_default_device",
    "init_weights",
    "load_generator",
]

__version__ = "1.0.0"
