"""Dataset download and loading."""

from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from car_gan.config import NORM_MEAN, NORM_STD, Config


def download_dataset(config: Config) -> Path:
    """Download the Stanford Cars dataset from Kaggle (skipped if already present).

    Requires Kaggle API credentials: either ``kaggle.json`` in the working directory
    or the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.
    """
    import opendatasets as od  # optional dependency, only needed for downloading

    config.data_dir.mkdir(parents=True, exist_ok=True)
    od.download(config.dataset_url, data_dir=str(config.data_dir))
    return config.dataset_dir


def build_transforms(image_size: int) -> T.Compose:
    """Resize, centre-crop to a square and normalise to [-1, 1]."""
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(NORM_MEAN, NORM_STD),
    ])


def build_dataloader(config: Config) -> DataLoader:
    """Return a shuffled loader over every image in the dataset.

    The GAN is unconditional, so the ``cars_train`` and ``cars_test`` splits are both
    used as training data (~16k images) and the ImageFolder labels are ignored.
    """
    if not config.dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset not found at '{config.dataset_dir}'. "
            "Run with --download or call car_gan.data.download_dataset() first."
        )
    dataset = ImageFolder(str(config.dataset_dir), transform=build_transforms(config.image_size))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # keeps every batch the same size, which stabilises BatchNorm
        persistent_workers=config.num_workers > 0,
    )
