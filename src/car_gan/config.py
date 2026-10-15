"""Hyperparameters and paths shared by training, generation and the notebook."""

from dataclasses import dataclass
from pathlib import Path

# Images are normalised to [-1, 1] so they match the generator's Tanh output range.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


@dataclass
class Config:
    """Training configuration. Defaults reproduce the published checkpoints."""

    data_dir: Path = Path("data")
    dataset_url: str = "https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset"
    sample_dir: Path = Path("generated-images")
    checkpoint_dir: Path = Path("checkpoints")

    image_size: int = 64
    latent_size: int = 128
    batch_size: int = 128
    num_workers: int = 2

    epochs: int = 100
    lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    seed: int | None = None

    def __post_init__(self) -> None:
        # Accept plain strings for convenience.
        self.data_dir = Path(self.data_dir)
        self.sample_dir = Path(self.sample_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)

    @property
    def dataset_dir(self) -> Path:
        """Folder that ``opendatasets`` extracts the Kaggle archive into."""
        return self.data_dir / "stanford-cars-dataset"
