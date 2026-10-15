"""Device handling, image helpers, checkpoint I/O and video export."""

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from car_gan.config import NORM_MEAN, NORM_STD
from car_gan.models import Discriminator, Generator


def get_default_device() -> torch.device:
    """Pick CUDA, then Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo the dataset normalisation, mapping [-1, 1] back to [0, 1]."""
    mean = torch.tensor(NORM_MEAN, device=images.device).view(3, 1, 1)
    std = torch.tensor(NORM_STD, device=images.device).view(3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def image_grid(images: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Denormalise a batch and tile it into a single ``H x W x 3`` CPU tensor for plotting."""
    return make_grid(denormalize(images.detach()), nrow=nrow).permute(1, 2, 0).cpu()


def show_images(images: torch.Tensor, nmax: int = 64, nrow: int = 8) -> None:
    """Display up to ``nmax`` normalised images in a matplotlib grid."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image_grid(images[:nmax], nrow=nrow).numpy())
    plt.show()


@torch.no_grad()
def generate(generator: Generator, latent: torch.Tensor) -> torch.Tensor:
    """Run the generator in eval mode (BatchNorm uses running stats) and restore its mode."""
    was_training = generator.training
    generator.eval()
    try:
        return generator(latent)
    finally:
        generator.train(was_training)


def save_samples(generator: Generator, latent: torch.Tensor, path: str | Path, nrow: int = 8) -> Path:
    """Generate images from ``latent`` and save them as one PNG grid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(generate(generator, latent)), path, nrow=nrow)
    return path


def sample_latent(n: int, latent_size: int, device: torch.device | str = "cpu",
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw ``n`` latent vectors from a standard normal, shaped for the generator."""
    return torch.randn(n, latent_size, 1, 1, device=device, generator=generator)


def save_checkpoint(generator: Generator, discriminator: Discriminator, checkpoint_dir: str | Path) -> None:
    """Save both networks' state dicts as ``generator.pth`` / ``discriminator.pth``."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), checkpoint_dir / "generator.pth")
    torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator.pth")


def load_generator(path: str | Path, latent_size: int = 128,
                   device: torch.device | str = "cpu") -> Generator:
    """Load a generator checkpoint onto ``device`` (works for GPU-trained weights on CPU)."""
    model = Generator(latent_size)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def load_discriminator(path: str | Path, device: torch.device | str = "cpu") -> Discriminator:
    """Load a discriminator checkpoint onto ``device``."""
    model = Discriminator()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def make_video(image_dir: str | Path, out_path: str | Path, fps: int = 4, pattern: str = "epoch-*.png") -> Path:
    """Stitch the per-epoch sample grids in ``image_dir`` into an MJPG ``.avi`` video."""
    import cv2

    files = sorted(Path(image_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No images matching '{pattern}' in '{image_dir}'")

    first = cv2.imread(str(files[0]))
    height, width = first.shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    try:
        for f in files:
            frame = cv2.imread(str(f))
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    return out_path
