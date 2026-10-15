"""Fast CPU tests: model shapes, checkpoint compatibility, a training step and the CLI."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from car_gan import Config, Discriminator, Generator, init_weights, load_generator
from car_gan.cli import main
from car_gan.data import build_dataloader
from car_gan.trainer import GANTrainer
from car_gan.utils import denormalize, sample_latent

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"


def test_generator_output_shape_and_range():
    g = Generator(latent_size=128)
    out = g(sample_latent(4, 128))
    assert out.shape == (4, 3, 64, 64)
    assert out.min() >= -1 and out.max() <= 1


def test_discriminator_output_shape_and_range():
    d = Discriminator()
    out = d(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 1)
    assert ((out > 0) & (out < 1)).all()


def test_init_weights_statistics():
    g = Generator()
    g.apply(init_weights)
    std = g[0].weight.std().item()
    assert 0.015 < std < 0.025


def test_denormalize_maps_to_unit_range():
    x = torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1).expand(1, 3, 2, 2)
    y = denormalize(x)
    assert torch.allclose(y[0, :, 0, 0], torch.tensor([0.0, 0.5, 1.0]))


@pytest.mark.skipif(not (CHECKPOINTS / "generator.pth").exists(), reason="no published checkpoint")
def test_published_checkpoints_load():
    g = load_generator(CHECKPOINTS / "generator.pth")
    d = Discriminator()
    d.load_state_dict(torch.load(CHECKPOINTS / "discriminator.pth", map_location="cpu", weights_only=True))
    assert g(sample_latent(2, 128)).shape == (2, 3, 64, 64)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """A fake ImageFolder with the same layout as the Kaggle download."""
    root = tmp_path / "data" / "stanford-cars-dataset" / "cars_train"
    root.mkdir(parents=True)
    for i in range(10):
        Image.new("RGB", (80, 60), color=(i * 20, 100, 200)).save(root / f"{i:05d}.jpg")
    return tmp_path


def test_training_epoch_writes_samples_and_checkpoints(tiny_dataset: Path):
    config = Config(data_dir=tiny_dataset / "data", sample_dir=tiny_dataset / "samples",
                    checkpoint_dir=tiny_dataset / "ckpt", batch_size=4, num_workers=0)
    trainer = GANTrainer(Generator(), Discriminator(), config, torch.device("cpu"))
    history = trainer.fit(build_dataloader(config), epochs=1)

    assert len(history.losses_g) == 1
    assert (config.sample_dir / "epoch-0000.png").exists()
    assert (config.sample_dir / "epoch-0001.png").exists()
    assert (config.checkpoint_dir / "generator.pth").exists()
    assert trainer.generator.training, "sampling must not leave the generator in eval mode"


def test_missing_dataset_has_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="--download"):
        build_dataloader(Config(data_dir=tmp_path))


@pytest.mark.skipif(not (CHECKPOINTS / "generator.pth").exists(), reason="no published checkpoint")
def test_cli_generate(tmp_path: Path):
    out = tmp_path / "grid.png"
    main(["generate", "--checkpoint", str(CHECKPOINTS / "generator.pth"),
          "--output", str(out), "--num", "16", "--seed", "0", "--device", "cpu"])
    assert Image.open(out).size == (8 * 66 + 2, 2 * 66 + 2)  # 8 per row, 2px padding
