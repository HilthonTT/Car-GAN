"""Command-line interface: ``car-gan train | generate | video``."""

import argparse
import json
from pathlib import Path

import torch

from car_gan.config import Config
from car_gan.utils import get_default_device


def _train(args: argparse.Namespace) -> None:
    from car_gan.data import build_dataloader, download_dataset
    from car_gan.models import Discriminator, Generator, init_weights
    from car_gan.trainer import GANTrainer

    config = Config(
        data_dir=args.data_dir,
        sample_dir=args.sample_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    if config.seed is not None:
        torch.manual_seed(config.seed)
    if args.download:
        download_dataset(config)

    device = torch.device(args.device) if args.device else get_default_device()
    print(f"Using device: {device}")

    generator = Generator(config.latent_size)
    discriminator = Discriminator()
    if args.resume:
        generator.load_state_dict(torch.load(config.checkpoint_dir / "generator.pth",
                                             map_location="cpu", weights_only=True))
        discriminator.load_state_dict(torch.load(config.checkpoint_dir / "discriminator.pth",
                                                 map_location="cpu", weights_only=True))
    else:
        generator.apply(init_weights)
        discriminator.apply(init_weights)

    trainer = GANTrainer(generator, discriminator, config, device)
    history = trainer.fit(build_dataloader(config), start_epoch=args.start_epoch,
                          checkpoint_every=args.checkpoint_every)

    history_path = config.checkpoint_dir / "history.json"
    history_path.write_text(json.dumps(history.__dict__, indent=2))
    print(f"Checkpoints saved to '{config.checkpoint_dir}', history to '{history_path}'")


def _generate(args: argparse.Namespace) -> None:
    from car_gan.utils import load_generator, sample_latent, save_samples

    device = torch.device(args.device) if args.device else get_default_device()
    generator = load_generator(args.checkpoint, device=device)
    rng = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    latent = sample_latent(args.num, generator.latent_size, device, generator=rng)
    path = save_samples(generator, latent, args.output, nrow=args.nrow)
    print(f"Saved {args.num} images to '{path}'")


def _video(args: argparse.Namespace) -> None:
    from car_gan.utils import make_video

    path = make_video(args.image_dir, args.output, fps=args.fps)
    print(f"Saved video to '{path}'")


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    parser = argparse.ArgumentParser(prog="car-gan", description="Train and sample a DCGAN on Stanford Cars.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the GAN from scratch or resume from a checkpoint.")
    train.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    train.add_argument("--sample-dir", type=Path, default=defaults.sample_dir)
    train.add_argument("--checkpoint-dir", type=Path, default=defaults.checkpoint_dir)
    train.add_argument("--download", action="store_true", help="Download the dataset from Kaggle first.")
    train.add_argument("--epochs", type=int, default=defaults.epochs)
    train.add_argument("--batch-size", type=int, default=defaults.batch_size)
    train.add_argument("--lr", type=float, default=defaults.lr)
    train.add_argument("--num-workers", type=int, default=defaults.num_workers)
    train.add_argument("--checkpoint-every", type=int, default=10)
    train.add_argument("--resume", action="store_true", help="Continue from the checkpoints in --checkpoint-dir.")
    train.add_argument("--start-epoch", type=int, default=1, help="Epoch number to start from (for naming when resuming).")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--device", default=None, help="e.g. cuda, cpu, mps (default: auto).")
    train.set_defaults(func=_train)

    gen = sub.add_parser("generate", help="Generate a grid of car images from a trained generator.")
    gen.add_argument("--checkpoint", type=Path, default=defaults.checkpoint_dir / "generator.pth")
    gen.add_argument("--output", type=Path, default=Path("samples.png"))
    gen.add_argument("--num", type=int, default=64)
    gen.add_argument("--nrow", type=int, default=8)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--device", default=None)
    gen.set_defaults(func=_generate)

    video = sub.add_parser("video", help="Turn the per-epoch sample grids into a training timelapse.")
    video.add_argument("--image-dir", type=Path, default=defaults.sample_dir)
    video.add_argument("--output", type=Path, default=Path("gans_training.avi"))
    video.add_argument("--fps", type=int, default=4)
    video.set_defaults(func=_video)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
