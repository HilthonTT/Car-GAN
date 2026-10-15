"""Adversarial training loop."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from car_gan.config import Config
from car_gan.models import Discriminator, Generator
from car_gan.utils import sample_latent, save_checkpoint, save_samples


@dataclass
class History:
    """Per-epoch averages of the losses and discriminator scores."""

    losses_g: list[float] = field(default_factory=list)
    losses_d: list[float] = field(default_factory=list)
    real_scores: list[float] = field(default_factory=list)
    fake_scores: list[float] = field(default_factory=list)


class GANTrainer:
    """Trains a :class:`Generator` / :class:`Discriminator` pair with the standard BCE GAN loss."""

    def __init__(self, generator: Generator, discriminator: Discriminator,
                 config: Config, device: torch.device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.device = device
        self.opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=config.betas)
        self.opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=config.betas)
        # A fixed latent batch makes the saved samples comparable from epoch to epoch.
        self.fixed_latent = sample_latent(64, config.latent_size, device)

    def train_discriminator(self, real_images: torch.Tensor) -> tuple[float, float, float]:
        """One discriminator step. Returns ``(loss, mean D(real), mean D(fake))``."""
        n = real_images.size(0)
        self.opt_d.zero_grad(set_to_none=True)

        real_preds = self.discriminator(real_images)
        real_loss = F.binary_cross_entropy(real_preds, torch.ones_like(real_preds))

        # detach(): the generator is not updated here, so don't backprop into it.
        fake_images = self.generator(sample_latent(n, self.config.latent_size, self.device)).detach()
        fake_preds = self.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))

        loss = real_loss + fake_loss
        loss.backward()
        self.opt_d.step()
        return loss.item(), real_preds.mean().item(), fake_preds.mean().item()

    def train_generator(self, batch_size: int) -> float:
        """One generator step: push D(G(z)) towards "real". Returns the generator loss."""
        self.opt_g.zero_grad(set_to_none=True)

        fake_images = self.generator(sample_latent(batch_size, self.config.latent_size, self.device))
        preds = self.discriminator(fake_images)
        loss = F.binary_cross_entropy(preds, torch.ones_like(preds))

        loss.backward()
        self.opt_g.step()
        return loss.item()

    def fit(self, dataloader: DataLoader, epochs: int | None = None, start_epoch: int = 1,
            save_every: int = 1, checkpoint_every: int = 10) -> History:
        """Train for ``epochs`` epochs, saving sample grids and periodic checkpoints."""
        epochs = epochs if epochs is not None else self.config.epochs
        history = History()
        self.generator.train()
        self.discriminator.train()

        save_samples(self.generator, self.fixed_latent,
                     self.config.sample_dir / f"epoch-{start_epoch - 1:04d}.png")

        for epoch in range(start_epoch, start_epoch + epochs):
            sums = torch.zeros(4)
            num_batches = 0
            for real_images, _ in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
                real_images = real_images.to(self.device, non_blocking=True)
                loss_d, real_score, fake_score = self.train_discriminator(real_images)
                loss_g = self.train_generator(real_images.size(0))
                sums += torch.tensor([loss_g, loss_d, real_score, fake_score])
                num_batches += 1

            loss_g, loss_d, real_score, fake_score = (sums / max(num_batches, 1)).tolist()
            history.losses_g.append(loss_g)
            history.losses_d.append(loss_d)
            history.real_scores.append(real_score)
            history.fake_scores.append(fake_score)
            print(f"Epoch [{epoch}/{start_epoch + epochs - 1}] "
                  f"loss_g: {loss_g:.4f}  loss_d: {loss_d:.4f}  "
                  f"real_score: {real_score:.4f}  fake_score: {fake_score:.4f}")

            if epoch % save_every == 0:
                save_samples(self.generator, self.fixed_latent,
                             self.config.sample_dir / f"epoch-{epoch:04d}.png")
            if epoch % checkpoint_every == 0:
                save_checkpoint(self.generator, self.discriminator, self.config.checkpoint_dir)

        save_checkpoint(self.generator, self.discriminator, self.config.checkpoint_dir)
        return history


def plot_history(history: History, save_path: str | None = None) -> None:
    """Plot losses and discriminator scores per epoch."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.losses_d, label="Discriminator")
    ax1.plot(history.losses_g, label="Generator")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Losses")
    ax1.legend()
    ax2.plot(history.real_scores, label="Real score  D(x)")
    ax2.plot(history.fake_scores, label="Fake score  D(G(z))")
    ax2.set(xlabel="Epoch", ylabel="Mean discriminator output", title="Scores")
    ax2.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.show()
