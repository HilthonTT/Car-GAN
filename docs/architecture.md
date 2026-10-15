<img src="images/logo.svg" alt="Car-GAN logo" width="64" height="64" align="right">

# Architecture

Car-GAN is a standard DCGAN ([Radford et al., 2015](https://arxiv.org/abs/1511.06434)). Two convolutional
networks are trained against each other:

- the **generator** `G` turns a random noise vector `z` into a 64×64 RGB image;
- the **discriminator** `D` outputs the probability that an image is a real photo rather than one made by `G`.

```
z ~ N(0, I)          ┌───────────┐   fake image    ┌───────────────┐
(128 × 1 × 1) ─────▶ │ Generator │ ─────────────▶  │               │
                     └───────────┘   3 × 64 × 64   │ Discriminator │ ──▶ P(real) ∈ (0, 1)
real image ──────────────────────────────────────▶ │               │
(3 × 64 × 64, scaled to [-1, 1])                   └───────────────┘
```

Both networks are defined in [`src/car_gan/models.py`](../src/car_gan/models.py).

## Generator

Upsamples a 128-dimensional latent vector with transposed convolutions. Every layer except the last is
followed by batch normalisation and ReLU; the output uses `tanh` so pixel values fall in [-1, 1], the same
range as the normalised training images.

| # | Layer | Output shape |
| --- | --- | --- |
| in | latent vector `z` | 128 × 1 × 1 |
| 1 | ConvTranspose 4×4, stride 1, pad 0 → BatchNorm → ReLU | 512 × 4 × 4 |
| 2 | ConvTranspose 4×4, stride 2, pad 1 → BatchNorm → ReLU | 256 × 8 × 8 |
| 3 | ConvTranspose 4×4, stride 2, pad 1 → BatchNorm → ReLU | 128 × 16 × 16 |
| 4 | ConvTranspose 4×4, stride 2, pad 1 → BatchNorm → ReLU | 64 × 32 × 32 |
| 5 | ConvTranspose 4×4, stride 2, pad 1 → Tanh | 3 × 64 × 64 |

About 3.8 M parameters.

## Discriminator

The mirror image of the generator: strided convolutions halve the spatial size at each step. It uses
LeakyReLU (slope 0.2) instead of ReLU and ends in a sigmoid.

| # | Layer | Output shape |
| --- | --- | --- |
| in | image | 3 × 64 × 64 |
| 1 | Conv 4×4, stride 2, pad 1 → BatchNorm → LeakyReLU(0.2) | 64 × 32 × 32 |
| 2 | Conv 4×4, stride 2, pad 1 → BatchNorm → LeakyReLU(0.2) | 128 × 16 × 16 |
| 3 | Conv 4×4, stride 2, pad 1 → BatchNorm → LeakyReLU(0.2) | 256 × 8 × 8 |
| 4 | Conv 4×4, stride 2, pad 1 → BatchNorm → LeakyReLU(0.2) | 512 × 4 × 4 |
| 5 | Conv 4×4, stride 1, pad 0 → Flatten → Sigmoid | 1 |

About 2.8 M parameters.

> The original DCGAN paper leaves batch norm out of the discriminator's first layer. This project keeps it,
> because the published checkpoints were trained that way and their state-dict layout depends on it.

## Weight initialisation

`init_weights` draws convolution weights from N(0, 0.02) and batch-norm scales from N(1, 0.02), with
batch-norm biases at 0, as in the DCGAN paper. Apply it with `model.apply(init_weights)` before training
from scratch.

## Loss

Both networks use binary cross-entropy (BCE).

**Discriminator step** — push `D(x)` towards 1 for real images and `D(G(z))` towards 0 for fakes:

```
L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
```

The fake images are detached here so no gradients flow into the generator.

**Generator step** — the "non-saturating" loss: push `D(G(z))` towards 1, i.e. fool the discriminator:

```
L_G = BCE(D(G(z)), 1)
```

One discriminator step and one generator step are taken per batch, each with a fresh noise sample.
Both use Adam with learning rate 2e-4 and betas (0.5, 0.999).

## Checkpoint format

`checkpoints/generator.pth` and `checkpoints/discriminator.pth` are plain PyTorch `state_dict`s. Because
`Generator` and `Discriminator` subclass `nn.Sequential`, the keys are layer indices (`0.weight`,
`1.running_mean`, ...), the same as the original notebook, so older checkpoints stay compatible.

Load them with `car_gan.load_generator(path, device=...)`. This passes `map_location`, so weights trained on a
GPU load on a CPU-only machine.

## Sampling

`car_gan.utils.generate` and `save_samples` switch the generator to **eval mode** while sampling. Batch norm
then uses the running statistics collected during training, so an image depends only on its own latent vector
and not on the rest of the batch. The previous training mode is restored afterwards.
