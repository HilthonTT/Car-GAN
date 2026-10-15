<img src="images/logo.svg" alt="Car-GAN logo" width="64" height="64" align="right">

# Training guide

## Data pipeline

The [Stanford Cars dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) is
downloaded with [`opendatasets`](https://github.com/JovianHQ/opendatasets) into
`data/stanford-cars-dataset/`. It contains 16,185 photos of 196 car models, split into `cars_train` and
`cars_test`.

The GAN is **unconditional**: it doesn't use the model labels. Both splits are therefore loaded together
with `torchvision.datasets.ImageFolder` as one pool of training images.

Each image goes through these steps ([`src/car_gan/data.py`](../src/car_gan/data.py)):

1. `Resize(64)`: the shorter side is scaled to 64 px.
2. `CenterCrop(64)`: crop to a 64×64 square.
3. `ToTensor()`: convert to [0, 1].
4. `Normalize(0.5, 0.5)`: scale to [-1, 1], the range of the generator's `tanh` output.

The loader shuffles every epoch and uses `drop_last=True`, so every batch has the same size. With batch size
128 that gives 126 batches per epoch.

## Hyperparameters

| Setting | Value | Where |
| --- | --- | --- |
| Image size | 64 × 64 | `Config.image_size` |
| Latent size | 128 | `Config.latent_size` |
| Batch size | 128 | `--batch-size` |
| Optimiser | Adam, lr 2e-4, betas (0.5, 0.999), both networks | `--lr`, `Config.betas` |
| Epochs | 100 | `--epochs` |

These match the published checkpoints. The image size is fixed by the architecture: changing it would need
more or fewer up/down-sampling layers.

## Outputs of a run

| Path | Contents |
| --- | --- |
| `generated-images/epoch-0000.png` ... | 8×8 grid from the same 64 latent vectors, before training and after every epoch |
| `checkpoints/generator.pth`, `discriminator.pth` | Weights, saved every `--checkpoint-every` epochs and at the end |
| `checkpoints/history.json` | Per-epoch averages of `losses_g`, `losses_d`, `real_scores`, `fake_scores` |

Because the latent vectors are fixed, the sample grids show how the *same* cars change over training. Use
`car-gan video` to turn them into a timelapse.

## Reading the metrics

Each epoch prints averages over all its batches:

- **`loss_d`**: discriminator loss. Very close to 0 for many epochs means `D` is winning easily and `G` is
  getting little useful gradient.
- **`loss_g`**: generator loss. GAN losses do **not** steadily decrease. They oscillate as the two
  networks compete, and that is normal.
- **`real_score`**: mean `D(x)` on real images. Ideally high but not pinned at 1.0.
- **`fake_score`**: mean `D(G(z))` on generated images. Ideally low but not pinned at 0.0.

Judge quality mainly by looking at the sample grids. The losses tell you whether training is *stable*, not
whether the images are *good*.

In the original 100-epoch run (see `docs/images/losses.png`), the discriminator's loss collapsed now and then
(for example epochs 27 and 33, where `fake_score` jumped close to 1). Training recovered within an epoch each
time. Those curves were logged from the last batch of each epoch, so they are noisier than the per-epoch
averages the trainer now records.

## Resuming

```bash
car-gan train --resume --start-epoch 101 --epochs 50
```

This loads the weights from `--checkpoint-dir` and continues numbering the sample images from epoch 101.
Optimiser state isn't saved, so Adam's moment estimates restart from zero. Expect a few noisy batches.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `FileNotFoundError: Dataset not found` | Run once with `--download`, or point `--data-dir` at the folder that contains `stanford-cars-dataset/`. |
| Kaggle asks for a username/key | Put `kaggle.json` in the repository root, or set `KAGGLE_USERNAME` / `KAGGLE_KEY`. |
| `CUDA out of memory` | Lower `--batch-size` (e.g. 64). |
| DataLoader hangs or crashes on Windows / in Jupyter | Use `--num-workers 0` (or `Config(num_workers=0)` in the notebook). |
| Every sample looks almost the same (mode collapse) | Restart from a different `--seed`, or lower the learning rate (e.g. `--lr 1e-4`). |
| `fake_score` stays near 0 and `loss_g` keeps rising | The discriminator is overpowering the generator. Lower `--lr` or restart training. |
| Training is very slow | Check that the log prints `Using device: cuda`. If it says `cpu`, install the CUDA build of PyTorch. |

## Hardware

The original run used a single CUDA GPU. Training on a CPU works but is impractically slow for 100 epochs.
It is fine for smoke tests (see `tests/test_car_gan.py`).
