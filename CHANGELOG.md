# Changelog

## 1.0.0 — Package refactor

### Structure
- Moved the training code out of the notebook into an installable package, `src/car_gan/`
  (`config`, `data`, `models`, `trainer`, `utils`, `cli`).
- Added a `car-gan` command-line tool with `train`, `generate` and `video` subcommands.
- New layout: `notebooks/` for the notebook, `checkpoints/` for the pretrained weights, `assets/` for the
  training video, `docs/` for guides and images, and `tests/` for the test suite.
- Added `pyproject.toml` / `requirements.txt` with optional extras. Replaced the Visual Studio `.gitignore`
  with a Python one that also ignores the dataset and `kaggle.json`.
- Rewrote the README and added `docs/architecture.md` and `docs/training.md`.
- The pretrained checkpoints are unchanged and still load: the state-dict keys are the same.

### Bug fixes
- **`save_samples(show=True)` raised `NameError`**: it referenced a `fake_images` variable that was never
  defined in that function. It only ran in the original session because of leftover kernel state.
- **Preview images were displayed without denormalisation**, so they looked dark and clipped.
- **Checkpoints failed to load on CPU-only machines**: `torch.load` had no `map_location`. It now uses
  `map_location` and `weights_only=True`.
- **Sampling built autograd graphs**: generating sample grids every epoch ran without `torch.no_grad()`,
  which wasted memory. Sampling also ran in training mode, so batch-norm statistics depended on the batch.
- **The discriminator step backpropagated into the generator**: the fake images are now detached.
- **Fake batches didn't match the real batch size**: the size was hard-coded to `batch_size`, while the last
  real batch was smaller. Batch sizes now match and `drop_last=True` is used.
- **Metrics came from the last batch only**: each epoch now records the average over all batches.
- **The video writer** paired an MP4V codec with an `.avi` container (OpenCV warned and fell back) and used a
  hard-coded 530×530 frame size. It now uses MJPG and reads the frame size from the images.
- The score plot's y-axis was labelled "Loss". It now reads "Mean discriminator output".
- `num_workers=4` with functions defined in the notebook could hang on Windows. Workers now default to 2,
  and the dataset code lives in an importable module.

### Improvements
- DCGAN weight initialisation (`init_weights`).
- Automatic device selection now includes Apple Silicon (MPS).
- Periodic checkpointing, `--resume`, seeding and `history.json` metrics export.
- A pytest suite covering shapes, checkpoint loading, a training epoch and the CLI.
