# Hydra + Lightning Training Pipeline

A config-driven deep learning pipeline for **supervised image classification** and **self-supervised learning (SimCLR)**. The project uses **Hydra** for configuration and **PyTorch Lightning** for training. Logger and callbacks are defined in YAML; each run writes to a timestamped experiment directory (logs, checkpoints, config).

## What it does

- **Supervised:** Trains classifiers on **MNIST**, **CIFAR-10**, and **ImageNet** with a simple MLP or **ResNet-18/ResNet-50**. Model, datamodule, and optimizer are chosen from config.
- **SimCLR:** Self-supervised pretraining on **CIFAR-10** or **ImageNet** (two-view augmentation, contrastive loss). Config uses `ssl: true` and overrides `trainer.callbacks` (ModelCheckpoint only; no EarlyStopping).
- **ImageNet** is loaded from WebDataset `.tar` shards (streaming). Use `misc/` scripts to convert raw ImageNet into that format.
- **Training** saves best and last checkpoints under `checkpoints/`; callbacks (ModelCheckpoint, EarlyStopping) are configured in YAML.
- **Testing** uses `experiment_dir` to find checkpoints; the model is instantiated from config and only the checkpoint **state dict** is loaded. Test runs on both **last** and **best** checkpoints and prints metrics (classification report for supervised models).
- **Scripts:** `run_experiment.ps1` runs train then test for one config; `run_infra_test.ps1` runs all non-ImageNet configs (MNIST, ResNet-18/50 CIFAR-10, SimCLR CIFAR-10) in sequence and skips test for SimCLR (no test dataloader).

## Project structure

```text
configs/
  config.yaml              # Default: MNIST + simple classifier, logger, trainer.callbacks
  resnet18_cifar10.yaml    # CIFAR-10 + ResNet-18
  resnet50_cifar10.yaml    # CIFAR-10 + ResNet-50
  resnet50_imagenet.yaml   # ImageNet + ResNet-50 (WebDataset .tar shards)
  simclr_cifar10.yaml      # SimCLR on CIFAR-10 (ssl: true, callbacks override)
  simclr_imagenet.yaml     # SimCLR on ImageNet (ssl: true, callbacks override)

src/
  train.py                 # Hydra entrypoint: instantiate logger, callbacks, datamodule, model; Trainer.fit()
  test.py                  # Load best + last checkpoints from experiment_dir; Trainer.test() for each
  module/                  # LightningModules
    simple_classifier.py   # MLP for MNIST
    resnet18.py            # ResNet-18 classifier
    resnet50.py            # ResNet-50 classifier
    simclr.py              # SimCLR (ResNet-50 backbone + projection head)
  datamodule/
    default/               # Supervised datamodules
      mnist.py
      cifar10.py
      imagenet.py          # ImageNet via WebDataset
    simclr_cifar10.py      # Two-view CIFAR-10 (TwoViewDataset)
    simclr_imagenet.py     # Two-view ImageNet (decode in pipeline)
  utils/
    data_utils.py          # TransformWrapper, TwoViewDataset

scripts/
  run_experiment.ps1       # Single run: train then test; overrides + experiment_dir; output to experiments/<name>_<timestamp>/run.log
  run_infra_test.ps1       # Batch run: train (and test for supervised) for config, resnet18_cifar10, resnet50_cifar10, simclr_cifar10; test skipped for SimCLR

experiments/               # One folder per run (when using script or experiment_dir)
  <name>_<timestamp>/
    run.log                # train + test stdout/stderr
    lightning_logs/        # TensorBoard
    checkpoints/           # last.ckpt, best-*.ckpt
    .hydra/                # Hydra config snapshot

misc/                      # ImageNet → WebDataset
  prepare_imagenet.ps1
  imagenet_to_webdataset.py
  imagenet_to_webdataset.ps1
```

## Setup

```bash
conda create -n tp python=3.11 -y
conda activate tp
pip install -r requirements.txt
```

## Running

**Train only (default MNIST config):**

```bash
python src/train.py
```

Override from the CLI:

```bash
python src/train.py trainer.max_epochs=5 datamodule.batch_size=256
```

**Use a specific config:**

```bash
python src/train.py --config-name=resnet18_cifar10
python src/train.py --config-name=simclr_cifar10
```

**Test after training** (requires `experiment_dir` so checkpoints can be found):

```bash
python src/test.py experiment_dir=./experiments/mnist_20260314_123456
```

Test runs on both `last.ckpt` and any `best-*.ckpt` in `experiment_dir/checkpoints/`.

**PowerShell script** (train + test, one log file):

```powershell
.\scripts\run_experiment.ps1
```

Edit `$experimentName`, `$configName`, `$dataDir`, and the `$overrides` array. The script sets `experiment_dir` and `hydra.run.dir` so artifacts go under `experiments/<name>_<timestamp>/`, runs `train.py` then `test.py` with the same overrides, and appends test output to `run.log`.

**Infra test script** (all non-ImageNet configs in one go):

```powershell
.\scripts\run_infra_test.ps1
```

Runs config (MNIST), resnet18_cifar10, resnet50_cifar10, and simclr_cifar10 in sequence. Test is skipped for SimCLR. Edit `$dataRoot`, `$max_epochs`, etc. at the top of the script.

**ImageNet:** Data must be WebDataset `.tar` shards (e.g. `imagenet-train-000000.tar`, `imagenet-val-000000.tar`). Use `misc/prepare_imagenet.ps1` and `misc/imagenet_to_webdataset.py` to build them. Set `datamodule.data_dir` to the directory containing the tars, then run with `--config-name=resnet50_imagenet` or `--config-name=simclr_imagenet`.

## Config

- **Logger** and **trainer.callbacks** are in config; Hydra instantiates them. Base config uses `TensorBoardLogger` and callbacks: `ModelCheckpoint` (best + last) and `EarlyStopping` (val_loss). SimCLR configs override `trainer.callbacks` with only `ModelCheckpoint` (monitor `train_loss`).
- **Model** and **datamodule** use `_target_` (full class path). Optimizer lives under `model.optimizer` with `_partial_: true`.
- **ssl:** Set to `true` in SimCLR configs; used to select callback behavior (e.g. no EarlyStopping).
- Extra configs inherit from `config.yaml` via `defaults: [config, _self_]` and override `model`, `datamodule`, and optionally `trainer.callbacks`.

## Reproducibility

- Full config is saved under the experiment directory (e.g. `.hydra/` when using the script).
- `pl.seed_everything(cfg.seed)` is called at startup.
