# Hydra + Lightning Training Pipeline

A config-driven deep learning pipeline for **supervised image classification**, **self-supervised learning (SimCLR)**, and **linear probing** of SSL backbones. The project uses **Hydra** for configuration and **PyTorch Lightning** for training. Logger and callbacks are defined in YAML; each run can write to a timestamped experiment directory (logs, checkpoints, config snapshot).

## What it does

- **Supervised:** Trains classifiers on **MNIST**, **CIFAR-10**, and **ImageNet** with a simple MLP or **ResNet-18/ResNet-50**. Model, datamodule, and optimizer are chosen from config.
- **SimCLR:** Self-supervised pretraining on **CIFAR-10** or **ImageNet** (two-view augmentation, contrastive loss). Config uses `ssl: true` and overrides `trainer.callbacks` (ModelCheckpoint only; no EarlyStopping).
- **Linear probe (CIFAR-10):** After SimCLR pretraining, train a **frozen** ResNet-50 backbone plus a **linear head** on labeled CIFAR-10 (`LinearProbeClassifier`). Set `model.pretrained_ckpt` to a SimCLR Lightning checkpoint. Test metrics are written to `linear_probe_metrics.json` in the Hydra run directory.
- **ImageNet** is loaded from WebDataset `.tar` shards (streaming). Use `misc/` scripts to convert raw ImageNet into that format.
- **Training** saves best and last checkpoints under `checkpoints/`; callbacks (ModelCheckpoint, EarlyStopping) are configured in YAML.
- **Testing** uses `experiment_dir` to find checkpoints; the model is instantiated from config and only the checkpoint **state dict** is loaded. Test runs on both **last** and **best** checkpoints and prints metrics (classification report for supervised and linear-probe models).
- **Scripts:** Per-config PowerShell helpers under `scripts/` (e.g. `simclr_cifar10.ps1`, `linear_probe_cifar10.ps1`) set `experiment_dir` and run `train.py` / `test.py`. `run_infra_test.ps1` runs several non-ImageNet configs in sequence and skips test for SimCLR (no test dataloader).

## Project structure

```text
configs/
  config.yaml              # Default: MNIST + simple classifier, logger, trainer.callbacks
  resnet18_cifar10.yaml    # CIFAR-10 + ResNet-18
  resnet50_cifar10.yaml    # CIFAR-10 + ResNet-50
  resnet50_imagenet.yaml   # ImageNet + ResNet-50 (WebDataset .tar shards)
  simclr_cifar10.yaml      # SimCLR on CIFAR-10 (ssl: true, callbacks override)
  simclr_imagenet.yaml     # SimCLR on ImageNet (ssl: true, callbacks override)
  linear_probe_cifar10.yaml  # Frozen SimCLR backbone + linear head on CIFAR-10

src/
  train.py                 # Hydra entrypoint: instantiate logger, callbacks, datamodule, model; Trainer.fit()
  test.py                  # Load best + last checkpoints from experiment_dir; Trainer.test() for each
  module/                  # LightningModules
    simple_classifier.py   # MLP for MNIST
    resnet18.py            # ResNet-18 classifier
    resnet50.py            # ResNet-50 classifier
    simclr.py              # SimCLR (ResNet-50 backbone + projection head)
    linear_probe.py        # Linear probe: frozen SSL backbone + linear classifier (CIFAR-10)
  datamodule/
    default/               # Supervised datamodules
      mnist.py
      cifar10.py
      imagenet.py          # ImageNet via WebDataset
    simclr_cifar10.py      # Two-view CIFAR-10 (TwoViewDataset)
    simclr_imagenet.py     # Two-view ImageNet (decode in pipeline)
  utils/
    data_utils.py          # TransformWrapper, TwoViewDataset

visualizations/
  reporting.py             # Plot confusion matrix from linear_probe_metrics.json (matplotlib)

scripts/
  simple_classifier_mnist.ps1
  resnet18_cifar10.ps1
  resnet50_cifar10.ps1
  resnet50_imagenet.ps1
  simclr_cifar10.ps1
  simclr_imagenet.ps1
  linear_probe_cifar10.ps1 # Linear probe: set $pretrainedCkpt to SimCLR checkpoint; train + test
  run_infra_test.ps1       # Batch run: MNIST, resnet18/50 CIFAR-10, simclr_cifar10; test skipped for SimCLR

experiments/               # One folder per run (when using script or experiment_dir)
  <name>_<timestamp>/
    run.log                # train + test stdout/stderr
    lightning_logs/        # TensorBoard
    checkpoints/           # last.ckpt, best-*.ckpt
    .hydra/                # Hydra config snapshot
    linear_probe_metrics.json  # After linear probe test (under Hydra output dir when using that workflow)

misc/                      # ImageNet → WebDataset
  prepare_imagenet.ps1
  imagenet_to_webdataset.py
  imagenet_to_webdataset.ps1

reports/                   # Optional: figures (e.g. confusion matrix PNG from visualizations/reporting.py)
  figures/
```

## Setup

```bash
conda create -n tp python=3.11 -y
conda activate tp
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

Adjust the PyTorch index URL if you use a different CUDA build or CPU-only wheels.

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
python src/train.py --config-name=linear_probe_cifar10 model.pretrained_ckpt=/path/to/simclr/checkpoint.ckpt
```

**Linear probe workflow**

1. Pretrain with SimCLR, e.g. `python src/train.py --config-name=simclr_cifar10 experiment_dir=./experiments/my_simclr ...`
2. Train the probe with `model.pretrained_ckpt` pointing at the SimCLR checkpoint (backbone weights are loaded from `backbone.*` in that file).
3. **Hydra CLI:** Checkpoint filenames from Lightning often contain `=` (e.g. `best-epoch=00-val_loss=...`). Those characters break unquoted overrides. Prefer **`last.ckpt`**, quote the value, or use forward slashes, e.g. `model.pretrained_ckpt="/abs/path/to/last.ckpt"`.

**Test after training** (requires `experiment_dir` so checkpoints can be found):

```bash
python src/test.py experiment_dir=./experiments/mnist_20260314_123456
```

Test runs on both `last.ckpt` and any `best-*.ckpt` in `experiment_dir/checkpoints/`.

**PowerShell scripts** (train + test, one log under `experiments/<name>_<timestamp>/run.log`):

```powershell
.\scripts\simclr_cifar10.ps1
.\scripts\linear_probe_cifar10.ps1
```

Edit variables at the top of each script (`$dataDir`, `$pretrainedCkpt` for linear probe, epochs, batch size). The scripts set `experiment_dir` and `hydra.run.dir` so artifacts stay in one folder.

**Infra test script** (several non-ImageNet configs in one go):

```powershell
.\scripts\run_infra_test.ps1
```

Runs default config (MNIST), `resnet18_cifar10`, `resnet50_cifar10`, and `simclr_cifar10` in sequence. Test is skipped for SimCLR. Edit `$dataRoot`, `$max_epochs`, etc. at the top of the script.

**Visualization (linear probe):** After testing, point `visualizations/reporting.py` at `linear_probe_metrics.json` to save a confusion matrix PNG (default: `reports/figures/confusion_matrix.png`).

```bash
python visualizations/reporting.py path/to/linear_probe_metrics.json
```

**ImageNet:** Data must be WebDataset `.tar` shards (e.g. `imagenet-train-000000.tar`, `imagenet-val-000000.tar`). Use `misc/prepare_imagenet.ps1` and `misc/imagenet_to_webdataset.py` to build them. Set `datamodule.data_dir` to the directory containing the tars, then run with `--config-name=resnet50_imagenet` or `--config-name=simclr_imagenet`.

## Config

- **Logger** and **trainer.callbacks** are in config; Hydra instantiates them. Base config uses `TensorBoardLogger` and callbacks: `ModelCheckpoint` (best + last) and `EarlyStopping` (val_loss). SimCLR configs override `trainer.callbacks` with only `ModelCheckpoint` (monitor `train_loss`).
- **Model** and **datamodule** use `_target_` (full class path). Optimizer lives under `model.optimizer` with `_partial_: true`.
- **ssl:** Set to `true` in SimCLR configs; used to select callback behavior (e.g. no EarlyStopping).
- **Linear probe:** `linear_probe_cifar10.yaml` sets `ssl: false` and requires `model.pretrained_ckpt` (override or set in YAML) to a SimCLR checkpoint.
- Extra configs inherit from `config.yaml` via `defaults: [config, _self_]` and override `model`, `datamodule`, and optionally `trainer.callbacks`.

## Reproducibility

- Full config is saved under the experiment directory (e.g. `.hydra/` when using the script).
- `pl.seed_everything(cfg.seed)` is called at startup.
