# Hydra + Lightning Training Pipeline

A config-driven deep learning training pipeline for image classification. The project uses **Hydra** for configuration and **PyTorch Lightning** for training, with a single experiment directory per run that stores logs, config snapshots, and script output.

## What it does

- Trains image classifiers on **MNIST** and **CIFAR-10** using a unified configuration and training pipeline.
- Lets you choose **model**, **datamodule**, and **optimizer** from YAML; no code changes needed to switch between a simple MLP and **ResNet-18/ResNet-50**.
- Runs training and test in one go; after testing, prints a single **sklearn classification report** over the full test set.
- Writes each run into a timestamped folder under `experiments/` (config copy, Lightning logs, run log).
- Ships helper scripts in `misc/` to prepare **ImageNet** and convert it into an indexed WebDataset format for future ingestion.

## Project structure

```text
configs/                  # Hydra configs (one per experiment type)
  config.yaml             # Default: MNIST + simple classifier
  resnet18_cifar10.yaml   # CIFAR-10 + ResNet-18
  resnet50_cifar10.yaml   # CIFAR-10 + ResNet-50

src/
  train.py                # Entrypoint: Hydra + Lightning
  module/                 # LightningModules (models)
    simple_classifier.py  # Shared training logic + MLP head
    resnet18.py
    resnet50.py
  datamodule/             # LightningDataModules
    mnist.py
    cifar10.py
  utils/
    data_utils.py         # e.g. TransformWrapper

scripts/
  run_experiment.ps1      # Run from project root, writes to experiments/<name>_<timestamp>

experiments/              # One folder per run (created by script)
  mnist_2026.../
    run.log
    config.yaml
    lightning_logs/

misc/                     # ImageNet preparation + WebDataset conversion
  prepare_imagenet.ps1
  imagenet_to_webdataset.py
```

## Setup

```bash
conda create -n tp python=3.11 -y
conda activate tp
pip install -r requirements.txt
```

## Running

**From project root (default MNIST config):**

```bash
python src/train.py
```

Uses `configs/config.yaml` by default. Override anything from the CLI, for example:

```bash
python src/train.py trainer.max_epochs=5 datamodule.batch_size=256
```

**Use a specific experiment config** (e.g. CIFAR-10 + ResNet-18):

```bash
python src/train.py --config-name=resnet18_cifar10
```

**Using the PowerShell script** (creates a timestamped experiment dir and redirects output to `run.log`):

```powershell
.\scripts\run_experiment.ps1
```

Edit `$experimentName`, `$configName`, `$dataDir`, and the overrides array in the script to change the run. The script sets `experiment_dir` and `hydra.run.dir` so all artifacts go under `experiments/<name>_<timestamp>/`, and measures wall-clock runtime for each experiment.

## Config

- **Model and datamodule** are specified with `_target_` (full class path); Hydra instantiates them.
- **Optimizer** is under `model.optimizer` with `_partial_: true` so it is created in `configure_optimizers()` with `self.parameters()`.
- Extra experiment configs (e.g. `resnet18_cifar10.yaml`) inherit from `config.yaml` via `defaults: [config, _self_]` and override only `model` and `datamodule`.

## Reproducibility

- Each run’s full config is saved under the experiment directory (and by Hydra in `.hydra/` when using the script).
- `pl.seed_everything(cfg.seed)` is called at startup.
