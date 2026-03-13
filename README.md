# MNIST Training Pipeline

A config-driven deep learning training pipeline for image classification. The project uses **Hydra** for configuration and **PyTorch Lightning** for training, with a single experiment directory per run that stores logs, config snapshots, and script output.

## What it does

- Trains a small classifier on MNIST (or other datasets via config).
- Lets you choose **model**, **datamodule**, and **optimizer** from YAML; no code changes needed to switch setups.
- Runs training and test in one go; after testing, prints a single **sklearn classification report** over the full test set.
- Writes each run into a timestamped folder under `experiments/` (config copy, Lightning logs, run log).

## Project structure

```text
configs/              # Hydra configs (one per experiment type)
  config.yaml         # Default: MNIST + simple classifier
  resnet18_cifar10.yaml
  resnet50_cifar10.yaml
src/
  train.py            # Entrypoint: Hydra + Lightning
  module/             # LightningModules (models)
    simple_classifier.py
  datamodule/         # LightningDataModules
    mnist.py
  utils/
    data_utils.py     # e.g. TransformWrapper
scripts/
  run_experiment.ps1  # Run from project root, writes to experiments/<name>_<timestamp>
experiments/          # One folder per run (created by script)
  mnist_20260312_173551/
    run.log
    config.yaml
    lightning_logs/
```

## Setup

```bash
conda create -n tp python=3.11 -y
conda activate tp
pip install -r requirements.txt
```

## Running

**From project root:**

```bash
python src/train.py
```

Uses `configs/config.yaml` by default. Override anything from the CLI:

```bash
python src/train.py trainer.max_epochs=5 data.batch_size=256
```

**Using the PowerShell script** (creates a timestamped experiment dir and redirects output to `run.log`):

```powershell
.\scripts\run_experiment.ps1
```

Edit `$experimentName`, `$configName`, and the overrides array in the script to change the run. The script sets `experiment_dir` and `hydra.run.dir` so all artifacts go under `experiments/<name>_<timestamp>/`.

## Config

- **Model and datamodule** are specified with `_target_` (full class path); Hydra instantiates them.
- **Optimizer** is under `model.optimizer` with `_partial_: true` so it is created in `configure_optimizers()` with `self.parameters()`.
- Extra experiment configs (e.g. `resnet18_cifar10.yaml`) can inherit from `config.yaml` via `defaults: [config]` and override only `model` and `datamodule`.

## Reproducibility

- Each run’s full config is saved under the experiment directory (and by Hydra in `.hydra/` when using the script).
- `pl.seed_everything(cfg.seed)` is called at startup.
