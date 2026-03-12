## Training Pipeline Project

This repository contains a small but realistic **deep learning training pipeline** built with **Hydra** and **PyTorch Lightning**. The goal is to showcase both **machine learning** and **software engineering** skills: configuration management, reproducibility, and clean separation of data/model/training code.

### Features

- **Hydra-driven configuration**
  - Single `config.yaml` entrypoint with override support from CLI or PowerShell.
  - Easy experiment changes via config groups and/or scripted overrides.
- **PyTorch Lightning training loop**
  - `LightningModule` for a simple MNIST classifier.
  - `LightningDataModule` for data loading, splitting, and transforms.
- **Reproducibility**
  - Hydra automatically snapshots configs per run in `.hydra/`.
  - Full config is also saved next to Lightning logs.

### Project Structure

```text
training_pipeline/
  configs/
    config.yaml           # Hydra root config
  src/
    data/
      mnist_datamodule.py
    models/
      simple_classifier.py
  train.py                # Hydra + Lightning entrypoint
  run_experiment.ps1      # Example PowerShell runner with overrides
  requirements.txt
```

### Environment Setup (Conda)

```bash
conda create -n tp python=3.11 -y
conda activate tp
cd C:\Projects\AI\new-project
pip install -r training_pipeline\requirements.txt
```

You can verify everything is installed with:

```bash
python -c "import torch, pytorch_lightning as pl, hydra; print('ok')"
```

### Running Training

From the `training_pipeline` directory:

```bash
cd training_pipeline
python train.py
```

This will:

- Load the Hydra config from `configs/config.yaml`.
- Train the MNIST classifier for the configured number of epochs.
- Write Hydra run artifacts (including full config) under `outputs/...`.
- Write Lightning logs and checkpoints under `lightning_logs/...`.

### Using Overrides

You can change hyperparameters from the command line without editing YAML:

```bash
python train.py data.batch_size=128 trainer.max_epochs=5 model.lr=1e-4
```

There is also a sample PowerShell script that builds overrides and appends a timestamp to the experiment name:

```powershell
.\run_experiment.ps1
```

This pattern makes it easy to:

- Keep **default configs** stable.
- Track experiments via **named configs and timestamps**.
- Reproduce runs using the saved config snapshots.

