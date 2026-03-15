import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl


torch.set_float32_matmul_precision("medium")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@hydra.main(config_path=str(PROJECT_ROOT / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    experiment_dir = Path(cfg.experiment_dir) if cfg.get("experiment_dir") else None
    if experiment_dir is None:
        raise ValueError("Set experiment_dir=... (same as training) to run test.")

    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints dir at {checkpoint_dir}. Train first.")

    last_ckpt = checkpoint_dir / "last.ckpt"
    best_ckpts = sorted(checkpoint_dir.glob("best-*.ckpt"))
    to_run = []
    if last_ckpt.exists():
        to_run.append(("last", last_ckpt))
    for p in best_ckpts:
        to_run.append((p.stem, p))

    if not to_run:
        raise FileNotFoundError(f"No last.ckpt or best-*.ckpt in {checkpoint_dir}.")

    if experiment_dir is not None:
        cfg.logger.save_dir = str(experiment_dir / "lightning_logs")
    logger = hydra.utils.instantiate(cfg.logger)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("test")

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
    )

    for name, ckpt_path in to_run:
        print(f"\n--- Testing checkpoint: {name} ({ckpt_path.name}) ---")
        model = hydra.utils.instantiate(cfg.model)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()