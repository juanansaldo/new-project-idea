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

    callbacks = hydra.utils.instantiate(cfg.trainer.get("callbacks", [])) or []

    experiment_dir = Path(cfg.experiment_dir) if cfg.get("experiment_dir") else None
    if experiment_dir is not None:
        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for cb in callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                cb.dirpath = str(checkpoint_dir)

    if experiment_dir is not None:
        cfg.logger.save_dir = str(experiment_dir / "lightning_logs")
    logger = hydra.utils.instantiate(cfg.logger)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()