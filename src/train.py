import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


torch.set_float32_matmul_precision("medium")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@hydra.main(config_path=str(PROJECT_ROOT / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    experiment_dir = Path(cfg.experiment_dir) if cfg.get("experiment_dir") else None
    if experiment_dir is not None:
        save_dir = str(experiment_dir / "lightning_logs")
    else:
        save_dir = "lightning_logs"

    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=cfg.experiment_name,
    )

    datamodule = hydra.utils.instantiate(cfg.datamodule, **cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=logger,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()