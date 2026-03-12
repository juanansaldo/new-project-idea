import hydra
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.data.mnist_datamodule import MNISTDataModule
from src.models.simple_classifier import SimpleMNISTClassifier


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    datamodule = MNISTDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    model = SimpleMNISTClassifier(lr=cfg.model.lr)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    trainer.fit(model=model, datamodule=datamodule)

    # Save config next to lightning logs
    log_dir = Path(trainer.default_root_dir) / trainer.logger.name / f"version_{trainer.logger.version}"
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "config.yaml")


if __name__ == "__main__":
    main()