import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # pl.seed_everthing(cfg.seed)


if __name__ == "__main__":
    main()