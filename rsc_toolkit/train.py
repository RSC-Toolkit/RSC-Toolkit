import hydra
from omegaconf import DictConfig, OmegaConf


def hydra_main(overrides=None):
    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def _main(cfg: DictConfig) -> None:
        print(OmegaConf.to_yaml(cfg))

    _main(overrides)

if __name__ == "__main__":
    hydra_main()
