import gc

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from main import setup_run_env, clean_empty_configs, update_config_from_data
from src.hydra import parse_hyperparams
from src.resolvers import register_custom_resolvers
from src.trainer import Trainer


@hydra.main(config_path="conf", config_name="experiment", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_run_env(cfg)
    cfg = clean_empty_configs(cfg)

    datamodule = instantiate(cfg.dataset, _convert_="all")
    datamodule.setup()  # call setup as we need to know the number of batches
    cfg = update_config_from_data(cfg, datamodule)
    engine = instantiate(cfg.engine, _convert_="all")
    if cfg.get("matmul_precision", None) is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    trainer = Trainer(cfg)
    trainer.logger.log_hyperparams(parse_hyperparams(cfg))
    trainer.test(model=engine, datamodule=datamodule)
    trainer.logger.finalize("success")
    del trainer, engine, datamodule
    gc.collect()


if __name__ == "__main__":
    register_custom_resolvers()
    main()
