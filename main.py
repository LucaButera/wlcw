import gc

import hydra
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, open_dict
from tsl.data import SpatioTemporalDataModule

from src.hydra import parse_hyperparams
from src.resolvers import register_custom_resolvers
from src.trainer import Trainer


def update_config_from_data(
    cfg: DictConfig, datamodule: SpatioTemporalDataModule
) -> DictConfig:
    n_nodes = datamodule.trainset.n_nodes
    input_map = datamodule.trainset.input_map

    def get_size(key):
        return input_map[key].shape[-1] if key in input_map else 0

    with open_dict(cfg):
        cfg.engine.model.update(
            input_size=get_size("x"),
            exog_size=get_size("u"),
            n_nodes=n_nodes,
            horizon=datamodule.torch_dataset.horizon,
        )
        if cfg.get("gpi") is not None:
            cfg.gpi.update(
                history_input_size=get_size("h"),
                static_input_size=get_size("v"),
            )
        if cfg.engine._target_.split(".")[-1] == "MultiTaskTemporalPredictor":
            cfg.engine.update(task_names=list(datamodule.datamodules.keys()))
    return cfg


def setup_run_env(cfg: DictConfig):
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))


def clean_empty_configs(cfg: DictConfig) -> DictConfig:
    if not cfg.get("gpi"):
        cfg.gpi = None
    return cfg


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
    trainer.fit(engine, datamodule=datamodule)
    trainer.test(datamodule=datamodule)
    trainer.logger.finalize("success")
    del trainer, engine, datamodule
    gc.collect()


if __name__ == "__main__":
    register_custom_resolvers()
    main()
