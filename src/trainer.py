from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning import Trainer as _Trainer_
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import DummyLogger
from torch import cuda


def _get_logger(cfg: DictConfig):
    if cfg.trainer.logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=HydraConfig.get().runtime.output_dir,
            name="",
            version="",
        )
    else:
        raise ValueError(f"Unknown logger {cfg.trainer.logger}")
    return logger


class Trainer(_Trainer_):
    def __init__(self, cfg: DictConfig):
        callbacks = []
        if cfg.trainer.get("monitor", None) is not None:
            if cfg.trainer.get("patience", None) is not None:
                callbacks.append(
                    EarlyStopping(
                        monitor=cfg.trainer.monitor,
                        patience=cfg.trainer.patience,
                    )
                )
            callbacks.append(
                ModelCheckpoint(
                    dirpath="checkpoints",
                    every_n_epochs=None,
                    monitor=cfg.trainer.monitor,
                    save_top_k=1,
                    mode="min",
                    save_last=True,
                    save_weights_only=False,
                )
            )
        callbacks.append(
            LearningRateMonitor(
                logging_interval="step",
            )
        )
        if cuda.is_available():
            accelerator = "gpu"
        else:
            accelerator = "cpu"
        if cfg.trainer.get("logger") is not None:
            logger = _get_logger(cfg)
        else:
            logger = DummyLogger()
        trainer_kwargs = {
            k: v
            for k, v in cfg.trainer.items()
            if k not in ["monitor", "patience", "logger"]
        }
        super().__init__(
            callbacks=callbacks,
            accelerator=accelerator,
            logger=logger,
            **trainer_kwargs,
        )
