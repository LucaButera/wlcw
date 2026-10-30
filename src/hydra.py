from omegaconf import DictConfig, OmegaConf


def target_classname(cfg: DictConfig) -> str:
    name = cfg._target_.split(".")[-1]
    if name == "MultiInductiveTemporalDataModule":
        name = f"multitask{sorted(cfg.datamodules.keys())}"
    return name


def parse_hyperparams(cfg: DictConfig) -> dict[str, any]:
    if cfg.get("gpi") is not None:
        gpi = target_classname(cfg.gpi).lower()
        emb_size = cfg.gpi.emb_size
    else:
        gpi = "none"
        emb_size = 0
    hyperparams = {
        "engine": target_classname(cfg.engine).lower(),
        "dataset": target_classname(cfg.dataset)
        .replace("InductiveDataModule", "")
        .replace("DataModule", "")
        .lower(),
        "model": target_classname(cfg.model).lower(),
        "hidden_size": cfg.model.get("hidden_size", 0),
        "gpi": gpi.lower(),
        "embedding_size": emb_size,
        "lr": cfg.engine.optim_kwargs.lr,
        "seed": cfg.get("seed") or "none",
        "hydra_cfg": OmegaConf.to_container(cfg),
    }
    return hyperparams
