from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from tsl import logger
from tsl.data import BatchMapItem, SynchMode
from tsl.data.datamodule.spatiotemporal_datamodule import (
    StageOptions,
)
from tsl.data.loader import StaticGraphLoader
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import TabularDataset

from src.data.prototypes.history_temporal_dataset import TemporalDataset
from src.data.splitters import InductiveBlockSplitter


class InductiveTemporalDataModule(LightningDataModule):
    """DataModule supporting both standard and decoupled inductive temporal forecasting.

    Terminology mapping to 'Why Do Time Series Models Need Long Context Windows?' (Butera et al., 2026):
      - `window`: Recent observation window length W (fed to base predictor F).
      - `horizon`: Forecasting horizon H.
      - `history`: Past context window length C (fed to process identification module G).
      - `history_first_sample_shift`: Temporal gap S between past context and current window (t - t').
    """

    def __init__(
        self,
        dataset: TabularDataset,
        val_len: int | float | list[int] | np.ndarray,
        test_len: int | float | list[int] | np.ndarray,
        val_nodes: int | float | list[int] | np.ndarray | None,
        test_nodes: int | float | list[int] | np.ndarray | None,
        add_temporal_encoding: bool,
        history: int,
        mask_as_exogenous: bool,
        window: int,
        horizon: int,
        delay: int = 0,
        stride: int = 1,
        window_lag: int = 1,
        horizon_lag: int = 1,
        precision: int | str = 32,
        mask_scaling: bool = True,
        batch_size: int = 64,
        transductive_only: bool = False,
        history_first_sample_shift: int | tuple[int, int] = 0,
        train_len: list[int] | None = None,
        workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self._mask_as_exogenous = mask_as_exogenous
        self._add_temporal_encoding = add_temporal_encoding
        self._transductive_only = transductive_only

        splitter = InductiveBlockSplitter(
            train_len=train_len,
            val_len=val_len,
            test_len=test_len,
            val_nodes=val_nodes,
            test_nodes=test_nodes,
            transductive_only=self._transductive_only,
        )

        torch_dataset = TemporalDataset(
            target=dataset.dataframe(),
            mask=dataset.mask,
            covariates=dataset.covariates,
            transform=None,
            window=window,
            horizon=horizon,
            delay=delay,
            stride=stride,
            window_lag=window_lag,
            horizon_lag=horizon_lag,
            precision=precision,
            history=history,
            history_shift=history_first_sample_shift,
        )

        scalers = {
            "target": StandardScaler(axis=(0, 1)),
            **{
                k: StandardScaler(
                    axis=(0, 1) if torch_dataset.patterns[k] == "t n f" else 0
                )
                for k in self.get_covariate_keys(
                    torch_dataset,
                    filters=[
                        lambda k: not (
                            k.endswith("_mask") or k in ["mask", "dist", "t_enc"]
                        )
                    ],
                )
            },
        }

        if self._add_temporal_encoding:
            t_enc = self.compute_temporal_encoding(dataset)
            torch_dataset.add_covariate(
                name="t_enc",
                value=t_enc,
                pattern="t n f",
                add_to_input_map=False,
            )
        u = self.get_covariate_keys(torch_dataset, temporal_only=True)
        if u:
            torch_dataset.update_input_map(
                u=BatchMapItem(
                    u,
                    preprocess=True,
                    pattern="t n f",
                    synch_mode=SynchMode.WINDOW,
                )
            )
        if history > 0:
            h = self.get_history_keys(torch_dataset)
            torch_dataset.update_input_map(
                h=BatchMapItem(
                    h,
                    preprocess=True,
                    pattern="t n f",
                    synch_mode=SynchMode.HISTORY,
                ),
            )

        self.torch_dataset = torch_dataset
        # splitting
        self.splitter = splitter
        self.trainset = self.valset = self.testset = None
        self.fair_valset = self.fair_testset = None
        # scaling
        self.scalers = dict() if scalers is None else scalers
        self.mask_scaling = mask_scaling
        # data loaders
        self.batch_size = batch_size
        self.workers = workers
        self.pin_memory = pin_memory
        # cache directory
        self._cache_dir = Path(dataset.root, "cache")
        if hasattr(dataset, "get_hash"):
            self._cache_dir = self._cache_dir.joinpath(dataset.get_hash())

    def __getattr__(self, item):
        ds = self.__dict__.get("torch_dataset")
        if ds is not None and hasattr(ds, item):
            return getattr(ds, item)
        else:
            raise AttributeError(item)

    def __repr__(self):
        return (
            "{}(train_len={}, val_len={}, test_len={}, "
            "scalers=[{}], batch_size={})".format(
                self.__class__.__name__,
                self.train_len,
                self.val_len,
                self.test_len,
                ", ".join(self.scalers.keys()),
                self.batch_size,
            )
        )

    @property
    def trainset(self) -> TemporalDataset | None:
        return self._trainset

    @property
    def valset(
        self,
    ) -> dict[Literal["transd", "ind"], TemporalDataset] | TemporalDataset | None:
        return self._valset

    @property
    def testset(
        self,
    ) -> dict[Literal["transd", "ind"], TemporalDataset] | TemporalDataset | None:
        return self._testset

    @trainset.setter
    def trainset(self, value: TemporalDataset | None):
        self._add_set("train", value)

    @valset.setter
    def valset(
        self,
        value: dict[Literal["transd", "ind"], TemporalDataset] | TemporalDataset | None,
    ):
        self._add_set("val", value)

    @testset.setter
    def testset(
        self,
        value: dict[Literal["transd", "ind"], TemporalDataset] | TemporalDataset | None,
    ):
        self._add_set("test", value)

    @property
    def train_len(self) -> int | None:
        return self._get_set_len("train")

    @property
    def val_len(self) -> list[int] | int | None:
        return self._get_set_len("val")

    @property
    def test_len(self) -> list[int] | int | None:
        return self._get_set_len("test")

    @property
    def fair_trainset(self) -> TemporalDataset | None:
        return self.trainset

    @property
    def fair_valset(self) -> TemporalDataset | None:
        return self._fair_valset

    @property
    def fair_testset(self) -> TemporalDataset | None:
        return self._fair_testset

    @fair_valset.setter
    def fair_valset(self, value: TemporalDataset | None):
        self._add_set("fair_val", value)

    @fair_testset.setter
    def fair_testset(self, value: TemporalDataset | None):
        self._add_set("fair_test", value)

    def _get_set_len(
        self, split_type: str
    ) -> dict[Literal["transd", "ind"], int] | int | None:
        _set = getattr(self, f"{split_type}set")
        if _set is None:
            return None
        if isinstance(_set, dict):
            return {k: len(v) for k, v in _set.items()}
        return len(_set)

    def _add_set(
        self,
        split_type,
        _set: dict[Literal["transd", "ind"], TemporalDataset] | TemporalDataset | None,
    ):
        assert split_type in [
            "train",
            "val",
            "test",
            "fair_val",
            "fair_test",
        ]
        assert not (
            split_type in ["train", "fair_val", "fair_test"] and isinstance(_set, dict)
        ), "Fair sets cannot be a multiple datasets."
        setattr(self, f"_{split_type}set", _set)

    def setup(self, stage: StageOptions = None):
        # splitting
        splits = self.splitter.split(self.torch_dataset)

        self.trainset = splits["train_set"]
        self.trainset.history_shift_mode = "random"

        if self._transductive_only:
            self.valset = splits["val_set"]
            self.testset = splits["test_set"]
            self.valset.history_shift_mode = "upto"
            self.testset.history_shift_mode = "upto"
        else:
            self.valset = {
                "transd": splits["val_set"],
                "ind": splits["val_set_inductive"],
            }
            self.testset = {
                "transd": splits["test_set"],
                "ind": splits["test_set_inductive"],
            }
            for k in ["transd", "ind"]:
                self.valset[k].history_shift_mode = "upto"
                self.testset[k].history_shift_mode = "upto"
            self.fair_valset = splits["val_nodes_at_train_time"]
            self.fair_testset = splits["test_nodes_at_train_time"]

        # set scalers
        for key, scaler in self.scalers.items():
            if key not in self.torch_dataset:
                raise RuntimeError(
                    "Cannot find a tensor to scale matching " f"key '{key}'."
                )
            if stage == "predict":
                logger.info(f"Set scaler for {key}: {scaler}")
            else:  # fit scalers before training
                data, mask = self._get_fair_data(key=key, return_mask=self.mask_scaling)
                scaler = scaler.fit(data, mask=mask, keepdims=True)
                logger.info(f"Fit and set scaler for {key}: {scaler}")
            for dset in [self.torch_dataset, self.trainset, self.valset, self.testset]:
                if dset is not None:
                    if isinstance(dset, dict):
                        for dsk in dset:
                            dset[dsk].add_scaler(key, scaler)
                    else:
                        dset.add_scaler(key, scaler)
            # Scalers are optional for auxiliary sets as they might not have
            # some dynamically generated features.
            for dset in [self.fair_valset, self.fair_testset]:
                if dset is not None and key in dset:
                    dset.add_scaler(key, scaler)

    def get_dataloader(
        self,
        split: Literal["train", "val", "test"] = None,
        shuffle: bool = False,
        batch_size: int | None = None,
    ) -> list[DataLoader] | DataLoader | None:
        if split is None:
            dataset = self.torch_dataset
        elif split in ["train", "val", "test"]:
            dataset = getattr(self, f"{split}set")
        else:
            raise ValueError(
                "Argument `split` must be one of " "'train', 'val', or 'test'."
            )
        if dataset is None:
            return None
        pin_memory = self.pin_memory if split == "train" else None
        if isinstance(dataset, dict):
            return [
                StaticGraphLoader(
                    dataset[dsk],
                    batch_size=batch_size or self.batch_size,
                    shuffle=shuffle,
                    drop_last=split == "train",
                    num_workers=self.workers,
                    pin_memory=pin_memory,
                )
                for dsk in ["transd", "ind"]
            ]
        else:
            return StaticGraphLoader(
                dataset,
                batch_size=batch_size or self.batch_size,
                shuffle=shuffle,
                drop_last=split == "train",
                num_workers=self.workers,
                pin_memory=pin_memory,
            )

    def train_dataloader(
        self, shuffle: bool = True, batch_size: int | None = None
    ) -> DataLoader | None:
        return self.get_dataloader("train", shuffle, batch_size)

    def val_dataloader(
        self, shuffle: bool = False, batch_size: int | None = None
    ) -> list[DataLoader] | DataLoader | None:
        return self.get_dataloader("val", shuffle, batch_size)

    def test_dataloader(
        self, shuffle: bool = False, batch_size: int | None = None
    ) -> list[DataLoader] | DataLoader | None:
        return self.get_dataloader("test", shuffle, batch_size)

    # END OF COMMON INTERFACE WITH ST_DATAMODULE

    def compute_temporal_encoding(self, dataset: TabularDataset) -> np.ndarray:
        day = dataset.datetime_encoded("day").values
        weekday = dataset.datetime_onehot("weekday").values
        t_enc = np.tile(
            np.concatenate([day, weekday], axis=-1)[:, None],
            (1, dataset.n_nodes, 1),
        )
        return t_enc

    def get_covariate_keys(
        self,
        dataset: TemporalDataset | None = None,
        temporal_only: bool = False,
        filters: list[Callable[[str], bool]] | None = None,
        allowed_patterns: list[str] | None = None,
    ) -> list[str]:
        dataset = dataset if dataset is not None else self.torch_dataset
        filters = [] if filters is None else filters
        if not self._mask_as_exogenous:
            filters.append(lambda k: not (k.endswith("_mask") or k == "mask"))
        if not self._add_temporal_encoding:
            filters.append(lambda k: k != "t_enc")
        keys = (
            dataset.covariates.keys() if not temporal_only else dataset.exogenous.keys()
        )
        return list(
            sorted(
                k
                for k in keys
                if (
                    (
                        allowed_patterns is None
                        or dataset.patterns[k] in allowed_patterns
                    )
                    and all(f(k) for f in filters)
                )
            )
        )

    def get_history_keys(self, dataset: TemporalDataset | None = None) -> list[str]:
        return ["target"] + self.get_covariate_keys(dataset, temporal_only=True)

    def _get_fair_data(
        self, key: str, return_mask: bool = False, split: str = "train"
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        assert split in ["train", "val", "test"]
        # Prepare the data given the kind of splits
        dset = getattr(self, f"fair_{split}set")
        # Get the data and mask
        mask = None
        data = getattr(dset, key)
        if return_mask:
            if key == "target" and dset.has_mask:
                mask = dset.get_mask()
            elif f"{key}_mask" in dset:
                mask = getattr(dset, f"{key}_mask")
        return data if not return_mask else (data, mask)
