from typing import Sequence

import numpy as np
from tsl.datasets import DatetimeDataset
from tsl.datasets.large_st import SubsetType

from env import CACHE
from src.data.datasets.largest import LargeST
from src.data.prototypes.inductive_temporal_datamodule import (
    InductiveTemporalDataModule,
)
from src.data.utils import datetime_encoded


class LargeSTInductiveDataModule(InductiveTemporalDataModule):
    def __init__(
        self,
        subset: SubsetType = "CA",
        year: Sequence[int] | int = 2019,
        freq: str = "15T",
        val_len: int | float = 0.1,
        test_len: int | float = 0.2,
        val_nodes: int | float | None = 0.1,
        test_nodes: int | float | None = 0.2,
        add_temporal_encoding: bool = True,
        temporal_encoding_type: str | list[str] = "day",
        history: int = 0,
        mask_as_exogenous: bool = True,
        window: int = 12,
        horizon: int = 12,
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
        dataset = LargeST(
            root=str(CACHE / "LargeST"),
            imputation_mode="zero",
            subset=subset,
            year=year,
            freq=freq,
        )
        del dataset.adj
        del dataset.metadata

        self.temporal_encoding_type = temporal_encoding_type

        super().__init__(
            dataset=dataset,
            val_len=val_len,
            test_len=test_len,
            val_nodes=val_nodes,
            test_nodes=test_nodes,
            add_temporal_encoding=add_temporal_encoding,
            history=history,
            mask_as_exogenous=mask_as_exogenous,
            window=window,
            horizon=horizon,
            delay=delay,
            stride=stride,
            window_lag=window_lag,
            horizon_lag=horizon_lag,
            precision=precision,
            mask_scaling=mask_scaling,
            batch_size=batch_size,
            transductive_only=transductive_only,
            history_first_sample_shift=history_first_sample_shift,
            train_len=train_len,
            workers=workers,
            pin_memory=pin_memory,
        )

    def compute_temporal_encoding(self, dataset: DatetimeDataset) -> np.ndarray:
        sin_cos = datetime_encoded(dataset, self.temporal_encoding_type).values
        weekday = dataset.datetime_onehot("weekday").values
        t_enc = np.tile(
            np.concatenate([sin_cos, weekday], axis=-1)[:, None],
            (1, dataset.n_nodes, 1),
        )
        return t_enc
