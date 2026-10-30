import numpy as np

from env import CACHE
from src.data.datasets.multi_pvarx import MultiPVARXDataset
from src.data.prototypes.inductive_temporal_datamodule import (
    InductiveTemporalDataModule,
)


class MultiPVARXInductiveDataModule(InductiveTemporalDataModule):
    def __init__(
        self,
        num_nodes: int = 1000,
        num_steps: int = 10000,
        num_features: int = 1,
        coeff_bound: float = 1.0,
        sigma_noise: float | list[float] = 0.2,
        seed: int | None = 42,
        temporal_order: int = 2,
        target_dims: int | list[int] | tuple | None = 0,
        exogenous_dims: int | list[int] | tuple | None = (1, 2),
        val_len: int | float = 0.1,
        test_len: int | float = 0.2,
        val_nodes: int | float | None = 0.1,
        test_nodes: int | float | None = 0.2,
        history: int = 0,
        mask_as_exogenous: bool = False,
        window: int = 6,
        horizon: int = 1,
        delay: int = 0,
        stride: int = 1,
        window_lag: int = 1,
        horizon_lag: int = 1,
        precision: int | str = 32,
        mask_scaling: bool = True,
        batch_size: int = 128,
        transductive_only: bool = False,
        history_first_sample_shift: int | tuple[int, int] = 0,
        train_len: list[int] | None = None,
        workers: int = 0,
        pin_memory: bool = False,
        cache_dataset: bool = True,
    ):
        params = self._get_params(
            temporal_order=temporal_order,
            num_nodes=num_nodes,
            coeff_bound=coeff_bound,
            num_features=num_features,
            seed=seed,
        )

        dataset = MultiPVARXDataset(
            root=str(CACHE / "MultiPVARX"),
            num_nodes=num_nodes,
            num_steps=num_steps,
            sigma_noise=sigma_noise,
            seed=seed,
            target_dims=target_dims,
            exogenous_dims=exogenous_dims,
            params_as_exogenous=False,
            cache=cache_dataset,
            nonlinearity="tanh",
            **params,
        )
        del dataset.optimal_pred

        super().__init__(
            dataset=dataset,
            val_len=val_len,
            test_len=test_len,
            val_nodes=val_nodes,
            test_nodes=test_nodes,
            add_temporal_encoding=False,
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

    @staticmethod
    def _get_params(
        temporal_order: int,
        num_nodes: int,
        coeff_bound: float,
        num_features: int,
        seed: int | None,
    ) -> dict:
        assert temporal_order > 1, "Temporal order must be greater than 1"

        # Using legacy random state as lightning seed_everything uses it
        rng = np.random if seed is None else np.random.RandomState(seed)

        # Take params from 0.5 to coeff_bound and then flip randomly
        # avoid having too small coefficients which yield almost flat series
        params = rng.uniform(
            0.5, coeff_bound, (num_nodes, temporal_order, num_features)
        )
        params = np.where(
            rng.uniform(0, 1, (num_nodes, temporal_order, num_features)) > 0.5,
            params,
            -params,
        )

        # Feature parameters with higher self correlation
        if num_features > 1:
            feature_params = rng.uniform(
                low=0,
                high=1 / (num_features - 1),
                size=(num_features, num_features),
            )
            feature_params[range(num_features), range(num_features)] = 1.0
        else:
            feature_params = None

        return dict(
            params=params,
            feature_params=feature_params,
        )
