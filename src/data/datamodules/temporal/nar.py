from env import CACHE
from src.data.datasets.nar import NARDataset
from src.data.prototypes.inductive_temporal_datamodule import (
    InductiveTemporalDataModule,
)


class NARInductiveDataModule(InductiveTemporalDataModule):
    def __init__(
        self,
        order: int,
        num_nodes: int = 1000,
        num_steps: int = 10000,
        alpha: float = 1.0,
        non_linear: bool = True,
        sigma_noise: float | list[float] = 0.1,
        unit_circle_sampling: bool = False,
        save_coeffs: bool = False,
        seed: int | None = 42,
        val_len: int | float = 0.1,
        test_len: int | float = 0.2,
        val_nodes: int | float | None = 0.1,
        test_nodes: int | float | None = 0.2,
        history: int = 0,
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
    ):
        dataset = NARDataset(
            order=order,
            num_nodes=num_nodes,
            num_steps=num_steps,
            noise_std=sigma_noise,
            alpha=alpha,
            non_linear=non_linear,
            unit_circle_sampling=unit_circle_sampling,
            seed=seed,
            root=str(CACHE / "NAR"),
            save_coeffs=save_coeffs,
        )
        if save_coeffs:
            self.nar_info = {
                "num_nodes": dataset.num_nodes,
                "coeffs": dataset.coeffs,
                "noise_std": dataset.noise_std,
                "order": dataset.order,
            }

        super().__init__(
            dataset=dataset,
            val_len=val_len,
            test_len=test_len,
            val_nodes=val_nodes,
            test_nodes=test_nodes,
            add_temporal_encoding=False,
            history=history,
            mask_as_exogenous=False,
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
