import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tsl.datasets import TabularDataset


class NARDataset(TabularDataset):
    def __init__(
        self,
        order: int,
        num_nodes: int = 1000,
        num_steps: int = 10000,
        noise_std: float = 0.1,
        alpha: float = 1.0,
        non_linear: bool = True,
        seed: int = 42,
        root: str | None = None,
        save_coeffs: bool = False,
        unit_circle_sampling: bool = False,
        burn_in: int = 20,
    ):
        self.root = root
        self.order = order
        self.num_nodes = num_nodes
        self.num_steps = num_steps
        self.noise_std = noise_std
        self.alpha = alpha
        self.non_linear = non_linear
        self.seed = seed
        self.unit_circle_sampling = unit_circle_sampling
        self.burn_in = burn_in
        if self.unit_circle_sampling:
            assert (
                not self.non_linear
            ), "Non-linear NAR with unit circle sampling is not supported."
            assert (
                self.order == 2
            ), "Unit circle sampling is only supported for order 2."
        data, coeffs = self._generate_all_series_parallel()
        if save_coeffs:
            self.coeffs = coeffs
        super().__init__(
            target=data,
            mask=np.ones_like(data, dtype=bool),
            name="NAR",
        )

    @staticmethod
    def generate_single_series(
        order: int,
        num_steps: int,
        alpha: float,
        non_linear: bool,
        noise_std: float,
        unit_circle_sampling: bool,
        seed: int,
        burn_in: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        non_lin = np.tanh if non_linear else lambda x: x
        rng = np.random.default_rng(seed)
        if unit_circle_sampling:
            for _ in range(100):
                c = rng.uniform(0, 2 * np.pi, size=1)
                c = np.array([np.cos(c), np.sin(c)]).squeeze()
                if c[0] + c[1] < 1 and c[0] - c[1] < 1 and c[1] - c[0] < 1:
                    break
            else:
                raise ValueError(
                    "Could not find suitable coefficients after 100 attempts"
                )
        else:
            bound = alpha * (1 + 0.1 * np.log(order)) / np.sqrt(order)
            c = rng.uniform(-bound, bound, size=order)
        total_steps = num_steps + burn_in * order
        x = np.zeros(total_steps)
        x[:order] = rng.normal(0, noise_std, size=order)
        for t in range(order, total_steps):
            x[t] = non_lin(np.dot(c, x[t - order : t][::-1])) + rng.normal(0, noise_std)
        return x[burn_in * order :], c

    def _generate_all_series_parallel(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        seeds = rng.integers(0, 1_000_000_000, size=self.num_nodes)

        args = [
            (
                self.order,
                self.num_steps,
                self.alpha,
                self.non_linear,
                self.noise_std,
                self.unit_circle_sampling,
                int(seeds[i]),
                self.burn_in,
            )
            for i in range(self.num_nodes)
        ]
        with ProcessPoolExecutor() as executor:
            result = list(executor.map(NARDataset.generate_single_series, *zip(*args)))
            data, coeffs = zip(*result)
        return np.array(data).transpose(1, 0), np.array(coeffs)

    def get_hash(self) -> str:
        s = str(
            (
                self.order,
                self.num_nodes,
                self.num_steps,
                self.noise_std,
                self.alpha,
                self.non_linear,
                self.seed,
                self.unit_circle_sampling,
                self.burn_in,
            )
        )
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
