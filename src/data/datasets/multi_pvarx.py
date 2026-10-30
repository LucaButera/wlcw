import hashlib
import os.path
from pathlib import Path

import numpy as np
import torch
from tsl import logger
from tsl.datasets import GaussianNoiseSyntheticDataset

from src.data.simulation.pvar import MultiPolyVARX


class MultiPVARXDataset(GaussianNoiseSyntheticDataset):
    def __init__(
        self,
        root: str,
        num_nodes: int,
        num_steps: int,
        nonlinearity: str | None = "tanh",
        num_features: int | None = 1,
        temporal_order: int | None = 2,
        coeff_bound: float | tuple[float, float] | None = 2.0,
        sigma_noise: float | list[float] | np.ndarray = 0.2,
        seed: int | None = 42,
        params: np.ndarray | list | None = None,
        feature_params: np.ndarray | list | None = None,
        dependent_features: bool = True,
        symmetric_dependency: bool = False,
        target_dims: int | list[int] | tuple | np.ndarray | None = None,
        exogenous_dims: int | list[int] | tuple | np.ndarray | None = None,
        params_as_exogenous: bool = False,
        name: str | None = None,
        cache: bool = True,
    ):
        assert seed is not None, (
            "Currently a None seed breaks reproducibility "
            "due to low-level implementation not using the global random state."
        )

        self.name = "MultiPVARX" if name is None else name
        self.seed = seed
        self.root = root
        if not isinstance(sigma_noise, float):
            sigma_noise = np.array(sigma_noise, dtype=np.float32)

        if params is not None:
            logger.warning(
                "Building filter from given parameters, some arguments might be ignored."
            )
            _filter = MultiPolyVARX.from_params(
                filter_params=torch.tensor(params, dtype=torch.float32),
                feature_params=torch.tensor(feature_params, dtype=torch.float32)
                if feature_params is not None
                else None,
                nonlinearity=nonlinearity,
            )
        else:
            logger.info("Building filter with standard initialization.")
            _filter = MultiPolyVARX(
                temporal_order=temporal_order,
                n_nodes=num_nodes,
                n_features=num_features,
                dependent_features=dependent_features,
                symmetric_dependency=symmetric_dependency,
                coeff_bound=coeff_bound,
                seed=self.seed,
                nonlinearity=nonlinearity,
            )

        super().__init__(
            num_features=_filter.n_features,
            num_nodes=num_nodes,
            num_steps=num_steps,
            connectivity=None,
            min_window=_filter.temporal_order,
            model=_filter,
            sigma_noise=sigma_noise,
            seed=self.seed,
            name=self.name,
        )
        if cache and not self.cache_file.exists():
            if not self.cache_file.parent.exists():
                self.cache_file.parent.mkdir(parents=False, exist_ok=False)
            self.save(filename=str(self.cache_file))

        # Rearrange the data between target and covariates
        target_dims, exogenous_dims = self._parse_feature_dims(
            target_dims, exogenous_dims
        )
        if exogenous_dims is not None:
            assert (
                target_dims is not None
            ), "target_dims should be provided if exogenous_dims is not None"
            self.add_exogenous("u", self.target[..., exogenous_dims])
        if target_dims is not None:
            self.set_target(self.target[..., target_dims])
            self.set_mask(self.mask[..., target_dims])
            self.add_exogenous("optimal_pred", self.optimal_pred[..., target_dims])
        if params_as_exogenous:
            v = self.model.weight.detach().cpu().clone().view(num_nodes, -1)
            self.add_covariate("v", v, pattern="n f")

    def _parse_feature_dims(
        self,
        target_dims: int | list[int] | tuple | np.ndarray | None,
        exogenous_dims: int | list[int] | tuple | np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        def convert_type(
            dims: int | list[int] | np.ndarray | None,
        ) -> np.ndarray | None:
            if dims is None:
                return dims
            elif isinstance(dims, int):
                dims = np.array([dims])
            else:
                dims = np.array(dims)
            # this also sorts the array
            dims = np.unique(dims)
            return dims

        def check_compatibility(dims: np.ndarray | None):
            if isinstance(dims, np.ndarray):
                assert np.all(
                    dims < self.n_channels
                ), f"dims {dims} should all be less than {self.n_channels}"
            elif dims is None:
                return
            else:
                raise RuntimeError("dims should be a list or np.ndarray")

        target_dims = convert_type(target_dims)
        exogenous_dims = convert_type(exogenous_dims)
        check_compatibility(target_dims)
        check_compatibility(exogenous_dims)
        if isinstance(target_dims, np.ndarray) and isinstance(
            exogenous_dims, np.ndarray
        ):
            assert (
                np.intersect1d(target_dims, exogenous_dims).size == 0
            ), "target_dims and exogenous_dims should not overlap"
        elif target_dims is None and exogenous_dims is not None:
            raise RuntimeError(
                "target_dims should be provided if exogenous_dims is not None"
            )
        return target_dims, exogenous_dims

    def get_hash(self) -> str:
        s = str(
            (
                self._num_nodes,
                self._num_steps,
                str(self.model.weight.detach().cpu().numpy().tolist()),
                str(self.model.features_weight.detach().cpu().numpy().tolist())
                if self.model.features_weight is not None
                else None,
                str(self.sigma_noise.tolist())
                if isinstance(self.sigma_noise, np.ndarray)
                else self.sigma_noise,
                self.seed,
            )
        )
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @property
    def cache_file(self) -> Path:
        return Path(self.root).joinpath(f"{self.name}_{self.get_hash()}.npz").absolute()

    def save(self, filename: str):
        if not hasattr(self, "target"):
            target, optimal_pred, mask = self.load_raw()
        else:
            target, optimal_pred, mask = self.target, self.optimal_pred, self.mask
        if not os.path.isabs(filename):
            this_dir = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(this_dir, filename)
        if not filename.endswith(".npz"):
            filename += ".npz"
        logger.info(f"Saving synthetic dataset to: {filename}")
        np.savez_compressed(
            filename, target=target, optimal_pred=optimal_pred, mask=mask
        )

    def load(self):
        if not self.cache_file.exists():
            return self.load_raw()
        else:
            filename = str(self.cache_file)
            if not filename.endswith(".npz"):
                filename += ".npz"
            if not os.path.isabs(filename):
                this_dir = os.path.dirname(os.path.realpath(__file__))
                filename = os.path.join(this_dir, filename)
            logger.warning(f"Loading synthetic dataset from: {filename}")
            content = np.load(filename)
            target = content["target"]
            optimal_pred = content["optimal_pred"]
            mask = content["mask"]
            return target, optimal_pred, mask


if __name__ == "__main__":
    from env import CACHE

    sd = 42
    n_nodes = 200
    n_feat = 3
    t_ord = 2
    # N
    s_noise = [0.4] * n_feat

    dataset = MultiPVARXDataset(
        root=str(CACHE / "MultiPVARX"),
        num_nodes=n_nodes,
        num_steps=5000,
        num_features=n_feat,
        temporal_order=t_ord,
        coeff_bound=2.0,
        sigma_noise=s_noise,
        seed=sd,
        params=None,
        feature_params=None,
        dependent_features=False,
        symmetric_dependency=False,
        cache=False,
        target_dims=0,
        exogenous_dims=[1, 2],
    )

    x, y, m = dataset.target, dataset.optimal_pred, dataset.mask
    u = dataset.u if hasattr(dataset, "u") else None

    import matplotlib.pyplot as plt

    f, axs = plt.subplots(
        min([dataset.n_nodes, 6]), 2 if u is None else 3, sharex=True, sharey=True
    )
    for i, ax in enumerate(axs):
        if i == 0:
            ax[0].set_title("Optimal pred")
            ax[1].set_title("Target")
            if u is not None:
                ax[2].set_title("Covariates")
        ax[0].plot(y[:100, i])
        ax[1].plot(x[:100, i])
        if u is not None:
            ax[2].plot(u[:100, i])
    plt.tight_layout()
    plt.show()
    for i in range(x.shape[-1]):
        plt.imshow(x[:1000, :, i].T, aspect="auto")
        plt.title(f"Target feature {i}")
        plt.grid()
        plt.show()
    if u is not None:
        for i in range(u.shape[-1]):
            plt.imshow(u[:1000, :, i].T, aspect="auto")
            plt.title(f"Covariate feature {i}")
            plt.grid()
            plt.show()
    print("done")
