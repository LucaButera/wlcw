import math

import torch
from torch import nn, Tensor
from tsl import logger

from src.utils import is_symmetric


class PolyVARX(nn.Module):
    def __init__(
        self,
        temporal_order: int,
        n_features: int = 1,
        dependent_features: bool = True,
        symmetric_dependency: bool = False,
        coeff_bound: float | tuple[float, float] | None = None,
        seed: int | None = None,
        reset_params: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.dependent_features = dependent_features
        self.symmetric_dependency = symmetric_dependency
        self.temporal_order = temporal_order
        self.seed = seed

        self._coeff_bound = self._get_coeff_bounds(coeff_bound, self.temporal_order)

        self.weight = nn.Parameter(Tensor(self.n_features, self.temporal_order))
        if self.dependent_features and n_features > 1:
            self.features_weight = nn.Parameter(
                torch.Tensor(self.n_features, self.n_features)
            )
            self._features_coeff_bound = self._get_coeff_bounds(
                coeff_bound,
                self.n_features,
            )
        else:
            self.register_parameter("features_weight", None)
        if reset_params:
            self.reset_parameters()

    @staticmethod
    def _get_coeff_bounds(
        bound: float | tuple[float, float] | None, n_features: int
    ) -> tuple[float, float]:
        if bound is None:
            bound = 1 / math.sqrt(n_features)
        if isinstance(bound, float):
            assert bound > 0
            bound = (-bound, bound)
        else:
            assert len(bound) == 2
        return bound

    def _make_feature_weights_symmetric(
        self, weight: Tensor, check: bool = True, dims: tuple[int, int] = (-2, -1)
    ) -> Tensor:
        with torch.no_grad():
            i, j = torch.triu_indices(self.n_features, self.n_features)
            weight[..., i, j] = weight.transpose(*dims)[..., i, j]
        if check:
            assert is_symmetric(weight), "Weight matrix is not symmetric."
        return weight

    def reset_parameters(self, generator: torch.Generator | None = None):
        if self.seed is not None and generator is None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        nn.init.uniform_(self.weight, *self._coeff_bound, generator=generator)
        if self.features_weight is not None:
            nn.init.uniform_(
                self.features_weight, *self._features_coeff_bound, generator=generator
            )
            if self.symmetric_dependency:
                self.features_weight = self._make_feature_weights_symmetric(
                    self.features_weight
                )

    @staticmethod
    def _parse_params(
        filter_params: Tensor,
        feature_params: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, dict[str, int | bool]]:
        if filter_params.ndim == 1:
            logger.warning(
                "PVAR filter parameters are the same for all features, "
                "adding dummy feature dimension."
            )
            filter_params = filter_params.unsqueeze(0)
        kwargs = {
            "temporal_order": filter_params.shape[-1],  # p
            "n_features": filter_params.shape[-2],  # f
        }
        if feature_params is not None:
            assert (
                feature_params.shape[-2]
                == feature_params.shape[-1]
                == filter_params.shape[-2]
            )  # f
            kwargs["dependent_features"] = True
            kwargs["symmetric_dependency"] = is_symmetric(feature_params)
        else:
            kwargs["dependent_features"] = False
            kwargs["symmetric_dependency"] = False
        return filter_params, feature_params, kwargs

    def _set_parameters(
        self,
        filter_params: Tensor,
        feature_params: Tensor | None,
    ):
        self.weight.data.copy_(filter_params)
        if feature_params is not None:
            self.features_weight.data.copy_(feature_params)

    @classmethod
    def from_params(
        cls,
        filter_params: Tensor,
        feature_params: Tensor | None = None,
    ):
        filter_params, feature_params, kwargs = cls._parse_params(
            filter_params, feature_params
        )

        model = cls(
            reset_params=False,
            **kwargs,
        )
        model._set_parameters(filter_params, feature_params)
        return model

    def forward(self, x: Tensor):
        assert x.shape[-3] >= self.temporal_order  # time steps
        assert x.shape[-1] == self.n_features  # node features

        # consider only the relevant time steps
        # [b, t>=p, n, f] -> [b, p, n, f]
        x = x[:, -self.temporal_order :]
        # apply corresponding temporal linear combination to each node and feature
        # [b, p, n, f] -> [b, n, f]
        h = torch.einsum("...pnf, fp -> ...nf", x, self.weight)
        # maybe add dependency on other features
        if self.features_weight is not None:
            h = torch.einsum("...nf, hf -> ...nh", h, self.features_weight)
        # add dummy temporal dimension as expected in the output
        # [b, n, f] -> [b, t=1, n, f]
        h = h.unsqueeze(-3)
        return h


class MultiPolyVARX(PolyVARX):
    def __init__(
        self,
        temporal_order: int,
        n_features: int = 1,
        n_nodes: int = 1,
        nonlinearity: str | None = "tanh",
        dependent_features: bool = True,
        symmetric_dependency: bool = False,
        coeff_bound: float | tuple[float, float] | None = None,
        seed: int | None = None,
        reset_params: bool = True,
    ):
        super().__init__(
            temporal_order=temporal_order,
            n_features=n_features,
            dependent_features=dependent_features,
            symmetric_dependency=symmetric_dependency,
            coeff_bound=coeff_bound,
            seed=seed,
            reset_params=False,
        )
        self.n_nodes = n_nodes
        if nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        elif nonlinearity == "sigmoid":
            self.nonlinearity = torch.sigmoid
        elif nonlinearity == "relu":
            self.nonlinearity = torch.relu
        elif nonlinearity is None:
            self.nonlinearity = None
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

        self._coeff_bound = self._get_coeff_bounds(coeff_bound, self.temporal_order)

        self.weight = nn.Parameter(
            Tensor(self.n_nodes, self.n_features, self.temporal_order)
        )

        if reset_params:
            self.reset_parameters()

    @classmethod
    def from_params(
        cls,
        filter_params: Tensor,
        feature_params: Tensor | None = None,
        nonlinearity: str | None = "tanh",
    ):
        filter_params, feature_params, kwargs = cls._parse_params(
            filter_params, feature_params
        )
        model = cls(
            nonlinearity=nonlinearity,
            reset_params=False,
            **kwargs,
        )
        model._set_parameters(filter_params, feature_params)
        return model

    @staticmethod
    def _parse_params(
        filter_params: Tensor,
        feature_params: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, dict[str, int | bool]]:
        if filter_params.ndim == 2:
            logger.warning(
                "PVAR filter parameters are the same for all features, "
                "adding dummy feature dimension."
            )
            filter_params = filter_params.unsqueeze(1)
        kwargs = {
            "temporal_order": filter_params.shape[-1],  # p
            "n_features": filter_params.shape[-2],  # f
            "n_nodes": filter_params.shape[0],  # n
        }
        if feature_params is not None:
            assert (
                feature_params.shape[-2]
                == feature_params.shape[-1]
                == filter_params.shape[-2]
            )  # f
            kwargs["dependent_features"] = True
            kwargs["symmetric_dependency"] = is_symmetric(feature_params)
        else:
            kwargs["dependent_features"] = False
            kwargs["symmetric_dependency"] = False
        return filter_params, feature_params, kwargs

    def forward(self, x: Tensor):
        assert (
            x.shape[-2] == self.n_nodes
        ), f"Expected {self.n_nodes} nodes, got {x.shape[-2]}."
        assert (
            x.shape[-3] >= self.temporal_order
        ), f"Expected at least {self.temporal_order} time steps, got {x.shape[-3]}."
        assert (
            x.shape[-1] == self.n_features
        ), f"Expected {self.n_features} features, got {x.shape[-1]}."

        # consider only the relevant time steps
        # [b, t>=p, n, f] -> [b, p, n, f]
        x = x[:, -self.temporal_order :]
        # apply corresponding temporal linear combination to each node and feature
        # [b, p, n, f] -> [b, n, f]
        h = torch.einsum("...pnf, nfp -> ...nf", x, self.weight)
        # maybe add dependency on other features
        if self.features_weight is not None:
            h = torch.einsum("...nf, hf -> ...nh", h, self.features_weight)
        # apply nonlinearity if specified
        if self.nonlinearity is not None:
            h = self.nonlinearity(h)
        # add dummy temporal dimension as expected in the output
        # [b, n, f] -> [b, t=1, n, f]
        h = h.unsqueeze(-3)
        return h
