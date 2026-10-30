import math

import torch
from torch import Tensor

from src.nn.gpi import GPIModule
from src.nn.models.temporal.prototypes import GlobalTemporalModel
from src.nn.models.chronos.model import Chronos2WrappedModel


class Chronos2Model(GlobalTemporalModel):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        window: int,
        output_size: int | None = None,
        exog_size: int = 0,
        gpi: GPIModule | None = None,
        **kwargs,
    ) -> None:
        assert input_size == 1, "Chronos only supports univariate time series."
        assert exog_size == 0, "Chronos does not support exogenous features."
        assert output_size in (
            None,
            1,
        ), "Chronos only supports univariate time series."
        assert gpi is None, "Chronos does not support GPI."
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            output_size=output_size,
            exog_size=exog_size,
            gpi=None,
            **kwargs,
        )

        self.module = Chronos2WrappedModel(horizon=horizon, context=window)
        self._num_output_patches = math.ceil(
            horizon / self.module.module.chronos_config.output_patch_size
        )
        assert self.emb_size == 0, "Chronos does not support GPI."

    def temporal_processing(
        self,
        x: Tensor,
        u: Tensor | None = None,
        emb: Tensor | None = None,
        h: Tensor | None = None,
    ) -> Tensor:
        # Handle additional inputs
        assert emb is None, "Chronos does not support embeddings."
        if h is not None:
            assert self.emb_size == 0, "Cannot use both h and emb."
            x = torch.cat([h, x], dim=1)
        quantiles = self.module(
            x.squeeze(-1), num_output_patches=self._num_output_patches
        ).quantile_preds
        medians = quantiles[:, quantiles.shape[1] // 2, : self.horizon].unsqueeze(-1)
        return medians
