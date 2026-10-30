import math

import torch
from torch import Tensor

from src.nn.gpi import GPIModule
from src.nn.models.temporal.prototypes import GlobalTemporalModel
from src.nn.models.timer.module import TimerXLModule


class TimerXLModel(GlobalTemporalModel):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        output_size: int | None = None,
        exog_size: int = 0,
        gpi: GPIModule | None = None,
        **kwargs,
    ) -> None:
        assert input_size == 1, "TimerXLModel only supports univariate time series."
        assert exog_size == 0, "TimerXLModel does not support exogenous features."
        assert output_size in (
            None,
            1,
        ), "TimerXLModel only supports univariate time series."
        assert gpi is None, "TimerXLModel does not support GPI."
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            output_size=output_size,
            exog_size=exog_size,
            gpi=None,
            **kwargs,
        )

        self.module = TimerXLModule.from_pretrained("timer_xl_pretrained")
        assert self.horizon <= self.module.patch_size, (
            "Horizon should be less than or equal to patch_size."
            f" Got horizon={self.horizon}, patch_size={self.module.patch_size}.",
        )
        assert self.emb_size == 0, "TimerXLModel does not support GPI."

    def _circular_pad(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        assert (
            t % self.module.patch_size != 0
        ), "Input length should not be a multiple of patch_size for padding."
        # Compute pad length (smallest multiple of patch_size > t)
        pad_len = self.module.patch_size - (t % self.module.patch_size)
        # Repeat enough times so we can slice extra elements
        pad = x.repeat(1, math.ceil(pad_len / t), 1)[:, -pad_len:, :]
        return torch.cat([pad, x], dim=1)

    def temporal_processing(
        self,
        x: Tensor,
        u: Tensor | None = None,
        emb: Tensor | None = None,
        h: Tensor | None = None,
    ) -> Tensor:
        # Handle additional inputs
        assert emb is None, "TimerXLModel does not support embeddings."
        if h is not None:
            assert self.emb_size == 0, "Cannot use both h and emb."
            x = torch.cat([h, x], dim=1)
        if x.shape[1] % self.module.patch_size != 0:
            x = self._circular_pad(x)
        outputs = self.module(x)
        if self.horizon == self.module.patch_size:
            return outputs[:, -self.module.patch_size :]
        else:
            return outputs[
                :, -self.module.patch_size : -self.module.patch_size + self.horizon
            ]
