import torch
from torch import Tensor

from src.nn.gpi import GPIModule
from src.nn.models.temporal.prototypes import GlobalTemporalModel
from src.nn.models.timesfm.configs import ForecastConfig
from src.nn.models.timesfm.module import TimesFM_2p5_200M_torch


class TimesFMModel(GlobalTemporalModel):
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
        assert input_size == 1, "TimesFM only supports univariate time series."
        assert exog_size == 0, "TimesFM does not support exogenous features."
        assert output_size in (
            None,
            1,
        ), "TimesFM only supports univariate time series."
        assert gpi is None, "TimesFM does not support GPI."
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            output_size=output_size,
            exog_size=exog_size,
            gpi=None,
            **kwargs,
        )

        self.module = TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        # Register the underlying PyTorch module so its parameters are visible to PyTorch/Lightning
        self.module_nn = self.module.model
        # Freeze parameters since TimesFM is used zero-shot and decode is run in no_grad mode
        for p in self.module_nn.parameters():
            p.requires_grad = False

        self.module.compile(
            ForecastConfig(
                max_context=window,
                max_horizon=horizon,
                normalize_inputs=False,
                use_continuous_quantile_head=False,
                force_flip_invariance=False,
                infer_is_positive=False,
                fix_quantile_crossing=False,
            )
        )
        assert self.emb_size == 0, "TimesFM does not support GPI."

    def temporal_processing(
        self,
        x: Tensor,
        u: Tensor | None = None,
        emb: Tensor | None = None,
        h: Tensor | None = None,
    ) -> Tensor:
        # Handle additional inputs
        assert emb is None, "TimesFM does not support embeddings."
        if h is not None:
            assert self.emb_size == 0, "Cannot use both h and emb."
            x = torch.cat([h, x], dim=1)
        x_squeezed = x.squeeze(-1)
        context_len = x_squeezed.shape[1]
        p = self.module.model.p
        pad_len = (p - (context_len % p)) % p
        if pad_len > 0:
            x_padded = torch.nn.functional.pad(x_squeezed, (pad_len, 0), value=0.0)
            mask = torch.cat(
                [
                    torch.ones(
                        (x_squeezed.shape[0], pad_len),
                        dtype=torch.bool,
                        device=x.device,
                    ),
                    torch.zeros_like(x_squeezed, dtype=torch.bool),
                ],
                dim=1,
            )
        else:
            x_padded = x_squeezed
            mask = torch.zeros_like(x_squeezed, dtype=torch.bool)

        point_forecast, quantile_forecast = self.module.compiled_decode(
            self.horizon, x_padded, mask
        )
        return point_forecast.unsqueeze(-1)
