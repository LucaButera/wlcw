import torch
from einops import rearrange
from torch import nn, Tensor

from src.nn.gpi import GPIModule
from src.nn.layers.temporal import MixerLayer, FeatureMixing
from src.nn.models.temporal.prototypes import GlobalTemporalModel


class TSMixerModel(GlobalTemporalModel):
    """Based on https://github.com/ditschuk/pytorch-tsmixer/tree/main"""

    def __init__(
        self,
        input_size: int,
        horizon: int,
        window: int,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 64,
        gpi: GPIModule | None = None,
        num_blocks: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            output_size=output_size,
            exog_size=exog_size,
            gpi=gpi,
            **kwargs,
        )

        self.window = window
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_blocks = num_blocks

        self.fc_time = nn.Linear(self.window, self.horizon)
        self.fc_features = nn.Linear(self.hidden_size, self.output_size)

        if self.emb_size + self.input_size + self.exog_size > 1:
            self.feature_mixing = FeatureMixing(
                window=self.window,
                input_size=self.input_size + self.exog_size,
                output_size=self.hidden_size,
                static_size=self.emb_size,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
            )
        else:
            self.register_module("feature_mixing", None)

        self.mixer = nn.ModuleList(
            [
                MixerLayer(
                    input_size=(self.input_size + self.exog_size)
                    if (self.feature_mixing is None and i == 0)
                    else self.hidden_size,
                    output_size=self.hidden_size,
                    window=self.window,
                    hidden_size=self.hidden_size,
                    static_size=self.emb_size,
                    dropout=self.dropout,
                )
                for i in range(self.num_blocks)
            ]
        )

    def temporal_processing(
        self,
        x: Tensor,
        u: Tensor | None = None,
        emb: Tensor | None = None,
        h: Tensor | None = None,
    ) -> Tensor:
        if u is not None:
            x = torch.cat([x, u], dim=-1)
        # Apply conditional feature mixing to the input
        if self.feature_mixing is not None:
            x = self.feature_mixing(x, v=emb)
        # Process the concatenated data through the mixer layers
        for mixing_layer in self.mixer:
            x = mixing_layer(x, v=emb)
        # Final linear transformation to produce the forecast
        x = self.fc_features(x)
        x = rearrange(self.fc_time(rearrange(x, "b t f -> b f t")), "b f t -> b t f")
        return x
