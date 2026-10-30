# Adapted from GluonTS
# https://ts.gluon.ai/stable/_modules/gluonts/
# torch/model/patch_tst/module.html#PatchTSTModel
import torch
from einops import rearrange
from torch import nn, Tensor

from src.nn.gpi import GPIModule
from src.nn.layers.temporal import PatchTSTBackbone
from src.nn.models.temporal.prototypes import GlobalTemporalModel


class PatchTSTModel(GlobalTemporalModel):
    """
    Module implementing the PatchTST model. https://arxiv.org/abs/2211.14730
    """

    _layer_norm_eps: float = 1e-5

    def __init__(
        self,
        input_size: int,
        horizon: int,
        window: int,
        patch_len: int = 16,
        stride: int = 8,
        pad_sequence: bool = True,
        num_heads: int = 16,  # 4
        dim_feedforward: int = 256,  # 128
        dropout: float = 0.2,
        num_encoder_layers: int = 3,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 128,  # 16
        gpi: GPIModule | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            output_size=output_size,
            exog_size=exog_size,
            gpi=gpi,
            **kwargs,
        )

        self.hidden_size = hidden_size

        self.backbone = PatchTSTBackbone(
            input_size=self.input_size + self.exog_size,
            window=window,
            patch_len=patch_len,
            stride=stride,
            pad_sequence=pad_sequence,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_encoder_layers=num_encoder_layers,
            hidden_size=self.hidden_size,
            static_size=self.emb_size,
        )
        self.readout = nn.Linear(
            self.hidden_size * self.backbone.patch_num,
            self.horizon * self.output_size,
        )

    def reset_parameters(self) -> None:
        self.backbone.reset_parameters()
        self.readout.reset_parameters()

    def temporal_processing(
        self,
        x: Tensor,
        u: Tensor | None = None,
        emb: Tensor | None = None,
        h: Tensor | None = None,
    ) -> Tensor:
        if u is not None:
            x = torch.cat([x, u], dim=-1)
        # process window with backbone, x: [b, t, f] -> [b, p, d]
        x = self.backbone(x, emb)
        # flatten patches, x: [b, p, d] -> [b, p * d]
        x = rearrange(x, "b p d -> b (p d)")
        # readout projection, x: [b, p * d] -> [b, h * o]
        x = self.readout(x)
        # reshape to output shape, x: [b, h * o] -> [b, h, o]
        x = rearrange(x, "b (o h) -> b h o", h=self.horizon, o=self.output_size)
        return x
