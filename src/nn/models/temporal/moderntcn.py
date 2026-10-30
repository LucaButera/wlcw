import math

import torch
from einops import rearrange
from torch import nn, Tensor

from src.nn.gpi import GPIModule
from src.nn.layers.temporal import ModernTCNBlock
from src.nn.models.temporal.prototypes import GlobalTemporalModel


class ModernTCNModel(GlobalTemporalModel):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        window: int,
        output_size: int | None = None,
        exog_size: int = 0,
        patch_len: int = 8,
        stride: int = 4,
        downsample_ratio: int = 2,
        ff_hidden_size_ratio: int = 8,
        num_stages: int = 4,
        num_blocks: int = 1,
        large_kernel_size: int = 51,
        small_kernel_size: int = 5,
        hidden_size: int = 64,
        kernel_already_merged: bool = False,
        dropout: float = 0.1,
        gpi: GPIModule | None = None,
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
        assert num_stages > 0, "num_stages should be greater than 0"
        self.patch_len = patch_len
        self.stride = stride
        self.downsample_ratio = downsample_ratio
        self.num_stages = num_stages
        self.hidden_size = hidden_size
        self.window = window

        if self.emb_size > 0 and self.emb_size != self.hidden_size:
            self.emb_projection = nn.Linear(
                in_features=self.emb_size,
                out_features=self.hidden_size,
            )
        else:
            self.emb_projection = nn.Identity()

        # stem layer & down sampling layers
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv1d(
                    self.input_size + self.exog_size,
                    self.hidden_size,
                    kernel_size=self.patch_len,
                    stride=self.stride,
                ),
                nn.BatchNorm1d(self.hidden_size),
            )
        )
        for i in range(self.num_stages - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(self.hidden_size),
                    nn.Conv1d(
                        self.hidden_size,
                        self.hidden_size,
                        kernel_size=self.downsample_ratio,
                        stride=self.downsample_ratio,
                    ),
                )
            )
        # backbone
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            self.stages.append(
                nn.Sequential(
                    *(
                        ModernTCNBlock(
                            large_kernel_size=large_kernel_size,
                            small_kernel_size=small_kernel_size,
                            hidden_size=self.hidden_size,
                            ff_hidden_size_ratio=ff_hidden_size_ratio,
                            kernel_already_merged=kernel_already_merged,
                            dropout=dropout,
                        )
                        for _ in range(num_blocks)
                    )
                )
            )
        # readout layer
        readout_input_size = self.hidden_size * math.ceil(
            (self.window // self.stride)
            / pow(self.downsample_ratio, (self.num_stages - 1))
        )
        self.readout = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(readout_input_size, self.horizon * self.output_size),
        )

    def backbone_forward(self, x: Tensor, emb: Tensor | None = None) -> Tensor:
        # x: [b, f, t], emb: [b, emb_size]
        for i in range(self.num_stages):
            t = x.shape[-1]
            if i == 0:
                if self.patch_len != self.stride:
                    # stem layer padding
                    pad_len = self.patch_len - self.stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if t % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (t % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)
            if i == 0 and self.emb_size > 0:
                x = x + self.emb_projection(emb).unsqueeze(-1)
            x = self.stages[i](x)
        return x

    def temporal_processing(
        self,
        x: Tensor,
        u: Tensor | None = None,
        emb: Tensor | None = None,
        h: Tensor | None = None,
    ) -> Tensor:
        if u is not None:
            x = torch.cat([x, u], dim=-1)
        # process window with backbone, x: [b, t, f] -> [b, d, p]
        x = self.backbone_forward(rearrange(x, "b t f -> b f t"), emb=emb)
        # readout projection, x: [b, d, p] -> [b, h * o]
        x = self.readout(x)
        # reshape to output shape, x: [b, h * o] -> [b, h, o]
        x = rearrange(x, "b (o h) -> b h o", h=self.horizon, o=self.output_size)
        return x
