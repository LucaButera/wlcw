import torch
from einops import rearrange
from torch import nn, Tensor

from src.nn.gpi import GPIModule
from src.nn.layers.temporal import LRUBlock
from src.nn.models.temporal.prototypes import GlobalTemporalModel


class LRUModel(GlobalTemporalModel):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        window: int,
        output_size: int | None = None,
        exog_size: int = 0,
        num_blocks: int = 6,
        hidden_size: int = 64,
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
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.window = window

        if self.emb_size > 0 and self.emb_size != 2 * self.hidden_size:
            self.emb_projection = nn.Linear(
                in_features=self.emb_size,
                out_features=2 * self.hidden_size,  # double size for complex embeddings
            )
        else:
            self.emb_projection = nn.Identity()

        self.input_encoder = nn.Linear(
            in_features=self.input_size + self.exog_size,
            out_features=self.hidden_size,
        )
        self.backbone = nn.ModuleList(
            [
                LRUBlock(
                    input_size=self.hidden_size,
                    output_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    dropout=self.dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.readout = nn.Linear(self.hidden_size, self.horizon * self.output_size)

    def temporal_processing(
        self,
        x: Tensor,
        u: Tensor | None = None,
        emb: Tensor | None = None,
        h: Tensor | None = None,
    ) -> Tensor:
        if u is not None:
            x = torch.cat([x, u], dim=-1)
        if self.emb_size > 0:
            emb = torch.view_as_complex(
                self.emb_projection(emb).view(emb.shape[0], self.hidden_size, 2)
            )
        # input encoding, x: [b, t, f] -> [b, t, d]
        x = self.input_encoder(x)
        for block in self.backbone:
            # x: [b, t, d] -> [b, t, d]
            x = block(x, h0=emb)
        # readout projection, x: [b, t, d] -> [b, h * o]
        x = self.readout(x[..., -1, :])  # take the last time step
        # reshape to output shape, x: [b, h * o] -> [b, h, o]
        x = rearrange(x, "b (o h) -> b h o", h=self.horizon, o=self.output_size)
        return x
