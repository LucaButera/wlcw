from torch import nn, Tensor
from tsl.nn import maybe_cat_exog
from tsl.nn.blocks import MLPDecoder
from tsl.nn.models import BaseModel

from src.nn.gpi import GPIModule


class TemporalModel(BaseModel):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: int | None = None,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 32,
        gpi: GPIModule | None = None,
        activation: str = "elu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size
        self.activation = activation
        self.dropout = dropout
        # GPI MODULE
        if gpi is not None:
            self.gpi = gpi
        else:
            self.register_module("gpi", None)
        # ENCODER
        self.encoder_input = input_size + exog_size
        self.encoder = nn.Linear(self.encoder_input, hidden_size)
        if self.dropout > 0:
            self.encoder = nn.Sequential(self.encoder, nn.Dropout(p=self.dropout))
        # DECODER
        self.decoder_input = hidden_size
        self.decoder = MLPDecoder(
            input_size=self.decoder_input,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            horizon=self.horizon,
            activation=self.activation,
            dropout=self.dropout,
        )

    def forward(
        self,
        x: Tensor,
        u: Tensor | None = None,
        node_index: Tensor | None = None,
        h: Tensor | None = None,
        v: Tensor | None = None,
        emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        # x: [batches steps nodes features]
        x = maybe_cat_exog(x, u)
        if self.gpi is not None and emb is None:
            emb = self.gpi(h=h, v=v)
        elif emb is not None:
            assert (
                not self.training
            ), "External embeddings are not supported in training."

        # ENCODER   ###########################################################
        x = self.encoder(x)

        # TEMPORAL PROCESSING   ####################################
        out = self.temporal_processing(
            x=x,
            emb=emb,
        )
        # DECODER   ###########################################################
        out = self.decoder(out)
        return out, emb

    def temporal_processing(
        self,
        x: Tensor,
        emb: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError(
            "The 'temporal_processing' method must be implemented in the subclass."
        )


class GlobalTemporalModel(BaseModel):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 32,
        gpi: GPIModule | None = None,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size
        # GPI MODULE
        if gpi is not None:
            self.gpi = gpi
        else:
            self.register_module("gpi", None)

    @property
    def emb_size(self) -> int:
        return self.gpi.emb_size if self.gpi is not None else 0

    def forward(
        self,
        x: Tensor,
        u: Tensor | None = None,
        node_index: Tensor | None = None,
        h: Tensor | None = None,
        v: Tensor | None = None,
        emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.gpi is not None and emb is None:
            emb = self.gpi(h=h, v=v)
        elif emb is not None:
            assert (
                not self.training
            ), "External embeddings are not supported in training."

        # TEMPORAL PROCESSING   ####################################
        out = self.temporal_processing(
            x=x.squeeze(2),
            u=u.squeeze(2) if u is not None else None,
            emb=emb.squeeze(1) if emb is not None else None,
            h=h.squeeze(2) if h is not None else None,
        ).unsqueeze(2)
        return out, emb

    def temporal_processing(
        self,
        x: Tensor,
        u: Tensor | None = None,
        emb: Tensor | None = None,
        h: Tensor | None = None,
    ) -> Tensor:
        # x: [b, t, f], u: [b, t, f], emb: [b, e]
        raise NotImplementedError(
            "The 'temporal_processing' method must be implemented in the subclass."
        )
