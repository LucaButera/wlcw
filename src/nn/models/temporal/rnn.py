import torch
from einops import repeat
from torch import Tensor, nn
from tsl.nn import maybe_cat_exog

from src.nn.gpi import GPIModule
from src.nn.layers.indexed import IndexedLinear, IndexedMLPDecoder
from src.nn.layers.temporal import RNNWithInit
from src.nn.models.temporal.prototypes import TemporalModel


class RNNModel(TemporalModel):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: int | None = None,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 32,
        gpi: GPIModule | None = None,
        n_layers: int = 1,
        activation: str = "elu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        local: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            gpi=gpi,
            activation=activation,
            dropout=dropout,
        )
        self.local = local

        if self.local:
            assert self.gpi is None, "GPI modules are not supported with 'local' RNNs."
            self.rnn = nn.ModuleList(
                RNNWithInit(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    n_layers=n_layers,
                    return_only_last_state=True,
                    cell="gru",
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
                for _ in range(n_nodes)
            )
            self.encoder = IndexedLinear(
                self.n_nodes, self.encoder_input, hidden_size, dropout=self.dropout
            )
            self.decoder = IndexedMLPDecoder(
                n_nodes=self.n_nodes,
                input_size=self.decoder_input,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                horizon=self.horizon,
                activation=self.activation,
                dropout=self.dropout,
            )
        else:
            self.rnn = RNNWithInit(
                input_size=hidden_size,
                hidden_size=hidden_size,
                n_layers=n_layers,
                return_only_last_state=True,
                cell="gru",
                dropout=dropout,
                layer_norm=layer_norm,
            )
            if self.gpi is not None and self.gpi.emb_size != self.rnn.rnn.hidden_size:
                raise ValueError(
                    "Embedding size must match hidden size when using GPI with RNN."
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
        if self.local:
            # x: [batches steps nodes features]
            assert (
                emb is None
            ), "External embeddings are not supported with 'local' RNNs."
            assert node_index is not None, "node_index must be provided for local RNNs."
            x = maybe_cat_exog(x, u)
            x = self.encoder(x, node_index)
            out = torch.empty(x.shape[0], x.shape[2], self.hidden_size, device=x.device)
            for i, rnn in enumerate(self.rnn):
                mask = node_index.squeeze(1) == i
                if mask.any():
                    xi = x[mask]
                    out_i = rnn(xi, hx=None)
                    out[mask] = out_i
            out = self.decoder(out, node_index)
            return out, emb
        else:
            return super().forward(
                x=x,
                u=u,
                node_index=node_index,
                h=h,
                v=v,
                emb=emb,
            )

    def temporal_processing(
        self,
        x: Tensor,
        emb: Tensor | None = None,
    ) -> Tensor:
        if emb is not None:
            hx = repeat(
                emb,
                "b 1 f -> d b f",
                d=self.rnn.rnn.num_layers * (2 if self.rnn.rnn.bidirectional else 1),
            )
        else:
            hx = None
        return self.rnn(x, hx=hx)
