from einops import rearrange, repeat
from torch import Tensor, nn
from tsl.nn.layers import Dense

from src.nn.layers.temporal import RNNWithInit, PatchTSTBackbone


class GPIModule(nn.Module):
    """Generative Process Identification (GPI) module G(·).

    Implements the decoupled function G: R^{C x d} -> R^{d_gpi} (Eq. 8a in
    'Why Do Time Series Models Need Long Context Windows?', Butera et al., 2026).
    Maps past context observations x_{t'-C:t'} (and optional static covariates)
    into a process-identifying representation used to condition the base predictor F.
    """

    def __init__(
        self,
        hidden_size: int,
        emb_size: int,
        history_input_size: int | None = None,
        static_input_size: int | None = None,
    ):
        super().__init__()

        self.history_input_size = (
            None if history_input_size == 0 else history_input_size
        )
        self.static_input_size = None if static_input_size == 0 else static_input_size
        assert any(
            input_size is not None
            for input_size in [
                self.history_input_size,
                self.static_input_size,
            ]
        ), "At least one of `history_input_size`, or `static_input_size` must be provided."

        self.hidden_size = hidden_size
        self.emb_size = emb_size

    def forward(
        self,
        h: Tensor | None = None,
        v: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError()


class RNNGPIModule(GPIModule):
    def __init__(
        self,
        hidden_size: int,
        emb_size: int,
        history_input_size: int | None = None,
        static_input_size: int | None = None,
        n_layers: int = 1,
        activation: str = "elu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__(
            hidden_size=hidden_size,
            emb_size=emb_size,
            history_input_size=history_input_size,
            static_input_size=static_input_size,
        )

        self.rnn = RNNWithInit(
            input_size=history_input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            return_only_last_state=True,
            cell="gru",
            dropout=dropout,
            layer_norm=layer_norm,
        )

        if static_input_size is not None:
            self.static_encoder = Dense(
                input_size=static_input_size,
                output_size=hidden_size,
                activation=activation,
                dropout=dropout,
            )
        else:
            self.register_module("static_encoder", None)

    def forward(
        self,
        h: Tensor | None = None,
        v: Tensor | None = None,
    ):
        if v is not None:
            hx = self.static_encoder(v)
            hx = repeat(
                hx,
                "b 1 f -> d b f",
                d=self.rnn.rnn.num_layers * (2 if self.rnn.rnn.bidirectional else 1),
            )
        else:
            hx = None
        o = self.rnn(h, hx=hx)
        return o


class PatchTSTGPIModule(GPIModule):
    def __init__(
        self,
        emb_size: int,
        history_input_size: int,
        history_length: int,
        static_input_size: int | None = None,
        hidden_size: int = 128,  # 16
        patch_len: int = 16,
        stride: int = 8,
        pad_sequence: bool = True,
        num_heads: int = 16,  # 4
        dim_feedforward: int = 256,  # 128
        dropout: float = 0.2,
        num_encoder_layers: int = 3,
    ):
        super().__init__(
            hidden_size=hidden_size,
            emb_size=emb_size,
            history_input_size=history_input_size,
            static_input_size=static_input_size,
        )
        assert (
            static_input_size is None or static_input_size == 0
        ), "Static variables not supported in PatchTSTGPIModule."

        self.backbone = PatchTSTBackbone(
            input_size=self.history_input_size,
            window=history_length,
            patch_len=patch_len,
            stride=stride,
            pad_sequence=pad_sequence,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_encoder_layers=num_encoder_layers,
            hidden_size=self.hidden_size,
            static_size=0,
        )
        self.readout = nn.Linear(
            self.hidden_size * self.backbone.patch_num,
            self.emb_size,
        )

    def forward(
        self,
        h: Tensor | None = None,
        v: Tensor | None = None,
    ):
        if v is not None:
            raise NotImplementedError(
                "Static variables not supported in PatchTSTGPIModule."
            )
        # process window with backbone, x: [b, t, f] -> [b, p, d]
        h = self.backbone(h.squeeze(2))  # assuming h is [b, t, 1, f]
        # flatten patches, x: [b, p, d] -> [b, p * d]
        h = rearrange(h, "b p d -> b (p d)")
        # readout projection, x: [b, p * d] -> [b, emb_d]
        h = self.readout(h)
        # add dummy node dimension
        h = h.unsqueeze(1)  # [b, 1, emb_d]
        return h
