import math
from collections import OrderedDict

import torch
from einops import rearrange
from torch import Tensor, nn
from tsl.nn import maybe_cat_exog
from tsl.nn.blocks import RNN
from tsl.nn.layers import Dense

from src.nn.associative_scan import associative_scan, linear_recurrence_op


class LayerNormGRUCell(nn.GRUCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.ln_i2h = nn.LayerNorm(2 * hidden_size)
        self.ln_h2h = nn.LayerNorm(2 * hidden_size)
        self.ln_cell_1 = nn.LayerNorm(hidden_size)
        self.ln_cell_2 = nn.LayerNorm(hidden_size)

    def forward(self, input: Tensor, hx: Tensor | None = None):
        if input.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        ret = self._gru_cell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            ret = ret.squeeze(0)

        return ret

    def _gru_cell(
        self,
        x: Tensor,
        h: Tensor,
        w_ih: Tensor,
        w_hh: Tensor,
        b_ih: Tensor | None = None,
        b_hh: Tensor | None = None,
    ):
        x_chunks = nn.functional.linear(input=x, weight=w_ih, bias=b_ih)
        h_chunks = nn.functional.linear(input=h, weight=w_hh, bias=b_hh)
        z, r = torch.sigmoid(
            self.ln_i2h(x_chunks[..., : self.hidden_size * 2])
            + self.ln_h2h(h_chunks[..., : self.hidden_size * 2])
        ).chunk(2, dim=-1)
        n = torch.tanh(
            self.ln_cell_1(x_chunks[..., self.hidden_size * 2 :])
            + r * self.ln_cell_2(h_chunks[..., self.hidden_size * 2 :])
        )
        h_t = (1 - z) * n + z * h
        return h_t


class LayerNormGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if bidirectional:
            raise NotImplementedError("Bidirectional LayerNormGRU is not implemented.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.cells = nn.ModuleList(
            [
                LayerNormGRUCell(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    bias,
                    device,
                    dtype,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, input: Tensor, hx: Tensor | None = None):
        if self.batch_first:
            input = input.transpose(0, 1)

        h_n = input.new_zeros(len(self.cells), input.shape[1], self.hidden_size)
        output = input
        for layer_idx, cell in enumerate(self.cells):
            h = hx[layer_idx] if hx is not None else None
            outputs = input.new_zeros(*input.shape[:2], self.hidden_size)
            for t in range(input.shape[0]):
                h = cell(output[t], h)
                outputs[t] = h
            output = outputs
            if self.dropout > 0.0 and layer_idx < len(self.cells) - 1:
                output = nn.functional.dropout(
                    output, p=self.dropout, training=self.training
                )
            h_n[layer_idx] = h

        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_n


class RNNWithInit(RNN):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        exog_size: int = None,
        output_size: int = None,
        n_layers: int = 1,
        return_only_last_state: bool = False,
        cell: str = "gru",
        bias: bool = True,
        dropout: float = 0.0,
        layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            exog_size=exog_size,
            output_size=output_size,
            n_layers=n_layers,
            return_only_last_state=return_only_last_state,
            cell=cell,
            bias=bias,
            dropout=dropout,
            **kwargs,
        )
        if layer_norm:
            # if layer_norm is True, override the cell type
            if cell == "gru":
                cell = LayerNormGRU
            elif cell == "lstm":
                raise NotImplementedError("LayerNormLSTM is not implemented.")
            else:
                raise NotImplementedError(f'layer norm "{cell}" cell not implemented.')
            self.rnn = cell(
                input_size=self.rnn.input_size,
                hidden_size=self.rnn.hidden_size,
                num_layers=self.rnn.num_layers,
                bias=self.rnn.bias,
                dropout=self.rnn.dropout,
            )

    def forward(
        self, x: Tensor, u: Tensor | None = None, hx: Tensor | None = None
    ) -> Tensor:
        """Process the input sequence :obj:`x` with optional exogenous variables
        :obj:`u`.

        Args:
            x (Tensor): Input data.
            u (Tensor): Exogenous data.

        Shapes:
            x: :math:`(B, T, N, F_x)` where :math:`B` is the batch dimension,
                :math:`T` is the number of time steps, :math:`N` is the number
                of nodes, and :math:`F_x` is the number of input features.
            u: :math:`(B, T, N, F_u)` or :math:`(B, T, F_u)` where :math:`B` is
                the batch dimension, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes (optional), and :math:`F_u` is
                the number of exogenous features.
        """
        # x: [batches, steps, nodes, features]
        x = maybe_cat_exog(x, u)
        b, *_ = x.size()
        x = rearrange(x, "b s n f -> s (b n) f")
        x, *_ = self.rnn(x, hx=hx)
        # [steps batches * nodes, features] -> [steps batches, nodes, features]
        x = rearrange(x, "s (b n) f -> b s n f", b=b)
        if self.return_only_last_state:
            x = x[:, -1]
        if self.readout is not None:
            return self.readout(x)
        return x


class PatchTSTBackbone(nn.Module):
    _layer_norm_eps: float = 1e-5

    def __init__(
        self,
        input_size: int,
        window: int,
        patch_len: int = 16,
        stride: int = 8,
        pad_sequence: bool = True,
        num_heads: int = 16,  # 4
        dim_feedforward: int = 256,  # 128
        dropout: float = 0.2,
        num_encoder_layers: int = 3,
        hidden_size: int = 128,  # 16
        static_size: int = 0,
    ) -> None:
        super().__init__()

        self.window = window
        self.patch_len = patch_len
        self.stride = stride
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.static_size = static_size

        self.patch_num = int((self.window - self.patch_len) / self.stride + 1)
        if pad_sequence:
            self.sequence_padding = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1
        else:
            self.sequence_padding = nn.Identity()

        # custom: avoid having all channels interact directly in patch projection
        if self.input_size > 1:
            self.input_encoder = nn.Linear(self.input_size, self.input_size)
        else:
            self.input_encoder = nn.Identity()

        self.patch_proj = nn.Linear(
            self.patch_len * self.input_size,
            self.hidden_size,
        )

        if self.static_size > 0 and self.static_size != self.hidden_size:
            self.static_proj = nn.Linear(self.static_size, self.hidden_size)
        else:
            self.static_proj = nn.Identity()

        self.positional_encoding = nn.Parameter(
            nn.init.uniform_(Tensor(self.patch_num, self.hidden_size), -0.02, 0.02),
            requires_grad=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                layer_norm_eps=self._layer_norm_eps,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(hidden_size, eps=self._layer_norm_eps),
        )

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.positional_encoding, -0.02, 0.02)
        self.patch_proj.reset_parameters()
        self.encoder.reset_parameters()

    def forward(self, x: Tensor, v: Tensor | None = None) -> Tensor:
        # first interact over feature dimension
        x = self.input_encoder(x)
        # prepare for patching, # x: [b, t, f] -> [b, f, t]
        x = rearrange(x, "b t f -> b f t")
        # x: [b, f, t] -> [b, f, p, l]
        x = self.sequence_padding(x).unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )
        # flatten features, x: [b, f, p, l] -> [b, p, f * l]
        x = rearrange(x, "b f p l -> b p (f l)")
        # project patches + positional encodings, x: [b, p, f * l] -> [b, p, d]
        x = self.patch_proj(x) + self.positional_encoding.unsqueeze(0)

        # add static features, x: [b, p, d] -> [b, p, d]
        if v is not None:
            x = x + self.static_proj(v).unsqueeze(1).expand(-1, self.patch_num, -1)

        # transformer encoder, # x: [b, p, d] -> [b, p, d]
        x = self.encoder(x)
        return x


class TimeMixing(nn.Module):
    def __init__(
        self,
        window: int,
        input_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm([window, input_size])
        self.fc = Dense(
            input_size=window, output_size=window, activation="relu", dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = rearrange(self.fc(rearrange(x, "b t f -> b f t")), "b f t -> b t f")
        return self.norm(x + x_res)


class FeatureMixing(nn.Module):
    def __init__(
        self,
        window: int,
        input_size: int,
        output_size: int,
        hidden_size: int,
        static_size: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if static_size > 0:
            self.fc_static = nn.Linear(static_size, output_size)
            input_size += output_size
        else:
            self.register_module("fc_static", None)

        self.norm = nn.LayerNorm([window, output_size])
        self.fc = nn.Sequential(
            Dense(
                input_size=input_size,
                output_size=hidden_size,
                activation="relu",
                dropout=dropout,
            ),
            Dense(
                input_size=hidden_size,
                output_size=output_size,
                activation=None,
                dropout=dropout,
            ),
        )
        self.out_proj = (
            nn.Linear(input_size, output_size)
            if input_size != output_size
            else nn.Identity()
        )

    def forward(self, x: Tensor, v: Tensor | None = None) -> Tensor:
        if self.fc_static is not None:
            v = self.fc_static(v).unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, v], dim=-1)
        x_proj = self.out_proj(x)
        x = self.fc(x)
        return self.norm(x_proj + x)


class MixerLayer(nn.Module):
    def __init__(
        self,
        window: int,
        input_size: int,
        output_size: int,
        hidden_size: int,
        static_size: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.time_mixing = TimeMixing(
            window=window,
            input_size=input_size,
            dropout=dropout,
        )
        self.feature_mixing = FeatureMixing(
            window=window,
            input_size=input_size,
            output_size=output_size,
            static_size=static_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

    def forward(self, x: Tensor, v: Tensor | None = None) -> Tensor:
        x = self.time_mixing(x)
        x = self.feature_mixing(x, v=v)
        return x


class ModernTCNReparamLargeKernelConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel_size: int,
        kernel_already_merged: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel_size = small_kernel_size
        # We assume the conv does not change the feature map size, so padding = k//2.
        # Otherwise, you may configure padding as you wish,
        # and change the padding of small_conv accordingly.
        if kernel_already_merged:
            self.lkb_reparam = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv1d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=kernel_size // 2,
                                dilation=1,
                                groups=groups,
                                bias=False,
                            ),
                        ),
                        ("bn", nn.BatchNorm1d(out_channels)),
                    ]
                )
            )
            if small_kernel_size is not None:
                assert (
                    small_kernel_size <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv1d(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=small_kernel_size,
                                    stride=stride,
                                    padding=small_kernel_size // 2,
                                    dilation=1,
                                    groups=groups,
                                    bias=False,
                                ),
                            ),
                            ("bn", nn.BatchNorm1d(out_channels)),
                        ]
                    )
                )

    def forward(self, x: Tensor):
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)
        return out

    @staticmethod
    def fuse_bn_to_conv(conv: nn.Conv1d, bn: nn.BatchNorm1d):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def get_equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
        eq_k, eq_b = self.fuse_bn_to_conv(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self.fuse_bn_to_conv(
                self.small_conv.conv, self.small_conv.bn
            )
            # The small kernel is padded to the same size as the large kernel.
            pad_len = (self.kernel_size - self.small_kernel_size) // 2
            pad_left = torch.zeros(*small_k.shape[:2], pad_len)
            pad_right = torch.zeros(*small_k.shape[:2], pad_len)
            small_k = torch.cat([pad_left, small_k, pad_right], dim=-1)
            # Combine the large kernel and small kernel.
            eq_b += small_b
            eq_k += small_k
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(
            in_channels=self.lkb_origin.conv.in_channels,
            out_channels=self.lkb_origin.conv.out_channels,
            kernel_size=self.lkb_origin.conv.kernel_size,
            stride=self.lkb_origin.conv.stride,
            padding=self.lkb_origin.conv.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.lkb_origin.conv.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")


class ModernTCNBlock(nn.Module):
    def __init__(
        self,
        large_kernel_size: int,
        small_kernel_size: int,
        hidden_size: int,
        ff_hidden_size_ratio: int,
        kernel_already_merged: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        ff_hidden_size = hidden_size * ff_hidden_size_ratio
        self.dw_and_ffn = nn.Sequential(
            ModernTCNReparamLargeKernelConv(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=large_kernel_size,
                stride=1,
                groups=hidden_size,
                small_kernel_size=small_kernel_size,
                kernel_already_merged=kernel_already_merged,
            ),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=ff_hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
            ),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Conv1d(
                in_channels=ff_hidden_size,
                out_channels=hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.dw_and_ffn(x)


class LRU(nn.Module):
    # Adapted from https://github.com/Gothos/LRU-pytorch/blob/main/LRU_pytorch/LRU.py
    # and https://github.com/NicolasZucchet/minimal-LRU/blob/main/lru/model.py
    # and https://github.com/i404788/s5-pytorch/blob/master/s5/s5_model.py

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.283,  # 2 * pi
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Stability parameters
        self.nu_log = nn.Parameter(
            torch.log(
                -0.5
                * torch.log(
                    torch.rand(self.hidden_size) * (r_max**2 - r_min**2) + r_min**2
                )
            )
        )
        self.theta_log = nn.Parameter(
            torch.log(max_phase * torch.rand(self.hidden_size))
        )
        self.gamma_log = nn.Parameter(
            torch.log(
                torch.sqrt(
                    1
                    - torch.abs(
                        torch.exp(
                            -torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log)
                        )
                    )
                    ** 2
                )
            )
        )
        # LRU parameters
        self.B = nn.Parameter(
            torch.randn((self.hidden_size, self.input_size), dtype=torch.cfloat)
            / math.sqrt(2 * self.input_size)
        )
        self.C = nn.Parameter(
            torch.randn((self.output_size, self.hidden_size), dtype=torch.cfloat)
            / math.sqrt(self.hidden_size)
        )
        self.D = nn.Parameter(
            torch.randn([self.output_size, self.input_size])
            / math.sqrt(self.input_size)
        )

    def forward(self, x: Tensor, h0: Tensor | None = None):
        # x: [b, t, f]
        # Compute LRU stability parameters
        lambdas = torch.exp(
            -torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log)
        ).tile(*x.shape[:2], 1)
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)
        # Compute input contribution
        u = torch.einsum("btf,hf->bth", x.to(torch.cfloat), self.B * gammas)

        if h0 is not None:
            u[:, 0] = h0 * lambdas[:, 0] + u[:, 0]

        _, states = associative_scan(linear_recurrence_op, (lambdas, u), axis=1)

        # Readout: out = C * h_t + D * x_t
        out = torch.einsum("bth,oh->bto", states, self.C).real + torch.einsum(
            "btf,of->bto", x, self.D
        )
        if out.isnan().any():
            print("oh-oh")
        return out


class LRUBlock(nn.Module):
    """Single layer, with one LRU module, GLU, dropout and layer norm"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.lru = LRU(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            **kwargs,
        )
        self.lin1 = nn.Linear(
            in_features=output_size,
            out_features=output_size,
        )
        self.lin2 = nn.Linear(
            in_features=output_size,
            out_features=output_size,
        )
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        inputs = x
        x = self.dropout(nn.functional.gelu(self.lru(self.norm(x), h0=h0)))
        x = self.dropout(self.lin1(x) * nn.functional.sigmoid(self.lin2(x)))  # GLU
        return x + inputs
