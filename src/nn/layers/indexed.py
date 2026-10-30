import math

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from tsl.nn import utils, maybe_cat_exog


class IndexedLinear(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        dropout: float = 0.0,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_nodes = n_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((n_nodes, out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((n_nodes, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.register_module("dropout", None)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(fan_in)
        w_bound = math.sqrt(3.0) * std
        nn.init.uniform_(self.weight, -w_bound, w_bound)
        if self.bias is not None:
            b_bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -b_bound, b_bound)

    def forward(self, x: Tensor, node_index: Tensor) -> Tensor:
        w = self.weight[node_index.squeeze(1)]  # [B, out, in]
        squeezed = x.dim() == 4
        if squeezed:
            x = x.squeeze(2)
        out = torch.bmm(x, w.transpose(1, 2))
        if self.bias is not None:
            b = self.bias[node_index.squeeze(1)]
            out = out + b.unsqueeze(1)
        if self.dropout is not None:
            out = self.dropout(out)
        if squeezed:
            out = out.unsqueeze(2)
        return out


class IndexedMLP(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        exog_size: int | None = None,
        n_layers: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        if exog_size is not None:
            input_size += exog_size

        layers = [
            IndexedLinear(
                n_nodes=n_nodes,
                in_features=input_size if i == 0 else hidden_size,
                out_features=hidden_size,
                bias=True,
                dropout=0.0,
            )
            for i in range(n_layers)
        ]
        self.mlp = nn.ModuleList(layers)
        self.activation = utils.get_layer_activation(activation)()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if output_size is not None:
            self.readout = IndexedLinear(
                n_nodes=n_nodes,
                in_features=hidden_size,
                out_features=output_size,
                bias=True,
                dropout=0.0,
            )
        else:
            self.register_parameter("readout", None)

    def reset_parameters(self) -> None:
        for module in self.mlp._modules.values():
            module.reset_parameters()
        if self.readout is not None:
            self.readout.reset_parameters()

    def forward(self, x: Tensor, node_index: Tensor, u: Tensor | None = None):
        x = maybe_cat_exog(x, u)
        for i, layer in enumerate(self.mlp):
            x = layer(x, node_index)
            x = self.activation(x)
            x = self.dropout(x)
        if self.readout is not None:
            x = self.readout(x, node_index)
        return x


class IndexedMLPDecoder(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        horizon: int = 1,
        n_layers: int = 1,
        receptive_field: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.receptive_field = receptive_field
        self.readout = IndexedMLP(
            n_nodes=n_nodes,
            input_size=receptive_field * input_size,
            hidden_size=hidden_size,
            output_size=output_size * horizon,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
        )
        self.rearrange = Rearrange("b n (h f) -> b h n f", f=output_size, h=horizon)

    def forward(self, h: Tensor, node_index: Tensor) -> Tensor:
        # h: [batches (steps) nodes features]
        if h.dim() == 4:
            # take last step representation
            h = rearrange(h[:, -self.receptive_field :], "b t n f -> b n (t f)")
        else:
            assert self.receptive_field == 1
        out = self.readout(h, node_index)
        return self.rearrange(out)
