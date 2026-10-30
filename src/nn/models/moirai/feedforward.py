from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
import math


class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * in_dim
        out_dim = out_dim or in_dim

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bias = bias
        self.ffn_dropout_p = ffn_dropout_p

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout1 = nn.Dropout(ffn_dropout_p)
        self.dropout2 = nn.Dropout(ffn_dropout_p)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,  # "... in_dim" | float
    ) -> torch.Tensor:  # "... out_dim" | float
        x = self._in_proj(x)
        return self.dropout2(self.fc2(self.dropout1(x)))

    def _in_proj(
        self,
        x: torch.Tensor,  # "... in_dim" | float
    ) -> torch.Tensor:  # "... out_dim" | float
        return self.activation(self.fc1(x))


class GatedLinearUnitFeedForward(FeedForward):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__(
            in_dim,
            hidden_dim=hidden_dim or self.adjust_hidden_dim(4 * in_dim),
            out_dim=out_dim,
            activation=activation,
            bias=bias,
            ffn_dropout_p=ffn_dropout_p,
        )
        self.fc_gate = nn.Linear(self.in_dim, self.hidden_dim, bias=self.bias)

    @staticmethod
    def adjust_hidden_dim(dim):
        return (int(dim * 2 / 3) + 7) // 8 * 8

    def _in_proj(
        self,
        x: torch.Tensor,  # "... in_dim" | float
    ) -> torch.Tensor:  # "... out_dim" | float
        return self.activation(self.fc_gate(x)) * self.fc1(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dims,
        hidden_dims,
        output_dims,
    ):
        super(ResidualBlock, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        # Hidden Layer
        self.hidden_layer = nn.Linear(input_dims, hidden_dims)
        self.silu = nn.SiLU()

        # Output Layer
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        # Residual Layer
        self.residual_layer = nn.Linear(input_dims, output_dims)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        hidden = self.hidden_layer(x)
        hidden = self.silu(hidden)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual
