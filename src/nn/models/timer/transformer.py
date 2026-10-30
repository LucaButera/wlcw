from torch import nn, Tensor


class TimerLayer(nn.Module):
    def __init__(
        self,
        attention,
        hidden_size: int,
        ff_size: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(TimerLayer, self).__init__()
        ff_size = ff_size or 4 * hidden_size
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size, out_channels=ff_size, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=ff_size, out_channels=hidden_size, kernel_size=1
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = (
            nn.functional.relu if activation == "relu" else nn.functional.gelu
        )

    def forward(
        self, x: Tensor, n_vars: int, n_tokens: int, attn_mask: Tensor | None = None
    ) -> Tensor:
        new_x = self.attention(
            x,
            x,
            x,
            n_vars=n_vars,
            n_tokens=n_tokens,
            attn_mask=attn_mask,
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class TimerBlock(nn.Module):
    def __init__(
        self, attn_layers: list[nn.Module], norm_layer: nn.Module | None = None
    ):
        super(TimerBlock, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(
        self, x: Tensor, n_vars: int, n_tokens: int, attn_mask: Tensor | None = None
    ) -> Tensor:
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x = attn_layer(x, n_vars, n_tokens, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
