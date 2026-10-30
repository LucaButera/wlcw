import abc
import math

import torch
from einops import repeat, rearrange
from torch import nn, Tensor

from src.nn.models.timer.projection import QueryKeyProjection, RotaryProjection


class TimerMultivariateMask:
    def __init__(self, B: int, n_vars: int, n_tokens: int, device: str = "cpu"):
        mask_shape = [B, 1, n_tokens, n_tokens]
        with torch.no_grad():
            self._mask1 = torch.ones((n_vars, n_vars), dtype=torch.bool, device=device)
            self._mask2 = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1
            )
            self._mask = torch.kron(self._mask1, self._mask2)

    @property
    def mask(self):
        return self._mask


class AttentionBias(nn.Module, abc.ABC):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    @abc.abstractmethod
    def forward(self, query_id: Tensor, kv_id: Tensor) -> Tensor: ...


class BinaryAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int):
        super().__init__(dim, num_heads)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)

    def forward(self, query_id: Tensor, kv_id: Tensor) -> Tensor:
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        weight = rearrange(self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = ~ind * weight[:1] + ind * weight[1:]
        return bias


class TimeAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = True,
        scale=None,
        dropout: float = 0.1,
        hidden_size: int = 512,
        num_heads: int = 8,
        max_len: int = 100,
    ):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(dropout)
        self.qk_proj = QueryKeyProjection(
            dim=hidden_size,
            num_heads=num_heads,
            proj_layer=RotaryProjection,
            kwargs=dict(max_len=max_len),
            partial_factor=(0.0, 0.5),
        )
        self.attn_bias = BinaryAttentionBias(dim=hidden_size, num_heads=num_heads)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Tensor | None,
        n_vars: int,
        n_tokens: int,
    ) -> Tensor:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens * n_vars)
        seq_id = repeat(seq_id, "n -> b h n", b=B, h=H)

        queries, keys = self.qk_proj(queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1.0 / math.sqrt(E)

        var_id = repeat(torch.arange(n_vars), "C -> (C n_tokens)", n_tokens=n_tokens)
        var_id = repeat(var_id, "L -> b h L", b=B, h=1).to(queries.device)

        attn_bias = self.attn_bias(var_id, var_id)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TimerMultivariateMask(
                    B, n_vars, n_tokens, device=queries.device
                )
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        scores = torch.einsum("bhle,bhse->bhls", queries, keys)
        scores += attn_mask
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        hidden_size: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (hidden_size // n_heads)
        d_values = d_values or (hidden_size // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(hidden_size, d_keys * n_heads)
        self.key_projection = nn.Linear(hidden_size, d_keys * n_heads)
        self.value_projection = nn.Linear(hidden_size, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, hidden_size)
        self.n_heads = n_heads

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Tensor | None,
        n_vars: int | None = None,
        n_tokens: int | None = None,
    ) -> Tensor:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            n_vars=n_vars,
            n_tokens=n_tokens,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)
