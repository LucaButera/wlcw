from functools import partial
import math
from typing import Callable

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from src.nn.models.moirai.feedforward import FeedForward, GatedLinearUnitFeedForward
from src.nn.models.moirai.attention import AttentionBias, QueryKeyProjection


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        bias: bool = True,
        norm_layer: type[nn.Module] | partial[nn.Module] | None = nn.LayerNorm,
        softmax_scale: float | None = None,
        attn_dropout_p: float = 0.0,
        var_attn_bias: Callable[[], AttentionBias] | None = None,
        time_attn_bias: Callable[[], AttentionBias] | None = None,
        var_qk_proj: Callable[[], QueryKeyProjection] | None = None,
        time_qk_proj: Callable[[], QueryKeyProjection] | None = None,
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.heads_per_group = num_heads // num_groups
        self.var_attn_bias = var_attn_bias() if var_attn_bias is not None else None
        self.time_attn_bias = time_attn_bias() if time_attn_bias is not None else None
        self.var_qk_proj = var_qk_proj() if var_qk_proj is not None else None
        self.time_qk_proj = time_qk_proj() if time_qk_proj is not None else None

        self.softmax_scale = softmax_scale or 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.v_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.q_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.attn_dropout_p = attn_dropout_p
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _get_var_id(
        self,
        query: torch.Tensor,  # "*batch group hpg q_len dim" | float
        key: torch.Tensor,  # "*batch group hpg kv_len dim" | float
        query_var_id: torch.Tensor | None,  # "*batch q_len" | int
        kv_var_id: torch.Tensor | None,  # "*batch kv_len" | int
    ) -> tuple[
        torch.Tensor | None,  # "*batch #group #hpg q_len" | int
        torch.Tensor | None,  # "*batch #group #hpg kv_len" | int
    ]:
        if self.var_attn_bias is not None or self.var_qk_proj is not None:
            if query_var_id is None:
                query_var_id = repeat(
                    torch.zeros((), device=query.device, dtype=torch.long),
                    f" -> {' '.join(map(str, query.shape[:-4]))} 1 1 {query.shape[-2]}",
                )
            else:
                query_var_id = rearrange(query_var_id, "... q_len -> ... 1 1 q_len")

            if kv_var_id is None:
                kv_var_id = repeat(
                    torch.zeros((), device=key.device, dtype=torch.long),
                    f" -> {' '.join(map(str, key.shape[:-4]))} 1 1 {key.shape[-2]}",
                )
            else:
                kv_var_id = rearrange(kv_var_id, "... kv_len -> ... 1 1 kv_len")

        return query_var_id, kv_var_id

    def _get_time_id(
        self,
        query: torch.Tensor,  # "*batch group hpg q_len dim" | float
        key: torch.Tensor,  # "*batch group hpg kv_len dim" | float
        query_time_id: torch.Tensor | None,  # "*batch q_len" | int
        kv_time_id: torch.Tensor | None,  # "*batch kv_len" | int
    ) -> tuple[
        torch.Tensor | None,  # "*batch 1 1 q_len" | int
        torch.Tensor | None,  # "*batch 1 1 kv_len" | int
    ]:
        if self.time_attn_bias is not None or self.time_qk_proj is not None:
            if query_time_id is None:
                query_time_id = repeat(
                    torch.arange(
                        query.shape[-2], device=query.device, dtype=torch.long
                    ),
                    f"q_len -> {' '.join(map(str, query.shape[:-4]))} 1 1 q_len",
                )
            else:
                query_time_id = rearrange(query_time_id, "... q_len -> ... 1 1 q_len")

            if kv_time_id is None:
                kv_time_id = repeat(
                    torch.arange(key.shape[-2], device=key.device, dtype=torch.long),
                    f"kv_len -> {' '.join(map(str, key.shape[:-4]))} 1 1 kv_len",
                )
            else:
                kv_time_id = rearrange(kv_time_id, "... kv_len-> ... 1 1 kv_len")

        return query_time_id, kv_time_id

    def _update_attn_mask(
        self,
        attn_mask: torch.Tensor,  # "*batch q_len kv_len" | bool
        query: torch.Tensor,  # "*batch group hpg q_len dim" | float
        key: torch.Tensor,  # "*batch group hpg kv_len dim" | float
        query_var_id: torch.Tensor | None = None,  # "*batch 1 1 q_len" | int
        kv_var_id: torch.Tensor | None = None,  # "*batch 1 1 kv_len" | int
        query_time_id: torch.Tensor | None = None,  # "*batch 1 1 q_len" | int
        kv_time_id: torch.Tensor | None = None,  # "*batch 1 1 kv_len" | int
    ) -> torch.Tensor | None:  # "*batch #group #hpg q_len kv_len" | float | bool
        if attn_mask is not None:
            attn_mask = rearrange(
                attn_mask,
                "... q_len kv_len -> ... 1 1 q_len kv_len",
            )

        attn_bias = 0
        if self.var_attn_bias is not None:
            attn_bias = attn_bias + self.var_attn_bias(
                query,
                key,
                query_id=query_var_id,
                kv_id=kv_var_id,
            )

        if self.time_attn_bias is not None:
            attn_bias = attn_bias + self.time_attn_bias(
                query,
                key,
                query_id=query_time_id,
                kv_id=kv_time_id,
            )

        attn_mask = (
            attn_mask
            if isinstance(attn_bias, int)
            else (
                attn_bias
                if attn_mask is None
                else attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
            )
        )
        return attn_mask

    def _qk_proj(
        self,
        query: torch.Tensor,  # "*batch group hpg q_len dim" | float
        key: torch.Tensor,  # "*batch group hpg kv_len dim" | float
        query_var_id: torch.Tensor | None = None,  # "*batch #group #hpg q_len" | int
        kv_var_id: torch.Tensor | None = None,  # "*batch #group #hpg kv_len" | int
        query_time_id: torch.Tensor | None = None,  # "*batch #group #hpg q_len" | int
        kv_time_id: torch.Tensor | None = None,  # "*batch #group #hpg kv_len" | int
    ) -> tuple[
        torch.Tensor,  # "*batch group hpg q_len dim" | float
        torch.Tensor,  # "*batch group hpg kv_len dim" | float
    ]:
        if self.var_qk_proj is not None:
            query, key = self.var_qk_proj(
                query, key, query_id=query_var_id, kv_id=kv_var_id
            )

        if self.time_qk_proj is not None:
            query, key = self.time_qk_proj(
                query, key, query_id=query_time_id, kv_id=kv_time_id
            )

        return query, key

    def forward(
        self,
        query: torch.Tensor,  # "*batch q_len dim" | float
        key: torch.Tensor,  # "*batch kv_len dim" | float
        value: torch.Tensor,  # "*batch kv_len dim" | float
        attn_mask: torch.Tensor | None = None,  # "*batch q_len kv_len" | bool
        query_var_id: torch.Tensor | None = None,  # "*batch q_len" | int
        kv_var_id: torch.Tensor | None = None,  # "*batch kv_len" | int
        query_time_id: torch.Tensor | None = None,  # "*batch q_len" | int
        kv_time_id: torch.Tensor | None = None,  # "*batch kv_len" | int
    ) -> torch.Tensor:  # "*batch q_len dim" | float
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = self.q_norm(
            rearrange(
                query,
                "... q_len (group hpg dim) -> ... group hpg q_len dim",
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )
        key = self.k_norm(
            repeat(
                key,
                "... kv_len (group dim) -> ... group hpg kv_len dim",
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )
        value = repeat(
            value,
            "... kv_len (group dim) -> ... group hpg kv_len dim",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )

        query_var_id, kv_var_id = self._get_var_id(query, key, query_var_id, kv_var_id)
        query_time_id, kv_time_id = self._get_time_id(
            query,
            key,
            query_time_id,
            kv_time_id,
        )

        attn_mask = self._update_attn_mask(
            attn_mask,
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )

        query, key = self._qk_proj(
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p,
            scale=self.softmax_scale,
        )
        out = rearrange(out, "... group hpg q_len dim -> ... q_len (group hpg dim)")
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        self_attn: GroupedQueryAttention,
        ffn: FeedForward,
        norm1: nn.Module | None,
        norm2: nn.Module | None,
        post_attn_dropout_p: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.dropout_p = post_attn_dropout_p

        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1 or nn.Identity()
        self.norm2 = norm2 or nn.Identity()
        self.dropout = nn.Dropout(post_attn_dropout_p)

    def forward(
        self,
        x: torch.Tensor,  # "*batch time_len dim" | float
        attn_mask: torch.Tensor | None = None,  # "*batch time_len time_len" | bool
        var_id: torch.Tensor | None = None,  # "*batch time_len" | int
        time_id: torch.Tensor | None = None,  # "*batch time_len" | int
    ) -> torch.Tensor:  # "*batch time_len dim" | float
        if self.pre_norm:
            sa = self._sa_block(
                self.norm1(x),
                attn_mask,
                var_id=var_id,
                time_id=time_id,
            )
            x = x + sa
            x = x + self.ffn(self.norm2(x))
        else:
            sa = self._sa_block(
                x,
                attn_mask,
                var_id=var_id,
                time_id=time_id,
            )
            x = self.norm1(x + sa)
            x = self.norm2(x + self.ffn(x))

        return x

    def _sa_block(
        self,
        x: torch.Tensor,  # "*batch time_len dim" | float
        attn_mask: torch.Tensor | None = None,  # "*batch time_len time_len" | bool
        var_id: torch.Tensor | None = None,  # "*batch time_len" | int
        time_id: torch.Tensor | None = None,  # "*batch time_len" | int
    ) -> torch.Tensor:  # "*batch time_len dim" | float
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            query_var_id=var_id,
            kv_var_id=var_id,
            query_time_id=time_id,
            kv_time_id=time_id,
        )
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int | None = None,
        num_groups: int | None = None,
        pre_norm: bool = True,
        attn_dropout_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: Callable[[int], nn.Module] | None = nn.LayerNorm,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_glu: bool = True,
        use_qk_norm: bool = True,
        var_attn_bias_layer: Callable[[int, int, int], AttentionBias] | None = None,
        time_attn_bias_layer: Callable[[int, int, int], AttentionBias] | None = None,
        var_qk_proj_layer: Callable[[int, int, int], QueryKeyProjection] | None = None,
        time_qk_proj_layer: Callable[[int, int, int], QueryKeyProjection] | None = None,
        shared_var_attn_bias: bool = False,
        shared_time_attn_bias: bool = False,
        shared_var_qk_proj: bool = False,
        shared_time_qk_proj: bool = False,
        d_ff: int | None = None,
    ):
        super().__init__()
        num_heads = num_heads or d_model // 64
        num_groups = num_groups or num_heads  # defaults to mha

        var_attn_bias = self.get_layer(
            d_model,
            num_heads,
            num_groups,
            var_attn_bias_layer,
            shared_var_attn_bias,
        )
        time_attn_bias = self.get_layer(
            d_model,
            num_heads,
            num_groups,
            time_attn_bias_layer,
            shared_time_attn_bias,
        )
        var_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, var_qk_proj_layer, shared_var_qk_proj
        )
        time_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, time_qk_proj_layer, shared_time_qk_proj
        )

        get_self_attn = partial(
            GroupedQueryAttention,
            dim=d_model,
            num_heads=num_heads,
            num_groups=num_groups,
            bias=False,
            norm_layer=norm_layer if use_qk_norm else None,
            softmax_scale=None,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )
        get_ffn = partial(
            GatedLinearUnitFeedForward if use_glu else FeedForward,
            in_dim=d_model,
            hidden_dim=d_ff,
            out_dim=None,
            activation=activation,
            bias=False,
            ffn_dropout_p=dropout_p,
        )
        get_encoder_layer_norm = partial(norm_layer, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self_attn=get_self_attn(),
                    ffn=get_ffn(),
                    norm1=get_encoder_layer_norm(),
                    norm2=get_encoder_layer_norm(),
                    pre_norm=pre_norm,
                    post_attn_dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = norm_layer(d_model)

    @staticmethod
    def get_layer(
        dim: int,
        num_heads: int,
        num_groups: int,
        layer: Callable,
        shared_layer: bool,
    ) -> Callable[[], nn.Module] | None:
        if layer is None:
            return None
        if shared_layer:
            module = layer(dim=dim, num_heads=num_heads, num_groups=num_groups)
            return lambda: module
        return partial(layer, dim=dim, num_heads=num_heads, num_groups=num_groups)

    def forward(
        self,
        x: torch.Tensor,  # "*batch time_len dim" | float
        attn_mask: torch.Tensor | None = None,  # "*batch time_len time_len" | bool
        var_id: torch.Tensor | None = None,  # "*batch time_len" | int
        time_id: torch.Tensor | None = None,  # "*batch time_len" | int
    ) -> torch.Tensor:  # "*batch time_len dim" | float
        for layer in self.layers:
            x = layer(
                x,
                attn_mask,
                var_id=var_id,
                time_id=time_id,
            )
        return self.norm(x)
