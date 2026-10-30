from functools import partial

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from src.nn.models.moirai.attention import (
    QueryKeyProjection,
    BinaryAttentionBias,
    RotaryProjection,
)
from src.nn.models.moirai.feedforward import ResidualBlock
from src.nn.models.moirai.norm import RMSNorm
from src.nn.models.moirai.transformer import TransformerEncoder


def packed_attention_mask(
    sample_id: torch.Tensor,  # "*batch seq_len" | int
) -> torch.Tensor:  # "*batch seq_len seq_len" | bool
    sample_id = sample_id.unsqueeze(-1)
    attention_mask = sample_id.eq(sample_id.mT)
    return attention_mask


def packed_causal_attention_mask(
    sample_id: torch.Tensor,  # "*batch seq_len" | int
    time_id: torch.Tensor,  # "*batch seq_len" | int
) -> torch.Tensor:  # "*batch seq_len seq_len" | bool
    attention_mask = packed_attention_mask(sample_id)
    expanded_id1 = time_id.unsqueeze(-2)
    expanded_id2 = time_id.unsqueeze(-1)
    compare_res = expanded_id1 <= expanded_id2
    attention_mask = attention_mask * compare_res
    return attention_mask


class Moirai2Module(
    nn.Module,
    PyTorchModelHubMixin,
):
    """
    Contains components of Moirai, to ensure implementation is identical across models.
    Subclasses huggingface_hub.PyTorchModelHubMixin to support loading from HuggingFace Hub.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_layers: int,
        patch_size: int,
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        num_predict_token: int = 1,
        quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ):
        """
        :param d_model: model hidden dimensions
        :param num_layers: number of transformer layers
        :param patch_size: patch size
        :param max_seq_len: maximum sequence length for inputs
        :param attn_dropout_p: dropout probability for attention layers
        :param dropout_p: dropout probability for all other layers
        :param scaling: whether to apply scaling (standardization)
        :param num_quantiles: number of quantile levels
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_predict_token = num_predict_token
        self.max_seq_len = max_seq_len
        self.quantile_levels = quantile_levels
        self.num_quantiles = len(quantile_levels)

        self.in_proj = ResidualBlock(
            input_dims=patch_size * 2,
            hidden_dims=d_model,
            output_dims=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=d_ff,
        )
        self.out_proj = ResidualBlock(
            input_dims=d_model,
            hidden_dims=d_model,
            output_dims=num_predict_token * self.num_quantiles * patch_size,
        )

    def forward(
        self,
        target: torch.Tensor,  # "*batch seq_len patch" | float
        observed_mask: torch.Tensor,  # "*batch seq_len patch" | bool
        sample_id: torch.Tensor,  # "*batch seq_len" | int
        time_id: torch.Tensor,  # "*batch seq_len" | int
        variate_id: torch.Tensor,  # "*batch seq_len" | int
    ) -> torch.Tensor:
        """
        Defines the forward pass of MoiraiDecoderModule.
        This method expects processed inputs.

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :return: predictive distribution
        """
        input_tokens = torch.cat([target, observed_mask.to(torch.float32)], dim=-1)
        reprs = self.in_proj(input_tokens)

        reprs = self.encoder(
            reprs,
            packed_causal_attention_mask(sample_id, time_id),
            time_id=time_id,
            var_id=variate_id,
        )
        preds = self.out_proj(reprs)
        return preds
