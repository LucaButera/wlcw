import os

import torch
from torch import nn
from tsl.utils import download_url

from env import CACHE
from src.nn.models.timer.attention import AttentionLayer, TimeAttention
from src.nn.models.timer.transformer import TimerBlock, TimerLayer


class TimerXLModule(nn.Module):
    """
    Timer-XL: Long-Context Transformers for Unified Time Series Forecasting

    Paper: https://arxiv.org/abs/2410.04803

    GitHub: https://github.com/thuml/Timer-XL

    Citation: @article{liu2024timer,
        title={Timer-XL: Long-Context Transformers for Unified Time Series Forecasting},
        author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
        journal={arXiv preprint arXiv:2410.04803},
        year={2024}
    }
    """

    _weights = {
        "timer_xl_pretrained": {
            "url": "https://drive.switch.ch/index.php/s/6ovZYN31ZPWXGVa/download",
            "path": str(CACHE.joinpath("weights", "Timer", "timer_xl_pretrained.pth")),
        },
    }

    def __init__(
        self,
        patch_size: int = 96,
        hidden_size: int = 1024,
        dropout: float = 0.1,
        ff_size: int = 2048,
        num_heads: int = 8,
        num_layers: int = 8,
        activation: str = "relu",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(self.patch_size, hidden_size)
        self.blocks = TimerBlock(
            attn_layers=[
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(
                            mask_flag=True,
                            dropout=dropout,
                            hidden_size=hidden_size,
                            num_heads=num_heads,
                        ),
                        hidden_size=hidden_size,
                        n_heads=num_heads,
                    ),
                    hidden_size=hidden_size,
                    ff_size=ff_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )
        self.head = nn.Linear(hidden_size, patch_size)

    @classmethod
    def from_pretrained(cls, key: str) -> "TimerXLModule":
        path = cls._get_weights_path(key)
        model = cls()
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def _get_weights_path(cls, key: str) -> str:
        ref = cls._weights.get(key, None)
        if ref is None:
            raise ValueError(f"Pretrained model '{key}' is not available.")
        path, url = ref["path"], ref["url"]
        if not os.path.exists(path):
            sub_dir, filename = os.path.split(path)
            os.makedirs(sub_dir, exist_ok=True)
            download_url(url, sub_dir, filename)
        return path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, C = x.shape
        # [B, C, L]
        x = x.permute(0, 2, 1)
        # [B, C, N, P]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        N = x.shape[2]
        # [B, C, N, D]
        embed_out = self.embedding(x)
        # [B, C * N, D]
        embed_out = embed_out.reshape(B, C * N, -1)
        embed_out = self.blocks(embed_out, n_vars=C, n_tokens=N)
        # [B, C * N, P]
        dec_out = self.head(embed_out)
        # [B, C, N * P]
        dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)
        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out
