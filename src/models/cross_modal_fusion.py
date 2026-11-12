"""Gated cross-modal attention and progressive multimodal fusion blocks."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossModalBlock(nn.Module):
    """Cross-modal attention with adaptive gating for noisy interactions."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        query: [B, Lq, D], key_value: [B, Lk, D]
        key_padding_mask: [B, Lk], where True marks tokens to ignore.
        """
        q0 = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, _ = self.cross_attn(
            q0,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        gate = self.gate(torch.cat([query, attn_out], dim=-1))
        x = query + gate * attn_out
        x = self.norm_out(x + self.ffn(x))
        return x


def masked_mean_tokens(
    hidden: torch.Tensor, pad_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Return a masked mean over tokens."""
    if pad_mask is None:
        return hidden.mean(dim=1)
    m = (~pad_mask).unsqueeze(-1).to(hidden.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (hidden * m).sum(dim=1) / denom


def info_nce_symmetric(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives."""
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    t = max(temperature, 1e-6)
    logits_12 = (z1 @ z2.T) / t
    logits_21 = (z2 @ z1.T) / t
    targets = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (
        F.cross_entropy(logits_12, targets)
        + F.cross_entropy(logits_21, targets)
    )
