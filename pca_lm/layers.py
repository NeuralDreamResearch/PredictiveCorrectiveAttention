"""Primitives shared by the mixer blocks: RMSNorm, RoPE, SwiGLU FFN,
and a memory-frugal chunked cross-entropy.

Extracted verbatim from the training code so a released checkpoint
loads against exactly the ops it was trained with."""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even.")

        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)        # [seq_len, dim]

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: [B, H, T, D]
    cos, sin: [T, D]
    """
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, T, D]
    sin = sin.unsqueeze(0).unsqueeze(0)   # [1, 1, T, D]

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    fl = logits.view(-1, logits.size(-1))
    ft = targets.reshape(-1)

    total = fl.size(0)
    loss_sum = torch.zeros((), device=logits.device, dtype=torch.float32)

    for i in range(0, total, chunk_size):
        loss_sum = loss_sum + F.cross_entropy(
            fl[i : i + chunk_size],
            ft[i : i + chunk_size],
            reduction="sum",
        )

    return loss_sum / total

