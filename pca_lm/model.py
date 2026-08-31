#!/usr/bin/env python3
"""
Predictive-Corrective Attention (PCA) language model v9 — hybrid PCA/Kalman + quadratic attention.

Architecture
------------
Layers alternate in a 4:1 ratio: four PCA/Kalman blocks followed by one
standard causal softmax-attention block, repeated.

    idx  0  1  2  3  4   5  6  7  8  9
         P  P  P  P  A   P  P  P  P  A

The PCA block carries a compressed [d_k, d_v] recurrent state, so its cost is
linear in sequence length but its capacity is fixed. The softmax block has
unbounded per-token recall but quadratic cost. Interleaving gives exact recall
checkpoints every 5 layers while keeping 80% of the layers linear -- the same
motivation as Griffin/Jamba-style hybrids, and it lets the context length grow
without the cost blowing up.

Changes from v8 (see legacy/)
-----------------------------
  * hybrid 4:1 PCA:attention (was: all PCA)
  * n_heads 4 -> 8. More heads measurably helped, and it also shrinks the
    recurrent state: per layer it is B*H*d_k*d_v = B*d_model^2/H, so doubling
    the heads HALVES both state memory and scan FLOPs.
  * n_layers 6 -> 10 (keeps the 4:1 pattern whole; depth beats width here)
  * max_seq_len 256 -> 512, rebuilt from the existing token cache by joining
    adjacent rows (no re-tokenisation needed)
  * the scan is the exact chunkwise form in pca_scan.py, not a Python loop
"""

import math
import os
from dataclasses import dataclass, field

import torch
import torch.utils.checkpoint

import torch.nn as nn
import torch.nn.functional as F

from .pca_scan import pca_scan_chunked
from .losses import fused_ce_zloss

# Generic layers are unchanged from v8 and imported rather than duplicated.
from .layers import (
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    SwiGLUFFN,
    chunked_cross_entropy,
)


@dataclass
class ConfigV9:
    # ── Model ─────────────────────────────────────────────────
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 10
    d_ff: int = 1408
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dropout: float = 0.05          # less regularisation: the run is data-rich

    # ── Mixing layout ─────────────────────────────────────────
    # "depth"      : whole blocks alternate -- 4 PCA blocks then 1 attention
    #                block. Exact recall exists at only 2 of 10 depths, so
    #                anything a PCA layer discards is unrecoverable before the
    #                next attention layer sees it.
    # "head_split" : every block runs BOTH mixers, splitting the heads between
    #                them and concatenating. Exact recall is available at every
    #                depth for roughly the same parameter count, because the
    #                width is re-divided rather than a path being added.
    #
    # Measured on the depth layout at step 4,750: removing both attention
    # layers cost +0.79 CE, versus +0.45 for an average pair of middle PCA
    # layers -- attention was 1.7x more load-bearing per layer while occupying
    # only 2 of 10 depths. That scarcity is what head_split removes.
    mixer_layout: str = "head_split"     # "depth" | "head_split"

    # depth layout only: layer i is attention iff (i+1) % attn_every == 0
    attn_every: int = 5            # 4 PCA : 1 attention

    # head_split only: how many of n_heads run softmax attention. The rest run
    # the PCA recurrence. 2 of 8 keeps the quadratic cost low while giving
    # every layer an exact-recall path.
    n_attn_heads: int = 2
    n_mamba_heads: int = 0   # selective-SSM heads; 0 keeps the 2-way split

    # Normalise q and k per head before the dot product. The PCA path already
    # does this (it is what keeps the recurrent state bounded); the softmax
    # path did not. It removes the main source of attention-logit blow-up and
    # is what makes the higher learning rate safe.
    qk_norm: bool = True

    # z-loss coefficient: penalises log-sum-exp drift in the softmax. 0 = off.
    z_loss_coef: float = 1e-4

    # ── Loss transform ────────────────────────────────────────
    # "none" | "log" | "log_ema".  Optimises  log(CE + c)  instead of CE.
    #
    # This is a MONOTONE transform, so the optimum is identical; it only
    # rescales the gradient by 1/(CE+c). Two things follow:
    #
    #  * Adam is scale-invariant -- m/(sqrt(v)+eps) is unchanged when every
    #    gradient is multiplied by a constant -- so with AdamW this is close
    #    to a no-op. It bites much harder with Lion (sign updates) or SGD.
    #  * With gradient accumulation, "log" weights each micro-batch by
    #    1/(CE_i + c), so HARDER batches get LESS weight (anti-focal).
    #    "log_ema" uses a detached EMA of CE instead, giving the same
    #    intended lr-vs-loss coupling with no per-batch reweighting.
    #    Prefer log_ema.
    #
    # The transform is compensated:
    #     L = (ce_ref + c) * log((CE + c) / (ce_ref + c))
    # so dL/dCE == 1 exactly at CE = ce_ref, i.e. it does not silently act as
    # a learning-rate cut at the current operating point. Set ce_ref to the CE
    # you are starting from (3.92 at step 3,000 of the first v9 run).
    #
    # Choosing c -- it sets how much the gradient grows as CE falls:
    #     CE 3.9 -> 3.0 amplification:  c=0  1.30x   c=0.5  1.26x
    #                                   c=1  1.22x   c=5    1.10x
    # and bounds the worst case at (ce_ref + c)/c as CE -> 0. c=1.0 gives a
    # gentle 1.22x with a hard ceiling of 4.9x, which is the safe default.
    loss_transform: str = "none"
    loss_log_c: float = 1.0
    ce_ref: float = 3.92
    ce_ema_beta: float = 0.98
    ce_chunk: int = 1024           # rows per loss chunk
    # Recompute logits in backward instead of storing them. Saves memory but
    # costs a second [rows x vocab] matmul; measured below.
    # At T=2048 the logits are [B, 2048, 50257] and the loss is a much larger
    # share of the step, so recomputing it in backward is worth the ~5%.
    ce_checkpoint: bool = True
    # Recompute each block's activations in the backward pass instead of
    # storing them. Costs one extra forward (~30% compute) and is what makes
    # a 500M model fit on an 8 GB card at a micro-batch above 1.
    grad_checkpoint: bool = False

    # ── PCA / Kalman ──────────────────────────────────────────
    p_decay: float = 1.0
    # Per-head decay ladder. Left free, the learned decay COLLAPSES: measured
    # on the 63.8M checkpoint at 2B tokens, all 60 PCA heads at all 10 depths
    # converged to A ~ 0.9972, a single ~360-token horizon -- the architecture
    # claims multi-timescale retention and was delivering one timescale.
    # Pinning each head to its own band forces the spread to persist. Values
    # are decay-per-token; horizon ~ 1/(1-A).
    #     0.9 -> 10 tok    0.99  -> 100     0.998 -> 500
    #     0.95 -> 20       0.995 -> 200     0.999 -> 1000
    pca_decay_ladder: tuple = (0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995)
    # Each head may move within band_x of its rung, in HORIZON space, so the
    # decay stays input-selective without a head drifting onto its neighbour.
    pca_decay_band: float = 2.0
    q_max: float = 0.01
    # Chunk length trades throughput against precision, and the right value
    # depends on sequence length. The scan loops T/chunk times SEQUENTIALLY, so
    # at T=2048 a chunk of 16 means 128 steps per layer and throughput halves
    # (4,007 tok/s vs 9,477 at C=64).
    #
    # Accuracy vs the reference loop, with aggressively-decaying A
    # (mean 0.64, min 5e-6) -- the regime the trained model actually occupies:
    #     C=16  7.5e-5     C=32  4.2e-4     C=64  4.2e-4     C=128  1.7e-1
    # So 64 is free relative to 32, and 128 falls off a cliff.
    #
    # NOTE: what matters is that training and inference use the SAME value.
    # The v8 checkpoint appeared to degrade at C=32 only because it was
    # evaluated with a different chunk than it was trained with; a consistent
    # 4e-4 perturbation is harmless. scan_chunk is stored in the checkpoint
    # config, so infer.py picks it up automatically.
    scan_chunk: int = 64

    # ── RoPE ──────────────────────────────────────────────────
    rope_base: float = 10000.0
    rope_max_len: int = 2048

    def is_attn(self, i: int) -> bool:
        if self.mixer_layout != "depth":
            return False
        return (i + 1) % self.attn_every == 0

    @property
    def n_attn_layers(self) -> int:
        return sum(self.is_attn(i) for i in range(self.n_layers))

    @property
    def n_pca_heads(self) -> int:
        return self.n_heads - self.n_attn_heads - self.n_mamba_heads


# ═══════════════════════════════════════════════════════════════
# Quadratic causal self-attention
# ═══════════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    """Standard multi-head causal attention with RoPE, via fused SDPA."""

    def __init__(self, cfg: ConfigV9):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_k = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.W_qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.d_k)
            self.k_norm = RMSNorm(self.d_k)

        self.rotary = RotaryEmbedding(
            dim=self.d_k,
            max_seq_len=max(cfg.max_seq_len + 8, cfg.rope_max_len),
            base=cfg.rope_base,
        )

    def forward(self, x):
        B, T, D = x.shape
        H, d_k = self.n_heads, self.d_k

        q, k, v = self.W_qkv(x).split(D, dim=-1)
        q = q.view(B, T, H, d_k).transpose(1, 2)
        k = k.view(B, T, H, d_k).transpose(1, 2)
        v = v.view(B, T, H, d_k).transpose(1, 2)

        # QK-norm goes before RoPE so the rotation acts on unit-scale vectors.
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        cos, sin = self.rotary(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Flash / mem-efficient kernel; never materialises the T x T matrix.
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        o = o.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(self.drop(o))


# ═══════════════════════════════════════════════════════════════
# PCA / Kalman attention
# ═══════════════════════════════════════════════════════════════
class DiagonalPCAAttention(nn.Module):
    def __init__(self, cfg: ConfigV9):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_k = cfg.d_model // cfg.n_heads
        self.d_v = self.d_k
        self.p_decay = cfg.p_decay
        self.q_max = cfg.q_max
        self.chunk = cfg.scan_chunk

        if self.d_k % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        D = cfg.d_model
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.W_A = nn.Linear(D, D, bias=True)
        self.W_Q = nn.Linear(D, D, bias=True)
        self.W_r = nn.Linear(D, cfg.n_heads, bias=True)
        self.gate = nn.Linear(D, D, bias=False)
        self.W_o = nn.Linear(D, D, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.rotary = RotaryEmbedding(
            dim=self.d_k,
            max_seq_len=max(cfg.max_seq_len + 8, cfg.rope_max_len),
            base=cfg.rope_base,
        )
        self._init_pca()

    def _init_pca(self):
        # A near 1 = slow decay / long memory. sigmoid(6) ~ 0.9975.
        nn.init.zeros_(self.W_A.weight)
        nn.init.constant_(self.W_A.bias, 6.0)
        nn.init.zeros_(self.W_Q.weight)
        nn.init.constant_(self.W_Q.bias, -5.0 - math.log(self.d_k))
        nn.init.zeros_(self.W_r.weight)
        nn.init.constant_(self.W_r.bias, -2.0)

    def forward(self, x):
        B, T, D = x.shape
        H, d_k, d_v = self.n_heads, self.d_k, self.d_v

        q = self.W_q(x).view(B, T, H, d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, d_v).transpose(1, 2)

        cos, sin = self.rotary(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Normalising q/k is what keeps the linear-attention state bounded.
        q = F.normalize(q, dim=-1) * math.sqrt(d_k)
        k = F.normalize(k, dim=-1)

        A = torch.sigmoid(self.W_A(x)).view(B, T, H, d_k).transpose(1, 2)
        Q = F.softplus(self.W_Q(x)).view(B, T, H, d_k).transpose(1, 2)
        Q = Q * self.q_max / (Q + self.q_max)          # bound the process noise
        r = F.softplus(self.W_r(x)).view(B, T, H).transpose(1, 2)
        g = torch.sigmoid(self.gate(x)).view(B, T, H, d_v).transpose(1, 2)

        # Runs in fp32 internally regardless of the ambient autocast dtype.
        out, unc = pca_scan_chunked(
            k.float(), v.float(), q.float(),
            A.float(), (A * A).float(), Q.float(), r.float(), g.float(),
            self.p_decay, chunk=self.chunk,
        )
        out = out.to(x.dtype)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(self.drop(out)), unc.transpose(1, 2).to(x.dtype)



# ═══════════════════════════════════════════════════════════════
# Head-split mixer: PCA recurrence and softmax attention in ONE block
# ═══════════════════════════════════════════════════════════════
def decay_bands(ladder, n_heads, band):
    """Per-head (A_min, A_max) from a decay ladder, as [n_heads, 1, 1].

    Interpolates geometrically in horizon space when the head count does not
    match the ladder length, so the spread is preserved at any width.
    """
    import numpy as _np
    lad = _np.asarray(ladder, dtype=_np.float64)
    hor = 1.0 / (1.0 - lad)
    if n_heads != len(lad):
        hor = _np.exp(_np.interp(_np.linspace(0, 1, n_heads),
                                 _np.linspace(0, 1, len(hor)), _np.log(hor)))
    lo = torch.tensor(1.0 - 1.0 / (hor / band), dtype=torch.float32)
    hi = torch.tensor(1.0 - 1.0 / (hor * band), dtype=torch.float32)
    lo = lo.clamp(1e-4, 1 - 1e-6)
    hi = hi.clamp(1e-4, 1 - 1e-6)
    return lo.view(-1, 1, 1), hi.view(-1, 1, 1)


class HybridHeadMixer(nn.Module):
    """
    Splits the heads of a single block between the two mixing mechanisms.

        heads 0 .. Hp-1   -> diagonal PCA / Kalman recurrence (linear in T)
        heads Hp .. H-1   -> causal softmax attention          (quadratic in T)

    Both operate on the same input and the same head dimension, and their
    outputs are concatenated back to d_model before a shared output
    projection. Every layer therefore has an exact-recall path alongside the
    compressed recurrent state, instead of exact recall appearing at only
    every 5th depth.

    Cost is roughly neutral: the projections are sized to their own head
    groups, so total width is unchanged. The quadratic term is paid on
    n_attn_heads only -- 2 of 8 heads here, a quarter of a full attention
    layer, but present at every depth.

    The two paths keep SEPARATE q/k normalisation on purpose. The PCA path
    uses F.normalize, which is what bounds its recurrent state and is
    load-bearing for stability; the attention path uses RMSNorm QK-norm.
    Sharing one of them would silently change the recurrence's conditioning.
    """

    def __init__(self, cfg: ConfigV9):
        super().__init__()
        D = cfg.d_model
        self.H = cfg.n_heads
        self.Ha = cfg.n_attn_heads
        self.Hp = cfg.n_pca_heads
        self.Hm_cfg = cfg.n_mamba_heads
        if self.Hp + self.Hm_cfg + self.Ha != self.H:
            raise ValueError(
                f"head split must sum to n_heads: {self.Hp} PCA + "
                f"{self.Hm_cfg} mamba + {self.Ha} attn != {self.H}")
        if self.Hp < 1 or self.Ha < 1:
            raise ValueError("head_split needs at least one head of each kind")

        self.d_k = D // self.H
        if self.d_k % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        Dp = self.Hp * self.d_k         # width owned by the PCA heads
        Da = self.Ha * self.d_k         # width owned by the attention heads

        self.p_decay = cfg.p_decay
        self.q_max = cfg.q_max
        self.chunk = cfg.scan_chunk
        self.dropout = cfg.dropout

        # ── PCA head projections ──────────────────────────────
        self.W_q = nn.Linear(D, Dp, bias=False)
        self.W_k = nn.Linear(D, Dp, bias=False)
        self.W_v = nn.Linear(D, Dp, bias=False)
        self.W_A = nn.Linear(D, Dp, bias=True)
        self.W_Q = nn.Linear(D, Dp, bias=True)
        self.W_r = nn.Linear(D, self.Hp, bias=True)
        self.gate = nn.Linear(D, Dp, bias=False)


        # ── attention head projections ────────────────────────
        self.W_qkv = nn.Linear(D, 3 * Da, bias=False)
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.d_k)
            self.k_norm = RMSNorm(self.d_k)

        self.W_o = nn.Linear(D, D, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.rotary = RotaryEmbedding(
            dim=self.d_k,
            max_seq_len=max(cfg.max_seq_len + 8, cfg.rope_max_len),
            base=cfg.rope_base,
        )
        self._init_pca()

    def _init_pca(self):
        nn.init.zeros_(self.W_A.weight)
        nn.init.constant_(self.W_A.bias, 6.0)
        nn.init.zeros_(self.W_Q.weight)
        nn.init.constant_(self.W_Q.bias, -5.0 - math.log(self.d_k))
        nn.init.zeros_(self.W_r.weight)
        nn.init.constant_(self.W_r.bias, -2.0)

    def forward(self, x):
        B, T, D = x.shape
        Hp, Ha, d_k = self.Hp, self.Ha, self.d_k

        cos, sin = self.rotary(T)

        # ── PCA heads ─────────────────────────────────────────
        q = self.W_q(x).view(B, T, Hp, d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, Hp, d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, Hp, d_k).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = F.normalize(q, dim=-1) * math.sqrt(d_k)
        k = F.normalize(k, dim=-1)

        A = torch.sigmoid(self.W_A(x)).view(B, T, Hp, d_k).transpose(1, 2)
        Q = F.softplus(self.W_Q(x)).view(B, T, Hp, d_k).transpose(1, 2)
        Q = Q * self.q_max / (Q + self.q_max)
        r = F.softplus(self.W_r(x)).view(B, T, Hp).transpose(1, 2)
        g = torch.sigmoid(self.gate(x)).view(B, T, Hp, d_k).transpose(1, 2)

        o_p, unc = pca_scan_chunked(
            k.float(), v.float(), q.float(),
            A.float(), (A * A).float(), Q.float(), r.float(), g.float(),
            self.p_decay, chunk=self.chunk,
        )
        o_p = o_p.to(x.dtype)                       # [B, Hp, T, d_k]

        # ── attention heads ───────────────────────────────────
        qa, ka, va = self.W_qkv(x).split(Ha * d_k, dim=-1)
        qa = qa.view(B, T, Ha, d_k).transpose(1, 2)
        ka = ka.view(B, T, Ha, d_k).transpose(1, 2)
        va = va.view(B, T, Ha, d_k).transpose(1, 2)
        if self.qk_norm:
            qa, ka = self.q_norm(qa), self.k_norm(ka)
        qa, ka = apply_rotary_pos_emb(qa, ka, cos, sin)

        o_a = F.scaled_dot_product_attention(
            qa, ka, va, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )                                            # [B, Ha, T, d_k]

        # ── recombine ─────────────────────────────────────────
        # Order is fixed: PCA heads, then Mamba, then attention. W_o's columns
        # are laid out to match, so changing the order silently permutes a
        # trained checkpoint's projection.
        o = torch.cat([o_p, o_a], dim=1)             # [B, H, T, d_k]
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(self.drop(o)), unc.transpose(1, 2).to(x.dtype)


class HybridHeadBlock(nn.Module):
    def __init__(self, cfg: ConfigV9):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = HybridHeadMixer(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x):
        a, unc = self.attn(self.norm1(x))
        x = x + a
        return x + self.ffn(self.norm2(x)), unc


# ═══════════════════════════════════════════════════════════════
# Blocks
# ═══════════════════════════════════════════════════════════════
class PCABlock(nn.Module):
    def __init__(self, cfg: ConfigV9):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = DiagonalPCAAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x):
        a, unc = self.attn(self.norm1(x))
        x = x + a
        return x + self.ffn(self.norm2(x)), unc


class AttnBlock(nn.Module):
    def __init__(self, cfg: ConfigV9):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x)), None


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════
class HybridPCALanguageModel(nn.Module):
    def __init__(self, cfg: ConfigV9):
        super().__init__()
        self.cfg = cfg

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)

        if cfg.mixer_layout == "head_split":
            self.layers = nn.ModuleList(
                [HybridHeadBlock(cfg) for _ in range(cfg.n_layers)])
        else:
            self.layers = nn.ModuleList([
                AttnBlock(cfg) if cfg.is_attn(i) else PCABlock(cfg)
                for i in range(cfg.n_layers)
            ])
        self.norm = RMSNorm(cfg.d_model)

        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight        # tied

        self.ce_ema = None            # running CE estimate for loss_transform

        self.apply(self._init_weights)

        # apply() would clobber the PCA biases, so restore them, and scale the
        # residual output projections by depth.
        for layer in self.layers:
            if isinstance(layer, (PCABlock, HybridHeadBlock)):
                layer.attn._init_pca()
            nn.init.normal_(layer.attn.W_o.weight,
                            std=0.02 / math.sqrt(2.0 * cfg.n_layers))
            nn.init.normal_(layer.ffn.w2.weight,
                            std=0.02 / math.sqrt(2.0 * cfg.n_layers))

    def _transform(self, ce):
        """Apply the configured loss transform. Identity unless enabled."""
        cfg = self.cfg
        mode = cfg.loss_transform

        if mode == "none":
            return ce

        c = cfg.loss_log_c
        ref = cfg.ce_ref + c                      # compensation: dL/dCE = 1 at ce_ref

        if mode == "log":
            # Literal log(CE + c). Reweights micro-batches by 1/(CE_i + c).
            return ref * torch.log((ce + c) / ref)

        if mode == "log_ema":
            # Same gradient scale, but from a detached running estimate, so
            # every micro-batch in an accumulation window is scaled equally.
            with torch.no_grad():
                cur = ce.detach()
                if self.ce_ema is None:
                    self.ce_ema = cur.clone()
                else:
                    b = cfg.ce_ema_beta
                    self.ce_ema.mul_(b).add_(cur, alpha=1.0 - b)
                scale = ref / (self.ce_ema + c)
            return ce * scale

        raise ValueError(f"unknown loss_transform: {mode!r}")

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, targets=None):
        x = self.embed_drop(self.tok_embed(input_ids))

        unc_mean = torch.zeros((), device=x.device, dtype=torch.float32)
        unc_last = torch.zeros((), device=x.device, dtype=torch.float32)
        n_pca = 0

        use_ckpt = (getattr(self.cfg, "grad_checkpoint", False)
                    and self.training and torch.is_grad_enabled())
        for layer in self.layers:
            if use_ckpt:
                # use_reentrant=False keeps this compatible with DDP's
                # autograd hooks; the reentrant version double-counts them.
                x, unc = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False)
            else:
                x, unc = layer(x)
            if unc is not None:                 # attention blocks have none
                unc_mean = unc_mean + unc.float().mean()
                unc_last = unc_last + unc[:, -1, :].float().mean()
                n_pca += 1

        x = self.norm(x)

        n = max(1, n_pca)

        if targets is None:
            return self.head(x), None, unc_mean / n, unc_last / n

        # Training path: never materialise [B, T, vocab]. fused_ce_zloss
        # recomputes each logit chunk in the backward pass instead, which is
        # the single largest memory saving in the step.
        ce, z = fused_ce_zloss(x, self.head.weight, targets,
                               z_coef=self.cfg.z_loss_coef,
                               chunk=self.cfg.ce_chunk,
                               use_checkpoint=self.cfg.ce_checkpoint)

        loss = self._transform(ce) + self.cfg.z_loss_coef * z

        # `ce` is returned separately: perplexity must come from it, not from
        # the regularised total.
        self.last_ce = ce.detach()
        self.last_z = z.detach()

        return None, loss, unc_mean / n, unc_last / n

    @torch.inference_mode()
    def generate_with_uncertainty(self, input_ids, max_new_tokens=100, temperature=0.8):
        self.eval()
        gen = input_ids.clone()
        sigmas = []
        for _ in range(max_new_tokens):
            idx = gen[:, -self.cfg.max_seq_len:]
            logits, _, _, unc_last = self(idx)
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            gen = torch.cat([gen, nxt], dim=1)
            sigmas.append(unc_last.item())
        return gen, sigmas


def resolve_checkpoint(path):
    """Accept a .pt, a manifest.json, or a directory holding either.

    The weights ship split into <25 MB parts so they can be uploaded through
    GitHub's web UI, so the common case is that only the parts exist on a
    fresh clone. Joining is done once and cached next to the parts.
    """
    import glob

    if os.path.isdir(path):
        pt = glob.glob(os.path.join(path, "*.pt"))
        if pt:
            return pt[0]
        man = glob.glob(os.path.join(path, "*.manifest.json"))
        if not man:
            raise FileNotFoundError(f"no .pt or .manifest.json under {path}")
        path = man[0]

    if path.endswith(".manifest.json"):
        from .shards import join
        joined = join(path)
        print(f"  reassembled {os.path.basename(joined)} from parts")
        return joined

    if not os.path.exists(path):
        # A bare .pt name whose parts are present but not yet joined.
        man = os.path.splitext(path)[0] + ".manifest.json"
        if os.path.exists(man):
            from .shards import join
            joined = join(man)
            print(f"  reassembled {os.path.basename(joined)} from parts")
            return joined
        raise FileNotFoundError(path)

    return path


def load_pretrained(path, device="cpu"):
    """Load a released checkpoint, joining split parts if needed.

    Returns (model, config).
    """
    ck = torch.load(resolve_checkpoint(path), map_location="cpu",
                    weights_only=False)
    cfg = ConfigV9(**{k: v for k, v in ck["config"].items()
                      if k in ConfigV9.__dataclass_fields__})
    model = HybridPCALanguageModel(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg
