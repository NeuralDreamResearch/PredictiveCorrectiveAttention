"""
Memory-efficient cross-entropy with an optional z-loss term.

Two problems with the v8 loss:

1. It materialised the full [B, T, vocab] logit tensor and then let
   F.cross_entropy build an fp32 log-softmax on top of it. At B=6, T=512,
   V=50257 that is ~309 MB of bf16 logits plus ~617 MB of fp32 softmax saved
   for backward -- comfortably the largest single allocation in the step, and
   it scales linearly with batch size. Chunking the loop, as v8 did, does not
   help: autograd still holds every chunk's saved tensor until backward.

   Here each chunk is wrapped in gradient checkpointing, so the forward keeps
   only the hidden states and recomputes the logits during backward. Peak
   memory becomes one chunk instead of the whole sequence.

2. Nothing constrained the softmax normaliser. z-loss (Chowdhery et al., PaLM;
   Zoph et al., ST-MoE) adds

       z_loss = coef * mean( logsumexp(logits)^2 )

   which pulls log Z toward 0. It stops the logits drifting to large absolute
   values, which is a common source of bf16/fp16 instability and of loss
   spikes at higher learning rates. Standard coefficient is 1e-4; it is a
   regulariser, not an objective, so it is reported separately from the
   cross-entropy that perplexity is computed from.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _ce_chunk(h, weight, targets, z_coef: float):
    """One chunk: hidden states -> logits -> (nll_sum, z_sum). Recomputed in
    backward, so nothing here is kept alive by the forward pass."""
    logits = F.linear(h, weight).float()

    lse = torch.logsumexp(logits, dim=-1)
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    nll = (lse - picked).sum()
    z = (lse * lse).sum() if z_coef > 0.0 else lse.new_zeros(())

    return torch.stack((nll, z))


def fused_ce_zloss(hidden, weight, targets, z_coef: float = 1e-4,
                   chunk: int = 1024, use_checkpoint: bool = True):
    """
    hidden : [B, T, d_model]   weight : [vocab, d_model] (may be tied)
    targets: [B, T]

    Returns (ce, z) as separate scalars. The training objective is
    ce + z_coef * z; perplexity must be computed from `ce` alone.
    """
    h = hidden.reshape(-1, hidden.size(-1))
    t = targets.reshape(-1)
    n = h.size(0)

    acc = torch.zeros(2, device=h.device, dtype=torch.float32)
    for i in range(0, n, chunk):
        hs, ts = h[i:i + chunk], t[i:i + chunk]
        if use_checkpoint and torch.is_grad_enabled() and hs.requires_grad:
            acc = acc + checkpoint(_ce_chunk, hs, weight, ts, z_coef,
                                   use_reentrant=False)
        else:
            acc = acc + _ce_chunk(hs, weight, ts, z_coef)

    return acc[0] / n, acc[1] / n
