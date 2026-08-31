#!/usr/bin/env python3
"""
Independently re-score a checkpoint on the FineWeb held-out validation tail.

The trainer stores best_val_loss inside the checkpoint, but that is its own
bookkeeping. This recomputes the number from the weights and the cache, using
the same split the trainer uses: the last `val_tokens` worth of rows in the
primary cache, taken `blocks_per_seq` at a time.

    python eval_val.py --ckpt <path> --device cuda:0
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from pca_lm import load_pretrained, resolve_checkpoint



def load(path, device):
    # Resolve first: `path` may be a directory or a manifest, and the parts
    # have to be joined before anything can torch.load them.
    path = resolve_checkpoint(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, cfg = load_pretrained(path, device)
    return model, cfg, ck


@torch.no_grad()
def score(model, blocks, starts, m, seq_len, device, amp, batch=4):
    tot, ntok = 0.0, 0
    for i in range(0, len(starts), batch):
        rows = []
        for s in starts[i:i + batch]:
            seq = np.concatenate([np.asarray(blocks[s + j]) for j in range(m)])
            rows.append(seq[:seq_len + 1].astype(np.int64))
        x = torch.tensor([r[:-1] for r in rows], device=device)
        y = torch.tensor([r[1:] for r in rows], device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(amp and device.startswith("cuda"))):
            out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        tot += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               y.reshape(-1), reduction="sum").item()
        ntok += y.numel()
    return tot / ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights",
                    help="a .pt, a .manifest.json, or the weights/ directory")
    ap.add_argument("--data", dest="cache", required=True,
                    help="path to the tokenised validation .npy")
    ap.add_argument("--train-blocks", type=int, default=1_167_314)
    ap.add_argument("--val-blocks", type=int, default=390)
    ap.add_argument("--blocks-per-seq", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", action="store_true", help="bf16, as the trainer evaluates")
    a = ap.parse_args()

    blocks = np.load(a.cache, mmap_mode="r")
    m = a.blocks_per_seq
    starts = list(range(a.train_blocks,
                        a.train_blocks + a.val_blocks - m + 1, m))
    model, cfg, ck = load(a.ckpt, a.device)
    ce = score(model, blocks, starts, m, a.seq_len, a.device, a.amp)
    print(f"\n  {os.path.basename(a.ckpt)}")
    print(f"    stored in ckpt : {ck.get("val_loss_nats", ck.get("best_val_loss", float("nan"))):.4f} nats  "
          f"PPL {math.exp(ck.get("val_loss_nats", ck.get("best_val_loss", float("nan")))):.2f}   (step {ck.get('step', ck.get('opt_step', 0)):,})")
    print(f"    recomputed     : {ce:.4f} nats  PPL {math.exp(ce):.2f}  "
          f"({len(starts)} windows x {a.seq_len} tok, "
          f"{'bf16' if a.amp else 'fp32'})")
    print(f"    delta          : {ce - ck.get("val_loss_nats", ck.get("best_val_loss", float("nan"))):+.4f} nats\n")


if __name__ == "__main__":
    main()
