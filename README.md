# PCA-LM — Predictive-Corrective Attention

A 63.8M-parameter language model built on **Predictive-Corrective Attention** —
a sequence mixer that restores the two ingredients modern linear-attention and
SSM architectures dropped: **predictive dynamics** and **uncertainty tracking**.

Existing methods are reactive. A token arrives, a key-value pair is formed, a
state is updated. PCA instead maintains a *belief* — a memory matrix plus its
covariance — predicts what the memory should be before the token arrives,
measures the surprise, and corrects with a **Kalman gain** rather than a
hand-tuned learning rate. Linear attention, DeltaNet, Mamba and Gated DeltaNet
all fall out as degenerate cases in the `P → ∞` limit, where unbounded prior
uncertainty reduces the optimal gain to the delta rule.

The recurrence is O(T) with O(1) state and runs in an exact chunkwise form, so
it is matmul-bound rather than a loop over timesteps.

| | |
|---|---|
| parameters | 63.8M (25.7M embedding) |
| layers | 10 × head-split (6 PCA + 2 attention heads each) |
| context | 512 tokens |
| tokenizer | GPT-2 BPE (50257) |
| training | 2.10B tokens, FineWeb-Edu |
| validation | **3.4691 nats · 1.0892 bits/byte · PPL 32.11** |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the derivation, the numerics, and —
importantly — [what the trained weights actually show](ARCHITECTURE.md#6-what-the-weights-actually-show),
including an ablation indicating that per head, attention is worth about 2.7
PCA heads, and that the learned decay collapsed to a single timescale.

This release implements the **diagonal** variant without the predictive bias
`B_θ`, and the retrieval uncertainty `σ²` is computed but not yet used by the
objective or decoding — see [What this release implements](ARCHITECTURE.md#3-what-this-release-implements).

## Install

```bash
pip install -r requirements.txt
```

The 255 MB checkpoint ships as **13 parts of ≤21 MB**, because GitHub's web
uploader rejects any single file over 25 MB. Nothing to do by hand — the first
load rejoins them (verifying a SHA-256) and caches the result:

```
weights/
  pca_lm_63m.manifest.json    sizes + checksums
  pca_lm_63m.part000 .. 012   upload these
```

To rejoin explicitly: `python split_weights.py join weights/pca_lm_63m.manifest.json`

## Generate

```bash
python infer.py --prompt "The key idea behind" --tokens 120
```

The first run prints `reassembled pca_lm_63m.pt from parts` and takes a few
extra seconds; later runs use the cached file.

## Use as a library

```python
import torch
from pca_lm import load_pretrained

model, cfg = load_pretrained("weights", device="cuda")   # dir, .pt or manifest
ids = torch.randint(0, cfg.vocab_size, (1, 128), device="cuda")

logits, _, unc_mean, unc_last = model(ids)   # no targets -> logits
loss, z = model(ids, targets=ids)[:2]        # targets -> fused CE + z-loss
```

`unc_mean` / `unc_last` are the PCA heads' predictive uncertainty `σ²`. They
are exposed but unused by the objective — see Limitations.

## Reproduce the reported number

```bash
python eval_val.py --data <fineweb_cache>.npy
```

The released code reproduces the training-time validation loss to 5e-5 nats.

## Layout

```
pca_lm/
  model.py              config, head-split mixer, blocks, LM head
  pca_scan.py           exact chunkwise recurrence (fp32-pinned)
  pca_kalman_triton.py  fused Kalman-gain kernel (optional)
  layers.py             RMSNorm, RoPE, SwiGLU
  losses.py             chunked cross-entropy + z-loss
  shards.py             byte-level split/join for the weight parts
infer.py                sampling
eval_val.py             re-score a checkpoint
split_weights.py        split/join CLI
weights/                13 parts + manifest (255 MB joined, no optimizer state)
```

## Status

This is a research artifact, not a product. It has **no control run** — no
plain transformer of matched size was trained on the same data — so it does not
establish that the mechanism beats attention. It is released because the
negative results are as informative as the positive ones.

## License

No license file is included; add one before publishing. Without it the default
is "all rights reserved", which is probably not what you want for a public
repo. MIT or Apache-2.0 are the usual choices.
