# Predictive-Corrective Attention (PCA)

## 1. The gap this fills

Modern linear-attention and SSM architectures are **reactive**: a token arrives,
a key-value pair is computed, a state is updated. None of them predict what the
state *should* be before the token arrives; none track how uncertain they are
about their own memory; none compute an optimal balance between trusting the
internal model and trusting the new observation.

The Kalman filter did all three in 1960. Deep learning has since rediscovered
fragments of it — the delta rule is a degenerate Kalman update, Mamba's
selection is a degenerate Kalman gain — without assembling the whole.

| Era | Mechanism | Has | Missing |
|---|---|---|---|
| 1960 | Kalman filter | predict, correct, uncertainty, optimal gain | not learned, not high-dim |
| 1960 | Delta rule | correction | prediction, uncertainty |
| 2017 | Transformer | parallel retrieval | state, uncertainty; O(T²) |
| 2020 | Linear attention | accumulation | correction, uncertainty |
| 2023 | Mamba | selective state transition | correction, uncertainty |
| 2024 | DeltaNet | error correction | prediction, uncertainty |
| 2024 | Gated DeltaNet | gate + correct | prediction, uncertainty |

PCA restores the two missing ingredients — **predictive dynamics** and
**uncertainty tracking** — and derives the update gain instead of designing it.

## 2. The framework

State is a *belief*: a memory matrix plus its covariance, `B_t = (S_t, P_t)`.

**Predict** — before the token is seen:

    Ŝ_t = A_θ(x_t) S_{t-1} + B_θ(x_t)
    P̂_t = A_θ(x_t) P_{t-1} A_θ(x_t)ᵀ + Q_θ(x_t)

**Innovation** — the surprise:

    e_t = v_t − Ŝ_tᵀ k_t

**Optimal gain** — not a hand-tuned learning rate:

    K_t = P̂_t k_t (k_tᵀ P̂_t k_t + r_t)^{-1}

**Correct**:

    S_t = Ŝ_t + K_t e_tᵀ
    P_t = (I − K_t k_tᵀ) P̂_t

**Retrieve**, with a confidence for free:

    o_t = g_t ⊙ (S_tᵀ q_t)
    σ²_t = q_tᵀ P_t q_t + r_t

`A_θ`, `Q_θ`, `r_t`, `g_t` are all projections of the input, so decay, process
noise, observation noise and output gate are input-selective.

### Existing methods as degenerate cases

| Architecture | A_θ | B_θ | Q_θ | P_t | K_t |
|---|---|---|---|---|---|
| Linear attention | I | 0 | 0 | → ∞ | k_t |
| DeltaNet | I | 0 | 0 | → ∞ | β_t k_t |
| Mamba | diag(α_t) | 0 | 0 | — | k_t |
| Gated DeltaNet | diag(α_t) | 0 | 0 | → ∞ | β_t k_t |
| Kalman (1960) | F fixed | Bu_t | Q fixed | full | optimal |
| **PCA** | learned | learned | learned | full/structured | optimal |

`P → ∞` is the informative limit: with unbounded prior uncertainty the Kalman
gain collapses to a fixed multiple of `k_t`, which is exactly the delta rule.
The existing methods are what you get when you decline to track uncertainty.

## 3. What this release implements

**The diagonal variant, without `B_θ`.** Concretely, per head:

    P̂_t = A_t² ⊙ P_{t-1} ⊙ p_decay + Q_t          (no B_θ term)
    K_t  = P̂_t k_t / (k_tᵀ P̂_t k_t + r_t)
    P_t  = (1 − K_t ⊙ k_t) ⊙ P̂_t                   clamped to [1e-5, 10]
    S_t  = (I − K_t k_tᵀ) diag(A_t) S_{t-1} + K_t v_tᵀ
    y_t  = g_t ⊙ (S_tᵀ q_t)

| Framework component | Status here |
|---|---|
| `A_θ(x)` input-dependent decay | ✅ diagonal, `sigmoid(W_A x)` |
| `Q_θ(x)` process noise | ✅ `softplus(W_Q x)`, capped |
| `r_t` observation noise | ✅ `softplus(W_r x)`, per head |
| Diagonal `P_t`, Kalman gain | ✅ |
| `σ²_t` retrieval uncertainty | ⚠️ computed and logged, **not used** by the loss or decoding |
| `B_θ(x)` predictive bias | ❌ not implemented |
| Full / low-rank `P_t` | ❌ diagonal only |

So of the three claimed novelties: predictive dynamics are present but without
`B_θ`; surprise-driven updating is present implicitly through the innovation;
**uncertainty-aware retrieval is computed but unexploited**. The
adaptive-compute, calibrated-sampling and OOD-detection applications that
`σ²` enables are untested.

## 4. Making it parallel

The state update is a generalised delta rule — rank-1 erase, rank-1 write, over
a diagonal decay — which admits an exact chunkwise (WY / UT-transform) form.
With `G_t = Π_{s≤t} A_s` and `S̃_t = diag(G_t)^{-1} S_t`, per chunk of C tokens:

    (I + L) D = V − k̃ S̃_0 ,   L[t,s] = k̃_t · K̃_s   (strictly lower)
    S̃_C = S̃_0 + K̃ᵀ D
    y    = g ⊙ ( q̃ S̃_0 + tril(q̃ K̃ᵀ, 0) D )

Only T/C steps are sequential. Exact, not an approximation.

**Numerics.** Every needed quantity is a ratio `G_t/G_s`, `s ≤ t`, always in
(0,1]; only the factorisation `G_t · (1/G_s)` is unstable. Trained decays reach
`A ≈ 1e-4`, so a raw cumprod underflows within ~32 tokens. Fixes, both exact:
log space with a floor, and centring the factorisation on the chunk's geometric
midpoint `√G_C`, halving the exponent range. The scan must run in true fp32 —
under autocast the centred factors (~1e25) are finite in fp32 but instantly
`inf` in fp16.

Chunk length is an accuracy parameter, not just a speed knob. Against the
reference loop: C=16 exact, C=32 drifts to 4e-4, C=128 breaks down (1.7e-1).

The covariance recurrence stays sequential — it touches only `[B,H,d_k]`
vectors — and is fused into a Triton kernel.

## 5. Head-split hybrid

Every block splits its heads rather than alternating whole layers:

    heads 0..Hp-1   → PCA recurrence        (linear in T)
    heads Hp..H-1   → causal softmax attention (quadratic in T)

Both read the same input and share one output projection, so exact recall is
available at every depth. The two paths keep **separate** q/k normalisation on
purpose: the PCA path uses `F.normalize`, which bounds its state and is
load-bearing for stability; the attention path uses RMSNorm QK-norm.

## 6. Released model

    layers        10 × head-split block
    heads         8 = 6 PCA + 2 attention, d_k 64
    d_model       512      d_ff 1408 (SwiGLU)
    context       512      vocab 50257 (GPT-2 BPE)
    params        63.8M (25.7M embedding, 38.1M non-embedding)
    training      2.10B tokens FineWeb-Edu, AdamW, cosine, bf16
    validation    3.4691 nats | 1.0892 bits/byte | PPL 32.11

## 7. What the trained weights show

Findings from this checkpoint, reported because they bear on whether the
mechanism earns its place.

**The PCA path is live, not gated off.** Output-gate norms are 14–19 at every
depth; the PCA columns of `W_o` carry mass comparable to the attention columns.

**Per head, attention is worth ~2.7 PCA heads.** Zeroing whole paths is
misleading (the split is 6/2); the matched control zeroes two heads either way,
in every layer:

| ablation | Δ val loss |
|---|---|
| 2 attention heads | **+1.078 nats** |
| 2 PCA heads (0–1) | +0.460 |
| 2 PCA heads (2–3) | +0.359 |
| 2 PCA heads (4–5) | +0.393 |

Zeroing the entire PCA path (6 heads) costs +5.03 nats vs +1.08 for the whole
attention path (2 heads) — PCA dominates the aggregate only because it holds
three times the width.

**The learned decay collapsed to a single timescale.** The framework's premise
is a diversity of retention horizons:

| layer | mean A | median horizon | p90 | p99 |
|---|---|---|---|---|
| 0 | 0.9971 | 346 tok | 375 | 414 |
| 3 | 0.9973 | 377 | 393 | 403 |
| 6 | 0.9973 | 367 | 390 | 403 |
| 9 | 0.9972 | 362 | 397 | 416 |

All 60 PCA heads, at all 10 depths, converged to ≈360 tokens. A likely cause is
the decay init: `W_A.bias = 6.0` puts `sigmoid` at 0.9975, deep in saturation
where the gradient is ~2e-3, so heads had little pressure to differentiate.
Constraining each head to its own decay band is the obvious remedy; it is not
applied in this release.

## 8. Limitations

- **No control run.** No plain transformer of matched size was trained on the
  same data, so "does this beat attention" is unanswered. 64% of non-embedding
  parameters (FFN, norms, output projection) are architecture-neutral.
- **Evaluated only at 512 context**, where the O(T) advantage is worth nothing
  and quadratic attention is cheap — i.e. not in the regime the mechanism is
  designed for.
- **`σ²` is unused**, so the framework's most distinctive claim is untested.
- `B_θ` and non-diagonal `P` are unimplemented.
- Single tokenizer, single corpus, one seed.

## 9. Predicted experiments (untested here)

The framework implies falsifiable predictions, none of which this release
evaluates:

1. **Noisy associative recall** — the gain should down-weight corrupted
   observations, where delta-rule methods incorporate them blindly.
2. **Topic-shift detection** — `‖e_t‖` should spike at boundaries, giving free
   unsupervised segmentation.
3. **Adaptive compute** — route only high-`σ²` tokens to softmax attention.
4. **OOD detection** — `σ²` systematically higher off-distribution.
5. **Few-shot extrapolation** — `A_θ` capturing pattern structure rather than
   stored instances.

## 10. References

The chunkwise form follows the delta-rule / WY-transform line on parallelising
linear recurrences (DeltaNet, GLA, Mamba-2 SSD). The Kalman parameterisation of
the write is the part that differs. The predictive-coding framing connects to
Friston's free-energy principle; the estimator itself is Kálmán (1960).
