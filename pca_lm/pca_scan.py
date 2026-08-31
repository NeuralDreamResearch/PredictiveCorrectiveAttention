"""
Chunkwise-parallel form of the diagonal Predictive-Corrective Attention
(PCA) scan.

The per-token recurrence in train.py is

    S_t = (I - K_t k_t^T) diag(A_t) S_{t-1} + K_t v_t^T
    y_t = g_t * (S_t^T q_t)

which is a *generalised delta rule*: a rank-1 erase plus a rank-1 write on top
of a diagonal decay. Recurrences of that shape have an exact chunkwise form
(the WY / UT transform), so the sequential Python loop over T can be replaced
by matmuls over chunks of C tokens with only T/C sequential steps.

Derivation (per chunk, local indices 1..C)
-----------------------------------------
Absorb the decay. With G_t = prod_{s<=t} A_s (inclusive cumprod inside the
chunk) and S~_t = diag(G_t)^-1 S_t:

    S~_t = (I - K~_t k~_t^T) S~_{t-1} + K~_t v_t^T
    k~_t = G_t * k_t,   K~_t = K_t / G_t,   q~_t = G_t * q_t

so S~_0 = S_0 and S_C = diag(G_C) S~_C.

Now (I - u w^T) S + u z^T = S + u (z - S^T w)^T, so with d_t := v_t - S~_{t-1}^T k~_t

    S~_t = S~_{t-1} + K~_t d_t^T          =>   S~_{t-1} = S~_0 + sum_{s<t} K~_s d_s^T
    d_t  = v_t - S~_0^T k~_t - sum_{s<t} (K~_s . k~_t) d_s

In matrix form with L[t,s] = k~_t . K~_s strictly lower triangular:

    (I + L) D = V - k~ S~_0        =>   D = tri_solve(I + L, V - k~ S~_0)
    S~_C = S~_0 + K~^T D
    y    = g * ( q~ S~_0 + tril(q~ K~^T, 0) D )       (diagonal included:
                                                       y_t uses S_t, post-update)

Exact, not an approximation.

The Kalman-gain recurrence for K_t and the uncertainty is left sequential: it
only touches [B,H,d_k] vectors, so it costs no activation memory for the
[d_k,d_v] state and is cheap to recompute in the backward pass.
"""

import torch

try:
    from .pca_kalman_triton import kalman_gain_scan_triton
    _HAVE_TRITON = True
except Exception:                                   # no triton / no GPU
    _HAVE_TRITON = False

# Numerics of the intra-chunk decay
# ---------------------------------
# Every quantity the chunk form needs is a RATIO G_t/G_s with s <= t, which is
# always in (0, 1]. Only the factorisation G_t * (1/G_s) is unstable: the
# trained model learns A with mean ~0.78 and min ~1e-4, so a raw cumprod
# underflows to zero within 32 tokens and K/G becomes inf.
#
# Two fixes, both exact:
#   1. Work in log space and floor the cumulative log-decay at LOG_FLOOR. A
#      chunk-internal decay of e^-120 is a complete reset; clamping the
#      cumulative sum keeps it monotone non-increasing, so every ratio it
#      produces is still correct to fp32 precision.
#   2. Centre the factorisation on the geometric midpoint sqrt(G_C):
#          G_t/G_s = (G_t/sqrt(G_C)) * (sqrt(G_C)/G_s)
#      Both factors then live in [e^-60, e^60] instead of [e^-120, 1], which
#      HALVES the exponent range and keeps everything inside fp32.
LOG_FLOOR = -120.0

# Chunk length matters for accuracy, not just speed. The centred exponents span
# roughly +/- C*|log A|/2, so a longer chunk widens the fp32 range the C x C
# products must cover. Measured against the reference loop on a trained
# checkpoint (val PPL 50.14): C=16 reproduces it exactly, C=32 drifts to 51.85.
# Clamping the exponents to +/-30 to buy headroom was tried and *hurt* (50.72) --
# ratios down to e^-60 carry real signal. So: keep C small and leave the
# exponents alone. C=16 keeps the range near e^+/-30, well inside fp32.

def kalman_gain_scan(k, A2, Q, r, q, p_decay: float):
    """Sequential part: diagonal covariance P, Kalman gain K, uncertainty.

    k, A2, Q, q : [B, H, T, d_k]      r : [B, H, T]
    returns  K : [B, H, T, d_k],  unc : [B, H, T]
    """
    B, H, T, d_k = k.shape
    P = torch.ones(B, H, d_k, device=k.device, dtype=k.dtype)

    Ks, uncs = [], []
    for t in range(T):
        k_t, r_t = k[:, :, t], r[:, :, t]

        P_hat = A2[:, :, t] * P * p_decay + Q[:, :, t]
        Pk = P_hat * k_t
        den = (k_t * Pk).sum(-1, keepdim=True) + r_t.unsqueeze(-1)
        K_t = Pk / (den + 1e-6)

        P = ((1.0 - K_t * k_t) * P_hat).clamp(1e-5, 10.0)

        Ks.append(K_t)
        uncs.append((q[:, :, t] * P * q[:, :, t]).sum(-1) + r_t)

    return torch.stack(Ks, dim=2), torch.stack(uncs, dim=2)


# The chunk form must run in true fp32. Under torch.amp.autocast every matmul
# is downcast to the autocast dtype regardless of input dtype, and the
# midpoint-centred factors legitimately reach ~1e25 -- finite in fp32/bf16 but
# instantly inf in fp16. Opting out here keeps the scan exact under any AMP
# setting; the caller casts the result back.
# torch.compile must NOT trace into this function. Inductor does not honour
# the autocast(enabled=False) guard below when it traces through, so the
# midpoint-centred factors (which legitimately reach ~1e25) get computed in
# bf16 and overflow -- measured as a loss error of 0.25 nats and a 52%
# gradient error, i.e. silently wrong training. Keeping it opaque costs the
# fusion inside the scan but preserves the fp32 contract; everything around it
# (norms, SwiGLU, residuals, the loss) still compiles.
@torch._dynamo.disable
@torch.amp.autocast("cuda", enabled=False)
def pca_scan_chunked(k, v, q, A, A2, Q, r, g, p_decay: float, chunk: int = 16):
    """Drop-in replacement for _pca_scan_impl. All tensors [B,H,T,d]; r [B,H,T]."""
    B, H, T, d_k = k.shape
    d_v = v.size(-1)

    if _HAVE_TRITON and k.is_cuda:
        K, unc = kalman_gain_scan_triton(k, A2, Q, r, q, p_decay)
    else:
        K, unc = kalman_gain_scan(k, A2, Q, r, q, p_decay)

    # Pad T up to a multiple of the chunk length.
    pad = (-T) % chunk
    if pad:
        z = lambda x, val=0.0: torch.nn.functional.pad(x, (0, 0, 0, pad), value=val)
        k, v, q, K = z(k), z(v), z(q), z(K)
        A = z(A, 1.0)          # pad decay with 1.0 so G stays flat
    Tp = T + pad
    N = Tp // chunk

    sh = (B, H, N, chunk, -1)
    kc, vc, qc, Kc = k.view(sh), v.view(sh), q.view(sh), K.view(sh)
    Ac = A.view(sh)

    logG = torch.log(Ac.clamp_min(1e-30)).cumsum(dim=3).clamp_min(LOG_FLOOR)
    logG_end = logG[:, :, :, -1:, :]                     # [B,H,N,1,d_k]
    half = 0.5 * logG_end

    e_G = torch.exp(logG)                                # G_t            (0,1]
    e_ctr = torch.exp(logG - half)                       # G_t/sqrt(G_C)
    e_inv = torch.exp(half - logG)                       # sqrt(G_C)/G_s
    e_end = torch.exp(logG_end - logG)                   # G_C/G_t        (0,1]

    kL, KL, qL = kc * e_ctr, Kc * e_inv, qc * e_ctr      # for the C x C matrices
    kS, qS = kc * e_G, qc * e_G                          # for the S_0 interaction
    KS = Kc * e_end                                      # for the state carry

    eye = torch.eye(chunk, device=k.device, dtype=k.dtype)
    L = (kL @ KL.transpose(-1, -2)).tril(-1) + eye       # [B,H,N,C,C]
    QK = (qL @ KL.transpose(-1, -2)).tril(0)             # [B,H,N,C,C]
    G_end = e_G[:, :, :, -1]                             # [B,H,N,d_k]

    S = torch.zeros(B, H, d_k, d_v, device=k.device, dtype=k.dtype)
    outs = []
    for n in range(N):
        rhs = vc[:, :, n] - kS[:, :, n] @ S
        D = torch.linalg.solve_triangular(L[:, :, n], rhs, upper=False, unitriangular=True)

        outs.append(qS[:, :, n] @ S + QK[:, :, n] @ D)
        S = G_end[:, :, n].unsqueeze(-1) * S + KS[:, :, n].transpose(-1, -2) @ D

    out = torch.cat(outs, dim=2).view(B, H, Tp, d_v)[:, :, :T]
    return g * out, unc
