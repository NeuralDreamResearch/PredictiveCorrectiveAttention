"""
Fused Triton kernel for the sequential Kalman-gain recurrence.

The chunkwise scan in pca_scan.py removes the sequential loop over the
[d_k, d_v] state, but the diagonal-covariance recurrence

    P_hat_t = A2_t * P_{t-1} * p_decay + Q_t
    K_t     = (P_hat_t * k_t) / (k_t . P_hat_t k_t + r_t + eps)
    P_t     = clamp((1 - K_t * k_t) * P_hat_t, 1e-5, 10)
    unc_t   = q_t . P_t q_t + r_t

is genuinely nonlinear and not associative, so it stays sequential. It only
touches d_k-sized vectors, though, so a single program per (batch, head) can
hold P in registers and run the whole T-step loop with one kernel launch
instead of ~8*T launches.

Backward is a hand-written reverse scan over the same loop; P_{t-1} is saved by
the forward pass (same footprint as K, which must be saved anyway) and
everything else is recomputed.
"""

import torch
import triton
import triton.language as tl

_EPS = tl.constexpr(1e-6)
_P_LO = tl.constexpr(1e-5)
_P_HI = tl.constexpr(10.0)


@triton.jit
def _fwd(k_ptr, A2_ptr, Q_ptr, r_ptr, q_ptr, K_ptr, P_ptr, unc_ptr,
         T: tl.constexpr, D: tl.constexpr, p_decay, BLOCK: tl.constexpr):
    bh = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    m = d < D
    base = bh * T * D + d
    rbase = bh * T

    P = tl.zeros([BLOCK], dtype=tl.float32) + 1.0

    for t in range(T):
        o = base + t * D
        k = tl.load(k_ptr + o, mask=m, other=0.0)
        a = tl.load(A2_ptr + o, mask=m, other=0.0)
        Qv = tl.load(Q_ptr + o, mask=m, other=0.0)
        qv = tl.load(q_ptr + o, mask=m, other=0.0)
        r = tl.load(r_ptr + rbase + t)

        Ph = a * P * p_decay + Qv
        Pk = Ph * k
        den = tl.sum(tl.where(m, k * Pk, 0.0)) + r + _EPS
        K = Pk / den
        P = tl.minimum(tl.maximum((1.0 - K * k) * Ph, _P_LO), _P_HI)

        tl.store(K_ptr + o, K, mask=m)
        tl.store(P_ptr + o, P, mask=m)
        tl.store(unc_ptr + rbase + t, tl.sum(tl.where(m, qv * P * qv, 0.0)) + r)


@triton.jit
def _bwd(k_ptr, A2_ptr, Q_ptr, r_ptr, q_ptr, P_ptr,
         gK_ptr, gunc_ptr,
         gk_ptr, gA2_ptr, gQ_ptr, gr_ptr, gq_ptr,
         T: tl.constexpr, D: tl.constexpr, p_decay, BLOCK: tl.constexpr):
    bh = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    m = d < D
    base = bh * T * D + d
    rbase = bh * T

    gP = tl.zeros([BLOCK], dtype=tl.float32)

    for i in range(T):
        t = T - 1 - i
        o = base + t * D
        k = tl.load(k_ptr + o, mask=m, other=0.0)
        a = tl.load(A2_ptr + o, mask=m, other=0.0)
        Qv = tl.load(Q_ptr + o, mask=m, other=0.0)
        qv = tl.load(q_ptr + o, mask=m, other=0.0)
        r = tl.load(r_ptr + rbase + t)
        gK_ext = tl.load(gK_ptr + o, mask=m, other=0.0)
        gu_sc = tl.load(gunc_ptr + rbase + t)

        # P_{t-1}: saved value from the previous step, or the initial ones.
        Pprev = tl.where(t > 0,
                         tl.load(P_ptr + o - D, mask=m & (t > 0), other=0.0),
                         1.0)

        # --- recompute forward ---
        Ph = a * Pprev * p_decay + Qv
        Pk = Ph * k
        den = tl.sum(tl.where(m, k * Pk, 0.0)) + r + _EPS
        K = Pk / den
        u = (1.0 - K * k) * Ph
        Pt = tl.minimum(tl.maximum(u, _P_LO), _P_HI)

        # --- backward ---
        # unc = sum(q*P*q) + r
        gPt = gP + gu_sc * qv * qv
        gq = gu_sc * 2.0 * qv * Pt
        gr = gu_sc

        # P = clamp(u)
        gu = tl.where((u > _P_LO) & (u < _P_HI), gPt, 0.0)

        # u = Ph * (1 - K*k)
        gPh = gu * (1.0 - K * k)
        gK = gK_ext - gu * Ph * k
        gk = -gu * Ph * K

        # K = Pk / den
        gPk = gK / den
        gden = -tl.sum(tl.where(m, gK * K, 0.0)) / den

        # den = sum(k*Pk) + r + eps
        gk += gden * Pk
        gPk += gden * k
        gr += gden

        # Pk = Ph * k
        gPh += gPk * k
        gk += gPk * Ph

        # Ph = a*Pprev*p_decay + Q
        gA2 = gPh * Pprev * p_decay
        gQ = gPh
        gP = gPh * a * p_decay

        tl.store(gk_ptr + o, gk, mask=m)
        tl.store(gA2_ptr + o, gA2, mask=m)
        tl.store(gQ_ptr + o, gQ, mask=m)
        tl.store(gq_ptr + o, gq, mask=m)
        tl.store(gr_ptr + rbase + t, gr)


def _num_warps(block: int) -> int:
    """
    One warp per 64 lanes, not the fixed 4 the kernel started with.

    The kernel is a serial loop over T on a d_k-wide vector, so with d_k=64 a
    num_warps=4 launch asks for 128 threads to move 64 elements and half of
    them idle. Measured on a 3070 at B=8,H=6,T=512,d_k=64:
        num_warps=1  0.66 ms   <-- 1.56x faster
        num_warps=2  1.02 ms
        num_warps=4  1.03 ms   (previous setting)
        num_warps=8  1.04 ms
    """
    return max(1, min(8, block // 64))


class _KalmanScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, A2, Q, r, q, p_decay):
        k, A2, Q, r, q = (x.contiguous().float() for x in (k, A2, Q, r, q))
        B, H, T, D = k.shape
        BLOCK = triton.next_power_of_2(D)

        K = torch.empty_like(k)
        P = torch.empty_like(k)
        unc = torch.empty_like(r)

        _fwd[(B * H,)](k, A2, Q, r, q, K, P, unc,
                       T=T, D=D, p_decay=float(p_decay), BLOCK=BLOCK,
                       num_warps=_num_warps(BLOCK))

        ctx.save_for_backward(k, A2, Q, r, q, P)
        ctx.p_decay = float(p_decay)
        return K, unc

    @staticmethod
    def backward(ctx, gK, gunc):
        k, A2, Q, r, q, P = ctx.saved_tensors
        B, H, T, D = k.shape
        BLOCK = triton.next_power_of_2(D)

        gk, gA2, gQ, gq = (torch.empty_like(k) for _ in range(4))
        gr = torch.empty_like(r)

        _bwd[(B * H,)](k, A2, Q, r, q, P,
                       gK.contiguous().float(), gunc.contiguous().float(),
                       gk, gA2, gQ, gr, gq,
                       T=T, D=D, p_decay=ctx.p_decay, BLOCK=BLOCK,
                       num_warps=_num_warps(BLOCK))

        return gk, gA2, gQ, gr, gq, None


def kalman_gain_scan_triton(k, A2, Q, r, q, p_decay: float):
    return _KalmanScan.apply(k, A2, Q, r, q, p_decay)
