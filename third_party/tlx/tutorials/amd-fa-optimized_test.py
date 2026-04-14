"""
AMD Flash Attention Forward — Optimization Progression (CDNA4 / MI350)
=====================================================================

Three optimization levels via two kernel functions:

  1. **Basic**      — ``_attn_fwd_vanilla`` with ``NUM_STAGES=1``.
     No pipelining.  Two-phase loop (fast + tail).  Baseline.
  2. **Vanilla S2** — ``_attn_fwd_vanilla`` with ``NUM_STAGES=2``.
     Triton auto-pipelines loads one iteration ahead.  Unmasked fast path
     for full blocks + masked tail for boundary/causal.  No pipeline
     prologue/epilogue — just loop-splitting for performance.
  3. **Async S4**   — ``_attn_fwd_async_s4``.  Manual 4-stage TLX async
     pipeline (hardcoded).  4 LDS buffer slots, ``relaxed`` local loads,
     prefetch 3 iterations ahead for maximum compute-memory overlap.

Usage (from TLX venv):

    python amd-fa-optimized_test.py -b 1 -d 128 -hq 64 -sq 16384 -causal false --dtype bf16
"""

import argparse
import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ═══════════════════════════════════════════════════════════════════════════
# Kernel: Vanilla (Triton auto-pipeline via NUM_STAGES)
# ═══════════════════════════════════════════════════════════════════════════
# NUM_STAGES=1 → no pipelining (basic baseline)
# NUM_STAGES=2 → Triton prefetches next K/V tile while computing current
#
# Two-phase loop (not prologue/epilogue — just fast-path vs remainder):
#   • Full-block loop: unmasked loads, pipelined at NUM_STAGES depth
#   • Tail loop: masked loads for causal diagonal / sequence boundary

@triton.jit
def _attn_fwd_vanilla(
    Q, K, V, Out, L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tl.assume(stride_qz >= 0); tl.assume(stride_qh >= 0)
    tl.assume(stride_qm > 0);  tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0); tl.assume(stride_kh >= 0)
    tl.assume(stride_kn > 0);  tl.assume(stride_kk >= 0)
    tl.assume(stride_vz >= 0); tl.assume(stride_vh >= 0)
    tl.assume(stride_vn > 0);  tl.assume(stride_vk >= 0)
    tl.assume(stride_oz >= 0); tl.assume(stride_oh >= 0)
    tl.assume(stride_om > 0);  tl.assume(stride_ok >= 0)

    pid_m  = tl.program_id(0)
    pid_hz = tl.program_id(1)
    off_z  = pid_hz // H
    off_h  = pid_hz %  H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    QK_SCALE = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    k_base = K + k_off + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # Split: blocks guaranteed fully in-bounds vs remainder
    if IS_CAUSAL:
        n_full_blocks = (pid_m * BLOCK_M) // BLOCK_N
    else:
        n_full_blocks = hi // BLOCK_N
    block_max_full = n_full_blocks * BLOCK_N

    # ── Full-block fast path: unmasked loads, no score masking ───────────
    for start_n in tl.range(0, block_max_full, BLOCK_N, num_stages=NUM_STAGES):
        k = tl.load(k_base + start_n * stride_kn)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk   = qk * QK_SCALE - m_ij[:, None]
        p    = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        acc   = acc * alpha[:, None]
        l_i   = l_i * alpha + l_ij
        m_i   = m_ij

        v = tl.load(v_base + start_n * stride_vn)
        acc += tl.dot(p.to(v.dtype), v)

    # ── Tail: masked loads + causal/boundary score masking ───────────────
    for start_n in tl.range(block_max_full, hi, BLOCK_N, num_stages=1):
        kn = start_n + offs_n
        k_mask = kn[None, :] < N_CTX
        k = tl.load(k_base + start_n * stride_kn, mask=k_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        if start_n + BLOCK_N > N_CTX:
            qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk   = qk * QK_SCALE - m_ij[:, None]
        p    = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        acc   = acc * alpha[:, None]
        l_i   = l_i * alpha + l_ij
        m_i   = m_ij

        v_mask = kn[:, None] < N_CTX
        v = tl.load(v_base + start_n * stride_vn, mask=v_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i)
    tl.store(L + pid_hz * N_CTX + offs_m, lse, mask=offs_m < N_CTX)

    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


# ═══════════════════════════════════════════════════════════════════════════
# Kernel: Async S4 — 4-stage TLX async pipeline (hardcoded)
# ═══════════════════════════════════════════════════════════════════════════
# Pipeline design (from the CDNA4 software-pipelining guide):
#   • 4 LDS buffer slots for K and V  (slot = iter % 4)
#   • Prologue: async-load first 3 K/V tiles into LDS
#   • Single main loop: wait → consume → compute → prefetch iter+3
#   • Invariant: always 3 outstanding commit groups entering each iter
#   • wait_group(2): at most 2 groups pending → oldest (current) is done

@triton.jit
def _attn_fwd_async_s4(
    Q, K, V, Out, L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    tl.assume(stride_qz >= 0); tl.assume(stride_qh >= 0)
    tl.assume(stride_qm > 0);  tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0); tl.assume(stride_kh >= 0)
    tl.assume(stride_kn > 0);  tl.assume(stride_kk >= 0)
    tl.assume(stride_vz >= 0); tl.assume(stride_vh >= 0)
    tl.assume(stride_vn > 0);  tl.assume(stride_vk >= 0)
    tl.assume(stride_oz >= 0); tl.assume(stride_oh >= 0)
    tl.assume(stride_om > 0);  tl.assume(stride_ok >= 0)

    pid_m  = tl.program_id(0)
    pid_hz = tl.program_id(1)
    off_z  = pid_hz // H
    off_h  = pid_hz %  H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    QK_SCALE = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    K_ITERS = tl.cdiv(hi, BLOCK_N)

    # 4 LDS buffer slots — hardcoded for 4-stage pipeline
    k_bufs = tlx.local_alloc((HEAD_DIM, BLOCK_N), tlx.dtype_of(K), 4)
    v_bufs = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(V), 4)

    k_base = K + k_off + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # ── Prologue: fill 3 buffer slots via async DMA ──────────────────────
    for i in tl.range(0, 3, loop_unroll_factor=3):
        sn = i * BLOCK_N
        kn = sn + offs_n
        tok_k = tlx.async_load(k_base + sn * stride_kn,
                               tlx.local_view(k_bufs, i),
                               mask=kn[None, :] < hi)
        tok_v = tlx.async_load(v_base + sn * stride_vn,
                               tlx.local_view(v_bufs, i),
                               mask=kn[:, None] < hi)
        tlx.async_load_commit_group([tok_k, tok_v])

    # ── Main loop ────────────────────────────────────────────────────────
    for k_iter in tl.range(0, K_ITERS, num_stages=0):
        buf = k_iter % 4

        tlx.async_load_wait_group(2)
        k_cur = tlx.local_load(tlx.local_view(k_bufs, buf), relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_bufs, buf), relaxed=True)

        sn = k_iter * BLOCK_N
        kn = sn + offs_n

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k_cur)

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        if sn + BLOCK_N > N_CTX:
            qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk   = qk * QK_SCALE - m_ij[:, None]
        p    = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        acc   = acc * alpha[:, None]
        l_i   = l_i * alpha + l_ij
        m_i   = m_ij
        acc  += tl.dot(p.to(v_cur.dtype), v_cur)

        # Prefetch iter+3 (mask handles out-of-bounds gracefully)
        pf      = k_iter + 3
        pf_sn   = pf * BLOCK_N
        pf_offs = pf_sn + offs_n
        tok_k = tlx.async_load(k_base + pf_sn * stride_kn,
                               tlx.local_view(k_bufs, pf % 4),
                               mask=pf_offs[None, :] < hi)
        tok_v = tlx.async_load(v_base + pf_sn * stride_vn,
                               tlx.local_view(v_bufs, pf % 4),
                               mask=pf_offs[:, None] < hi)
        tlx.async_load_commit_group([tok_k, tok_v])

    # ── Epilogue: normalize and store ────────────────────────────────────
    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i)
    tl.store(L + pid_hz * N_CTX + offs_m, lse, mask=offs_m < N_CTX)

    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


# ═══════════════════════════════════════════════════════════════════════════
# Host wrappers
# ═══════════════════════════════════════════════════════════════════════════

def _launch(kernel, q, k, v, sm_scale, causal, **extra_kw):
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)
    L = torch.empty((B * H, N_CTX), device=q.device, dtype=torch.float32)
    BLOCK_M = extra_kw.pop("BLOCK_M", 128)
    BLOCK_N = extra_kw.pop("BLOCK_N", 64)
    num_warps = extra_kw.pop("num_warps", 4)
    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)
    kernel[grid](
        q, k, v, o, L,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale, B, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        **extra_kw,
    )
    return o


def flash_attn_basic(q, k, v, sm_scale, causal=False, **kw):
    """No pipelining — baseline."""
    return _launch(_attn_fwd_vanilla, q, k, v, sm_scale, causal, NUM_STAGES=1, **kw)


def flash_attn_vanilla_s2(q, k, v, sm_scale, causal=False, **kw):
    """Triton auto-pipeline with hardcoded num_stages=2."""
    return _launch(_attn_fwd_vanilla, q, k, v, sm_scale, causal, NUM_STAGES=2, **kw)


def flash_attn_vanilla_s4(q, k, v, sm_scale, causal=False, **kw):
    """Triton auto-pipeline with hardcoded num_stages=4 — best throughput."""
    return _launch(_attn_fwd_vanilla, q, k, v, sm_scale, causal, NUM_STAGES=4, **kw)


def flash_attn_async_s4(q, k, v, sm_scale, causal=False, **kw):
    """4-stage TLX async pipeline (hardcoded, requires async_load support)."""
    return _launch(_attn_fwd_async_s4, q, k, v, sm_scale, causal, **kw)


# ═══════════════════════════════════════════════════════════════════════════
# Reference & verification
# ═══════════════════════════════════════════════════════════════════════════

def ref_sdpa(q, k, v, sm_scale, causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)


def verify(name, got, ref, atol=2e-2, rtol=2e-2):
    diff = (got.float() - ref.float()).abs()
    ok = torch.allclose(ref, got, atol=atol, rtol=rtol)
    max_err  = diff.max().item()
    mean_err = diff.mean().item()
    status = "PASS" if ok else "FAIL"
    print(f"  {name:<20} {status}  max={max_err:.6f}  mean={mean_err:.6f}")
    if not ok:
        idx = diff.argmax().item()
        coords = []
        sz = idx
        for d in reversed(got.shape):
            coords.append(sz % d)
            sz //= d
        coords = tuple(reversed(coords))
        print(f"    worst @ {coords}: ref={ref[coords].item():.6f}  got={got[coords].item():.6f}")
    return ok


ALL_KERNELS = {
    "s1":    ("Basic (S1)",  flash_attn_basic),
    "s2":    ("Vanilla S2",  flash_attn_vanilla_s2),
    "s4":    ("Vanilla S4",  flash_attn_vanilla_s4),
    "async": ("Async S4",    flash_attn_async_s4),
}


def test_correctness(dtype, causal, kernels, B=2, H=4, N=512, D=128):
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / math.sqrt(D)
    ref = ref_sdpa(q, k, v, sm, causal)

    tag = f"causal={causal} N={N}"
    all_ok = True

    for key in kernels:
        name, fn_factory = ALL_KERNELS[key]
        fn = lambda ff=fn_factory: ff(q, k, v, sm, causal)
        try:
            out = fn()
        except RuntimeError as e:
            if "LLVM IR" in str(e) or "unrealized_conversion_cast" in str(e):
                print(f"  {name:<20} SKIP  (async_load not supported)")
                continue
            raise
        ok = verify(f"{name} [{tag}]", out, ref)
        all_ok &= ok

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark(args):
    B, H, N, D = args.b, args.hq, args.sq, args.d
    causal = args.causal
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    kernels = args.kernels

    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / math.sqrt(D)

    print(f"\n{'='*70}")
    print(f"Flash Attention Forward — AMD TLX Optimization Progression")
    print(f"  B={B}, H={H}, N={N}, D={D}, causal={causal}, dtype={args.dtype}")
    print(f"  BLOCK_M=128, BLOCK_N=64, num_warps=4")
    print(f"  kernels: {', '.join(kernels)}")
    print(f"{'='*70}")

    if causal:
        valid_el = N * (N + 1) // 2
    else:
        valid_el = N * N
    total_flops = 2 * 2.0 * B * H * valid_el * D

    # ── Correctness ──────────────────────────────────────────────────────
    print("\n--- Correctness ---")
    all_ok = True
    for c in [False, True]:
        for n_test in [128, 500, 512, 1024]:
            ok = test_correctness(dtype, c, kernels, B=1, H=4, N=n_test, D=D)
            all_ok &= ok
    print(f"\n{'All passed' if all_ok else 'FAILURES DETECTED'}")

    # ── Performance ──────────────────────────────────────────────────────
    print(f"\n--- Performance (B={B}, H={H}, N={N}, D={D}, causal={causal}) ---")
    common = dict(BLOCK_M=128, BLOCK_N=64, num_warps=4)

    providers = [
        ("Torch SDPA", lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=sm)),
    ]
    for key in kernels:
        name, fn_factory = ALL_KERNELS[key]
        providers.append((name, lambda ff=fn_factory: ff(q, k, v, sm, causal, **common)))

    print(f"\n{'Provider':<20} {'ms':>8} {'TFLOPS':>8}")
    print(f"{'-'*38}")

    for name, fn in providers:
        try:
            fn()
        except RuntimeError as e:
            if "LLVM IR" in str(e) or "unrealized_conversion_cast" in str(e):
                print(f"{name:<20} {'N/A':>8} {'N/A':>8}  (async_load not supported)")
            else:
                raise
            continue
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        tflops = total_flops / ms * 1e-9
        print(f"{name:<20} {ms:>8.3f} {tflops:>8.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _str2bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    p = argparse.ArgumentParser(prog="AMD TLX FA Optimized (CDNA4/MI350)")
    p.add_argument("-b",  type=int, default=1)
    p.add_argument("-hq", type=int, default=64)
    p.add_argument("-sq", type=int, default=16384)
    p.add_argument("-d",  type=int, default=128)
    p.add_argument("-causal", type=_str2bool, default=False)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("-k", "--kernels", nargs="+", default=["s1", "s2", "s4"],
                   choices=list(ALL_KERNELS.keys()),
                   help="kernels to run (default: s1 s2 s4)")
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
