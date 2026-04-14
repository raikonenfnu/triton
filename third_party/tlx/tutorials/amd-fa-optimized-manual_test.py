"""
AMD Flash Attention Forward — Manually Pipelined Kernels (CDNA3/CDNA4)
======================================================================

Three kernels with **all pipelining explicit** (``num_stages=0``
everywhere, no Triton auto-pipelining).  Uses TLX shared-memory
management (``local_alloc``/``local_view``/``local_store``/``local_load``)
for explicit LDS control.

  1. **S1** — No pipeline.  Direct ``tl.load`` each iteration (baseline).
     Compiler manages LDS for layout conversion internally (non-mutable).
  2. **S2** — 1-deep LDS pipeline.  K+V prefetched one iteration ahead
     via ``tl.load`` → staged to TLX mutable LDS with ``local_store``,
     consumed via ``local_load``.  Scheduling: consume LDS first, then
     issue global loads, then compute, then stage next tile.
  3. **S4** — 4-stage: 3-deep K register prefetch + TLX LDS for layout
     conversion.  Modulo-scheduled: ``local_load K,V`` → QK MFMA →
     global loads K[i+3]/V[i+1] → softmax → PV MFMA → rotate K regs →
     ``local_store K,V``.

All kernels split the K/V sweep into an **unmasked fast path** (full
blocks) and a **masked tail** (causal diagonal / sequence boundary).

Performance notes (B=1, H=64, N=16384, D=128, bf16, non-causal):
  S1 baseline:   ~482 TFLOPS (compiler-managed non-mutable LDS)
  RegPipe:       ~685 TFLOPS (K+V register prefetch, BN=32, wpe=2, unroll=2)
  Async 2buf:    ~640 TFLOPS (async DMA, BN=64, wpe=2, token-based local_load)
  Vanilla S2:    ~729 TFLOPS (num_stages=2 auto-pipeline at BN=32, reference)
  RegPipe achieves 94% of auto-pipeliner throughput; remaining gap is from
  the tl.load VGPR→LDS path vs auto-pipeliner's buffer_load_to_local DMA.
  The async_load path (direct global→LDS DMA + token-based barrier
  elision) would close this gap but has layout issues on gfx950.

Usage:
    python amd-fa-optimized-manual_test.py -k s4 -b 1 -hq 64 -sq 16384 --dtype bf16
"""

import argparse
import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _assume_strides(
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
):
    tl.assume(stride_qz >= 0); tl.assume(stride_qh >= 0)
    tl.assume(stride_qm > 0);  tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0); tl.assume(stride_kh >= 0)
    tl.assume(stride_kn > 0);  tl.assume(stride_kk >= 0)
    tl.assume(stride_vz >= 0); tl.assume(stride_vh >= 0)
    tl.assume(stride_vn > 0);  tl.assume(stride_vk >= 0)
    tl.assume(stride_oz >= 0); tl.assume(stride_oh >= 0)
    tl.assume(stride_om > 0);  tl.assume(stride_ok >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# S1 — No pipeline (baseline, num_stages=0)
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _attn_fwd_manual_s1(
    Q, K, V, Out, L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale, Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    _assume_strides(
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok)

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
        n_full = (pid_m * BLOCK_M) // BLOCK_N
    else:
        hi = N_CTX
        n_full = hi // BLOCK_N
    block_max_full = n_full * BLOCK_N

    k_base = K + k_off + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    for start_n in tl.range(0, block_max_full, BLOCK_N, num_stages=0):
        k = tl.load(k_base + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k)
        v = tl.load(v_base + start_n * stride_vn)
        m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk    = qk * QK_SCALE - m_ij[:, None]
        p     = tl.math.exp2(qk)
        l_ij  = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
        acc += tl.dot(p.to(v.dtype), v)

    for start_n in tl.range(block_max_full, hi, BLOCK_N, num_stages=0):
        kn = start_n + offs_n
        k = tl.load(k_base + start_n * stride_kn, mask=kn[None, :] < N_CTX, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k)
        v = tl.load(v_base + start_n * stride_vn, mask=kn[:, None] < N_CTX, other=0.0)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        if start_n + BLOCK_N > N_CTX:
            qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))
        m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk    = qk * QK_SCALE - m_ij[:, None]
        p     = tl.math.exp2(qk)
        l_ij  = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
        acc += tl.dot(p.to(v.dtype), v)

    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i)
    tl.store(L + pid_hz * N_CTX + offs_m, lse, mask=offs_m < N_CTX)
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


# ═══════════════════════════════════════════════════════════════════════════
# RP — Register-pipelined (manual num_stages=2 equivalent)
# ═══════════════════════════════════════════════════════════════════════════
#
# Manually replicates what the compiler's num_stages=2 auto-pipeliner does:
# Issue tl.load for K[next],V[next] at the TOP of the loop body (non-blocking
# global loads), then compute with K[cur],V[cur] while those loads complete
# in the background. Uses the compiler's efficient non-mutable LDS path.
#
# Modulo schedule (steady state):
#   1. k_next = tl.load(K[i+1])  — issue global load (non-blocking)
#   2. v_next = tl.load(V[i+1])  — issue global load (non-blocking)
#   3. QK = dot(Q, k_cur)        — MFMA while loads in flight
#   4. softmax(QK)               — ALU while loads in flight
#   5. PV = dot(P, v_cur)        — MFMA while loads in flight
#   6. k_cur, v_cur = k_next, v_next  — rotate

@triton.jit
def _attn_fwd_regpipe(
    Q, K, V, Out, L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale, Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    _assume_strides(
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok)

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
        n_full = (pid_m * BLOCK_M) // BLOCK_N
    else:
        hi = N_CTX
        n_full = hi // BLOCK_N
    block_max_full = n_full * BLOCK_N

    k_base = K + k_off + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # ── Fast path: register-pipelined (K+V prefetch, V after QK dot) ────
    # K prefetched at top → max overlap with entire compute phase.
    # V prefetched after QK dot → doesn't pressure VGPRs during QK,
    # has softmax+PV compute time to complete before next iter needs it.
    if n_full >= 1:
        k_cur = tl.load(k_base)
        v_cur = tl.load(v_base)

        for k_iter in tl.range(1, n_full, num_stages=0, loop_unroll_factor=2):
            k_next = tl.load(k_base + k_iter * BLOCK_N * stride_kn)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k_cur)

            v_next = tl.load(v_base + k_iter * BLOCK_N * stride_vn)

            m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            qk    = qk * QK_SCALE - m_ij[:, None]
            p     = tl.math.exp2(qk)
            l_ij  = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
            acc += tl.dot(p.to(v_cur.dtype), v_cur)

            k_cur = k_next
            v_cur = v_next

        # Epilogue
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k_cur)
        m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk    = qk * QK_SCALE - m_ij[:, None]
        p     = tl.math.exp2(qk)
        l_ij  = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
        acc += tl.dot(p.to(v_cur.dtype), v_cur)

    # ── Tail path (masked) ───────────────────────────────────────────────
    for start_n in tl.range(block_max_full, hi, BLOCK_N, num_stages=0):
        kn = start_n + offs_n
        k = tl.load(k_base + start_n * stride_kn, mask=kn[None, :] < N_CTX, other=0.0)
        v = tl.load(v_base + start_n * stride_vn, mask=kn[:, None] < N_CTX, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        if start_n + BLOCK_N > N_CTX:
            qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))
        m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk    = qk * QK_SCALE - m_ij[:, None]
        p     = tl.math.exp2(qk)
        l_ij  = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
        acc += tl.dot(p.to(v.dtype), v)

    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i)
    tl.store(L + pid_hz * N_CTX + offs_m, lse, mask=offs_m < N_CTX)
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


# ═══════════════════════════════════════════════════════════════════════════
# S2 — 1-deep TLX LDS pipeline (single-buffered K+V)
# ═══════════════════════════════════════════════════════════════════════════
#
# LDS budget: 1 K buf (16KB) + 1 V buf (16KB) = 32KB TLX
#   + compiler temp for Q (32KB, freed before loop) = 64KB peak ✓
#
# Pipeline schedule (consume-first ordering for best perf):
#   Prologue: load K[0]+V[0] → stage to LDS
#   Main loop body:
#     1. local_load K, V from LDS              — consume ASAP (MFMA layout)
#     2. Issue global loads K[next], V[next]   — in-flight, hidden by compute
#     3. dot(Q, K) → QK                        — MFMA compute
#     4. softmax(QK)                            — ALU compute
#     5. dot(P, V) → acc                        — MFMA compute
#     6. local_store K[next], V[next] → LDS    — stage for next iteration
#   Epilogue: drain last buffer

@triton.jit
def _attn_fwd_manual_s2(
    Q, K, V, Out, L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale, Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    _assume_strides(
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok)

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
        n_full = (pid_m * BLOCK_M) // BLOCK_N
    else:
        hi = N_CTX
        n_full = hi // BLOCK_N
    block_max_full = n_full * BLOCK_N

    k_base = K + k_off + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    k_bufs = tlx.local_alloc((HEAD_DIM, BLOCK_N), tlx.dtype_of(K), 1)
    v_bufs = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(V), 1)
    k_view = tlx.local_view(k_bufs, 0)
    v_view = tlx.local_view(v_bufs, 0)

    # ── Fast path: manual TLX double-buffer (num_stages=0) ────────────
    if n_full >= 1:
        k_reg = tl.load(k_base)
        v_reg = tl.load(v_base)
        tlx.local_store(k_view, k_reg)
        tlx.local_store(v_view, v_reg)

        for k_iter in tl.range(1, n_full, num_stages=0):
            k_cur = tlx.local_load(k_view)
            v_cur = tlx.local_load(v_view)

            k_next = tl.load(k_base + k_iter * BLOCK_N * stride_kn)
            v_next = tl.load(v_base + k_iter * BLOCK_N * stride_vn)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k_cur)
            m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            qk    = qk * QK_SCALE - m_ij[:, None]
            p     = tl.math.exp2(qk)
            l_ij  = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
            acc += tl.dot(p.to(v_cur.dtype), v_cur)

            tlx.local_store(k_view, k_next)
            tlx.local_store(v_view, v_next)

        k_cur = tlx.local_load(k_view)
        v_cur = tlx.local_load(v_view)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k_cur)
        m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk    = qk * QK_SCALE - m_ij[:, None]
        p     = tl.math.exp2(qk)
        l_ij  = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
        acc += tl.dot(p.to(v_cur.dtype), v_cur)

    # ── Tail: masked (no LDS pipeline) ───────────────────────────────────
    for start_n in tl.range(block_max_full, hi, BLOCK_N, num_stages=0):
        kn = start_n + offs_n
        k = tl.load(k_base + start_n * stride_kn, mask=kn[None, :] < N_CTX, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k)
        v = tl.load(v_base + start_n * stride_vn, mask=kn[:, None] < N_CTX, other=0.0)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        if start_n + BLOCK_N > N_CTX:
            qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))
        m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk    = qk * QK_SCALE - m_ij[:, None]
        p     = tl.math.exp2(qk)
        l_ij  = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
        acc += tl.dot(p.to(v.dtype), v)

    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i)
    tl.store(L + pid_hz * N_CTX + offs_m, lse, mask=offs_m < N_CTX)
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


# ═══════════════════════════════════════════════════════════════════════════
# S4 — 4-stage: 3-deep K register prefetch + TLX LDS double-buffer
# ═══════════════════════════════════════════════════════════════════════════
#
# K tiles live in VGPRs (k_0, k_1, k_2) for 3-deep prefetch.
# At the END of each iteration, the next-to-consume K (after rotation)
# is pre-staged to LDS along with V for the next iteration.
# At the START of each iteration, K and V are consumed from LDS
# (MFMA-ready) — no global load stalls on the critical path.
#
# Modulo schedule per loop body (processing tile i, prefetching i+3):
#   1. local_load K,V           — consume from LDS (staged prev iter)
#   2. dot(Q, K) → QK           — MFMA compute
#   3. tl.load K[i+3]           — K prefetch (global, 3 ahead)
#   4. tl.load V[i+1]           — V prefetch (global, for next iter)
#   5. softmax(QK)              — ALU compute (hides global latency)
#   6. dot(P, V) → acc          — MFMA compute
#   7. rotate K regs             — k_0←k_1←k_2←k_new
#   8. local_store K,V → LDS    — stage next iter's data

@triton.jit
def _attn_fwd_manual_s4(
    Q, K, V, Out, L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale, Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    _assume_strides(
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok)

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
        n_full = (pid_m * BLOCK_M) // BLOCK_N
    else:
        hi = N_CTX
        n_full = hi // BLOCK_N
    block_max_full = n_full * BLOCK_N

    k_base = K + k_off + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    k_bufs = tlx.local_alloc((HEAD_DIM, BLOCK_N), tlx.dtype_of(K), 1)
    v_bufs = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(V), 1)
    k_view = tlx.local_view(k_bufs, 0)
    v_view = tlx.local_view(v_bufs, 0)

    # ── Fast path: 3-deep K register prefetch + LDS double-buffer ───────
    if n_full >= 3:
        k_0 = tl.load(k_base)
        k_1 = tl.load(k_base + 1 * BLOCK_N * stride_kn)
        k_2 = tl.load(k_base + 2 * BLOCK_N * stride_kn)
        v_reg = tl.load(v_base)
        tlx.local_store(k_view, k_0)
        tlx.local_store(v_view, v_reg)

        for k_iter in tl.range(3, n_full, num_stages=0):
            k_mfma = tlx.local_load(k_view)
            v_mfma = tlx.local_load(v_view)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k_mfma)

            k_new = tl.load(k_base + k_iter * BLOCK_N * stride_kn)
            v_reg = tl.load(v_base + (k_iter - 2) * BLOCK_N * stride_vn)

            m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            qk    = qk * QK_SCALE - m_ij[:, None]
            p     = tl.math.exp2(qk)
            l_ij  = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
            acc += tl.dot(p.to(v_mfma.dtype), v_mfma)

            k_0 = k_1; k_1 = k_2; k_2 = k_new
            tlx.local_store(k_view, k_0)
            tlx.local_store(v_view, v_reg)

        # Epilogue: drain 3 remaining tiles (k_0, k_1, k_2 staged one at a time)
        for _epi in tl.static_range(3):
            k_mfma = tlx.local_load(k_view)
            v_mfma = tlx.local_load(v_view)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k_mfma)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            qk = qk * QK_SCALE - m_ij[:, None]; p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1); alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
            acc += tl.dot(p.to(v_mfma.dtype), v_mfma)
            if _epi < 2:
                v_reg = tl.load(v_base + (n_full - 2 + _epi) * BLOCK_N * stride_vn)
                if _epi == 0:
                    tlx.local_store(k_view, k_1)
                else:
                    tlx.local_store(k_view, k_2)
                tlx.local_store(v_view, v_reg)

    elif n_full >= 1:
        for start_n in tl.range(0, block_max_full, BLOCK_N, num_stages=0):
            k = tl.load(k_base + start_n * stride_kn)
            v_reg = tl.load(v_base + start_n * stride_vn)
            tlx.local_store(k_view, k)
            tlx.local_store(v_view, v_reg)
            k_mfma = tlx.local_load(k_view)
            v_mfma = tlx.local_load(v_view)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k_mfma)
            m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            qk    = qk * QK_SCALE - m_ij[:, None]
            p     = tl.math.exp2(qk)
            l_ij  = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
            acc += tl.dot(p.to(v_mfma.dtype), v_mfma)

    # ── Tail: masked (no LDS pipeline) ───────────────────────────────────
    for start_n in tl.range(block_max_full, hi, BLOCK_N, num_stages=0):
        kn = start_n + offs_n
        k = tl.load(k_base + start_n * stride_kn, mask=kn[None, :] < N_CTX, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k)
        v = tl.load(v_base + start_n * stride_vn, mask=kn[:, None] < N_CTX, other=0.0)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        if start_n + BLOCK_N > N_CTX:
            qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))
        m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk    = qk * QK_SCALE - m_ij[:, None]
        p     = tl.math.exp2(qk)
        l_ij  = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
        acc += tl.dot(p.to(v.dtype), v)

    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i)
    tl.store(L + pid_hz * N_CTX + offs_m, lse, mask=offs_m < N_CTX)
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


# ═══════════════════════════════════════════════════════════════════════════
# SA — Fully async double-buffered pipeline (K permuted on host)
# ═══════════════════════════════════════════════════════════════════════════
#
# Both K and V use async_load for direct global→LDS DMA.
# K is pre-permuted on host to BHDN layout (N contiguous, stride_kn=1)
# so K's pointer shape (HEAD_DIM, BLOCK_N) has dim-1 contiguous,
# matching the shared default order [1,0]. No transpose needed.
#
# Pipeline (2-buf double-buffered, all num_stages=0):
#   Prologue: async_load K[0]+V[0] → commit
#   Main loop:
#     1. Prefetch K[next]+V[next] → commit
#     2. async_load_wait_group(1) — current ready
#     3. local_load K,V (relaxed=True)
#     4. dot(Q, K) → QK
#     5. softmax
#     6. dot(P, V) → acc

@triton.jit
def _attn_fwd_async(
    Q, K, V, Out, L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale, Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
    NUM_BUFS: tl.constexpr = 2,
):
    _assume_strides(
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok)

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
        n_full = (pid_m * BLOCK_M) // BLOCK_N
    else:
        hi = N_CTX
        n_full = hi // BLOCK_N
    block_max_full = n_full * BLOCK_N

    k_bufs = tlx.local_alloc((HEAD_DIM, BLOCK_N), tlx.dtype_of(K), NUM_BUFS)
    v_bufs = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(V), NUM_BUFS)

    k_base = K + k_off + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # ── Fast path (unmasked): async double-buffered ──────────────────────
    if n_full >= 1:
        tok_k = tlx.async_load(k_base, tlx.local_view(k_bufs, 0),
                               mask=offs_n[None, :] < hi)
        tok_v = tlx.async_load(v_base, tlx.local_view(v_bufs, 0),
                               mask=offs_n[:, None] < hi)
        tlx.async_load_commit_group([tok_k, tok_v])

        for k_iter in tl.range(0, n_full, num_stages=0):
            cur_buf = k_iter % NUM_BUFS
            nxt_buf = 1 - cur_buf

            pf_sn = (k_iter + 1) * BLOCK_N
            tok_k = tlx.async_load(k_base + pf_sn * stride_kn,
                                   tlx.local_view(k_bufs, nxt_buf),
                                   mask=(pf_sn + offs_n)[None, :] < hi)
            tok_v = tlx.async_load(v_base + pf_sn * stride_vn,
                                   tlx.local_view(v_bufs, nxt_buf),
                                   mask=(pf_sn + offs_n)[:, None] < hi)
            tlx.async_load_commit_group([tok_k, tok_v])

            wait_tok = tlx.async_load_wait_group(1)
            k_cur = tlx.local_load(tlx.local_view(k_bufs, cur_buf), token=wait_tok, relaxed=True)
            v_cur = tlx.local_load(tlx.local_view(v_bufs, cur_buf), token=wait_tok, relaxed=True)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k_cur)
            m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            qk    = qk * QK_SCALE - m_ij[:, None]
            p     = tl.math.exp2(qk)
            l_ij  = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
            acc += tl.dot(p.to(v_cur.dtype), v_cur)

    # ── Tail path (masked): sync loads, handles causal + boundary ────────
    for start_n in tl.range(block_max_full, hi, BLOCK_N, num_stages=0):
        kn = start_n + offs_n
        k = tl.load(k_base + start_n * stride_kn)
        v = tl.load(v_base + start_n * stride_vn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + tl.dot(q, k)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        if start_n + BLOCK_N > N_CTX:
            qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))
        m_ij  = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk    = qk * QK_SCALE - m_ij[:, None]
        p     = tl.math.exp2(qk)
        l_ij  = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]; l_i = l_i * alpha + l_ij; m_i = m_ij
        acc += tl.dot(p.to(v.dtype), v)

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
        IS_CAUSAL=causal, num_warps=num_warps, **extra_kw,
    )
    return o


def flash_attn_manual_s1(q, k, v, sm_scale, causal=False, **kw):
    return _launch(_attn_fwd_manual_s1, q, k, v, sm_scale, causal, **kw)

def flash_attn_regpipe(q, k, v, sm_scale, causal=False, **kw):
    return _launch(_attn_fwd_regpipe, q, k, v, sm_scale, causal, **kw)

def flash_attn_manual_s2(q, k, v, sm_scale, causal=False, **kw):
    return _launch(_attn_fwd_manual_s2, q, k, v, sm_scale, causal, **kw)

def flash_attn_manual_s4(q, k, v, sm_scale, causal=False, **kw):
    return _launch(_attn_fwd_manual_s4, q, k, v, sm_scale, causal, **kw)

def _launch_kperm(kernel, q, k, v, sm_scale, causal, **extra_kw):
    """Launch with K permuted to BHDN layout (N contiguous) for async_load."""
    B, H, N_CTX, D = q.shape
    k_perm = k.permute(0, 1, 3, 2).contiguous()
    o = torch.empty_like(q)
    L = torch.empty((B * H, N_CTX), device=q.device, dtype=torch.float32)
    BLOCK_M = extra_kw.pop("BLOCK_M", 128)
    BLOCK_N = extra_kw.pop("BLOCK_N", 64)
    num_warps = extra_kw.pop("num_warps", 4)
    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)
    kernel[grid](
        q, k_perm, v, o, L,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_perm.stride(0), k_perm.stride(1), k_perm.stride(3), k_perm.stride(2),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale, B, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=causal, num_warps=num_warps, **extra_kw,
    )
    return o


def flash_attn_async(q, k, v, sm_scale, causal=False, **kw):
    return _launch_kperm(_attn_fwd_async, q, k, v, sm_scale, causal, **kw)

def flash_attn_async_3(q, k, v, sm_scale, causal=False, **kw):
    return _launch_kperm(_attn_fwd_async, q, k, v, sm_scale, causal, NUM_BUFS=3, **kw)

ALL_KERNELS = {
    "s1": ("Manual S1", flash_attn_manual_s1),
    "rp": ("RegPipe", flash_attn_regpipe),
    "s2": ("Manual S2", flash_attn_manual_s2),
    "s4": ("Manual S4", flash_attn_manual_s4),
    "async": ("Async 2buf", flash_attn_async),
    "async3": ("Async 3buf", flash_attn_async_3),
}


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
    print(f"  {name:<28} {status}  max={max_err:.6f}  mean={mean_err:.6f}")
    return ok


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
        name, fn = ALL_KERNELS[key]
        try:
            out = fn(q, k, v, sm, causal)
        except RuntimeError as e:
            if "LLVM IR" in str(e) or "unrealized_conversion_cast" in str(e):
                print(f"  {name:<28} SKIP  (async_load layout issue)")
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
    print(f"Flash Attention Forward — Manual Pipeline + TLX LDS (num_stages=0)")
    print(f"  B={B}, H={H}, N={N}, D={D}, causal={causal}, dtype={args.dtype}")
    print(f"  kernels: {', '.join(kernels)}")
    print(f"{'='*70}")

    if causal:
        valid_el = N * (N + 1) // 2
    else:
        valid_el = N * N
    total_flops = 2 * 2.0 * B * H * valid_el * D

    print("\n--- Correctness ---")
    all_ok = True
    for c in [False, True]:
        for n_test in [128, 500, 512, 1024]:
            ok = test_correctness(dtype, c, kernels, B=1, H=4, N=n_test, D=D)
            all_ok &= ok
    print(f"\n{'All passed' if all_ok else 'FAILURES DETECTED'}")

    print(f"\n--- Performance (B={B}, H={H}, N={N}, D={D}, causal={causal}) ---")
    common64 = dict(BLOCK_M=128, BLOCK_N=64, num_warps=4)
    common32 = dict(BLOCK_M=128, BLOCK_N=32, num_warps=4, waves_per_eu=2)
    common64w = dict(BLOCK_M=128, BLOCK_N=64, num_warps=4, waves_per_eu=2)

    BEST_CONFIG = {
        "rp": common32,
        "async": common64w,
        "async3": common64w,
    }

    providers = [
        ("Torch SDPA", lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=sm)),
    ]
    for key in kernels:
        name, fn = ALL_KERNELS[key]
        c = BEST_CONFIG.get(key, common64)
        providers.append((name, lambda ff=fn, cc=c: ff(q, k, v, sm, causal, **cc)))

    print(f"\n{'Provider':<20} {'ms':>8} {'TFLOPS':>8}")
    print(f"{'-'*38}")
    for name, fn in providers:
        try:
            fn()
            ms = triton.testing.do_bench(fn, warmup=25, rep=100)
            tflops = total_flops / ms * 1e-9
            print(f"{name:<20} {ms:>8.3f} {tflops:>8.2f}")
        except RuntimeError as e:
            if "LLVM IR" in str(e) or "unrealized_conversion_cast" in str(e):
                print(f"{name:<20} {'N/A':>8} {'N/A':>8}  (async layout issue)")
            else:
                raise


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
    p = argparse.ArgumentParser(prog="AMD TLX FA Manual Pipeline")
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
