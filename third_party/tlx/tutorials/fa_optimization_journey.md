# FA TLX Optimization Journey Log

## Session Info
- **Date**: 2026-04-13
- **Target HW**: MI350 / CDNA4 (gfx950)
- **TLX Branch**: `hardcode84/pr3-insert-require-layout` (commit d4ec8c6d7)
- **Goal**: Implement and optimize BF16 Flash Attention forward pass using TLX APIs

---

## Phase 1: TTGIR Analysis from FAv3

### Step 1.1: Capture FAv3 TTGIR
- **Action**: Ran FAv3 with `TRITON_CACHE_DIR` to capture compiled TTGIR
- **Key findings from FAv3 TTGIR**:
  - Uses **BLOCK_M=256, BLOCK_N=64, 8 warps** (warp-specialized)
  - **2 LDS buffers** for both K (128x64) and V (64x128) via `ttg.memdesc<2x...>`
  - **`padded_shared` layout** (e.g., `#ttg.padded_shared<[512:+8]>`) for K/V in LDS
  - **`amdg.buffer_load_to_local`** for direct global→LDS async copy
  - **Block ping-pong** with `cond_barrier`, `s_barrier`, `sched_barrier`, `s_setprio`
  - QK and PV dots interleaved: QK dot starts before previous PV dot's V is loaded
  - `ttg.async_commit_group` / `ttg.async_wait` with token-based pipeline control
- **Decision**: TLX-AMD uses 4 warps (no warp specialization), so we adapt patterns for 4-warp execution

---

## Phase 2: Initial Smoke Test

### Step 2.1: Test existing kernels on `hardcode84/pr3-insert-require-layout`
- **Action**: Built TLX from new branch (d4ec8c6d7), ran all 3 kernels
- **Results**:
  - **Vanilla**: PASS (max_err=0.001953 vs SDPA)
  - **RegPipelined**: PASS (max_err=0.001953 vs SDPA)
  - **AsyncPipelined**: FAIL — `builtin.unrealized_conversion_cast` at LLVM translation
- **Root cause**: `tlx.local_alloc` on AMD creates `swizzled_shared<{vec=1, ...}>` layout, which is degenerate. FAv3 uses `padded_shared` layout optimized for buffer loads. The layout propagation pass cannot lower the conversion from this swizzled shared to register layout.
- **Decision**: Proceed with Vanilla + RegPipelined; async kernel kept in code but skipped at runtime

### Step 2.2: Minimal async reproduction
- **Action**: Created `test_async_minimal.py` with just async_load + local_load — confirms failure is in TLX infrastructure, not our kernel logic
- **TTGIR shows**: `ttg.local_load %k_smem_cur {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> tensor<128x64xbf16, #blocked>` — the `#shared` with vec=1 can't be lowered

---

## Phase 3: Full Correctness Verification (G2 Gate)

### Step 3.1: Full test matrix — bf16 + fp16
- **Action**: Ran complete test matrix per specification

| dtype | causal | B | H | N | D | Vanilla | RegPipelined |
|-------|--------|---|---|---|---|---------|-------------|
| bf16 | False | 1 | 4 | 512 | 128 | PASS | PASS |
| bf16 | True | 1 | 4 | 512 | 128 | PASS | PASS |
| bf16 | False | 2 | 4 | 128 | 128 | PASS | PASS |
| bf16 | True | 2 | 4 | 128 | 128 | PASS | PASS |
| bf16 | False | 1 | 4 | 1024 | 128 | PASS | PASS |
| bf16 | True | 1 | 4 | 1024 | 128 | PASS | PASS |
| bf16 | False | 1 | 4 | 500 | 128 | PASS (non-aligned) | PASS |
| bf16 | False | 1 | 4 | 64 | 128 | PASS (N < BLOCK_M) | PASS |
| fp16 | False | 1 | 4 | 512 | 128 | PASS | PASS |
| fp16 | True | 1 | 4 | 512 | 128 | PASS | PASS |
| fp16 | False | 2 | 4 | 128 | 128 | PASS | PASS |
| fp16 | True | 2 | 4 | 128 | 128 | PASS | PASS |
| fp16 | False | 1 | 4 | 1024 | 128 | PASS | PASS |
| fp16 | True | 1 | 4 | 1024 | 128 | PASS | PASS |
| fp16 | False | 1 | 4 | 500 | 128 | PASS | PASS |
| fp16 | False | 1 | 4 | 64 | 128 | PASS | PASS |

- **Correctness**: All 32 test configurations PASS (atol=2e-2, rtol=2e-2)
- **Cross-check**: Both explicit ref and SDPA ref agree; kernels match both

### Step 3.2: Benchmark-scale SDPA cross-check (B=1, H=64, N=16384, D=128)
- **Vanilla**: PASS (max_err=0.000244)
- **RegPipelined**: PASS (max_err=0.000244)

---

## Phase 4: Performance Benchmark

### Step 4.1: Target scale benchmark (B=1, H=64, N=16384, D=128, bf16, non-causal)

| Provider | Time (ms) | TFLOPS | Notes |
|----------|-----------|--------|-------|
| Torch SDPA | 13.460 | 653.50 | PyTorch built-in |
| FAv3 (triton) | ~11.2 | 784.56 | 8 warps, warp-specialized, padded_shared |
| FAv3 (torch) | ~13.9 | 629.33 | PyTorch SDPA from FAv3 environment |
| **TLX Vanilla** | **13.055** | **673.75** | 4 warps, Triton auto-pipelining |
| TLX RegPipelined | 22.747 | 386.68 | 4 warps, manual global→VGPR→LDS |
| TLX AsyncPipelined | N/A | N/A | Layout propagation failure |

### Analysis
- **TLX Vanilla beats Torch SDPA** by ~3% (673.75 vs 653.50 TFLOPS)
- **FAv3 is ~16% faster** than TLX Vanilla (784.56 vs 673.75 TFLOPS), expected since FAv3 uses:
  - 8 warps (vs 4) for more MFMA occupancy
  - Warp specialization (`async_task`) for producer/consumer overlap
  - `padded_shared` layout optimized for MFMA access patterns
  - Block ping-pong scheduling with `sched_barrier` / `s_setprio`
- **RegPipelined is slower** than Vanilla (386.68 vs 673.75 TFLOPS) because:
  - Extra VGPR→LDS→VGPR roundtrip adds overhead
  - Every iteration applies causal+boundary masks (no split-loop optimization)
  - The Triton auto-pipeliner in Vanilla is more efficient for this pattern

---

## Key Learnings

1. **Triton auto-pipelining (`num_stages=2`) is effective**: For FA on CDNA4, letting Triton handle K/V prefetching via `num_stages` produces good results without manual pipeline management.

2. **Split-loop pattern is critical**: Separating full (no-mask) blocks from masked (causal+boundary) blocks removes per-element branching from the hot path.

3. **`tlx.async_load` not yet functional on AMD**: The `local_alloc` API creates `swizzled_shared` with vec=1, which the LLVM lowering cannot handle. FAv3 uses `padded_shared` layout. This is a TLX infrastructure gap.

4. **FAv3's advantage is warp specialization**: The ~16% gap vs FAv3 is mainly from 8-warp warp-specialized execution, not from the async copy itself. TLX-AMD doesn't support `async_task`.

5. **RegPipelined overhead**: Manual register pipelining adds VGPR pressure and stalls that outweigh the benefits at this problem size. The technique is more beneficial when the auto-pipeliner can't handle complex patterns.

---

## Final Results (B=1 H=64 N=16384 D=128 bf16 non-causal)

| Kernel | ms | TFLOPS | vs SDPA |
|--------|----|--------|---------|
| Torch SDPA | 13.460 | 653.50 | 1.00x |
| TLX Vanilla | 13.055 | 673.75 | 1.03x |
| TLX RegPipelined | 22.747 | 386.68 | 0.59x |
| TLX AsyncPipelined | N/A | N/A | blocked |
| FAv3 (reference) | ~11.2 | 784.56 | 1.20x |

---

## Next Steps / Recommendations

1. **Fix `local_alloc` AMD layout**: Switch from `swizzled_shared` to `padded_shared` for async copy buffers
2. **Add warp specialization to TLX**: Support `async_task` for producer/consumer overlap (requires TLX-AMD backend work)
3. **Explore BLOCK_M=256**: FAv3 uses larger M blocks; test if this helps with 4 warps
4. **Add `sched_barrier` / `s_setprio` intrinsics**: Enable fine-grained scheduling control in TLX

---

## Run: 2026-04-14T05:19:39.401601


### ✓ All correctness checks passed


### Correctness — Benchmark-scale (B=1 H=64 N=16384 D=128)

| Kernel | Ref | Status | max_err | mean_err |
|--------|-----|--------|---------|----------|
| Vanilla | SDPA | ✅ PASS | 0.000244 | 0.000000 |
| RegPipelined | SDPA | ✅ PASS | 0.000244 | 0.000000 |

### Benchmark — B=1 H=64 N=16384 D=128 causal=False bf16

| Provider | Time (ms) | TFLOPS |
|----------|-----------|--------|
| Torch SDPA | 13.442 | 654.39 |
| TLX Vanilla | 13.074 | 672.82 |
| TLX RegPipelined | 22.776 | 386.20 |

---

## Run: 2026-04-14T07:42:22.120645


### ✓ All correctness checks passed


### Correctness — Benchmark-scale (B=1 H=64 N=16384 D=128)

| Kernel | Ref | Status | max_err | mean_err |
|--------|-----|--------|---------|----------|
| Vanilla | SDPA | ✅ PASS | 0.000244 | 0.000000 |
| RegPipelined | SDPA | ✅ PASS | 0.000244 | 0.000000 |

### Benchmark — B=1 H=64 N=16384 D=128 causal=False bf16

| Provider | Time (ms) | TFLOPS |
|----------|-----------|--------|
| Torch SDPA | 13.457 | 653.64 |
| TLX Vanilla | 13.075 | 672.72 |
| TLX RegPipelined | 22.777 | 386.19 |

---

## Run: 2026-04-14T08:33:23.583517


### ✓ All correctness checks passed


### Correctness — Benchmark-scale (B=1 H=64 N=16384 D=128)

| Kernel | Ref | Status | max_err | mean_err |
|--------|-----|--------|---------|----------|
| Vanilla | SDPA | ✅ PASS | 0.000244 | 0.000000 |
| RegPipelined | SDPA | ✅ PASS | 0.000244 | 0.000000 |

### Benchmark — B=1 H=64 N=16384 D=128 causal=False bf16

| Provider | Time (ms) | TFLOPS |
|----------|-----------|--------|
| Torch SDPA | 13.451 | 653.93 |
| TLX Vanilla | 13.066 | 673.19 |
| TLX RegPipelined | 22.760 | 386.46 |
