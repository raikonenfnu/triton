def fa_gluon():
    import torch
    from triton.experimental import gluon
    import triton.experimental.gluon.language as gl

    @gluon.jit
    def attn_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_on,
        SM_SCALE: gl.constexpr,
        SEQLEN_Q: gl.constexpr,
        SEQLEN_K: gl.constexpr,
        NUM_Q_HEADS: gl.constexpr,
        NUM_K_HEADS: gl.constexpr,
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        HEAD_SZ: gl.constexpr,
        BATCH: gl.constexpr,
    ):
        BLOCK_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1 ,1], [1, 0])
        MFMA_QK_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 1])
        MFMA_PV_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 16], False, [1, 1])

        n_blocks_m = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

        # workgroup id ranging: 0,1,2,...., (BATCH * NUM_Q_HEADS * NUM_BLOCKS - 1)
        wid = gl.program_id(0)
        n_blocks_n = (seqlen_k + BLOCK_N - 1) // BLOCK_N

        # offsets
        off_q_head = wid % NUM_Q_HEADS
        off_k_head = off_q_head
        start_m = (wid // NUM_Q_HEADS) % n_blocks_m
        off_m = start_m * BLOCK_M

        off_z = (wid // (n_blocks_m * NUM_Q_HEADS)) % BATCH

        # q [BLOCK_M, HEAD_SZ]
        q_offs = (
            stride_qz * off_z
            + stride_qh * off_q_head
            + stride_qm * (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, BLOCK_LAYOUT)))[:, None]
            + stride_qk * (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, BLOCK_LAYOUT)))[None, :]
        )

        # k [BLOCK_N, HEAD_SZ]
        k_offs = (
            stride_kz * off_z
            + stride_kh * off_k_head
            + stride_kn * gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, BLOCK_LAYOUT))[:, None]
            + stride_kk * gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, BLOCK_LAYOUT))[None, :]
        )

        # v [BLOCK_N, BLOCK_DMODEL]
        v_offs = (
            stride_vz * off_z
            + stride_vh * off_k_head
            + stride_vn * gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, BLOCK_LAYOUT))[:, None]
            + stride_vk * gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, BLOCK_LAYOUT))[None, :]
        )

        m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, MFMA_PV_LAYOUT))
        l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, MFMA_PV_LAYOUT))
        acc = gl.zeros([BLOCK_M, HEAD_SZ], dtype=gl.float32, layout=MFMA_PV_LAYOUT)

        q_mask = (off_m + gl.arange(0, BLOCK_M))[:, None] < seqlen_q
        q = gl.amd.cdna4.buffer_load(q_ptr, q_offs, q_mask, 0.0)
        q = gl.convert_layout(q, gl.DotOperandLayout(0, MFMA_QK_LAYOUT, 4))

        block_min = 0
        block_max = n_blocks_n * BLOCK_N

        RCP_LN2: gl.constexpr = 1.4426950408889634

        for block_id in range(block_min, block_max, BLOCK_N):
            k = gl.amd.cdna4.buffer_load(k_ptr, k_offs)
            k = k.T
            k = gl.convert_layout(k, gl.DotOperandLayout(1, MFMA_QK_LAYOUT, 4))

            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=MFMA_QK_LAYOUT)
            qk = gl.amd.cdna4.mfma(q, k, qk)
            qk = gl.convert_layout(qk, MFMA_PV_LAYOUT)
            qk_scaled = qk * SM_SCALE * RCP_LN2

            # get max scores so far
            m_ij_scaled = gl.maximum(m_i, gl.max(qk_scaled, 1))

            # scale and subtract max
            q_shifted = qk_scaled - m_ij_scaled[:, None]

            # Compute scaled QK and softmax probabilities
            p = gl.exp2(q_shifted)

            # update l_ij before applying dropout
            l_ij = gl.sum(p, 1)

            # update output accumulator
            # alpha is an adjustment factor for acc and li as we loop and find new maxes
            # store the diff in maxes to adjust acc and li as we discover new maxes
            m_diff_scaled = m_i - m_ij_scaled
            alpha = gl.exp2(m_diff_scaled)
            acc = acc * alpha[:, None]

            v = gl.amd.cdna4.buffer_load(v_ptr, v_offs)
            v = gl.convert_layout(v, gl.DotOperandLayout(1, MFMA_PV_LAYOUT, 4))

            l_i = l_i * alpha + l_ij
            m_i = m_ij_scaled

            p = p.to(gl.float16)
            p = gl.convert_layout(p, gl.DotOperandLayout(0, MFMA_PV_LAYOUT, 4))
            acc = gl.amd.cdna4.mfma(p, v, acc)

            k_offs += BLOCK_N * stride_kn
            v_offs += BLOCK_N * stride_vn


        l_recip = 1 / l_i[:, None]
        acc = acc * l_recip

        out_offs = (
            stride_oz * off_z
            + stride_oh * off_q_head
            + stride_om * (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, MFMA_PV_LAYOUT)))[:, None]
            + stride_on * (gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, MFMA_PV_LAYOUT)))[None, :]
        )

        out_mask = gl.full([BLOCK_M, HEAD_SZ], 1, dtype=gl.int1, layout=MFMA_PV_LAYOUT)
        out_mask &= (off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, MFMA_PV_LAYOUT)))[:, None] < seqlen_q

        op = acc.to(out_ptr.dtype.element_ty)
        gl.amd.cdna4.buffer_store(op, out_ptr, out_offs)

    def attn_fwd(
        BATCH: int,
        SEQLEN_Q: int,
        SEQLEN_K: int,
        NUM_Q_HEADS: int,
        NUM_K_HEADS: int,
        HEAD_SZ: int,
        BLOCK_M,
        BLOCK_N,
    ):
        q = 2 * torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device='cuda', dtype=torch.float16)
        k = 3 * torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device='cuda', dtype=torch.float16)
        v = 4 * torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device='cuda', dtype=torch.float16)
        sm_scale = 1.0 / (HEAD_SZ ** 0.5)

        q = q.permute(0, 2, 1, 3).contiguous() # [BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ]
        k = k.permute(0, 2, 1, 3).contiguous() # [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
        v = v.permute(0, 2, 1, 3).contiguous() # [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ]
        o = torch.zeros_like(q, dtype=torch.float32)

        grid = (BATCH * NUM_Q_HEADS * ((SEQLEN_Q + BLOCK_M - 1) // BLOCK_M),)
        pgm = attn_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            sm_scale,
            SEQLEN_Q,
            SEQLEN_K,
            NUM_Q_HEADS,
            NUM_K_HEADS,
            BLOCK_M,
            BLOCK_N,
            HEAD_SZ,
            BATCH,
            num_warps=1,
        )
        ref = torch.nn.functional.scaled_dot_product_attention(q,k,v)
        rtol = 0.02
        atol = 0.02
        torch.testing.assert_allclose(o, ref, rtol=rtol, atol=atol)
        print("PASS")

    attn_fwd(
        BATCH=16,
        SEQLEN_Q=64,
        SEQLEN_K=64,
        NUM_Q_HEADS=8,
        NUM_K_HEADS=8,
        HEAD_SZ=128,
        BLOCK_M=16,
        BLOCK_N=32,
    )


if __name__ == "__main__":
    fa_gluon()
