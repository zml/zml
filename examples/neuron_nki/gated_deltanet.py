"""Gated DeltaNet NKI kernel for Inferentia2 / Trainium.

Algorithm (per batch / per value-head, S timesteps):
    state[k, v] starts from h0
    for t in range(S):
        q_n = l2_normalize(q[t]) * scale            # (Dk,)
        k_n = l2_normalize(k[t])                    # (Dk,)
        state *= exp(g[t])                          # decay
        pred_v[v]  = sum_k state[k, v] * k_n[k]
        delta_v[v] = beta[t] * (v[t, v] - pred_v[v])
        state[k, v] += k_n[k] * delta_v[v]          # rank-1 update
        out[t, v]   = sum_k state[k, v] * q_n[k]

The state tile is kept resident in SBUF for the full sequence to avoid
HBM traffic per timestep. All three contractions go through the PE array
via ``nl.matmul``. Layout in SBUF is (partition = Dk, free = Dv) so
``nl.matmul(state, x_kp)`` computes the v-dim output naturally.
"""

import math

import nki.language as nl
import nki.isa as nisa


def gated_deltanet(q, k, v, g, beta, h0):
    batch_size, seq_len, num_q_heads, key_dim = q.shape
    v_batch_size, v_seq_len, num_value_heads, value_dim = v.shape

    assert q.shape == k.shape
    assert batch_size == v_batch_size
    assert seq_len == v_seq_len
    assert num_value_heads % num_q_heads == 0
    assert g.shape == (batch_size, seq_len, num_value_heads)
    assert beta.shape == (batch_size, seq_len, num_value_heads)
    assert h0.shape == (batch_size, num_value_heads, value_dim, key_dim)
    assert key_dim == nl.tile_size.pmax
    assert value_dim == nl.tile_size.pmax

    qk_head_repetition = num_value_heads // num_q_heads
    scale = 1.0 / math.sqrt(key_dim)
    eps = 1.0e-6

    output = nl.ndarray((batch_size, seq_len, value_dim, num_value_heads),
                        dtype=nl.float32, buffer=nl.shared_hbm)
    ht = nl.ndarray(h0.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    # All scratch tiles below are allocated once per kernel and reused.
    # PSUM scratch (small).
    kp_psum = nl.ndarray((key_dim, 1), dtype=nl.float32, buffer=nl.psum)
    mm_psum = nl.ndarray((value_dim, 1), dtype=nl.float32, buffer=nl.psum)
    tp_psum = nl.ndarray((1, value_dim), dtype=nl.float32, buffer=nl.psum)

    # Outer-product chunked update tiles.
    v_chunk = 32
    assert value_dim % v_chunk == 0
    n_v_chunks = value_dim // v_chunk
    update_psum_chunk = nl.ndarray((key_dim, v_chunk),
                                   dtype=nl.float32, buffer=nl.psum)
    update_sbuf_chunk = nl.ndarray((key_dim, v_chunk),
                                   dtype=nl.float32, buffer=nl.sbuf)

    # State tile (Dk part, Dv free) - one tile reused across all (b, vh).
    state = nl.ndarray((key_dim, value_dim), dtype=nl.float32, buffer=nl.sbuf)

    # Per-step SBUF scratch.
    q_row = nl.ndarray((1, key_dim), dtype=nl.float32, buffer=nl.sbuf)
    k_row = nl.ndarray((1, key_dim), dtype=nl.float32, buffer=nl.sbuf)
    q_sq = nl.ndarray((1, key_dim), dtype=nl.float32, buffer=nl.sbuf)
    k_sq = nl.ndarray((1, key_dim), dtype=nl.float32, buffer=nl.sbuf)
    q_sumsq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    k_sumsq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    q_inv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    k_inv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    q_scaled = nl.ndarray((1, key_dim), dtype=nl.float32, buffer=nl.sbuf)
    k_scaled = nl.ndarray((1, key_dim), dtype=nl.float32, buffer=nl.sbuf)
    q_kp = nl.ndarray((key_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    k_kp = nl.ndarray((key_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    v_row = nl.ndarray((1, value_dim), dtype=nl.float32, buffer=nl.sbuf)
    beta_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    g_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    decay = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    decay_b = nl.ndarray((key_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    predicted_v = nl.ndarray((value_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    predicted_v_T = nl.ndarray((1, value_dim), dtype=nl.float32, buffer=nl.sbuf)
    diff = nl.ndarray((1, value_dim), dtype=nl.float32, buffer=nl.sbuf)
    delta_v = nl.ndarray((1, value_dim), dtype=nl.float32, buffer=nl.sbuf)
    out_v = nl.ndarray((value_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    chunk = 32
    assert key_dim % chunk == 0 and value_dim % chunk == 0
    final_tile = nl.ndarray((chunk, chunk), dtype=nl.float32, buffer=nl.sbuf)

    for i_batch in nl.affine_range(batch_size):
        for i_value_head in nl.affine_range(num_value_heads):
            i_q_head = i_value_head // qk_head_repetition

            # --- Initialize SBUF state from h0 ---
            # HBM layout for h0[b, vh, :, :] is (value_dim, key_dim).
            # We load it transposed so SBUF state is (key_dim part, value_dim free).
            state_init = nl.load_transpose2d(h0[i_batch, i_value_head,
                                                0:value_dim, 0:key_dim])
            nisa.tensor_copy(dst=state, src=state_init)

            for i_seq in nl.sequential_range(seq_len):
                # ---- Load q/k rows for (b, t, qh) and L2-normalize ----
                nisa.dma_copy(dst=q_row, src=q[i_batch, i_seq,
                                               i_q_head:i_q_head+1, 0:key_dim])
                nisa.dma_copy(dst=k_row, src=k[i_batch, i_seq,
                                               i_q_head:i_q_head+1, 0:key_dim])

                nisa.tensor_tensor(dst=q_sq, data1=q_row, data2=q_row,
                                   op=nl.multiply)
                nisa.tensor_tensor(dst=k_sq, data1=k_row, data2=k_row,
                                   op=nl.multiply)
                nisa.tensor_reduce(dst=q_sumsq, data=q_sq, op=nl.add,
                                   axis=1, keepdims=True)
                nisa.tensor_reduce(dst=k_sumsq, data=k_sq, op=nl.add,
                                   axis=1, keepdims=True)
                # rsqrt(x + eps)
                nisa.activation(dst=q_inv, op=nl.rsqrt, data=q_sumsq, bias=eps)
                nisa.activation(dst=k_inv, op=nl.rsqrt, data=k_sumsq, bias=eps)

                nisa.tensor_scalar(dst=q_scaled, data=q_row,
                                   op0=nl.multiply, operand0=q_inv,
                                   op1=nl.multiply, operand1=scale)
                nisa.tensor_scalar(dst=k_scaled, data=k_row,
                                   op0=nl.multiply, operand0=k_inv)

                # ---- Decay: state *= exp(g[b, t, vh]) ----
                nisa.dma_copy(dst=g_tile, src=g[i_batch, i_seq:i_seq+1,
                                                i_value_head:i_value_head+1])
                nisa.activation(dst=decay, op=nl.exp, data=g_tile)
                nisa.tensor_copy(dst=decay_b,
                                 src=nl.broadcast_to(decay, (key_dim, 1)))
                nisa.tensor_scalar(dst=state, data=state,
                                   op0=nl.multiply, operand0=decay_b)

                # ---- predicted_v = state.T @ k_n  (k_n in (Dk part, 1 free)) ----
                nisa.nc_transpose(dst=kp_psum, data=k_scaled)
                nisa.tensor_copy(dst=k_kp, src=kp_psum)

                nisa.memset(dst=mm_psum, value=0.0)
                nisa.nc_matmul(mm_psum, stationary=state, moving=k_kp)
                nisa.tensor_copy(dst=predicted_v, src=mm_psum)

                # ---- Build delta_v in (1, Dv) layout for outer-product update ----
                nisa.dma_copy(dst=v_row, src=v[i_batch, i_seq,
                                                i_value_head:i_value_head+1,
                                                0:value_dim])
                nisa.dma_copy(dst=beta_f, src=beta[i_batch, i_seq:i_seq+1,
                                                    i_value_head:i_value_head+1])

                nisa.nc_transpose(dst=tp_psum, data=predicted_v)
                nisa.tensor_copy(dst=predicted_v_T, src=tp_psum)
                nisa.tensor_tensor(dst=diff, data1=v_row, data2=predicted_v_T,
                                   op=nl.subtract)
                nisa.tensor_scalar(dst=delta_v, data=diff,
                                   op0=nl.multiply, operand0=beta_f)

                # ---- Outer-product update: state[k,v] += k_n[k] * delta_v[v] ----
                # Chunked along v; each chunk uses a small (Dk, v_chunk) tile.
                # nc_matmul(stationary=k_scaled (1, Dk), moving=delta_v_chunk (1, vc))
                # contracts along partition (size 1) -> outer product (Dk, vc).
                for ivc in nl.affine_range(n_v_chunks):
                    vstart = ivc * v_chunk
                    nisa.memset(dst=update_psum_chunk, value=0.0)
                    nisa.nc_matmul(
                        update_psum_chunk,
                        stationary=k_scaled,
                        moving=delta_v[0:1, vstart:vstart+v_chunk])
                    nisa.tensor_copy(dst=update_sbuf_chunk, src=update_psum_chunk)
                    nisa.tensor_tensor(
                        dst=state[0:key_dim, vstart:vstart+v_chunk],
                        data1=state[0:key_dim, vstart:vstart+v_chunk],
                        data2=update_sbuf_chunk, op=nl.add)

                # ---- output[b, t, :, vh] = state.T @ q_n ----
                nisa.nc_transpose(dst=kp_psum, data=q_scaled)
                nisa.tensor_copy(dst=q_kp, src=kp_psum)

                nisa.memset(dst=mm_psum, value=0.0)
                nisa.nc_matmul(mm_psum, stationary=state, moving=q_kp)
                nisa.tensor_copy(dst=out_v, src=mm_psum)
                nl.store(output[i_batch, i_seq, 0:value_dim,
                                i_value_head:i_value_head+1], out_v)

            # ---- Persist final state back to HBM ----
            # SBUF state is (Dk part, Dv free); HBM ht layout is (Dv, Dk).
            # Transpose in 32x32 stream-transpose chunks (no PSUM needed).
            for ikk in nl.affine_range(key_dim // chunk):
                for ivv in nl.affine_range(value_dim // chunk):
                    nisa.nc_transpose(
                        dst=final_tile,
                        data=state[ikk*chunk:(ikk+1)*chunk,
                                   ivv*chunk:(ivv+1)*chunk])
                    nl.store(
                        ht[i_batch, i_value_head,
                           ivv*chunk:(ivv+1)*chunk,
                           ikk*chunk:(ikk+1)*chunk],
                        final_tile)

    return output, ht
