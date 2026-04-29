"""Gated DeltaNet NKI kernel for Inferentia2 / Trainium.

Algorithm (per batch / per q-head, processing the ``qk_head_repetition``
value-heads that share that q-head together, S timesteps):

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
HBM traffic per timestep. The `qk_head_repetition` value-heads that share
a q-head are *packed together* in the free dimension of a single SBUF
state tile of shape ``(Dk, qk_rep * Dv)``. This lets us:

* compute ``q_n`` / ``k_n`` (and their `(Dk,1)` matmul-moving forms) once
  per ``(qh, t)`` instead of once per ``(vh, t)``;
* fold the per-head ``state @ k_n`` and ``state @ q_n`` matmuls into a
  single wider matmul each; the moving operand stays `(Dk, 1)` and the
  free dim of the stationary operand grows by ``qk_rep``×.

All three contractions per step go through the PE array via
``nisa.nc_matmul``. Layout in SBUF is (partition = Dk, free = qk_rep*Dv).
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

    qk_rep = num_value_heads // num_q_heads
    pack_w = qk_rep * value_dim                  # packed state free dim
    scale = 1.0 / math.sqrt(key_dim)
    eps = 1.0e-6

    output = nl.ndarray((batch_size, seq_len, value_dim, num_value_heads),
                        dtype=nl.float32, buffer=nl.shared_hbm)
    ht = nl.ndarray(h0.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    # ------------------------------------------------------------------
    # All scratch tiles below are allocated once at kernel scope and reused
    # across every (batch, q-head, timestep). This is what keeps compile
    # time low and lets the compiler pin the tiles to fixed SBUF/PSUM banks.
    # ------------------------------------------------------------------
    # PSUM scratch.
    kp_psum = nl.ndarray((key_dim, 1), dtype=nl.float32, buffer=nl.psum)
    pv_psum = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.psum)
    out_psum = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.psum)
    out_t_psum = nl.ndarray((value_dim, 1), dtype=nl.float32, buffer=nl.psum)

    # Outer-product chunked update tile. v_chunk=64 keeps the PSUM tile at
    # 32 KB; smaller chunks would just multiply per-step issue overhead.
    v_chunk = 64
    assert pack_w % v_chunk == 0
    n_v_chunks = pack_w // v_chunk
    update_psum_chunk = nl.ndarray((key_dim, v_chunk),
                                   dtype=nl.float32, buffer=nl.psum)
    update_sbuf_chunk = nl.ndarray((key_dim, v_chunk),
                                   dtype=nl.float32, buffer=nl.sbuf)

    # Resident state tile: (Dk part, qk_rep * Dv free).  Reused across all
    # (b, qh) by overwriting it during the per-pair init.
    state = nl.ndarray((key_dim, pack_w), dtype=nl.float32, buffer=nl.sbuf)

    # Per-step SBUF scratch (q-head shared across the qk_rep value-heads).
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

    # Per-step value-head-local scratch (one tile per packed slot).
    v_row = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.sbuf)
    diff = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.sbuf)
    delta_v = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.sbuf)
    predicted_v_T = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.sbuf)
    out_v_T = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.sbuf)
    out_v_slot = nl.ndarray((value_dim, 1), dtype=nl.float32, buffer=nl.sbuf)

    g_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    beta_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    decay = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)

    # Final-state HBM transpose tile (32x32 keeps it as a stream transpose).
    chunk = 32
    assert key_dim % chunk == 0 and value_dim % chunk == 0
    final_tile = nl.ndarray((chunk, chunk), dtype=nl.float32, buffer=nl.sbuf)

    # ------------------------------------------------------------------
    # Main kernel body.
    # ------------------------------------------------------------------
    for i_batch in nl.affine_range(batch_size):
        for i_q_head in nl.sequential_range(num_q_heads):
            # ---- Initialize packed state from h0 for the qk_rep heads ----
            for ir in nl.affine_range(qk_rep):
                vh = i_q_head * qk_rep + ir
                # h0[b, vh, :, :] is (Dv, Dk); load_transpose2d -> (Dk, Dv).
                state_init = nl.load_transpose2d(
                    h0[i_batch, vh, 0:value_dim, 0:key_dim])
                nisa.tensor_copy(
                    dst=state[0:key_dim, ir*value_dim:(ir+1)*value_dim],
                    src=state_init)

            for i_seq in nl.sequential_range(seq_len):
                # ==== q/k normalize for this (qh, t) -- shared by all reps ====
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
                nisa.activation(dst=q_inv, op=nl.rsqrt, data=q_sumsq, bias=eps)
                nisa.activation(dst=k_inv, op=nl.rsqrt, data=k_sumsq, bias=eps)
                nisa.tensor_scalar(dst=q_scaled, data=q_row,
                                   op0=nl.multiply, operand0=q_inv,
                                   op1=nl.multiply, operand1=scale)
                nisa.tensor_scalar(dst=k_scaled, data=k_row,
                                   op0=nl.multiply, operand0=k_inv)

                # Move q/k into (Dk part, 1 free) form for matmul.  These two
                # transposes are issued *once* per (qh, t) and reused across
                # all qk_rep value-heads attached to this q-head.
                nisa.nc_transpose(dst=kp_psum, data=k_scaled)
                nisa.tensor_copy(dst=k_kp, src=kp_psum)
                nisa.nc_transpose(dst=kp_psum, data=q_scaled)
                nisa.tensor_copy(dst=q_kp, src=kp_psum)

                # ==== Decay: state[:, ir*Dv:(ir+1)*Dv] *= exp(g[t, vh]) ====
                # The broadcast_to result feeds tensor_scalar directly; no
                # intermediate SBUF tile needed.
                for ir in nl.affine_range(qk_rep):
                    vh = i_q_head * qk_rep + ir
                    nisa.dma_copy(dst=g_tile,
                                  src=g[i_batch, i_seq:i_seq+1, vh:vh+1])
                    nisa.activation(dst=decay, op=nl.exp, data=g_tile)
                    nisa.tensor_scalar(
                        dst=state[0:key_dim, ir*value_dim:(ir+1)*value_dim],
                        data=state[0:key_dim, ir*value_dim:(ir+1)*value_dim],
                        op0=nl.multiply,
                        operand0=nl.broadcast_to(decay, (key_dim, 1)))

                # ==== predicted_v (1, pack_w) = k_kp.T @ state ====
                # One matmul, contracts over Dk, free dim = pack_w.
                nisa.memset(dst=pv_psum, value=0.0)
                nisa.nc_matmul(pv_psum, stationary=k_kp, moving=state)
                nisa.tensor_copy(dst=predicted_v_T, src=pv_psum)

                # ==== Build delta_v(1, pack_w) per packed slot ====
                # Load v into the matching slot, then per-slot beta-scale.
                for ir in nl.affine_range(qk_rep):
                    vh = i_q_head * qk_rep + ir
                    nisa.dma_copy(
                        dst=v_row[0:1, ir*value_dim:(ir+1)*value_dim],
                        src=v[i_batch, i_seq, vh:vh+1, 0:value_dim])
                nisa.tensor_tensor(dst=diff, data1=v_row, data2=predicted_v_T,
                                   op=nl.subtract)
                for ir in nl.affine_range(qk_rep):
                    vh = i_q_head * qk_rep + ir
                    nisa.dma_copy(dst=beta_tile,
                                  src=beta[i_batch, i_seq:i_seq+1, vh:vh+1])
                    nisa.tensor_scalar(
                        dst=delta_v[0:1, ir*value_dim:(ir+1)*value_dim],
                        data=diff[0:1, ir*value_dim:(ir+1)*value_dim],
                        op0=nl.multiply, operand0=beta_tile)

                # ==== Outer-product update: state += k_kp ⊗ delta_v ====
                # Chunked along the packed-free dim to fit PSUM.
                for ivc in nl.affine_range(n_v_chunks):
                    vstart = ivc * v_chunk
                    nisa.memset(dst=update_psum_chunk, value=0.0)
                    nisa.nc_matmul(
                        update_psum_chunk,
                        stationary=k_scaled,
                        moving=delta_v[0:1, vstart:vstart+v_chunk])
                    nisa.tensor_copy(dst=update_sbuf_chunk,
                                     src=update_psum_chunk)
                    nisa.tensor_tensor(
                        dst=state[0:key_dim, vstart:vstart+v_chunk],
                        data1=state[0:key_dim, vstart:vstart+v_chunk],
                        data2=update_sbuf_chunk, op=nl.add)

                # ==== Output: out_v_T(1, pack_w) = q_kp.T @ state ====
                # Same trick as predicted_v: keep partition = 1 to dodge the
                # 128-partition matmul-dst limit. Then transpose each packed
                # slot back to (Dv, 1) for the HBM store.
                nisa.memset(dst=out_psum, value=0.0)
                nisa.nc_matmul(out_psum, stationary=q_kp, moving=state)
                nisa.tensor_copy(dst=out_v_T, src=out_psum)
                for ir in nl.affine_range(qk_rep):
                    vh = i_q_head * qk_rep + ir
                    nisa.nc_transpose(
                        dst=out_t_psum,
                        data=out_v_T[0:1, ir*value_dim:(ir+1)*value_dim])
                    nisa.tensor_copy(dst=out_v_slot, src=out_t_psum)
                    nl.store(
                        output[i_batch, i_seq, 0:value_dim, vh:vh+1],
                        out_v_slot)

            # ---- Persist final packed state back to HBM ----
            # SBUF state is (Dk part, qk_rep * Dv free); HBM ht layout is
            # (Dv, Dk) per head.  Stream-transpose 32x32 chunks per head.
            for ir in nl.affine_range(qk_rep):
                vh = i_q_head * qk_rep + ir
                for ikk in nl.affine_range(key_dim // chunk):
                    for ivv in nl.affine_range(value_dim // chunk):
                        nisa.nc_transpose(
                            dst=final_tile,
                            data=state[ikk*chunk:(ikk+1)*chunk,
                                       ir*value_dim + ivv*chunk:
                                       ir*value_dim + (ivv+1)*chunk])
                        nl.store(
                            ht[i_batch, vh,
                               ivv*chunk:(ivv+1)*chunk,
                               ikk*chunk:(ikk+1)*chunk],
                            final_tile)

    return output, ht
