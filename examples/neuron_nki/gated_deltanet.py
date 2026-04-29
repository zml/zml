"""Gated DeltaNet NKI kernel for Inferentia2 / Trainium.

Algorithm (per batch, per q-head, processing the ``qk_rep`` value-heads
that share that q-head together, S timesteps):

    state[k, v] starts from h0
    for t in range(S):
        q_n = l2_normalize(q[t]) * scale            # (Dk,)
        k_n = l2_normalize(k[t])                    # (Dk,)
        state *= exp(g[t])                          # decay
        pred_v[v]  = sum_k state[k, v] * k_n[k]
        delta_v[v] = beta[t] * (v[t, v] - pred_v[v])
        state[k, v] += k_n[k] * delta_v[v]          # rank-1 update
        out[t, v]   = sum_k state[k, v] * q_n[k]

Optimization strategy (vs a naive per-step formulation):

* **Bulk DMA preload of value-side inputs.** ``v``, ``g`` and ``beta``
  for the whole sequence (and for the ``qk_rep`` value-heads attached
  to the current q-head) are pulled into SBUF *once* at the start of
  the outer loop. The inner scan then never issues a DMA for these
  tensors, collapsing per-step DMA queue depth from 3*qk_rep to 0.
  ``q`` and ``k`` keep their per-step DMA + transpose -- they need
  partition-0 alignment for the matmul below, and a single
  ``(1, key_dim)`` DMA is essentially free.

* **Pre-computed decay.** ``decay = exp(g)`` is materialized once for
  the whole sequence into a ``(1, S * qk_rep)`` tile. The inner scan
  pulls a ``(1, 1)`` slice per packed slot and broadcasts to
  ``(key_dim, 1)`` for the in-place state-multiply -- one less
  per-step DMA + activation than the baseline.

* **Resident packed state.** The state for the ``qk_rep`` value-heads
  attached to a q-head is packed in the free dim of a single SBUF
  tile of shape ``(Dk, qk_rep * Dv)`` which stays resident across the
  whole sequence. The three contractions per step (predicted_v,
  rank-1 update, output) all run through the PE array via
  ``nisa.nc_matmul`` with this packed layout, folding ``qk_rep``
  per-head matmuls into one wider one.

* **Function-scope scratch.** All scratch tiles are allocated once at
  kernel scope and reused across every (batch, q-head, timestep). The
  PSUM allocator can pin them to fixed banks and the compiler doesn't
  re-cost them every iteration -- this is what keeps compile time
  bounded for the 16 q-heads x 16 timesteps unroll.

* **Structured loops.** Only the timestep loop is sequential; every
  inner sweep over ``qk_rep`` / chunks uses ``nl.affine_range``. No
  Python-level unrolling of large ranges.
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
    pack_w = qk_rep * value_dim
    scale = 1.0 / math.sqrt(key_dim)
    eps = 1.0e-6

    # PSUM update tile chunked along the packed-free dim.
    v_chunk = 256
    assert pack_w % v_chunk == 0
    n_v_chunks = pack_w // v_chunk

    # Final-state HBM transpose chunk size (keeps each block within a
    # single PSUM bank).
    chunk = 32
    assert key_dim % chunk == 0 and value_dim % chunk == 0

    output = nl.ndarray((batch_size, seq_len, value_dim, num_value_heads),
                        dtype=nl.float32, buffer=nl.shared_hbm)
    ht = nl.ndarray(h0.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    # ------------------------------------------------------------------
    # Function-scope scratch. Allocating once means stable PSUM/SBUF
    # bank assignment and short compile time.
    # ------------------------------------------------------------------
    # PSUM scratch.
    pv_psum = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.psum)
    out_psum = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.psum)
    out_t_psum = nl.ndarray((value_dim, 1),
                            dtype=nl.float32, buffer=nl.psum)
    update_psum_chunk = nl.ndarray((key_dim, v_chunk),
                                   dtype=nl.float32, buffer=nl.psum)
    # Bulk-transpose PSUM tile for the (Dk,S) form of q/k. Lifetime is
    # the preload phase only; the allocator reclaims its banks before
    # the scan starts using update_psum_chunk.
    bulk_kp_psum = nl.ndarray((key_dim, seq_len),
                              dtype=nl.float32, buffer=nl.psum)

    # Resident state tile.
    state = nl.ndarray((key_dim, pack_w), dtype=nl.float32, buffer=nl.sbuf)

    # Per-step q/k normalize scratch (partition=1 layout). The whole
    # sequence of q/k is loaded into q_all / k_all by ONE DMA each,
    # then sliced per-timestep into these scratch tiles.
    q_sq = nl.ndarray((1, key_dim), dtype=nl.float32, buffer=nl.sbuf)
    k_sq = nl.ndarray((1, key_dim), dtype=nl.float32, buffer=nl.sbuf)
    q_sumsq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    k_sumsq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    q_inv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    k_inv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)

    # Per-step value-side scratch.
    diff = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.sbuf)
    delta_v = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.sbuf)
    out_v_T = nl.ndarray((1, pack_w), dtype=nl.float32, buffer=nl.sbuf)
    out_v_slot = nl.ndarray((value_dim, 1),
                            dtype=nl.float32, buffer=nl.sbuf)

    # Per-step decay scalar. Sliced from decay_rows; broadcast to
    # (key_dim, 1) to scale the state tile.
    decay_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)

    # Final-state HBM transpose tile.
    final_tile = nl.ndarray((chunk, chunk),
                            dtype=nl.float32, buffer=nl.sbuf)

    # Bulk preload tiles. v / g / beta / decay use partition=1 layout
    # so any per-step slice stays at partition offset 0 and combines
    # with the (1, *) per-step scratch above.
    v_rows = nl.ndarray((1, seq_len * pack_w),
                        dtype=nl.float32, buffer=nl.sbuf)
    g_rows = nl.ndarray((1, seq_len * qk_rep),
                        dtype=nl.float32, buffer=nl.sbuf)
    beta_rows = nl.ndarray((1, seq_len * qk_rep),
                           dtype=nl.float32, buffer=nl.sbuf)
    decay_rows = nl.ndarray((1, seq_len * qk_rep),
                            dtype=nl.float32, buffer=nl.sbuf)
    # Pre-normalized (1, Dk) tiles for the whole sequence -- consumed
    # both by the bulk Dk-transpose below (for predicted_v / output
    # contractions) and as the rank-1 update's stationary operand.
    q_scaled_all = nl.ndarray((1, seq_len * key_dim),
                              dtype=nl.float32, buffer=nl.sbuf)
    k_scaled_all = nl.ndarray((1, seq_len * key_dim),
                              dtype=nl.float32, buffer=nl.sbuf)
    # (Dk, S) bulk-transposed q/k. The scan loop slices column i_seq.
    q_kp_all = nl.ndarray((key_dim, seq_len),
                          dtype=nl.float32, buffer=nl.sbuf)
    k_kp_all = nl.ndarray((key_dim, seq_len),
                          dtype=nl.float32, buffer=nl.sbuf)

    # Bulk preload scratch: q / k for the whole sequence as a single
    # (1, S*Dk) row, fed by ONE DMA per q-head. The per-step normalize
    # then slices columns out of these tiles.
    q_all = nl.ndarray((1, seq_len * key_dim),
                       dtype=nl.float32, buffer=nl.sbuf)
    k_all = nl.ndarray((1, seq_len * key_dim),
                       dtype=nl.float32, buffer=nl.sbuf)

    for i_batch in nl.affine_range(batch_size):
        for i_q_head in nl.affine_range(num_q_heads):
            vh_s = i_q_head * qk_rep
            # ============================================================
            # Bulk preload value-side inputs for this (batch, q_head).
            # ONE DMA per tensor instead of seq_len*qk_rep DMAs. The
            # SBUF dst stays (1, *) (partition 0 only) so downstream
            # per-step slices keep their (1, *) layout. The HBM source
            # is contiguous along (seq_len, qk_rep, value_dim) so the
            # DMA engine produces a single descriptor.
            # ============================================================
            nisa.dma_copy(
                dst=v_rows,
                src=v[i_batch, 0:seq_len, vh_s:vh_s+qk_rep, 0:value_dim])
            nisa.dma_copy(
                dst=g_rows,
                src=g[i_batch, 0:seq_len, vh_s:vh_s+qk_rep])
            nisa.dma_copy(
                dst=beta_rows,
                src=beta[i_batch, 0:seq_len, vh_s:vh_s+qk_rep])
            nisa.dma_copy(
                dst=q_all,
                src=q[i_batch, 0:seq_len, i_q_head:i_q_head+1, 0:key_dim])
            nisa.dma_copy(
                dst=k_all,
                src=k[i_batch, 0:seq_len, i_q_head:i_q_head+1, 0:key_dim])

            # ---- L2 normalize q / k for the whole sequence ----
            for it in nl.affine_range(seq_len):
                q_row_v = q_all[0:1, it*key_dim:(it+1)*key_dim]
                k_row_v = k_all[0:1, it*key_dim:(it+1)*key_dim]
                nisa.tensor_tensor(dst=q_sq, data1=q_row_v, data2=q_row_v,
                                   op=nl.multiply)
                nisa.tensor_tensor(dst=k_sq, data1=k_row_v, data2=k_row_v,
                                   op=nl.multiply)
                nisa.tensor_reduce(dst=q_sumsq, data=q_sq, op=nl.add,
                                   axis=1, keepdims=True)
                nisa.tensor_reduce(dst=k_sumsq, data=k_sq, op=nl.add,
                                   axis=1, keepdims=True)
                nisa.activation(dst=q_inv, op=nl.rsqrt,
                                data=q_sumsq, bias=eps)
                nisa.activation(dst=k_inv, op=nl.rsqrt,
                                data=k_sumsq, bias=eps)
                nisa.tensor_scalar(
                    dst=q_scaled_all[0:1, it*key_dim:(it+1)*key_dim],
                    data=q_row_v,
                    op0=nl.multiply, operand0=q_inv,
                    op1=nl.multiply, operand1=scale)
                nisa.tensor_scalar(
                    dst=k_scaled_all[0:1, it*key_dim:(it+1)*key_dim],
                    data=k_row_v,
                    op0=nl.multiply, operand0=k_inv)

            # Pre-compute decay = exp(g) for the full sequence in one op.
            nisa.activation(dst=decay_rows, op=nl.exp, data=g_rows)

            # ============================================================
            # Hoist q/k Dk-transpose out of the scan. Per-timestep we
            # need a (Dk, 1) tile for the predicted_v / output matmuls.
            # The PE array's stationary free-dim cap is 128, so we still
            # do S small transposes -- but they're now in their own loop
            # before the scan, freeing the scan body of two transposes
            # and two PSUM->SBUF copies per timestep, and letting the
            # compiler issue them in parallel with the bulk-preload DMAs.
            # ============================================================
            for it in nl.affine_range(seq_len):
                nisa.nc_transpose(
                    dst=bulk_kp_psum[0:key_dim, it:it+1],
                    data=q_scaled_all[0:1, it*key_dim:(it+1)*key_dim])
            nisa.tensor_copy(dst=q_kp_all, src=bulk_kp_psum)
            for it in nl.affine_range(seq_len):
                nisa.nc_transpose(
                    dst=bulk_kp_psum[0:key_dim, it:it+1],
                    data=k_scaled_all[0:1, it*key_dim:(it+1)*key_dim])
            nisa.tensor_copy(dst=k_kp_all, src=bulk_kp_psum)

            # ============================================================
            # Initialize the resident packed state from h0.
            # h0[b, vh] is (Dv, Dk); load_transpose2d -> (Dk, Dv).
            # ============================================================
            for ir in nl.affine_range(qk_rep):
                vh = i_q_head * qk_rep + ir
                state_init = nl.load_transpose2d(
                    h0[i_batch, vh, 0:value_dim, 0:key_dim])
                nisa.tensor_copy(
                    dst=state[0:key_dim, ir*value_dim:(ir+1)*value_dim],
                    src=state_init)

            # ============================================================
            # Main sequential scan over timesteps. Per-step body now
            # consumes PSUM directly (no PSUM->SBUF tensor_copy) wherever
            # the next consumer accepts a PSUM operand. This removes 3
            # tensor_copys + 4 update tensor_copys per step.
            # ============================================================
            for i_seq in nl.sequential_range(seq_len):
                # ---- Decay each packed slot by its scalar exp(g) ----
                for ir in nl.affine_range(qk_rep):
                    nisa.tensor_copy(
                        dst=decay_scalar,
                        src=decay_rows[0:1,
                                       i_seq*qk_rep + ir:
                                       i_seq*qk_rep + ir + 1])
                    nisa.tensor_scalar(
                        dst=state[0:key_dim, ir*value_dim:(ir+1)*value_dim],
                        data=state[0:key_dim, ir*value_dim:(ir+1)*value_dim],
                        op0=nl.multiply,
                        operand0=nl.broadcast_to(decay_scalar,
                                                 (key_dim, 1)))

                # ---- predicted_v(1, pack_w) = k_kp.T @ state ----
                nisa.memset(dst=pv_psum, value=0.0)
                nisa.nc_matmul(
                    pv_psum,
                    stationary=k_kp_all[0:key_dim, i_seq:i_seq+1],
                    moving=state)

                # ---- delta_v = beta * (v - predicted_v) ----
                # ``data2`` reads predicted_v straight from PSUM.
                nisa.tensor_tensor(
                    dst=diff,
                    data1=v_rows[0:1, i_seq*pack_w:(i_seq+1)*pack_w],
                    data2=pv_psum,
                    op=nl.subtract)
                for ir in nl.affine_range(qk_rep):
                    nisa.tensor_scalar(
                        dst=delta_v[0:1, ir*value_dim:(ir+1)*value_dim],
                        data=diff[0:1, ir*value_dim:(ir+1)*value_dim],
                        op0=nl.multiply,
                        operand0=beta_rows[0:1,
                                           i_seq*qk_rep + ir:
                                           i_seq*qk_rep + ir + 1])

                # ---- Outer-product update: state += k_scaled ⊗ delta_v ----
                # Chunked along the packed-free dim to fit PSUM. The
                # state add reads update_psum_chunk straight from PSUM.
                for ivc in nl.affine_range(n_v_chunks):
                    vstart = ivc * v_chunk
                    nisa.memset(dst=update_psum_chunk, value=0.0)
                    nisa.nc_matmul(
                        update_psum_chunk,
                        stationary=k_scaled_all[0:1,
                                                i_seq*key_dim:
                                                (i_seq+1)*key_dim],
                        moving=delta_v[0:1, vstart:vstart+v_chunk])
                    nisa.tensor_tensor(
                        dst=state[0:key_dim, vstart:vstart+v_chunk],
                        data1=state[0:key_dim, vstart:vstart+v_chunk],
                        data2=update_psum_chunk, op=nl.add)

                # ---- output(1, pack_w) = q_kp.T @ state ----
                # nc_transpose only accepts SBUF input, so we still
                # tensor_copy out_psum to SBUF here -- but only once,
                # then transpose qk_rep slots out of out_v_T.
                nisa.memset(dst=out_psum, value=0.0)
                nisa.nc_matmul(
                    out_psum,
                    stationary=q_kp_all[0:key_dim, i_seq:i_seq+1],
                    moving=state)
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

            # ============================================================
            # Persist final packed state back to HBM. Stream-transpose
            # ``chunk`` x ``chunk`` blocks; each block fits a single
            # PSUM bank.
            # ============================================================
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
