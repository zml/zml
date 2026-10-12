import math

import nki.isa as nisa
import nki.language as nl

NEG_INF = -3.3895313892515355e38


def decode(q, k, v, token_index):
    q_len = q.shape[0]
    num_q_heads = q.shape[1]
    d_head = q.shape[2]
    seq_len = k.shape[0]
    num_kv_heads = k.shape[1]
    heads_per_kv = num_q_heads // num_kv_heads
    scale = 1.0 / (d_head**0.5)

    assert q_len == 1
    assert k.shape == v.shape
    assert q.shape[2] == k.shape[2]
    assert num_q_heads % num_kv_heads == 0

    VALUE_TILE_TOKENS = nl.tile_size.pmax
    assert seq_len % VALUE_TILE_TOKENS == 0, (
        "seq_len "
        f"({seq_len}) must be a multiple of VALUE_TILE_TOKENS ({VALUE_TILE_TOKENS})"
    )
    # Largest tensor-engine K tile that exactly partitions this static sequence.
    BLOCK_K = math.gcd(seq_len, nl.tile_size.gemm_moving_fmax)
    assert seq_len % BLOCK_K == 0, (
        f"seq_len ({seq_len}) must be a multiple of BLOCK_K ({BLOCK_K})"
    )
    assert BLOCK_K % VALUE_TILE_TOKENS == 0, (
        "BLOCK_K "
        f"({BLOCK_K}) must be a multiple of VALUE_TILE_TOKENS ({VALUE_TILE_TOKENS})"
    )
    num_kv_blocks = seq_len // BLOCK_K
    value_tiles_per_k_tile = BLOCK_K // VALUE_TILE_TOKENS

    out = nl.ndarray((q_len, num_q_heads, d_head), dtype=q.dtype, buffer=nl.shared_hbm)
    token = nl.ndarray((1, 1), dtype=token_index.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=token, src=token_index[0:1, 0:1])

    positions = nl.ndarray((q_len, BLOCK_K), dtype=token_index.dtype, buffer=nl.sbuf)
    qk = nl.ndarray((heads_per_kv, BLOCK_K), dtype=nl.float32, buffer=nl.psum)
    qk_sbuf = nl.ndarray((heads_per_kv, BLOCK_K), dtype=nl.float32, buffer=nl.sbuf)
    masked_qk = nl.ndarray((heads_per_kv, BLOCK_K), dtype=nl.float32, buffer=nl.sbuf)
    neg_tile = nl.full(
        (heads_per_kv, BLOCK_K), NEG_INF, dtype=nl.float32, buffer=nl.sbuf
    )
    tile_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    new_row_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    old_max_delta = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    old_max_scale = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    row_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    neg_row_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    exp_tile = nl.ndarray((heads_per_kv, BLOCK_K), dtype=nl.float32, buffer=nl.sbuf)
    tile_sum = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    scaled_sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    new_sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    inverse_sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    running_out = nl.ndarray((heads_per_kv, d_head), dtype=nl.float32, buffer=nl.sbuf)
    scaled_running_out = nl.ndarray(
        (heads_per_kv, d_head), dtype=nl.float32, buffer=nl.sbuf
    )
    attn_out_psum = nl.ndarray((heads_per_kv, d_head), dtype=nl.float32, buffer=nl.psum)
    attn_out_sbuf = nl.ndarray((heads_per_kv, d_head), dtype=q.dtype, buffer=nl.sbuf)

    prob_t_psum = nl.ndarray(
        (VALUE_TILE_TOKENS, heads_per_kv), dtype=nl.float32, buffer=nl.psum
    )
    prob_t = nl.ndarray(
        (VALUE_TILE_TOKENS, heads_per_kv), dtype=v.dtype, buffer=nl.sbuf
    )
    v_sbuf = nl.ndarray((VALUE_TILE_TOKENS, d_head), dtype=v.dtype, buffer=nl.sbuf)

    for kv_head in nl.affine_range(num_kv_heads):
        q_head = kv_head * heads_per_kv

        q_sbuf = nl.ndarray((d_head, heads_per_kv), dtype=q.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(
            dst=q_sbuf,
            src=q[0, q_head : q_head + heads_per_kv, :],
        )
        nisa.activation(dst=q_sbuf, op=nl.copy, data=q_sbuf, scale=scale)

        nisa.iota(positions, [[1, BLOCK_K]])

        k_sbuf = nl.ndarray((d_head, BLOCK_K), dtype=k.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(dst=k_sbuf, src=k[nl.ds(0, BLOCK_K), kv_head, :])
        nisa.memset(dst=qk, value=0.0)
        nisa.nc_matmul(dst=qk, stationary=q_sbuf, moving=k_sbuf)
        nisa.tensor_copy(dst=qk_sbuf, src=qk)

        valid = nl.less_equal(positions, nl.broadcast_to(token, positions.shape))
        masked_qk = nl.where(
            nl.broadcast_to(valid, masked_qk.shape),
            qk_sbuf,
            neg_tile,
            dtype=nl.float32,
        )
        nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=masked_qk, axis=(1,))
        nisa.tensor_scalar(
            dst=neg_row_max,
            data=row_max,
            op0=nl.multiply,
            operand0=-1.0,
        )
        nisa.activation(
            dst=exp_tile,
            op=nl.exp,
            data=masked_qk,
            bias=neg_row_max,
            reduce_op=nl.add,
            reduce_res=sum_row,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        nisa.memset(dst=attn_out_psum, value=0.0)
        for value_tile_idx in nl.affine_range(value_tiles_per_k_tile):
            value_start = value_tile_idx * VALUE_TILE_TOKENS
            nisa.nc_transpose(
                dst=prob_t_psum,
                data=exp_tile[:, nl.ds(value_start, VALUE_TILE_TOKENS)],
            )
            nisa.tensor_copy(dst=prob_t, src=prob_t_psum)

            nisa.dma_copy(
                dst=v_sbuf,
                src=v[nl.ds(value_start, VALUE_TILE_TOKENS), kv_head, :],
            )
            nisa.nc_matmul(dst=attn_out_psum, stationary=prob_t, moving=v_sbuf)
        nisa.tensor_copy(dst=running_out, src=attn_out_psum)

        for k_tail_idx in nl.affine_range(num_kv_blocks - 1):
            k_block_idx = k_tail_idx + 1
            k_start = k_block_idx * BLOCK_K
            nisa.iota(positions, [[1, BLOCK_K]], offset=k_start)

            k_sbuf = nl.ndarray((d_head, BLOCK_K), dtype=k.dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=k_sbuf, src=k[nl.ds(k_start, BLOCK_K), kv_head, :])
            nisa.memset(dst=qk, value=0.0)
            nisa.nc_matmul(dst=qk, stationary=q_sbuf, moving=k_sbuf)
            nisa.tensor_copy(dst=qk_sbuf, src=qk)

            valid = nl.less_equal(positions, nl.broadcast_to(token, positions.shape))
            masked_qk = nl.where(
                nl.broadcast_to(valid, masked_qk.shape),
                qk_sbuf,
                neg_tile,
                dtype=nl.float32,
            )
            nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=masked_qk, axis=(1,))
            nisa.tensor_tensor(
                dst=new_row_max,
                data1=row_max,
                data2=tile_max,
                op=nl.maximum,
            )
            nisa.tensor_scalar(
                dst=old_max_delta,
                data=row_max,
                op0=nl.subtract,
                operand0=new_row_max,
            )
            nisa.activation(dst=old_max_scale, op=nl.exp, data=old_max_delta)
            nisa.tensor_scalar(
                dst=neg_row_max,
                data=new_row_max,
                op0=nl.multiply,
                operand0=-1.0,
            )
            nisa.activation(
                dst=exp_tile,
                op=nl.exp,
                data=masked_qk,
                bias=neg_row_max,
                reduce_op=nl.add,
                reduce_res=tile_sum,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )
            nisa.tensor_scalar(
                dst=scaled_sum_row,
                data=sum_row,
                op0=nl.multiply,
                operand0=old_max_scale,
            )
            nisa.tensor_tensor(
                dst=new_sum_row,
                data1=scaled_sum_row,
                data2=tile_sum,
                op=nl.add,
            )
            nisa.tensor_scalar(
                dst=scaled_running_out,
                data=running_out,
                op0=nl.multiply,
                operand0=old_max_scale,
            )

            nisa.memset(dst=attn_out_psum, value=0.0)
            for value_tile_idx in nl.affine_range(value_tiles_per_k_tile):
                value_start = value_tile_idx * VALUE_TILE_TOKENS
                nisa.nc_transpose(
                    dst=prob_t_psum,
                    data=exp_tile[:, nl.ds(value_start, VALUE_TILE_TOKENS)],
                )
                nisa.tensor_copy(dst=prob_t, src=prob_t_psum)

                nisa.dma_copy(
                    dst=v_sbuf,
                    src=v[nl.ds(k_start + value_start, VALUE_TILE_TOKENS), kv_head, :],
                )
                nisa.nc_matmul(dst=attn_out_psum, stationary=prob_t, moving=v_sbuf)

            nisa.tensor_tensor(
                dst=running_out,
                data1=scaled_running_out,
                data2=attn_out_psum,
                op=nl.add,
            )
            nisa.tensor_copy(dst=row_max, src=new_row_max)
            nisa.tensor_copy(dst=sum_row, src=new_sum_row)

        nisa.reciprocal(dst=inverse_sum_row, data=sum_row)
        nisa.tensor_scalar(
            dst=running_out,
            data=running_out,
            op0=nl.multiply,
            operand0=inverse_sum_row,
        )
        nisa.tensor_copy(dst=attn_out_sbuf, src=running_out)

        nisa.dma_copy(
            dst=out[0, q_head : q_head + heads_per_kv, :],
            src=attn_out_sbuf,
        )

    return out


def prefill(q, k, v):
    q_len = q.shape[0]
    num_q_heads = q.shape[1]
    d_head = q.shape[2]
    seq_len = k.shape[0]
    num_kv_heads = k.shape[1]
    heads_per_kv = num_q_heads // num_kv_heads
    scale = 1.0 / (d_head**0.5)

    assert q.shape[2] == k.shape[2]
    assert k.shape == v.shape
    assert num_q_heads % num_kv_heads == 0
    assert q_len == seq_len

    VALUE_TILE_TOKENS = nl.tile_size.pmax
    BLOCK_Q = nl.tile_size.pmax
    # Largest tensor-engine K tile that exactly partitions this static sequence.
    BLOCK_K = math.gcd(seq_len, nl.tile_size.gemm_moving_fmax)

    assert q_len % BLOCK_Q == 0, (
        f"q_len ({q_len}) must be a multiple of BLOCK_Q ({BLOCK_Q})"
    )
    assert seq_len % BLOCK_K == 0, (
        f"seq_len ({seq_len}) must be a multiple of BLOCK_K ({BLOCK_K})"
    )
    assert BLOCK_K % BLOCK_Q == 0, (
        f"BLOCK_K ({BLOCK_K}) must be a multiple of BLOCK_Q ({BLOCK_Q})"
    )
    assert BLOCK_K % VALUE_TILE_TOKENS == 0, (
        "BLOCK_K "
        f"({BLOCK_K}) must be a multiple of VALUE_TILE_TOKENS ({VALUE_TILE_TOKENS})"
    )

    num_q_blocks = q_len // BLOCK_Q
    q_blocks_per_k_tile = BLOCK_K // BLOCK_Q
    value_tiles_per_k_tile = BLOCK_K // VALUE_TILE_TOKENS

    out = nl.ndarray((q_len, num_q_heads, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    for kv_head in nl.affine_range(num_kv_heads):
        for q_idx in nl.affine_range(heads_per_kv):
            q_head = kv_head * heads_per_kv + q_idx

            for q_block_idx in nl.affine_range(num_q_blocks):
                q_start = q_block_idx * BLOCK_Q

                q_sbuf = nl.ndarray((d_head, BLOCK_Q), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_transpose(
                    dst=q_sbuf,
                    src=q[nl.ds(q_start, BLOCK_Q), q_head, :],
                )
                nisa.activation(dst=q_sbuf, op=nl.copy, data=q_sbuf, scale=scale)

                row_max = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                sum_row = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                running_out = nl.ndarray(
                    (BLOCK_Q, d_head), dtype=nl.float32, buffer=nl.sbuf
                )

                masked_qk = nl.ndarray(
                    (BLOCK_Q, BLOCK_K), dtype=nl.float32, buffer=nl.sbuf
                )
                qk_sbuf = nl.ndarray(
                    (BLOCK_Q, BLOCK_K), dtype=nl.float32, buffer=nl.sbuf
                )
                tile_max = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                new_row_max = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                old_max_delta = nl.ndarray(
                    (BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                old_max_scale = nl.ndarray(
                    (BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                neg_row_max = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                exp_tile = nl.ndarray(
                    (BLOCK_Q, BLOCK_K), dtype=nl.float32, buffer=nl.sbuf
                )
                tile_sum = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                scaled_sum_row = nl.ndarray(
                    (BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                new_sum_row = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                scaled_running_out = nl.ndarray(
                    (BLOCK_Q, d_head), dtype=nl.float32, buffer=nl.sbuf
                )
                attn_out_psum = nl.ndarray(
                    (BLOCK_Q, d_head), dtype=nl.float32, buffer=nl.psum
                )
                prob_t_psum = nl.ndarray(
                    (VALUE_TILE_TOKENS, BLOCK_Q), dtype=nl.float32, buffer=nl.psum
                )
                prob_t = nl.ndarray(
                    (VALUE_TILE_TOKENS, BLOCK_Q), dtype=v.dtype, buffer=nl.sbuf
                )
                qk_psum = nl.ndarray(
                    (BLOCK_Q, BLOCK_K), dtype=nl.float32, buffer=nl.psum
                )
                v_sbuf = nl.ndarray(
                    (VALUE_TILE_TOKENS, d_head), dtype=v.dtype, buffer=nl.sbuf
                )

                nisa.memset(dst=row_max, value=NEG_INF)
                nisa.memset(dst=sum_row, value=0.0)
                nisa.memset(dst=running_out, value=0.0)
                active_k_tiles = q_block_idx // q_blocks_per_k_tile + 1

                for k_block_idx in nl.affine_range(active_k_tiles):
                    k_start = k_block_idx * BLOCK_K
                    k_sbuf = nl.ndarray(
                        (d_head, BLOCK_K), dtype=k.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_transpose(
                        dst=k_sbuf,
                        src=k[nl.ds(k_start, BLOCK_K), kv_head, :],
                    )
                    nisa.memset(dst=qk_psum, value=0.0)
                    nisa.nc_matmul(dst=qk_psum, stationary=q_sbuf, moving=k_sbuf)
                    nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)
                    nisa.affine_select(
                        dst=masked_qk,
                        pattern=[[-1, BLOCK_K]],
                        offset=q_start - k_start,
                        channel_multiplier=1,
                        on_true_tile=qk_sbuf,
                        on_false_value=NEG_INF,
                        cmp_op=nl.greater_equal,
                    )
                    nisa.tensor_reduce(
                        dst=tile_max,
                        op=nl.maximum,
                        data=masked_qk,
                        axis=(1,),
                    )
                    nisa.tensor_tensor(
                        dst=new_row_max,
                        data1=row_max,
                        data2=tile_max,
                        op=nl.maximum,
                    )
                    nisa.tensor_scalar(
                        dst=old_max_delta,
                        data=row_max,
                        op0=nl.subtract,
                        operand0=new_row_max,
                    )
                    nisa.activation(dst=old_max_scale, op=nl.exp, data=old_max_delta)
                    nisa.tensor_scalar(
                        dst=neg_row_max,
                        data=new_row_max,
                        op0=nl.multiply,
                        operand0=-1.0,
                    )
                    nisa.activation(
                        dst=exp_tile,
                        op=nl.exp,
                        data=masked_qk,
                        bias=neg_row_max,
                        reduce_op=nl.add,
                        reduce_res=tile_sum,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce,
                    )
                    nisa.tensor_scalar(
                        dst=scaled_sum_row,
                        data=sum_row,
                        op0=nl.multiply,
                        operand0=old_max_scale,
                    )
                    nisa.tensor_tensor(
                        dst=new_sum_row,
                        data1=scaled_sum_row,
                        data2=tile_sum,
                        op=nl.add,
                    )
                    nisa.tensor_scalar(
                        dst=scaled_running_out,
                        data=running_out,
                        op0=nl.multiply,
                        operand0=old_max_scale,
                    )

                    nisa.memset(dst=attn_out_psum, value=0.0)
                    for value_tile_idx in nl.affine_range(value_tiles_per_k_tile):
                        value_start = value_tile_idx * VALUE_TILE_TOKENS
                        nisa.nc_transpose(
                            dst=prob_t_psum,
                            data=exp_tile[:, nl.ds(value_start, VALUE_TILE_TOKENS)],
                        )
                        nisa.tensor_copy(dst=prob_t, src=prob_t_psum)
                        nisa.dma_copy(
                            dst=v_sbuf,
                            src=v[
                                nl.ds(k_start + value_start, VALUE_TILE_TOKENS),
                                kv_head,
                                :,
                            ],
                        )
                        nisa.nc_matmul(
                            dst=attn_out_psum,
                            stationary=prob_t,
                            moving=v_sbuf,
                        )

                    nisa.tensor_tensor(
                        dst=running_out,
                        data1=scaled_running_out,
                        data2=attn_out_psum,
                        op=nl.add,
                    )
                    nisa.tensor_copy(dst=row_max, src=new_row_max)
                    nisa.tensor_copy(dst=sum_row, src=new_sum_row)

                inv_sum_row = nl.ndarray((BLOCK_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                normalized_out = nl.ndarray(
                    (BLOCK_Q, d_head), dtype=nl.float32, buffer=nl.sbuf
                )
                attn_out = nl.ndarray((BLOCK_Q, d_head), dtype=q.dtype, buffer=nl.sbuf)
                nisa.reciprocal(dst=inv_sum_row, data=sum_row)
                nisa.tensor_scalar(
                    dst=normalized_out,
                    data=running_out,
                    op0=nl.multiply,
                    operand0=inv_sum_row,
                )
                nisa.tensor_copy(dst=attn_out, src=normalized_out)

                nisa.dma_copy(
                    dst=out[nl.ds(q_start, BLOCK_Q), q_head, :],
                    src=attn_out,
                )

    return out
