import nki.isa as nisa
import nki.language as nl

_zml_orig_tensor_copy = nisa.tensor_copy


def _zml_tensor_copy_compat(dst, src, engine=None):
    if engine == nisa.scalar_engine:
        return nisa.activation(dst, op=nl.copy, data=src)
    if engine is None:
        return _zml_orig_tensor_copy(dst, src)
    return _zml_orig_tensor_copy(dst, src, engine=engine)


nisa.tensor_copy = _zml_tensor_copy_compat


def decode_tkg(q, k, v, token_index):
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

    out = nl.ndarray((q_len, num_q_heads, d_head), dtype=q.dtype, buffer=nl.shared_hbm)
    token = nl.load(token_index)[0, 0]

    positions = nl.ndarray((q_len, seq_len), dtype=token_index.dtype, buffer=nl.sbuf)
    nisa.iota(positions, [[1, seq_len]])
    token_broad = nl.broadcast_to(token, positions.shape)
    valid = nl.less_equal(positions, token_broad)
    zero_mask = nl.zeros((q_len, seq_len), dtype=nl.float32, buffer=nl.sbuf)
    neg_mask = nl.full(
        (q_len, seq_len), -3.3895313892515355e38, dtype=nl.float32, buffer=nl.sbuf
    )
    attn_mask = nl.where(valid, zero_mask, neg_mask, dtype=nl.float32)

    qk = nl.ndarray((heads_per_kv, seq_len), dtype=nl.float32, buffer=nl.psum)
    qk_sbuf = nl.ndarray((heads_per_kv, seq_len), dtype=nl.float32, buffer=nl.sbuf)
    row_max = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    norm_row = nl.ndarray((heads_per_kv, seq_len), dtype=nl.float32, buffer=nl.sbuf)
    exp_row = nl.ndarray((heads_per_kv, seq_len), dtype=nl.float32, buffer=nl.sbuf)
    sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    inverse_sum_row = nl.ndarray((heads_per_kv, 1), dtype=nl.float32, buffer=nl.sbuf)
    exp_t_psum = nl.ndarray((seq_len, heads_per_kv), dtype=nl.float32, buffer=nl.psum)
    exp_t = nl.ndarray((seq_len, heads_per_kv), dtype=nl.float32, buffer=nl.sbuf)
    v_t_psum = nl.ndarray((seq_len, d_head), dtype=nl.float32, buffer=nl.psum)
    v_t = nl.ndarray((seq_len, d_head), dtype=nl.float32, buffer=nl.sbuf)
    attn_out_psum = nl.ndarray((heads_per_kv, d_head), dtype=nl.float32, buffer=nl.psum)
    attn_out_sbuf = nl.ndarray((heads_per_kv, d_head), dtype=nl.float32, buffer=nl.sbuf)

    for kv_head in nl.sequential_range(num_kv_heads):
        q_head = kv_head * heads_per_kv

        q_sbuf = nl.load_transpose2d(
            q[0, q_head : q_head + heads_per_kv, :], dtype=q.dtype
        )
        q_sbuf = nl.multiply(q_sbuf, scale, dtype=q.dtype)
        k_sbuf = nl.load_transpose2d(k[:, kv_head, :], dtype=k.dtype)
        v_sbuf = nl.load_transpose2d(v[:, kv_head, :], dtype=v.dtype)

        nisa.nc_matmul(qk, q_sbuf, k_sbuf)
        nisa.tensor_copy(qk_sbuf, qk)
        qk_sbuf = nl.add(
            qk_sbuf, nl.broadcast_to(attn_mask, qk_sbuf.shape), dtype=nl.float32
        )

        nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=qk_sbuf, axis=(1,))
        nisa.tensor_scalar(
            dst=norm_row, data=qk_sbuf, op0=nl.subtract, operand0=row_max
        )
        nisa.activation(dst=exp_row, op=nl.exp, data=norm_row)
        nisa.tensor_reduce(dst=sum_row, op=nl.add, data=exp_row, axis=(1,))
        nisa.reciprocal(dst=inverse_sum_row, data=sum_row)

        nisa.nc_transpose(exp_t_psum, exp_row)
        nisa.tensor_copy(exp_t, exp_t_psum)
        nisa.nc_transpose(v_t_psum, v_sbuf)
        nisa.tensor_copy(v_t, v_t_psum)

        nisa.memset(attn_out_psum, 0.0)
        nisa.nc_matmul(attn_out_psum, exp_t, v_t)
        nisa.tensor_scalar(
            dst=attn_out_sbuf,
            data=attn_out_psum,
            op0=nl.multiply,
            operand0=inverse_sum_row,
        )

        nl.store(dst=out[0, q_head : q_head + heads_per_kv, :], value=attn_out_sbuf)

    return out
