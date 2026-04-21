import nki.isa as nisa
import nki.language as nl


def zml_attention_fwd(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = k.shape[1]

    assert q.shape == k.shape == v.shape
    assert d_head == 128
    assert seqlen_q == 128
    assert seqlen_kv == 128

    out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf_t = nl.load_transpose2d(v)

    qk = nl.ndarray((seqlen_q, seqlen_kv), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(qk, q_sbuf, k_sbuf)
    row_max = nl.broadcast_to(nl.max(qk, axis=1, keepdims=True), qk.shape)
    norm_row = nl.subtract(qk, row_max)
    exp_row = nl.exp(norm_row)
    sum_row = nl.broadcast_to(nl.sum(exp_row, axis=1, keepdims=True), exp_row.shape)
    scores = nl.multiply(exp_row, nl.reciprocal(sum_row))

    scores_sbuf = nl.ndarray(scores.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(scores_sbuf, scores)
    attn_out = nl.ndarray((seqlen_q, d_head), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(attn_out, scores_sbuf, v_sbuf_t)
    out_sbuf = nl.ndarray((seqlen_q, d_head), dtype=out.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(out_sbuf, attn_out)
    nl.store(dst=out, value=out_sbuf)

    return out


def zml_attention_decode_sample(q, k, v):
    d_head, num_q_heads, q_len = q.shape
    seq_len = k.shape[2]

    assert q_len == 1
    assert k.shape == v.shape
    assert k.shape[0] == d_head
    assert k.shape[1] == 8
    assert num_q_heads == 32
    assert d_head == 64
    assert seq_len == 128

    out = nl.ndarray((d_head, num_q_heads, q_len), dtype=q.dtype, buffer=nl.shared_hbm)
    qk = nl.ndarray((q_len, seq_len), dtype=nl.float32, buffer=nl.psum)
    qk_sbuf = nl.ndarray((q_len, seq_len), dtype=nl.bfloat16, buffer=nl.sbuf)
    out_sbuf = nl.ndarray((d_head, q_len), dtype=out.dtype, buffer=nl.sbuf)

    for q_head in nl.sequential_range(32):
        kv_head = q_head // 4
        q_sbuf = nl.load(q[:, q_head, :])
        k_sbuf = nl.load(k[:, kv_head, :])
        v_sbuf = nl.load(v[:, kv_head, :])

        nisa.nc_matmul(qk, q_sbuf, k_sbuf)
        nisa.tensor_copy(qk_sbuf, qk)
        row_max = nl.broadcast_to(nl.max(qk_sbuf, axis=1, keepdims=True), qk_sbuf.shape)
        norm_row = nl.subtract(qk_sbuf, row_max)
        exp_row = nl.exp(norm_row)
        sum_row = nl.broadcast_to(nl.sum(exp_row, axis=1, keepdims=True), exp_row.shape)
        scores = nl.multiply(exp_row, nl.reciprocal(sum_row))

        weighted_v = nl.multiply(v_sbuf, nl.broadcast_to(scores, v_sbuf.shape))
        reduced = nl.sum(weighted_v, axis=1, keepdims=True)
        nisa.tensor_copy(out_sbuf, reduced)
        nl.store(dst=out[:, q_head, :], value=out_sbuf)

    return out
