import math

import nki.isa as nisa
import nki.language as nl


def gated_deltanet(q, k, v, g, beta, h0):
    batch, time_steps, q_heads, key_dim = q.shape
    v_batch, v_time_steps, value_heads, value_dim = v.shape

    assert q.shape == k.shape
    assert batch == v_batch
    assert time_steps == v_time_steps
    assert q_heads == 1
    assert value_heads == 1
    assert g.shape == (batch, time_steps, value_heads)
    assert beta.shape == (batch, time_steps, value_heads)
    assert h0.shape == (batch, value_heads, value_dim, key_dim)

    out = nl.ndarray((batch, time_steps, value_dim, value_heads), dtype=nl.float32, buffer=nl.shared_hbm)
    ht = nl.ndarray(h0.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    h = nl.load(h0[0, 0, :, :], dtype=nl.float32)
    out_sbuf = nl.ndarray((value_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    scale = 1.0 / math.sqrt(key_dim)

    for t in nl.sequential_range(time_steps):
        q_src = nl.load(q[0, t, :, :])
        k_src = nl.load(k[0, t, :, :])
        v_src = nl.load_transpose2d(v[0, t, :, :])
        beta_src = nl.load(beta[:, t, :])

        q_t = nl.ndarray(q_src.shape, dtype=nl.float32, buffer=nl.sbuf)
        k_t = nl.ndarray(k_src.shape, dtype=nl.float32, buffer=nl.sbuf)
        v_t = nl.ndarray(v_src.shape, dtype=nl.float32, buffer=nl.sbuf)
        beta_t = nl.ndarray(beta_src.shape, dtype=nl.float32, buffer=nl.sbuf)

        nisa.tensor_copy(q_t, q_src)
        nisa.tensor_copy(k_t, k_src)
        nisa.tensor_copy(v_t, v_src)
        nisa.tensor_copy(beta_t, beta_src)

        g_t = nl.load(g[:, t, :], dtype=nl.float32)

        q_inv_norm = nl.rsqrt(nl.add(nl.sum(nl.multiply(q_t, q_t), axis=1, keepdims=True), 1.0e-6))
        k_inv_norm = nl.rsqrt(nl.add(nl.sum(nl.multiply(k_t, k_t), axis=1, keepdims=True), 1.0e-6))
        q_t = nl.multiply(nl.multiply(q_t, nl.broadcast_to(q_inv_norm, q_t.shape)), scale)
        k_t = nl.multiply(k_t, nl.broadcast_to(k_inv_norm, k_t.shape))

        h = nl.multiply(h, nl.broadcast_to(nl.exp(g_t), h.shape))
        predicted_v = nl.sum(nl.multiply(h, nl.broadcast_to(k_t, h.shape)), axis=1, keepdims=True)
        delta_v = nl.multiply(nl.broadcast_to(beta_t, v_t.shape), nl.subtract(v_t, predicted_v))
        h = nl.add(h, nl.multiply(nl.broadcast_to(k_t, h.shape), nl.broadcast_to(delta_v, h.shape)))

        o_t = nl.sum(nl.multiply(h, nl.broadcast_to(q_t, h.shape)), axis=1, keepdims=True)
        nisa.tensor_copy(out_sbuf, o_t)
        nl.store(out[0, t, :, :], value=out_sbuf)

    nl.store(ht[0, 0, :, :], value=h)

    return out, ht
