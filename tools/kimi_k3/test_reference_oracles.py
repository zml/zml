from __future__ import annotations

import numpy as np

from reference_oracles import decode_e8m0, kda_log_alpha, kda_scan, route, unpack_e2m1


def test_mxfp4_nibble_and_scale() -> None:
    np.testing.assert_array_equal(
        unpack_e2m1(np.asarray([[0x21, 0xF8]], dtype=np.uint8)),
        np.asarray([[0.5, 1.0, -0.0, -6.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        decode_e8m0(np.asarray([126, 127, 128], dtype=np.uint8)),
        np.asarray([0.5, 1.0, 2.0], dtype=np.float32),
    )


def test_router_bias_selects_but_does_not_weight() -> None:
    hidden = np.asarray([[1.0, -0.5]], dtype=np.float32)
    weight = np.asarray([[1.0, 0.0], [0.9, 0.0], [0.1, 0.0]], dtype=np.float32)
    result = route(hidden, weight, np.asarray([0.0, -10.0, 10.0]), top_k=2)
    assert set(result.ids[0]) == {0, 2}
    selected = np.take_along_axis(result.raw_scores, result.ids, axis=-1)
    np.testing.assert_allclose(result.weights, selected / selected.sum(axis=-1, keepdims=True))


def test_kda_continuation_matches_one_shot() -> None:
    rng = np.random.default_rng(31)
    q = rng.normal(size=(1, 5, 2, 4)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    v = rng.normal(size=q.shape).astype(np.float32)
    raw = rng.normal(size=q.shape).astype(np.float32)
    beta = rng.normal(size=q.shape[:-1]).astype(np.float32)
    a_log = rng.normal(size=(2,)).astype(np.float32)
    dt = rng.normal(size=(2, 4)).astype(np.float32)
    log_alpha = kda_log_alpha(raw, a_log, dt, -5.0)
    full, full_state = kda_scan(q, k, v, log_alpha, beta)
    first, state = kda_scan(q[:, :3], k[:, :3], v[:, :3], log_alpha[:, :3], beta[:, :3])
    tail, state = kda_scan(
        q[:, 3:], k[:, 3:], v[:, 3:], log_alpha[:, 3:], beta[:, 3:], state
    )
    np.testing.assert_allclose(np.concatenate((first, tail), axis=1), full, atol=1e-6)
    np.testing.assert_allclose(state, full_state, atol=1e-6)
