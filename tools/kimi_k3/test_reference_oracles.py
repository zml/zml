from __future__ import annotations

import numpy as np

from reference_oracles import (
    attention_residual_select,
    causal_conv_tail,
    causal_depthwise_conv1d,
    decode_e8m0,
    dequantize_mxfp4,
    expand_block32_scale,
    kda_log_alpha,
    kda_scan,
    l2_norm,
    mla_nope_join,
    mla_scale,
    mxfp4_linear,
    rms_norm,
    route,
    sigmoid,
    situ_glu,
    softmax,
    topk_descending,
    unpack_e2m1,
)


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


def test_common_primitives() -> None:
    value = np.asarray([[1.0, -2.0, 0.5, 4.0]], dtype=np.float32)
    weight = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    np.testing.assert_allclose(
        rms_norm(value, weight),
        value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + 1e-6) * weight,
        atol=1e-7,
    )
    np.testing.assert_allclose(np.sum(l2_norm(value) ** 2, axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(softmax(value).sum(axis=-1), 1.0, atol=1e-7)
    assert np.all((sigmoid(value) > 0.0) & (sigmoid(value) < 1.0))
    assert np.isfinite(situ_glu(value, -value)).all()
    values, ids = topk_descending(value, 2)
    np.testing.assert_array_equal(ids, [[3, 0]])
    np.testing.assert_array_equal(values, [[4.0, 1.0]])


def test_causal_depthwise_conv_and_tail() -> None:
    value = np.arange(1, 13, dtype=np.float32).reshape(1, 4, 3)
    kernel = np.asarray(
        [[[1.0, 2.0, 3.0]], [[-1.0, 0.0, 1.0]], [[0.5, 0.25, -0.5]]],
        dtype=np.float32,
    )
    output = causal_depthwise_conv1d(value, kernel)
    assert output.shape == value.shape
    np.testing.assert_allclose(output[:, 0], value[:, 0] * kernel[:, 0, -1])
    np.testing.assert_array_equal(causal_conv_tail(value, 3), value[:, -3:])
    short = causal_conv_tail(value[:, :2], 3)
    np.testing.assert_array_equal(short[:, 0], np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_array_equal(short[:, 1:], value[:, :2])


def test_mla_nope_scaling() -> None:
    content = np.ones((1, 2, 3, 4), dtype=np.float32)
    extra = np.full((1, 2, 3, 2), 2.0, dtype=np.float32)
    joined = mla_nope_join(content, extra)
    assert joined.shape == (1, 2, 3, 6)
    np.testing.assert_allclose(mla_scale(np.full((2, 2), 192.0)), np.sqrt(192.0))


def test_mxfp4_block_expansion_and_linear() -> None:
    packed = np.asarray([[0x21] * 16, [0xF8] * 16], dtype=np.uint8)
    scale = np.asarray([[127], [126]], dtype=np.uint8)
    expanded = expand_block32_scale(scale)
    assert expanded.shape == (2, 32)
    weight = dequantize_mxfp4(packed, scale)
    assert weight.shape == (2, 32)
    value = np.arange(64, dtype=np.float32).reshape(2, 32) / 16.0
    np.testing.assert_allclose(mxfp4_linear(value, packed, scale), value @ weight.T)


def test_attention_residual_masks_stale_sources() -> None:
    prefix = np.asarray([[1.0, 2.0, -1.0, 0.5]], dtype=np.float32)
    blocks = np.asarray(
        [[[0.5, -1.0, 2.0, 3.0], [1e6, -1e6, 1e6, -1e6]]], dtype=np.float32
    )
    norm = np.asarray([1.0, 0.75, 1.25, 0.5], dtype=np.float32)
    proj = np.asarray([-0.5, 0.25, 1.0, 0.75], dtype=np.float32)
    masked = attention_residual_select(prefix, blocks, [True, False], norm, proj)
    compact = attention_residual_select(prefix, blocks[:, :1], [True], norm, proj)
    np.testing.assert_allclose(masked.output, compact.output, atol=1e-6)
    np.testing.assert_allclose(masked.probabilities[:, [0, 2]], compact.probabilities)
    np.testing.assert_array_equal(masked.probabilities[:, 1], 0.0)


def test_attention_residual_scores_normalized_but_values_are_not() -> None:
    prefix = np.asarray([[100.0, 0.0]], dtype=np.float32)
    blocks = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    result = attention_residual_select(prefix, blocks, [True], [1.0, 1.0], [1.0, 0.0])
    np.testing.assert_allclose(result.probabilities, [[0.5, 0.5]], atol=1e-6)
    np.testing.assert_allclose(result.output, [[50.5, 0.0]], atol=1e-4)
