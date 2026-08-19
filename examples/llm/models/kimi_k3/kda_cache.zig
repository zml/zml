const zml = @import("zml");

/// Moonshot KDA temporal state for one logical layer.
///
/// Convolution histories retain the full kernel window in
/// `[batch, channel, kernel]` order, matching `ShortConvolution.step`.
/// The recurrence is deliberately V-first `[batch, head, value, key]`, which
/// is what Moonshot requests through `transpose_state_layout=true`.
pub const Cache = struct {
    q_conv: zml.Tensor,
    k_conv: zml.Tensor,
    v_conv: zml.Tensor,
    recurrent_state: zml.Tensor,
};
