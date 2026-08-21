const std = @import("std");

const zml = @import("zml");

/// Moonshot KimiRMSNorm: reduce in FP32, cast to the input dtype, then apply
/// the checkpoint weight directly (Kimi weights are not offset by one).
pub fn rmsNorm(input: zml.Tensor, weight: zml.Tensor, eps: f32) zml.Tensor {
    const normalized = zml.nn.rmsNorm(input, .d, eps);
    return normalized.convert(.f32)
        .mul(weight.convert(.f32).broad(normalized.shape()))
        .convert(input.dtype());
}

// Row-parallel projections reduce partial products across the model mesh.
// Accumulating those products in FP32 before the model-dtype boundary keeps
// single-rank and tensor-parallel recurrent sessions numerically aligned.
pub fn stableLinear(input: zml.Tensor, weight: zml.Tensor, axis: anytype) zml.Tensor {
    return input.convert(.f32)
        .dotWithPrecision(weight.convert(.f32), axis, .highest)
        .convert(input.dtype());
}

/// KDA q/k normalization.  The explicit FP32 conversion is required because
/// the shared normalizeL2 helper otherwise preserves the caller's reduction
/// dtype.
pub fn normalizeL2(input: zml.Tensor, eps: f32) zml.Tensor {
    return zml.nn.normalizeL2(input.convert(.f32), eps).convert(input.dtype());
}

/// K3 SiTU-GLU with the checkpoint contract beta=4, linear_beta=25.
pub fn situGlu(gate: zml.Tensor, up: zml.Tensor) zml.Tensor {
    const dtype = gate.dtype();
    const gate_f32 = gate.convert(.f32);
    const up_f32 = up.convert(.f32);
    const situ = gate_f32.scale(0.25).tanh().scale(4.0).mul(gate_f32.sigmoid());
    const linear = up_f32.scale(1.0 / 25.0).tanh().scale(25.0);
    return situ.mul(linear).convert(dtype);
}

pub fn sigmoid(input: zml.Tensor) zml.Tensor {
    return input.sigmoid();
}

pub fn softmax(input: zml.Tensor) zml.Tensor {
    return input.softmax(.d);
}

pub fn topKValues(input: zml.Tensor) zml.Tensor {
    return input.topK(.{ .top = .d }, 3, .{ .descending = true }).values;
}

pub fn topKIndices(input: zml.Tensor) zml.Tensor {
    return input.topK(.{ .top = .d }, 3, .{ .descending = true }).indices;
}

/// Depthwise correlation with left-only padding, matching Moonshot/Torch
/// causal Conv1d for [batch, sequence, channel] values.
pub fn causalDepthwiseConv1d(input: zml.Tensor, kernel: zml.Tensor) zml.Tensor {
    const channels = input.dim(.channel);
    const left_pad = kernel.dim(.kernel) - 1;
    return zml.Tensor.conv1d(input, kernel, .{
        .padding = &.{ left_pad, 0 },
        .input_batch_dimension = input.axis(.batch),
        .input_feature_dimension = input.axis(.channel),
        .input_spatial_dimensions = input.axis(.sequence),
        .kernel_output_feature_dimension = kernel.axis(.channel),
        .kernel_input_feature_dimension = kernel.axis(.one),
        .kernel_spatial_dimensions = kernel.axis(.kernel),
        .output_batch_dimension = input.axis(.batch),
        .output_feature_dimension = input.axis(.channel),
        .output_spatial_dimensions = input.axis(.sequence),
        .feature_group_count = channels,
    });
}

/// Preserve exactly the left context consumed by the next decode convolution.
pub fn causalConvTail(input: zml.Tensor, left_context: i64) zml.Tensor {
    const copy_len = @min(input.dim(.sequence), left_context);
    const tail = input.slice1d(.sequence, .{
        .start = input.dim(.sequence) - copy_len,
        .end = input.dim(.sequence),
    });
    if (copy_len == left_context) return tail;
    const padding_shape = zml.Shape.init(
        .{
            .batch = input.dim(.batch),
            .sequence = left_context - copy_len,
            .channel = input.dim(.channel),
        },
        input.dtype(),
    );
    const padding = zml.Tensor.constant(input.dtype().zero()).broad(padding_shape);
    return zml.Tensor.concatenate(&.{ padding, tail }, .sequence);
}

pub fn causalConvTail3(input: zml.Tensor) zml.Tensor {
    return causalConvTail(input, 3);
}

/// K3 asserts mla_use_nope=true: the 128 content and 64 extra dimensions are
/// concatenated without rotary transformation.
pub fn mlaNopeJoin(content: zml.Tensor, extra: zml.Tensor) zml.Tensor {
    return zml.Tensor.concatenate(&.{ content, extra }, .head_dim);
}

/// K3 q_head_dim is 128 NoPE + 64 extra = 192.
pub fn mlaScale(scores: zml.Tensor) zml.Tensor {
    return scores.scale(1.0 / std.math.sqrt(192.0));
}

/// Compressed-tensors stores the first logical E2M1 value in the low nibble.
/// StableHLO bitcast_convert exposes those two nibbles in that logical order.
pub fn unpackE2m1(packed_values: zml.Tensor) zml.Tensor {
    return packed_values.bitCast(.f4e2m1)
        .merge(.{ .d = .{ .kw, .bitcast } })
        .convert(.f32);
}

/// Decode biased E8M0 bytes as 2**(u8-127).  Keeping the source as u8 avoids
/// depending on a safetensors E8M0 dtype that is not present in this checkpoint.
pub fn decodeE8m0(scale: zml.Tensor) zml.Tensor {
    return scale.convert(.f32).addConstant(-127.0).scale(std.math.ln2).exp();
}

pub fn expandBlock32Scale(scale: zml.Tensor) zml.Tensor {
    return decodeE8m0(scale).stutter1d(scale.axis(.block), 32).renameTag(.block, .d);
}

/// Correctness-first MXFP4 dequantization.  Production optimized scaled-dot is
/// introduced only after this explicit reference path passes fixture parity.
pub fn dequantizeMxfp4(packed_values: zml.Tensor, scale: zml.Tensor) zml.Tensor {
    return unpackE2m1(packed_values).mul(expandBlock32Scale(scale));
}

pub fn slowMxfp4Linear(input: zml.Tensor, packed_values: zml.Tensor, scale: zml.Tensor) zml.Tensor {
    const input_f32 = input.convert(.f32);
    const weight = dequantizeMxfp4(packed_values, scale);
    const product_shape = zml.Shape.init(
        .{ .token = input_f32.dim(.token), .out = weight.dim(.out), .d = input_f32.dim(.d) },
        .f32,
    );
    // An explicit multiply/reduce is intentional for the slow oracle: the
    // general dot path may select TF32 on NVIDIA, which is appropriate for a
    // fast kernel but obscures strict dequantization parity at this milestone.
    const lhs = input_f32.reshape(.{ .token = input_f32.dim(.token), .out = 1, .d = input_f32.dim(.d) }).broad(product_shape);
    const rhs = weight.reshape(.{ .token = 1, .out = weight.dim(.out), .d = weight.dim(.d) }).broad(product_shape);
    return lhs.mul(rhs).sum(.d).squeeze(.d);
}

/// Native weight-only MXFP4 projection. The checkpoint bytes are reinterpreted
/// without dequantizing a BF16/F32 weight matrix: low-nibble-first E2M1 values
/// contract in blocks of 32 with their biased E8M0 scale bytes.
pub fn nativeMxfp4Linear(input: zml.Tensor, packed_values: zml.Tensor, scale: zml.Tensor) zml.Tensor {
    const weight = packed_values.bitCast(.f4e2m1)
        .merge(.{ .d = .{ .kw, .bitcast } });
    const native_scale = scale.bitCast(.f8e8m0);
    const acc = zml.nn.scaledDot(
        input.convert(.bf16),
        weight,
        null,
        native_scale,
        .d,
    );
    return acc.convert(input.dtype());
}
