const std = @import("std");
const zml = @import("zml");

const CompileArgs = @import("../../compiler.zig").CompileArgs;

const QwenImage21Rope = @This();

// Without this field, the LSP doesn't understand that the struct is a struct
_phantom: struct {},

pub fn init() QwenImage21Rope {
    return QwenImage21Rope{ ._phantom = .{} };
}

pub fn forward(
    _: QwenImage21Rope,
    x: zml.Tensor, // [BS, S, D]
    position_ids: zml.Tensor,
    args: CompileArgs,
) [2]zml.Tensor {
    const inv_freq = compute_inv_freq(args);
    const inv_freq_expanded = inv_freq.broad(.init(.{ .rope = 3, .bs = x.dim(.bs), .seq_len = 1, .h = args.head_dim / 2 }, .f32));
    const position_ids_expanded = position_ids
        .withTags(.{ .rope, .bs, .seq_len })
        .appendAxes(.{.h});

    const target_shape = zml.Shape.init(.{ .rope = 3, .bs = x.dim(.bs), .seq_len = x.dim(.seq_len), .h = args.head_dim / 2 }, .f32);
    const freqs = inv_freq_expanded
        .convert(.f32).broad(target_shape).mul(position_ids_expanded.convert(.f32).broad(target_shape));

    return [2]zml.Tensor{
        recomposition_frequencies(freqs.cos()).convert(x.dtype()),
        recomposition_frequencies(freqs.sin()).convert(x.dtype()),
    };
}

// Could be lazyonce
fn compute_inv_freq(args: CompileArgs) zml.Tensor {
    const ar = zml.Tensor.arange(.{ .end = @intCast(args.head_dim), .step = 2 }, .f32).div(zml.Tensor.scalar(args.head_dim, .f32));
    const freq = zml.Tensor.scalar(args.rope_theta, .f32).pow(ar);
    return zml.Tensor.scalar(1, .f32).div(freq).withTags(.{.h});
}

fn recomposition_frequencies(x: zml.Tensor) zml.Tensor {
    const x0 = x.slice(0, .single(0)).squeeze(0);
    return zml.Tensor.concatenate(&.{ x0, x0 }, .h).insertAxes(.seq_len, .{.bs});
}
