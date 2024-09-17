//! Common layer definition and functions for Neural Networks (NN)
const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const zml = @import("zml.zig");
const meta = @import("meta.zig");
const helpers = @import("helpers.zig");
const ops = @import("ops.zig");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.zml_tensor);

const cuda = @import("nn/cuda.zig");

pub const Linear = struct {
    weight: Tensor,
    bias: ?Tensor = null,

    pub fn forward(self: Linear, x: Tensor) Tensor {
        var y = x.dotGeneral(self.weight.convert(x.dtype()), &.{.{ -1, -1 }}, &.{});
        // If self.weight doesn't have tags, preserve tags from x.
        if (y.shape().tag(-1) == Shape.TagUnknown) {
            y._shape._tags.set(y.rank() - 1, x.shape().tag(-1));
        }

        // log.debug("Linear({*}): {d} -> {d} -> {d}", .{ self, x.dims(), y.dims(), if (self.bias) |bias| y.add(bias).dims() else y.dims() });
        return if (self.bias) |bias| y.add(bias.broadcastLeft(y.shape())) else y;
    }
};

pub const TokenEmbedding = struct {
    weight: Tensor,

    pub fn forward(self: TokenEmbedding, idx: Tensor) Tensor {
        meta.assert(idx.dtype().isInteger(), "TokenEmbedding expects an integer input, received: {}", .{idx});
        meta.assert(self.weight.rank() == 2, "TokenEmbedding expects it's weight Tensor to be a 2D matrix, got {}", .{self.weight});
        return self.weight.gather1d(0, idx, .{});
    }
};

pub const Activation = union(enum) {
    sigmoid,
    tanh,
    relu,
    leakyReLU: f32,
    silu,
    gelu,
    quick_gelu,

    pub fn forward(self: Activation, x: Tensor) Tensor {
        return switch (self) {
            .sigmoid => x.sigmoid(),
            .tanh => x.tanh(),
            .relu => x.relu(),
            .silu => x.silu(),
            .gelu => x.gelu(),
            .quick_gelu => x.quickGelu(),
            .leakyReLU => |slope| x.leakyReLU(slope),
        };
    }
};

pub fn chainModules(module_list: anytype, input: Tensor) Tensor {
    const T = @TypeOf(module_list);
    switch (@typeInfo(T)) {
        .Struct => |struct_info| {
            var x = input;
            inline for (struct_info.fields) |field| {
                x = @field(module_list, field.name).forward(x);
            }
            return x;
        },
        else => @compileError("chainModules only works on a struct with only containing 'module' struct."),
    }
}

/// Layer Normalization
pub const LayerNorm = struct {
    weight: Tensor,
    bias: ?Tensor = null,
    eps: f32 = 1e-5,

    pub fn forward(self: LayerNorm, x: Tensor) Tensor {
        const normed = normalizeVariance(x, self.eps);

        var out = normed.mul(self.weight.broadcastLeft(x.shape()));
        if (self.bias) |bias| out = out.add(bias.broadcastLeft(x.shape()));

        return out;
    }
};

/// Center and scale by the variance.
/// normalize(x, eps) = (x - mean(x)) / sqrt(var(x) + eps)
/// Work on the last axis.
pub fn normalizeVariance(x: Tensor, eps: f32) Tensor {
    const N: f32 = @floatFromInt(x.dim(-1));

    // Upcast to improve precision
    const xf32 = x.convert(.f32);
    const mean = xf32.sum(-1).scale(1.0 / N);
    const mean_dev = xf32.sub(mean.broadcastRight(xf32.shape()));
    const variance = mean_dev.mul(mean_dev).sum(-1).scale(1.0 / N);
    const rsqrt = Tensor.rsqrt(variance.addConstant(eps));

    return mean_dev.mul(rsqrt.broadcastRight(mean_dev.shape())).convert(x.dtype());
}

// ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
// Implementation equivalent to `nn.functional.normalize(tensor, dim=-1)` call
pub fn normalizeL2(input: Tensor, eps: f32) Tensor {
    const inv_norm = input.pow(Tensor.scalar(2, input.dtype())).sum(-1).addConstant(eps).rsqrt();
    return input.mul(inv_norm.broad(input.shape()));
}

test normalizeL2 {
    const platform = zml.testing.env();

    const input = try zml.Buffer.fromSlice(platform, .{ 2, 2 }, &[_]f32{ -0.9686, -1.0058, -1.7808, 0.6698 });

    const res = try zml.testing.compileAndCall(platform, zml.nn.normalizeL2, .{ input, 1e-12 });
    const expectation = zml.HostBuffer.fromSlice(.{ 2, 2 }, &[_]f32{ -0.6937, -0.7203, -0.9360, 0.3520 });
    try zml.testing.expectClose(expectation, res, 1e-4);
}

pub const RopeOpts = struct {
    /// There are two implementations corresponding to how to split `x` in real/imag parts.
    /// * Interleaved means that the real/imag of each scalar is contiguous.
    /// * Sequential means that you first read all real values then all imag values.
    pub const Implementation = enum { interleaved, sequential };

    impl: Implementation,
    freq_base: f32 = 10_000,
};

pub const CosSin = [2]Tensor;

/// Rotary position embedding modify queries and keys tensor before compute Q * K in self attention.
/// This biases a token to look at token near him.
/// The nice thing with this solution is that you can cache the modified queries and keys directly.
/// See: https://paperswithcode.com/method/rope
pub fn rope(x: Tensor, cos_sin_cache: CosSin, opts: RopeOpts) Tensor {
    const cos, const sin = cos_sin_cache;
    meta.assert(x.dim(-1) == 2 * cos.dim(-1), "Couldn't compute rope({}, {}, {})", .{ x, cos, sin });
    // broadcast cos / sin to .{ batch, .seq, .half_dim }
    const x_real, const x_imag = splitRealImg(x, opts.impl);
    const has_tags = cos.shape().tag(0) != Shape.TagUnknown;
    const b_cos = if (has_tags) cos.broad(x_real.shape()) else cos.broadcastLeft(x_real.shape());
    const b_sin = if (has_tags) sin.broad(x_real.shape()) else sin.broadcastLeft(x_real.shape());

    // apply rotation
    const y_real = x_real.mul(b_cos).sub(x_imag.mul(b_sin));
    const y_imag = x_real.mul(b_sin).add(x_imag.mul(b_cos));

    // flatten last dimensions
    const y = mergeRealImg(y_real, y_imag, opts.impl);

    return y;
}

pub fn ropeCosSin(sh: anytype, dtype: DataType, opts: RopeOpts) CosSin {
    const shape = Shape.init(sh, dtype);
    meta.assert(shape.rank() == 2, "ropeCosSin({}) shape need to exactly have 2 axes", .{shape});
    const seq_len, const head_dim = .{ shape.dim(0), shape.dim(1) };
    meta.assert(@mod(head_dim, 2) == 0, "ropeCosSin requires an even head_dim, got {}", .{head_dim});

    // compute sin and cos in f32 before downcasting to x type.
    const inv_freq = invFreq(head_dim, opts.freq_base, .f32);
    var inv_freq_pos = Tensor.outer(Tensor.arange(.{ .end = seq_len }, .f32), inv_freq).convert(shape.dtype());
    inv_freq_pos._shape._tags = shape._tags;
    const cos = inv_freq_pos.cos();
    const sin = inv_freq_pos.sin();
    return .{ cos, sin };
}

pub fn splitRealImg(x: Tensor, impl: RopeOpts.Implementation) [2]Tensor {
    const n = x.dim(-1);

    return switch (impl) {
        .sequential => .{
            x.slice1d(-1, .{ .end = @divExact(n, 2) }),
            x.slice1d(-1, .{ .start = @divExact(n, 2), .end = n }),
        },
        .interleaved => .{
            x.slice1d(-1, .{ .start = 0, .step = 2 }),
            x.slice1d(-1, .{ .start = 1, .step = 2 }),
        },
    };
}

pub fn mergeRealImg(x_real: Tensor, x_imag: Tensor, impl: RopeOpts.Implementation) Tensor {
    return switch (impl) {
        .sequential => Tensor.concatenate(&.{ x_real, x_imag }, -1),
        .interleaved => Tensor.concatenate(&.{
            x_real.appendAxes(.{.interleaved_real_img}),
            x_imag.appendAxes(.{.interleaved_real_img}),
        }, -1).flatten(-2),
    };
}

/// {exp( - n * ln(10_000) / N ) | n in [0..N] }
pub fn invFreq(N: i64, theta: f32, dtype: DataType) Tensor {
    const freq = -@log(theta) / @as(f32, @floatFromInt(N));
    const range = Tensor.arange(.{ .start = 0, .end = N, .step = 2 }, dtype).scale(freq);
    return range.exp();
}

test "real/img" {
    const platform = zml.testing.env();

    const Fns = struct {
        // fn testSplitMergeIsId(impl: RopeOpts.Implementation) Tensor {
        //     const x = Tensor.arange(.{ .end = 20 }, .f32).reshape(.{ 5, 4 });
        //     const real, const imag = splitRealImg(x, impl);
        //     const y = mergeRealImg(real, imag, impl);
        //     return y.cmp(.EQ, x).flatten(0).convert(.i32).sum(-1);
        // }

        fn testSplitMergeIsId(impl: RopeOpts.Implementation) Tensor {
            const x = Tensor.arange(.{ .end = 20 }, .f32).reshape(.{ 5, 4 });
            const real, const imag = splitRealImg(x, impl);
            const y = mergeRealImg(real, imag, impl);
            const real2, const imag2 = splitRealImg(y, impl);
            return real.cmp(.EQ, real2).flatten(0).convert(.i32).sum(-1).add(
                imag.cmp(.EQ, imag2).flatten(0).convert(.i32).sum(-1),
            );
        }

        fn testSplitSeqVoid(_: void) Tensor {
            const x = Tensor.arange(.{ .end = 20 }, .f32).reshape(.{ 5, 4 });
            const real, const imag = splitRealImg(x, .sequential);
            const x_real = Tensor.concatenate(&.{
                Tensor.arange(.{ .start = 0, .end = 20, .step = 4 }, .f32).reshape(.{ 5, 1 }),
                Tensor.arange(.{ .start = 1, .end = 20, .step = 4 }, .f32).reshape(.{ 5, 1 }),
            }, 1);
            const x_imag = Tensor.concatenate(&.{
                Tensor.arange(.{ .start = 2, .end = 20, .step = 4 }, .f32).reshape(.{ 5, 1 }),
                Tensor.arange(.{ .start = 3, .end = 20, .step = 4 }, .f32).reshape(.{ 5, 1 }),
            }, 1);

            return real.cmp(.EQ, x_real).flatten(0).convert(.i32).sum(-1).add(
                imag.cmp(.EQ, x_imag).flatten(0).convert(.i32).sum(-1),
            );
        }

        fn testSplitSeq() Tensor {
            const x = Tensor.arange(.{ .end = 20 }, .f32).reshape(.{ 5, 4 });
            const real, const imag = splitRealImg(x, .sequential);
            const x_real = Tensor.concatenate(&.{
                Tensor.arange(.{ .start = 0, .end = 20, .step = 4 }, .f32).reshape(.{ 5, 1 }),
                Tensor.arange(.{ .start = 1, .end = 20, .step = 4 }, .f32).reshape(.{ 5, 1 }),
            }, 1);
            const x_imag = Tensor.concatenate(&.{
                Tensor.arange(.{ .start = 2, .end = 20, .step = 4 }, .f32).reshape(.{ 5, 1 }),
                Tensor.arange(.{ .start = 3, .end = 20, .step = 4 }, .f32).reshape(.{ 5, 1 }),
            }, 1);

            return real.cmp(.EQ, x_real).flatten(0).convert(.i32).sum(-1).add(
                imag.cmp(.EQ, x_imag).flatten(0).convert(.i32).sum(-1),
            );
        }

        fn testSplitInterleaved() Tensor {
            const x = Tensor.arange(.{ .end = 20 }, .f32).reshape(.{ 5, 4 });
            const real, const imag = splitRealImg(x, .interleaved);
            const x_real = Tensor.arange(.{ .start = 0, .end = 20, .step = 2 }, .f32).reshape(.{ 5, 2 });
            const x_imag = Tensor.arange(.{ .start = 1, .end = 20, .step = 2 }, .f32).reshape(.{ 5, 2 });

            return real.cmp(.EQ, x_real).flatten(0).convert(.i32).sum(-1).add(
                imag.cmp(.EQ, x_imag).flatten(0).convert(.i32).sum(-1),
            );
        }
    };

    const d_interleaved = try zml.testing.compileAndCall(platform, Fns.testSplitMergeIsId, .{.interleaved});
    try testing.expectEqual(20, d_interleaved.getValue(i32));

    const d_sequential = try zml.testing.compileAndCall(platform, Fns.testSplitMergeIsId, .{.sequential});
    try testing.expectEqual(20, d_sequential.getValue(i32));

    // test the function that accepts 1 void argument
    const d_split_seq_void = try zml.testing.compileAndCall(platform, Fns.testSplitSeqVoid, .{{}});
    try testing.expectEqual(20, d_split_seq_void.getValue(i32));

    // test the function that takes NO arguments
    const d_split_seq = try zml.testing.compileAndCall(platform, Fns.testSplitSeq, .{});
    try testing.expectEqual(20, d_split_seq.getValue(i32));

    // now try compiling and calling ourselves
    {
        const mod = try zml.compileFn(std.testing.allocator, Fns.testSplitSeq, .{}, platform);
        defer mod.deinit();
        const ret = mod.call(.{});
        try testing.expectEqual(20, ret.getValue(i32));
    }
    const d_split_interleaved = try zml.testing.compileAndCall(platform, Fns.testSplitInterleaved, .{});
    try testing.expectEqual(20, d_split_interleaved.getValue(i32));
}

test "rope" {
    const platofrm = zml.testing.env();

    const TestRope = struct {
        fn forward(x: Tensor, opts: RopeOpts) Tensor {
            var input = x;
            {
                // Convert input to the requested format
                const real, const imag = splitRealImg(input, .sequential);
                input = mergeRealImg(real, imag, opts.impl);
            }
            const cos_sin = ropeCosSin(.{ input.dim(-2), input.dim(-1) }, input.dtype(), opts);
            var res = rope(input, cos_sin, opts).squeeze(0);

            {
                // Convert back to sequential
                const real, const imag = splitRealImg(res, opts.impl);
                res = mergeRealImg(real, imag, .sequential);
            }
            return res;
        }
    };

    // x is made such as the interleaved and sequential reps are the same.
    // So the two implementations should give the same results.
    const x = try zml.Buffer.fromSlice(platofrm, .{ 1, 5, 4 }, &[_]f32{ 1.0, 0.1, -1.0, -0.5 } ** 5);
    const res1 = try zml.testing.compileAndCall(platofrm, TestRope.forward, .{ x, RopeOpts{ .impl = .interleaved } });
    const res2 = try zml.testing.compileAndCall(platofrm, TestRope.forward, .{ x, RopeOpts{ .impl = .sequential } });

    try zml.testing.expectClose(res1, res2, 1e-4);
}

/// In neural network we generally care about the relative precision,
/// but on a given dimension, if the output is close to 0, then the precision
/// don't matter as much.
fn approxEq(comptime Float: type, l: Float, r: Float, tolerance: Float) bool {
    const closeRel = std.math.approxEqRel(Float, l, r, @floatCast(tolerance));
    const closeAbs = std.math.approxEqAbs(Float, l, r, @floatCast(tolerance / 2));
    return closeRel or closeAbs;
}

pub const UpsampleMode = enum {
    nearest,
    // TODO: Linear,
    // TODO: Bilinear,
    // TODO: Bicubic,
    // TODO: Trilinear,
};

/// Upsample
pub fn upsample(
    input: Tensor,
    opts: struct { mode: UpsampleMode, scale_factor: []const f64 },
) Tensor {
    // TODO(james): make `nearest` compatible with resizeBilinear and resizeBicubic, and wrap them here.
    // resize* have API which are more explicit, this assume you want to scale the N-2 last axes.
    meta.assert(3 <= input.rank() and input.rank() <= 5, "upsample is only implemented for (3,4,5)-D tensors, received {}", .{input});
    meta.assert(opts.scale_factor.len == 1 or opts.scale_factor.len == input.rank() - 2, "scale factors", .{});
    return switch (opts.mode) {
        .nearest => {
            var scale_factors: [3]f64 = undefined;
            switch (opts.scale_factor.len) {
                1 => {
                    for (0..input.rank() - 2) |i| scale_factors[i] = opts.scale_factor[0];
                },
                else => @memcpy(scale_factors[0..opts.scale_factor.len], opts.scale_factor),
            }
            return nearest(input, scale_factors[0 .. input.rank() - 2]);
        },
    };
}

pub fn nearest(input: Tensor, scale_factor: []const f64) Tensor {
    var out_shape = input.shape();
    for (scale_factor, 0..) |sf, i| {
        out_shape._dims.set(i + 2, @intFromFloat(@floor(@as(f64, @floatFromInt(out_shape.dim(i + 2))) * sf)));
    }
    // TODO(james): remove this implicit two batching dims
    var sd: [3]usize = undefined;
    var len_sd: usize = 0;
    for (2..input.rank()) |i| {
        if (input.dim(i) != out_shape.dim(i)) {
            sd[len_sd] = i;
            len_sd += 1;
        }
    }
    const spatial_dims = sd[0..len_sd];
    var res = input;
    for (spatial_dims) |d| {
        const n = out_shape.dim(d);
        const ratio = meta.divFloat(f32, input.dim(d), n);
        const offsets = Tensor.arange(.{ .end = n }, .f32).addConstant(0.5).scale(ratio).floor().convert(.i32);
        res = res.gather1d(d, offsets, .{ .indices_are_sorted = true });
    }
    return res;
}

test nearest {
    const platform = zml.testing.env();

    // 3D Tensor (basic)
    {
        const input_3d_basic = try zml.Buffer.fromArray(platform, [1][1][2]i32{.{.{ 1, 2 }}});
        const result = try zml.testing.compileAndCall(platform, upsample, .{ input_3d_basic, .{ .scale_factor = &.{3}, .mode = .nearest } });
        try std.testing.expectEqualSlices(i64, &.{ 1, 1, 6 }, result.dims());
        const expected: [1][1][6]i32 = .{.{.{ 1, 1, 1, 2, 2, 2 }}};
        try zml.testing.expectClose(zml.HostBuffer.fromArray(&expected), result, 0);
    }
    // 3D Tensor (advanced)
    {
        const input_3d_advanced = try zml.Buffer.fromArray(platform, [2][3][4]i32{
            .{ .{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 }, .{ 9, 10, 11, 12 } },
            .{ .{ 13, 14, 15, 16 }, .{ 17, 18, 19, 20 }, .{ 21, 22, 23, 24 } },
        });
        const result = try zml.testing.compileAndCall(platform, upsample, .{ input_3d_advanced, .{ .scale_factor = &.{2}, .mode = .nearest } });
        try std.testing.expectEqualSlices(i64, &.{ 2, 3, 8 }, result.dims());
        const expected: [2][3][8]i32 = .{
            .{
                .{ 1, 1, 2, 2, 3, 3, 4, 4 },
                .{ 5, 5, 6, 6, 7, 7, 8, 8 },
                .{ 9, 9, 10, 10, 11, 11, 12, 12 },
            },
            .{
                .{ 13, 13, 14, 14, 15, 15, 16, 16 },
                .{ 17, 17, 18, 18, 19, 19, 20, 20 },
                .{ 21, 21, 22, 22, 23, 23, 24, 24 },
            },
        };
        try zml.testing.expectClose(zml.HostBuffer.fromArray(&expected), result, 0);
    }
    // 4D Tensor (basic)
    {
        const input_4d_basic = try zml.Buffer.fromSlice(platform, .{ 1, 1, 2, 2 }, &[_]i32{ 1, 2, 3, 4 });
        const result = try zml.testing.compileAndCall(platform, upsample, .{ input_4d_basic, .{ .scale_factor = &.{ 3, 3 }, .mode = .nearest } });
        try std.testing.expectEqualSlices(i64, &.{ 1, 1, 6, 6 }, result.dims());
        const expected: [1][1][6][6]i32 = .{.{.{
            .{ 1, 1, 1, 2, 2, 2 },
            .{ 1, 1, 1, 2, 2, 2 },
            .{ 1, 1, 1, 2, 2, 2 },
            .{ 3, 3, 3, 4, 4, 4 },
            .{ 3, 3, 3, 4, 4, 4 },
            .{ 3, 3, 3, 4, 4, 4 },
        }}};
        try std.testing.expectEqual(expected, result.getValue([1][1][6][6]i32));
    }
    // 4D Tensor (advanced)
    {
        const input_4d_advanced = try zml.Buffer.fromArray(platform, [2][2][2][2]i32{ .{
            .{ .{ 1, 2 }, .{ 3, 4 } },
            .{ .{ 5, 6 }, .{ 7, 8 } },
        }, .{
            .{ .{ 9, 10 }, .{ 11, 12 } },
            .{ .{ 13, 14 }, .{ 15, 16 } },
        } });
        const result = try zml.testing.compileAndCall(platform, upsample, .{ input_4d_advanced, .{ .scale_factor = &.{ 2, 2 }, .mode = .nearest } });
        try std.testing.expectEqualSlices(i64, &.{ 2, 2, 4, 4 }, result.dims());
        const expected: [2][2][4][4]i32 = .{
            .{
                .{
                    .{ 1, 1, 2, 2 },
                    .{ 1, 1, 2, 2 },
                    .{ 3, 3, 4, 4 },
                    .{ 3, 3, 4, 4 },
                },
                .{
                    .{ 5, 5, 6, 6 },
                    .{ 5, 5, 6, 6 },
                    .{ 7, 7, 8, 8 },
                    .{ 7, 7, 8, 8 },
                },
            },
            .{
                .{
                    .{ 9, 9, 10, 10 },
                    .{ 9, 9, 10, 10 },
                    .{ 11, 11, 12, 12 },
                    .{ 11, 11, 12, 12 },
                },
                .{
                    .{ 13, 13, 14, 14 },
                    .{ 13, 13, 14, 14 },
                    .{ 15, 15, 16, 16 },
                    .{ 15, 15, 16, 16 },
                },
            },
        };
        try zml.testing.expectClose(zml.HostBuffer.fromArray(&expected), result, 0);
    }
    // 5D Tensor (basic)
    {
        const input_5d = try zml.Buffer.fromSlice(platform, .{ 1, 1, 1, 2, 2 }, &[_]i32{ 1, 2, 3, 4 });
        const result = try zml.testing.compileAndCall(platform, upsample, .{ input_5d, .{ .scale_factor = &.{2}, .mode = .nearest } });
        try std.testing.expectEqualSlices(i64, &.{ 1, 1, 2, 4, 4 }, result.dims());
        const expected: [1][1][2][4][4]i32 = .{
            .{
                .{
                    .{
                        .{ 1, 1, 2, 2 },
                        .{ 1, 1, 2, 2 },
                        .{ 3, 3, 4, 4 },
                        .{ 3, 3, 4, 4 },
                    },
                    .{
                        .{ 1, 1, 2, 2 },
                        .{ 1, 1, 2, 2 },
                        .{ 3, 3, 4, 4 },
                        .{ 3, 3, 4, 4 },
                    },
                },
            },
        };
        try zml.testing.expectClose(zml.HostBuffer.fromArray(&expected), result, 0);
    }
}

pub const ResizeOpts = struct { original_len: ?Tensor = null };

pub fn resizeBilinear(image: Tensor, axes: []const i8, dims: []const u63, opt: ResizeOpts) Tensor {
    var out = image;
    for (axes, dims) |a, d| {
        const child_opt: ResizeOpts = .{
            .original_len = if (opt.original_len) |o| o.choose1d(0, a) else null,
        };
        out = resizeLinear1d(out, a, d, child_opt);
    }
    return out;
}

pub fn resizeLinear1d(image: Tensor, axis: i8, new_len: u63, opt: ResizeOpts) Tensor {
    const og_len = opt.original_len orelse Tensor.scalar(image.dim(axis), .f32);
    const ratio = og_len.convert(.f32).scale(meta.divFloat(f32, 1, new_len));
    const scaled = Tensor.arange(.{ .end = new_len }, .f32).mul(ratio);
    const left = scaled.floor();
    const right = left.addConstant(1);

    const values = image.gatherSlices1d(axis, 2, left.convert(.i32), .{ .indices_are_sorted = true });
    const left_val, const right_val = helpers.mapTensors(
        Tensor.squeeze,
        values.convert(.f32).chunkExact(2, axis + 1),
        .{@as(i64, @intCast(axis + 1))},
    );
    const left_weight = right.sub(scaled).broadcast(left_val.shape(), &.{axis});
    const right_weight = scaled.sub(left).broadcast(left_val.shape(), &.{axis});

    return left_val.mul(left_weight).add(right_val.mul(right_weight)).convert(image.dtype());
}

/// Bicubic interpolation of the given image.
/// Warning as of May 2024 the cpu backend don't optimize this very well
/// and is not able to merge the weighting with the gather,
/// leading to 20x slow down compared to STB implementation.
pub fn resizeBicubic(image: Tensor, axes: []const i8, dims: []const u63, opt: ResizeOpts) Tensor {
    var out = image;
    for (axes, dims) |a, d| {
        const child_opt: ResizeOpts = .{
            .original_len = if (opt.original_len) |o| o.choose1d(0, a) else null,
        };
        out = resizeCubic1d(out, a, d, child_opt);
    }
    return out;
}

pub fn resizeCubic1d(image: Tensor, axis: i8, new_len: u63, opt: ResizeOpts) Tensor {
    // Extract neighboring pixels from the image.
    const og_len = opt.original_len orelse Tensor.scalar(image.dim(axis), .f32);
    const ratio = og_len.convert(.f32).scale(meta.divFloat(f32, 1, new_len));
    const scaled = Tensor.arange(.{ .end = new_len }, .f32).mul(ratio);
    const t = scaled.sub(scaled.floor());
    const pos = Tensor.stack(&.{
        Tensor.scalar(1, .f32).broadcast(t.shape(), &.{}),
        t,
        t.mul(t),
        t.pow(Tensor.scalar(3, .f32)),
    }, -1, .features);
    std.debug.assert(pos.dim(0) == new_len);
    std.debug.assert(pos.dim(1) == 4);

    const context = scaled.floor().addConstant(-1).convert(.i32).maximum(Tensor.scalar(0, .i32));
    const values = image.gatherSlices1d(axis, 4, context, .{ .indices_are_sorted = true });

    const weights_: [4][4]f32 = .{
        .{ 0, 1, 0, 0 },
        .{ -0.5, 0, 0.5, 0 },
        .{ 1, -2.5, 2, -0.5 },
        .{ -0.5, 1.5, -1.5, 0.5 },
    };
    const weights = zml.Tensor.constantTensor(zml.HostBuffer.fromArray(&weights_));

    // actually do the interpolation.
    // Note: ideally this matmul should be inlined with the gather, but that's currently not the case.
    var res = values.convert(.f32).dotGeneral(weights, &.{.{ axis + 1, 1 }}, &.{});
    res = pos.dotGeneral(res, &.{.{ 1, image.rank() }}, &.{.{ 0, axis }});

    // the current axis is outputted in first position because it's a batching dim, put it back in place.
    // if (axis != 0)
    //     res = res.transpose(Shape.range(image.rank()).swap(0, axis).dims());

    // verify the shape
    const res_shape = image.shape().set(axis, new_len);
    // log.debug("resizeCubic1d: ({}, {}, {}, {}) -> {}", .{ image, axis, new_len, opt, res });
    std.debug.assert(std.mem.eql(i64, res_shape.dims(), res.dims()));
    return res.convert(image.dtype());
}

/// Return causal attention masks for the given shape.
/// The last dimensions are
pub fn causalAttnMask(
    attn_shape_: anytype,
    dtype: DataType,
    attn_window_len: ?u32,
) Tensor {
    const attn_shape = Shape.init(attn_shape_, dtype);
    meta.assert(attn_shape.rank() == 2, "causalAttnMask({}) shape need to be exactly 2 axes", .{attn_shape});
    const qlen = attn_shape.dim(-2);
    const q_idx = Tensor.iota(attn_shape, .i32, -2);
    const klen = attn_shape.dim(-1);
    const k_idx = Tensor.iota(attn_shape, .i32, -1);

    // all elements > main diagonal must be 0
    // (q_idx - window_len < k_idx <= q_idx)
    var mask = k_idx.cmp(.LE, q_idx);
    if (attn_window_len) |window_len| {
        if (qlen >= window_len or klen >= window_len) {
            const window_mask = q_idx.cmp(.LT, k_idx.addConstant(window_len));
            mask = mask.logical(.AND, window_mask);
        }
    }
    mask = mask.convert(dtype);
    if (dtype.isFloat()) {
        // use log to convert "true" (ie 1) to 0, and "false" (ie 0) to -inf
        meta.guard(dtype.isFloat(), @src()); // -inf only exists for floats
        mask = mask.log();
    }
    return mask;
}

pub const SdpaOpts = struct {
    attn_mask: ?Tensor = null,
    scale: ?Tensor = null,
    bias: ?Tensor = null,
    allow_cudnn: bool = true,
    // TODO: put a callback instead of all this field,
    // so that
};

/// Scaled dot product attention.
///
/// **Shapes**:
///   - q, result: .{ .h, .q, .hd }
///   - k, v:      .{ .h, .k, .hd }
///
/// Where:
///   - .h is the number of head
///   - .q is the number of queries
///   - .k is the number of keys
///   - .hd is the head dimension
///
/// .h is allowed to differ from queries and keys as long as the key heads
/// can be repeated to match query heads.
pub fn sdpa(q_: Tensor, k_: Tensor, v_: Tensor, opts: SdpaOpts) Tensor {
    var q, var k, var v = .{ q_, k_, v_ };

    const err_template = "sdpa(q: {}, k: {}, v: {}, attn: {?}) is invalid ! ";
    const err_args = .{ q, k, v, opts.attn_mask };
    meta.assert(q.shape().hasTags(.{ .h, .q, .hd }), err_template ++ "q is missing tags {{.h, .q, .hd}}", err_args);
    meta.assert(k.shape().hasTags(.{ .h, .k, .hd }), err_template ++ "k is missing tags {{.h, .k, .hd}}", err_args);
    meta.assert(v.shape().hasTags(.{ .h, .k, .hd }), err_template ++ "v is missing tags {{.h, .k, .hd}}", err_args);

    if (opts.allow_cudnn and cuda.canUseCudnnSdpa(q.dim(.hd), q.dtype())) {
        return cuda.sdpa(q, k, v, opts);
    }

    if (q.dim(.h) != k.dim(.h)) {
        meta.assert(@mod(q.dim(.h), k.dim(.h)) == 0, err_template ++ "Different number of heads for keys and queries, but can't repeat keys.", err_args);
        // Note: we don't try to repeat queries.
        // Repeating keys is the interesting optimisation cause it reduces KV cache memory usage.
        const num_rep: u63 = @intCast(@divExact(q.dim(.h), k.dim(.h)));
        k, v = .{ k.repeat1d(.h, num_rep), v.repeat1d(.h, num_rep) };
    }
    const attn_mask = if (opts.attn_mask) |m| m else null;

    const dims = helpers.collectDims(.{ .h, .q, .k, .hd }, &.{ q, k, v, attn_mask }, .strict) catch {
        meta.panic(err_template ++ "Inputs have incompatible shapes.", err_args);
    };
    const sqrtHeadDim: f16 = 1.0 / std.math.sqrt(@as(f16, @floatFromInt(dims.hd)));
    const scale_logit = if (opts.scale) |s| s else Tensor.scalar(sqrtHeadDim, .f16);
    k = k.mul(scale_logit.convert(k.dtype()));

    var attn_weights = q.dot(k, .{.hd});
    // log.debug("attn_weights : {}", .{attn_weights});
    // log.debug("attn_mask : {?}", .{attn_mask});
    if (attn_mask) |mask| attn_weights = attn_weights.add(mask.broadcastLeft(attn_weights.shape()));

    attn_weights = attn_weights.convert(.f32);
    if (opts.bias) |bias| {
        attn_weights = attn_weights.add(bias);
    }
    attn_weights = attn_weights.softmax(.k).convert(q.dtype());

    var attn = attn_weights.dot(v, .{.k});
    return attn.transpose(q.shape());
}

pub const MemEfficientOps = struct {
    scale: ?f32 = null,
    query_chunk_size: u32,
    key_chunk_size: u32,
    opts: SdpaOpts = .{},
};

pub fn sdpaMemEfficient(q_: Tensor, k_: Tensor, v_: Tensor, opts: MemEfficientOps) Tensor {
    const q = q_.withTags(.{ .b, .hq, .sq, .hd });
    const k = k_.withTags(.{ .b, .hk, .sk, .hd });
    const v = v_.withTags(.{ .b, .hk, .sk, .hd });
    var sdpa_opts = opts.opts;
    if (sdpa_opts.attn_mask) |*attn_mask| attn_mask.* = attn_mask.withTags(.{ .sq, .sk });

    const sdpa_mem_efficient: SdpaMemEfficient = .{ .q = q, .k = k, .v = v, .opt = .{
        .query_chunk_size = @intCast(@min(q.dim(.sq), opts.query_chunk_size)),
        .key_chunk_size = @intCast(@min(k.dim(.sk), opts.key_chunk_size)),
        .scale = opts.scale,
        .opts = sdpa_opts,
    } };

    // TODO(Corentin): Maybe `withTags` could take a Shape to copy from.
    var result = sdpa_mem_efficient.forward();
    result._shape = q_.shape();
    return result;
}

const SdpaMemEfficient = struct {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    opt: MemEfficientOps,

    fn forward(self: SdpaMemEfficient) Tensor {
        const n_q_chunks = @divExact(self.q.dim(.sq), self.opt.query_chunk_size);
        const res = ops.for_(SdpaMemEfficient.nextQueriesChunk, self, .{ .nq = n_q_chunks });
        // TODO: should "for_" operate on an axis ?
        // res: (nq, b, nh, qlen / nq, dim) -> (b, nh, qlen, dim)
        return res.transpose(.{ 1, 2, 0, 3, 4 }).flatten(2);
        // return res.transpose(.{ .b, .hq, .nq, .sq, .hd }).merge(.{ .nq, .sq }, .sq);
    }

    fn nextQueriesChunk(self: SdpaMemEfficient, idx: Tensor) Tensor {
        const offset = idx.scale(self.opt.query_chunk_size);
        const q_chunk = self.q.dynamicSlice(.{ .sq = .{ .start = offset, .len = self.opt.query_chunk_size } });
        const attn_chunk = if (self.opt.opts.attn_mask) |attn_mask| attn_mask.dynamicSlice1d(0, self.opt.query_chunk_size, offset) else null;

        var chunk: SdpaMemEfficient = self;
        chunk.q = q_chunk;
        chunk.opt.opts.attn_mask = attn_chunk;
        return chunk.scanKeyVal();
    }

    fn scanKeyVal(self: SdpaMemEfficient) Tensor {
        const n_chunks = @divExact(self.k.dim(.sk), self.opt.key_chunk_size);
        const res = ops.for_(SdpaMemEfficient.nextKeyValChunk, self, .{ .k_chunk = n_chunks });
        const global_max = res.max_value.max(.k_chunk).broad(res.max_value.shape());
        const max_diffs = res.max_value.sub(global_max).exp();
        const attn = res.attn.mul(max_diffs.broad(res.attn.shape())).sum(.k_chunk).squeeze(.k_chunk);
        const exp_sum = res.exp_sum.mul(max_diffs.convert(.f32)).sum(.k_chunk).squeeze(.k_chunk).convert(attn.dtype());
        return attn.div(exp_sum.broad(self.q.shape()));
    }

    fn nextKeyValChunk(self: SdpaMemEfficient, idx: Tensor) PartialAttn {
        const offset = idx.scale(self.opt.key_chunk_size);
        const k_chunk = self.k.dynamicSlice(.{ .sk = .{ .start = offset, .len = self.opt.key_chunk_size } });
        const v_chunk = self.v.dynamicSlice(.{ .sk = .{ .start = offset, .len = self.opt.key_chunk_size } });
        const attn_chunk = if (self.opt.opts.attn_mask) |mask| mask.dynamicSlice1d(1, self.opt.key_chunk_size, offset) else null;

        return sdpaChunk(self.q, k_chunk, v_chunk, .{ .attn_mask = attn_chunk });
    }
};

pub const PartialAttn = struct {
    attn: Tensor,
    exp_sum: Tensor,
    max_value: Tensor,
};

/// Compute softmax over a chunk.
/// Returns intermediary results to allow aggregating later.
pub fn partialSoftmax(self: Tensor, axis: anytype) PartialAttn {
    const a = self.axis(axis);
    const max_val = self.max(a);
    const out = self.sub(max_val.broad(self.shape())).exp();
    return .{
        .attn = out,
        .exp_sum = out.convert(.f32).sum(a).squeeze(a),
        .max_value = max_val.squeeze(a),
    };
}

/// Compute sdpa on a chunk, and computes a partial softmax.
/// q: (B, H, Sq, H_dim) ⊙ k: (B, H, Sk, H_dim) -> qk: (B, H, Sq, Sk)
fn sdpaChunk(q: Tensor, k: Tensor, v: Tensor, opts: SdpaOpts) PartialAttn {
    // const bs, const num_head, const sk, const h_dim = q.dims[0..4];
    // TODO: rewrite using modern ZML

    // If we have more query heads (hq) than key heads (hk), repeat keys.
    const k_rep, const v_rep = if (q.dim(.hq) != k.dim(.hk)) blk: {
        const num_rep: u63 = @intCast(@divExact(q.dim(.hq), k.dim(.hk)));
        break :blk .{ k.repeat1d(0, num_rep).rename(.{ .hk = .hq }), v.repeat1d(0, num_rep).rename(.{ .hk = .hq }) };
    } else .{ k.rename(.{ .hk = .hq }), v.rename(.{ .hk = .hq }) };

    var qk = q.dot(k_rep, .{.hd});

    const sqrtHeadDim: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(q.dim(.hd))));
    qk = qk.scale(sqrtHeadDim);

    std.debug.assert(qk.rank() == q.rank());
    if (opts.attn_mask) |mask| {
        qk = qk.add(mask.broad(qk.shape()));
    }

    const partial = partialSoftmax(qk, -1);
    const attn = partial.attn.dot(v_rep, .{.sk});

    return .{
        .attn = attn,
        .exp_sum = partial.exp_sum,
        .max_value = partial.max_value,
    };
}
test "sdpaMemEfficient without mask" {
    if (true) return error.SkipZigTest;

    const platform = zml.testing.env();
    const allocator = std.testing.allocator;

    // Note we use small input vectors to have the tests run reasonably fast,
    // but don't expect speed ups with this small sizes.
    const rng = try zml.compileFn(allocator, Tensor.Rng.normal, .{ Shape.init(.{ 1, 10, 512, 64 }, .f16), .{ .mean = 0, .stddev = 1 } }, platform);
    defer rng.deinit();

    // Note: it's fine to pass undefined here, cause the arguments have already been baked into the executable.
    const q = rng.call(undefined);
    const k = rng.call(undefined);
    const v = rng.call(undefined);

    const ref_res = try zml.testing.compileAndCallWithTensors(platform, sdpa, .{
        q.shape().withTags(.{ .b, .h, .q, .hd }),
        k.shape().withTags(.{ .b, .h, .k, .hd }),
        v.shape().withTags(.{ .b, .h, .k, .hd }),
        .{ .attn_mask = null, .scale = null, .bias = null },
    }, .{ q, k, v, undefined });
    try std.testing.expectEqualSlices(i64, q.shape().dims(), ref_res.shape().dims());

    const opts: zml.ShapeOf(MemEfficientOps) = .{ .query_chunk_size = 256, .key_chunk_size = 128, .opts = .{ .attn_mask = null, .scale = null, .bias = null } };
    const res = try zml.testing.compileAndCallWithTensors(
        platform,
        sdpaMemEfficient,
        .{ q.shape(), k.shape(), v.shape(), opts },
        .{ q, k, v, undefined },
    );

    try zml.testing.expectClose(ref_res, res, 2e-3);
}

test "sdpaMemEfficient with mask" {
    if (true) return error.SkipZigTest;

    const platform = zml.testing.env();
    const allocator = std.testing.allocator;

    // Note we use small input vectors to have the tests run reasonably fast,
    // but don't expect speed ups with this small sizes.
    const rng = try zml.compileFn(allocator, Tensor.Rng.normal, .{ Shape.init(.{ 1, 10, 512, 64 }, .f16), .{ .mean = 0, .stddev = 1 } }, platform);
    defer rng.deinit();

    const rng_mask = try zml.compileFn(allocator, Tensor.Rng.normal, .{ Shape.init(.{ 512, 512 }, .f16), .{ .mean = 0, .stddev = 1 } }, platform);
    defer rng_mask.deinit();

    // Note: it's fine to pass undefined here, cause the arguments have already been backed into the executable.
    const q = rng.call(undefined);
    const k = rng.call(undefined);
    const v = rng.call(undefined);
    const mask = rng_mask.call(undefined);

    const ref_res = try zml.testing.compileAndCall(platform, sdpa, .{ q.withTags(.{ .b, .h, .q, .hd }), k.withTags(.{ .b, .h, .k, .hd }), v.withTags(.{ .b, .h, .k, .hd }), .{ .attn_mask = mask.withTags(.{ .q, .k }), .scale = null, .bias = null } });
    try std.testing.expectEqualSlices(i64, q.shape().dims(), ref_res.shape().dims());

    const res = try zml.testing.compileAndCall(platform, sdpaMemEfficient, .{ q, k, v, .{
        .query_chunk_size = 256,
        .key_chunk_size = 128,
        .scale = null,
        .opts = .{ .attn_mask = mask, .scale = null, .bias = null, .allow_cudnn = false },
    } });

    try zml.testing.expectClose(ref_res, res, 2e-3);
}

/// Options controlling generation. The default values correspond to greedy decoding.
pub const SamplingStrategy = struct {
    topk: u32 = 1,
    temperature: f32 = 1.0,
};

/// Given the output of the last layer of a LM with a `.voc` axis,
/// Compute indices for the next tokens, following the given sampling strategy.
/// Returns an integer tensor with a shape similar to the input, but without the .voc axis.
pub fn sampleTokens(activations: Tensor, opts: SamplingStrategy, rng: Tensor.Rng) struct { Tensor, Tensor.Rng } {
    if (opts.topk <= 1) {
        const next_tokens = activations.argMax(.voc, .i32).indices.squeeze(.voc);
        return .{ next_tokens, rng };
    }

    const topk = activations.topK(opts.topk, .voc, .{});
    // After the topk, we don't have .voc values, anymore, only topk.
    var x = topk.values.rename(.{ .voc = .topk });
    if (opts.temperature != 1.0) {
        x = x.scale(1 / opts.temperature);
    }

    // Gumbel reparametrization trick:
    // Adding gumbel noise and taking the argmax is equivalent
    // to sampling from the categorical distribution produced by the softmax.
    // https://en.wikipedia.org/wiki/Gumbel_distribution#Gumbel_reparametrization_tricks
    const next_rng, const gumbel_noise = rng.gumbel(x.shape());
    x = x.add(gumbel_noise);
    const topk_idx = x.argMax(.topk, .i32).indices;

    // topk_idx is indices into topk.values ! so in the range [0, topk]
    // Convert for the original indices from the full [0, voc] range.
    const next_tokens = topk.indices.gather1d(.voc, topk_idx.squeeze(.topk), .{}).squeeze(.voc);
    // log.debug("sampleTokens({}) -> {} -> {} -> {}", .{ activations, topk.indices, topk_idx, next_tokens });
    return .{ next_tokens, next_rng };
}

test {
    _ = cuda;
}
