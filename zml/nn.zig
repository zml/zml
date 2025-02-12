//! Common layer definition and functions for Neural Networks (NN)
const std = @import("std");
const stdx = @import("stdx");

const cuda = @import("nn/cuda.zig");
const helpers = @import("helpers.zig");
const meta = @import("meta.zig");
const ops = @import("ops.zig");
const zml = @import("zml.zig");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const assert = std.debug.assert;
const log = std.log.scoped(.@"zml/tensor");
const testing = std.testing;

test {
    _ = cuda;
    std.testing.refAllDecls(@This());
}

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
        stdx.debug.assert(idx.dtype().isInteger(), "TokenEmbedding expects an integer input, received: {}", .{idx});
        stdx.debug.assert(self.weight.rank() == 2, "TokenEmbedding expects it's weight Tensor to be a 2D matrix, got {}", .{self.weight});
        return self.weight.gatherValues(0, idx, .{});
    }
};

pub const Activation = union(enum) {
    sigmoid,
    tanh,
    relu,
    leakyReLU: f32,
    elu: f32,
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
            .elu => |alpha| elu(x, alpha),
            .quick_gelu => x.quickGelu(),
            .leakyReLU => |slope| x.leakyReLU(slope),
        };
    }
};

pub fn elu(x: Tensor, alpha: f32) Tensor {
    return x.cmp(.GE, Tensor.scalar(0, x.dtype())).select(
        x,
        x.exp().addConstant(-1).scale(alpha),
    );
}

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
        const ax = x.axis(-1);
        var out = normed.mul(self.weight.broadcast(x.shape(), &.{ax}));
        if (self.bias) |bias| out = out.add(bias.broadcast(x.shape(), &.{ax}));

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

/// Rotary position embedding modify queries and keys tensor before compute Q * K in self attention.
/// This biases a token to look at token near him.
/// The nice thing with rope is that you can cache the modified queries and keys directly.
/// See: https://paperswithcode.com/method/rope
///
/// Expected shapes of tensor:
/// - x: .{ .s, .hd } where .s is the sequence length and .hd the head dimension
/// - pos_idx: optional tensor which indicates which positions are needed.
///   When not set `rope` return all positions from 0 to x.dim(.s) which is the max seq len.
pub fn rope(x: Tensor, pos_idx: ?Tensor, opts: RopeOpts) Tensor {
    stdx.debug.assert(@mod(x.dim(.hd), 2) == 0, "rope expects a even head dim (.hd), got {}", .{x});

    const idx = if (pos_idx) |idx| blk: {
        stdx.debug.assert(x.shape().hasTags(.{.hd}), "rope expects x argument to have .hd axes got: rope(x={}, idx={})", .{ x, idx });
        break :blk idx;
    } else blk: {
        stdx.debug.assert(x.shape().hasTags(.{ .s, .hd }), "rope expects x argument to have both .s and .hd axes got: rope(x={})", .{x});
        break :blk Tensor.arange(.{ .end = x.dim(.s) }, .f32).withTags(.{.s});
    };
    const x_real, const x_imag = splitRealImg(x, opts.impl);

    // compute sin and cos in f32 before downcasting to x type.
    const inv_freq = invFreq(x.dim(.hd), opts.freq_base, .f32).withTags(.{.hd});
    const inv_freq_pos = Tensor.outer(idx.convert(.f32), inv_freq);
    const cos = inv_freq_pos.cos().convert(x.dtype()).broad(x_real.shape());
    const sin = inv_freq_pos.sin().convert(x.dtype()).broad(x_real.shape());

    // apply rotation
    const y_real = x_real.mul(cos).sub(x_imag.mul(sin));
    const y_imag = x_real.mul(sin).add(x_imag.mul(cos));

    // flatten last dimensions
    return mergeRealImg(y_real, y_imag, opts.impl);
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

test rope {
    const platform = zml.testing.env();

    const Local = struct {
        fn _fwd(x: Tensor, opts: RopeOpts) Tensor {
            var input = x;
            {
                // Convert input to the requested format
                const real, const imag = splitRealImg(input, .sequential);
                input = mergeRealImg(real, imag, opts.impl);
            }
            var res = rope(input, null, opts).squeeze(0);

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
    const x = try zml.Buffer.fromSlice(platform, .{ .b = 1, .s = 5, .hd = 4 }, &[_]f32{ 1.0, 0.1, -1.0, -0.5 } ** 5);
    const res1 = try zml.testing.compileAndCall(platform, Local._fwd, .{ x, RopeOpts{ .impl = .interleaved } });
    const res2 = try zml.testing.compileAndCall(platform, Local._fwd, .{ x, RopeOpts{ .impl = .sequential } });
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
    stdx.debug.assert(3 <= input.rank() and input.rank() <= 5, "upsample is only implemented for (3,4,5)-D tensors, received {}", .{input});
    stdx.debug.assert(opts.scale_factor.len == 1 or opts.scale_factor.len == input.rank() - 2, "scale factors", .{});
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
        const ratio = stdx.math.divFloat(f32, input.dim(d), n);
        const offsets = Tensor.arange(.{ .end = n }, .f32).addConstant(0.5).scale(ratio).floor().convert(.i32);
        res = res.gatherValues(d, offsets, .{ .indices_are_sorted = true });
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

pub const ResizeOpts = struct {
    /// scalar tensor containing the original dimension of the image.
    /// It can be different from the image shape,
    /// if the image has been padded.
    /// This allows to compile one module that handle different input image sizes.
    original_len: ?Tensor = null,

    /// Internal precision to do the interpolation.
    /// Result will always use the same dtype than the original.
    /// If not set, will use the image dtype, unless it's an integer type, in which case f32 will be used.
    precision: ?zml.DataType = null,
};

pub fn resizeBilinear(image: Tensor, resized_axes: anytype, opt: ResizeOpts) Tensor {
    const new_size, const tags_ = Shape.parseStruct(u63, resized_axes);
    var out = image;
    for (new_size.constSlice(), tags_.constSlice()) |d, t| {
        const ax = image.shape().axis(t);
        const child_opt: ResizeOpts = .{
            .original_len = if (opt.original_len) |o| o.choose1d(0, ax) else null,
        };
        out = resizeLinear1d(out, ax, d, child_opt);
    }
    return out;
}

test resizeBilinear {
    const platform = zml.testing.env();

    // Only test shapes
    var comp = try zml.module.CompilationContext.init(std.heap.page_allocator, "test", platform);
    defer comp.deinit();
    comp.activate();
    defer comp.deactivate();

    inline for (.{
        .{ .{ .a = 10, .b = 10 }, .{ .a = 20 }, .{ .a = 20, .b = 10 } },
        .{ .{ .a = 10, .b = 10 }, .{ .b = 5 }, .{ .a = 10, .b = 5 } },
        .{ .{ .a = 10, .b = 10 }, .{ .a = 20, .b = 5 }, .{ .a = 20, .b = 5 } },
    }) |testcase| {
        const x_shape, const resizing, const res_shape = testcase;
        const x = Tensor.constant(x_shape, .{ .f16 = 0 });
        const y = resizeBilinear(x, resizing, .{});
        try zml.testing.expectEqualShapes(Shape.init(res_shape, .f16), y.shape());
        try std.testing.expect(y.value().owner().verify());
    }
}

pub fn resizeLinear1d(image: Tensor, axis: i8, new_len: u63, opt: ResizeOpts) Tensor {
    const res_shape = image.shape().set(axis, new_len);

    const dtype = opt.precision orelse if (image.dtype().class() == .integer) .f32 else image.dtype();
    const og_len = opt.original_len orelse Tensor.scalar(image.dim(axis), dtype);
    const ratio = og_len.convert(dtype).scale(stdx.math.divFloat(f32, 1, new_len));
    const scaled = Tensor.arange(.{ .end = new_len }, dtype).mul(ratio);
    const left = scaled.floor();
    const right = left.addConstant(1);

    // TODO: check that two gather isn't too bad perf wise.
    // Normally we should use gatherSlices to collect the values 2 by 2,
    // but gatherSlices messes up with the order of axes.
    const left_val = image.gatherValues(axis, left.convert(.i32), .{ .indices_are_sorted = true }).convert(dtype);
    const right_val = image.gatherValues(axis, right.convert(.i32), .{ .indices_are_sorted = true }).convert(dtype);

    const left_weight = right.sub(scaled).broadcast(res_shape, &.{axis});
    const right_weight = scaled.sub(left).broadcast(res_shape, &.{axis});

    const res = left_val.mul(left_weight).add(right_val.mul(right_weight));
    return res.convert(image.dtype()).withTags(image.shape().tags());
}

/// Bicubic interpolation of the given image.
/// Warning as of May 2024 the cpu backend don't optimize this very well
/// and is not able to merge the weighting with the gather,
/// leading to 20x slow down compared to STB implementation.
pub fn resizeBicubic(image: Tensor, resized_axes: anytype, opt: ResizeOpts) Tensor {
    const new_size, const tags_ = Shape.parseStruct(u63, resized_axes);
    var out = image;
    for (new_size.constSlice(), tags_.constSlice()) |d, t| {
        const ax = image.shape().axis(t);
        const child_opt: ResizeOpts = .{
            .original_len = if (opt.original_len) |o| o.choose1d(0, ax) else null,
        };
        out = resizeCubic1d(out, ax, d, child_opt);
    }
    return out;
}

test resizeBicubic {
    const platform = zml.testing.env();

    // Only test shapes
    var comp = try zml.module.CompilationContext.init(std.heap.page_allocator, "test", platform);
    defer comp.deinit();
    comp.activate();
    defer comp.deactivate();

    inline for (.{
        .{ .{ .a = 10, .b = 10 }, .{ .a = 20 }, .{ .a = 20, .b = 10 } },
        .{ .{ .a = 10, .b = 10 }, .{ .b = 5 }, .{ .a = 10, .b = 5 } },
        .{ .{ .a = 10, .b = 10 }, .{ .a = 20, .b = 5 }, .{ .a = 20, .b = 5 } },
    }) |testcase| {
        const x_shape, const resizing, const res_shape = testcase;
        const x = Tensor.constant(x_shape, .{ .f16 = 0 });
        const y = resizeBicubic(x, resizing, .{});
        try zml.testing.expectEqualShapes(Shape.init(res_shape, .f16), y.shape());
        try std.testing.expect(y.value().owner().verify());
    }
}

pub fn resizeCubic1d(image: Tensor, axis: i8, new_len: u63, opt: ResizeOpts) Tensor {
    // Extract neighboring pixels from the image.
    const dtype = opt.precision orelse if (image.dtype().class() == .integer) .f32 else image.dtype();
    const og_len = opt.original_len orelse Tensor.scalar(image.dim(axis), dtype);

    const ratio = og_len.convert(dtype).scale(stdx.math.divFloat(f32, 1, new_len));
    const scaled = Tensor.arange(.{ .end = new_len }, dtype).mul(ratio);
    const t = scaled.sub(scaled.floor());
    const pos = Tensor.stack(&.{
        Tensor.constant(t.shape(), dtype.one()),
        t,
        t.mul(t),
        t.pow(Tensor.scalar(3, dtype)),
    }, .last, ._interpolated);

    std.debug.assert(pos.dim(0) == new_len);
    std.debug.assert(pos.dim(1) == 4);

    const neighbors = scaled.floor().addConstant(-1).convert(.i32).maximum(Tensor.scalar(0, .i32));

    const values = image.renameAxis(axis, ._neighbors).gatherSlices(
        Shape.init(.{ ._neighbors = 4 }, image.dtype()),
        neighbors.appendAxes(.{.coord}),
        .{ .indices_are_sorted = true },
    ).convert(dtype);

    const weights_: [4][4]f32 = .{
        .{ 0, 1, 0, 0 },
        .{ -0.5, 0, 0.5, 0 },
        .{ 1, -2.5, 2, -0.5 },
        .{ -0.5, 1.5, -1.5, 0.5 },
    };
    const weights = zml.Tensor.constantTensor(zml.HostBuffer.fromArray(&weights_)).convert(dtype).withTags(.{ ._interpolated, ._neighbors });

    // actually do the interpolation.
    // Note: ideally this matmul should be inlined with the gather, but that's currently not the case.
    // TODO: not being able to use dot here is a bit annoying.
    var res = values.dotGeneral(weights, &.{.{ values.axis(._neighbors), weights.axis(._neighbors) }}, &.{});
    res = pos.dotGeneral(res, &.{.{ pos.axis(._interpolated), res.axis(._interpolated) }}, &.{.{ 0, 0 }});

    // the current axis is outputted in first position because it's a batching dim, put it back in place.
    if (axis != 0) {
        res = res.swapAxes(0, axis);
    }

    // verify the shape
    const res_shape = image.shape().set(axis, new_len);
    // log.debug("resizeCubic1d: ({}, {}, {}, {}) -> {}", .{ image, axis, new_len, opt, res });
    std.debug.assert(std.mem.eql(i64, res_shape.dims(), res.dims()));
    return res.convert(image.dtype()).withTags(image.shape());
}

/// Return causal attention masks for the given shape.
/// The last dimensions are
pub fn causalAttnMask(
    attn_shape_: anytype,
    dtype: DataType,
    attn_window_len: ?u32,
) Tensor {
    const attn_shape = Shape.init(attn_shape_, dtype);
    stdx.debug.assert(attn_shape.rank() == 2, "causalAttnMask({}) shape need to be exactly 2 axes", .{attn_shape});
    const qlen = attn_shape.dim(-2);
    const q_idx = Tensor.iota(attn_shape, -2);
    const klen = attn_shape.dim(-1);
    const k_idx = Tensor.iota(attn_shape, -1);

    // all elements > main diagonal must be 0
    // (q_idx - window_len < k_idx <= q_idx)
    var mask = k_idx.cmp(.LE, q_idx);
    if (attn_window_len) |window_len| {
        if (qlen >= window_len or klen >= window_len) {
            const window_mask = q_idx.cmp(.LT, k_idx.addConstant(window_len));
            mask = mask.logical(.AND, window_mask);
        }
    }

    if (dtype.isFloat()) {
        const zeros = Tensor.constant(mask.shape(), dtype.zero());
        const minus_inf = Tensor.constant(mask.shape(), dtype.minValue());
        mask = Tensor.select(mask, zeros, minus_inf);
    } else {
        mask = mask.convert(dtype);
    }

    return mask;
}

pub const SdpaOpts = struct {
    attn_mask: ?Tensor = null,
    scale: ?Tensor = null,
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
    stdx.debug.assert(q.shape().hasTags(.{ .h, .q, .hd }), err_template ++ "q is missing tags {{.h, .q, .hd}}", err_args);
    stdx.debug.assert(k.shape().hasTags(.{ .h, .k, .hd }), err_template ++ "k is missing tags {{.h, .k, .hd}}", err_args);
    stdx.debug.assert(v.shape().hasTags(.{ .h, .k, .hd }), err_template ++ "v is missing tags {{.h, .k, .hd}}", err_args);

    if (opts.allow_cudnn and cuda.canUseCudnnSdpa(q.shape())) {
        return cuda.sdpa(q, k, v, opts);
    }

    if (q.dim(.h) != k.dim(.h)) {
        stdx.debug.assert(@mod(q.dim(.h), k.dim(.h)) == 0, err_template ++ "Different number of heads for keys and queries, but can't repeat keys.", err_args);
        // Note: we don't try to repeat queries.
        // Repeating keys is the interesting optimisation cause it reduces KV cache memory usage.
        const num_rep: u63 = @intCast(@divExact(q.dim(.h), k.dim(.h)));
        k, v = .{ k.repeat1d(.h, num_rep), v.repeat1d(.h, num_rep) };
    }
    const attn_mask = if (opts.attn_mask) |m| m else null;

    const dims = helpers.collectDims(.{ .h, .q, .k, .hd }, &.{ q, k, v, attn_mask }, .strict) catch {
        stdx.debug.panic(err_template ++ "Inputs have incompatible shapes.", err_args);
    };
    const sqrtHeadDim: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dims.hd)));
    const head_scaling = if (opts.scale) |s| s else Tensor.scalar(sqrtHeadDim, k.dtype());
    k = k.mul(head_scaling.convert(k.dtype()));

    var attn_weights = q.dot(k, .{.hd});
    // log.debug("attn_weights : {}", .{attn_weights});
    // log.debug("attn_mask : {?}", .{attn_mask});
    if (attn_mask) |mask| attn_weights = attn_weights.add(mask.broad(attn_weights.shape()));
    attn_weights = attn_weights.convert(.f32).softmax(.k).convert(q.dtype());

    var attn = attn_weights.dot(v, .{.k});
    return attn.transpose(q.shape());
}

pub const SdpaChunks = struct { q_chunk_size: u32, k_chunk_size: u32 };

pub fn sdpaMemEfficient(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sdpa_opts: SdpaOpts,
    chunking: SdpaChunks,
) Tensor {
    const sdpa_mem_efficient: SdpaMemEfficient = .{
        .q = q,
        .k = k,
        .v = v,
        .sdpa_opts = sdpa_opts,
        .chunking = .{
            .q_chunk_size = @intCast(@min(q.dim(.q), chunking.q_chunk_size)),
            .k_chunk_size = @intCast(@min(k.dim(.k), chunking.k_chunk_size)),
        },
    };

    return sdpa_mem_efficient.forward();
}

const SdpaMemEfficient = struct {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sdpa_opts: SdpaOpts,
    chunking: SdpaChunks,

    fn forward(self: SdpaMemEfficient) Tensor {
        stdx.debug.assert(@mod(self.q.dim(.q), self.chunking.q_chunk_size) == 0, "sdpaMemEfficient expects the chunk_size to exactly divise the seq_len, got: sdpaMemEfficient({}, {})", .{ self.q, self.chunking });
        stdx.debug.assert(@mod(self.k.dim(.k), self.chunking.k_chunk_size) == 0, "sdpaMemEfficient expects the chunk_size to exactly divise the seq_len, got: sdpaMemEfficient({}, {})", .{ self.k, self.chunking });
        const n_q_chunks: u32 = @intCast(@divExact(self.q.dim(.q), self.chunking.q_chunk_size));

        const ctx = zml.module.CompilationContext.current();
        const q_chunks = ctx.allocator().alloc(zml.Tensor, n_q_chunks) catch unreachable;
        defer ctx.allocator().free(q_chunks);
        for (0..n_q_chunks) |i| {
            const idx: u32 = @intCast(i);
            const q_slice: zml.Tensor.DynSlice = .{
                .start = Tensor.scalar(idx * self.chunking.q_chunk_size, .i32),
                .len = self.chunking.q_chunk_size,
            };
            const q_chunk = self.q.dynamicSlice(.{ .q = q_slice });
            const attn_chunk = if (self.sdpa_opts.attn_mask) |attn_mask| attn_mask.dynamicSlice(.{ .q = q_slice }) else null;

            var chunk: SdpaMemEfficient = self;
            chunk.q = q_chunk;
            chunk.sdpa_opts.attn_mask = attn_chunk;
            q_chunks[i] = chunk.scanKeyVal();
        }

        const res = zml.Tensor.concatenate(q_chunks, .q);
        return res.transpose(self.q.shape());
    }

    fn nextQueriesChunk(self: SdpaMemEfficient, idx: Tensor) Tensor {
        const q_slice: zml.Tensor.DynSlice = .{
            .start = idx.scale(self.chunking.q_chunk_size),
            .len = self.chunking.q_chunk_size,
        };
        const q_chunk = self.q.dynamicSlice(.{ .q = q_slice });
        const attn_chunk = if (self.sdpa_opts.attn_mask) |attn_mask| attn_mask.dynamicSlice(.{ .q = q_slice }) else null;

        var chunk: SdpaMemEfficient = self;
        chunk.q = q_chunk;
        chunk.sdpa_opts.attn_mask = attn_chunk;
        return chunk.scanKeyVal();
    }

    fn scanKeyVal(self: SdpaMemEfficient) Tensor {
        const n_chunks = @divExact(self.k.dim(.k), self.chunking.k_chunk_size);
        return if (n_chunks <= 4) {
            // Unrolled version
            var partial_softmax: ?PartialSoftmax = null;
            for (0..@intCast(n_chunks)) |idx| {
                const next = self.nextKeyValChunk(Tensor.scalar(idx, .i32));
                partial_softmax = if (partial_softmax) |prev| prev.merge(next) else next;
            }
            return partial_softmax.?.finalize();
        } else {
            // stablehlo.while version
            const partial_softmax, _ = zml.ops.while_(hasNextKeyValChunk, nextKeyValChunkMerge, self, .{ PartialSoftmax.zeros(self.q.shape(), .f32), Tensor.scalar(0, .i32) });
            return partial_softmax.finalize();
        };
    }

    fn nextKeyValChunkMerge(self: SdpaMemEfficient, prev: PartialSoftmax, idx: Tensor) struct { PartialSoftmax, Tensor } {
        const next = self.nextKeyValChunk(idx);
        return .{ prev.merge(next), idx.addConstant(1) };
    }

    fn nextKeyValChunk(self: SdpaMemEfficient, idx: Tensor) PartialSoftmax {
        const k_slice: zml.Tensor.DynSlice = .{
            .start = idx.scale(self.chunking.k_chunk_size),
            .len = self.chunking.k_chunk_size,
        };

        const k_chunk = self.k.dynamicSlice(.{ .k = k_slice });
        const v_chunk = self.v.dynamicSlice(.{ .k = k_slice });
        const attn_chunk = if (self.sdpa_opts.attn_mask) |mask| mask.dynamicSlice(.{ .k = k_slice }) else null;

        return sdpaChunk(self.q, k_chunk, v_chunk, .{ .attn_mask = attn_chunk });
    }

    pub fn hasNextKeyValChunk(self: SdpaMemEfficient, _: PartialSoftmax, idx: Tensor) zml.Tensor {
        const n_chunks = @divExact(self.k.dim(.k), self.chunking.k_chunk_size);
        return idx.cmp(.LT, Tensor.scalar(n_chunks, idx.dtype()));
    }
};

pub const PartialSoftmax = struct {
    values: Tensor,
    exp_sum: Tensor,
    max_value: Tensor,

    pub fn zeros(q_shape: Shape, exp_sum_precision: DataType) PartialSoftmax {
        return .{
            .values = Tensor.constant(q_shape, q_shape.dtype().zero()),
            .exp_sum = Tensor.constant(q_shape.setDim(.hd, 1), exp_sum_precision.zero()),
            .max_value = Tensor.constant(q_shape.setDim(.hd, 1), q_shape.dtype().minValue()),
        };
    }

    pub fn merge(self: PartialSoftmax, other: PartialSoftmax) PartialSoftmax {
        // Rescale self and other using the new global_max.
        const global_max = self.max_value.maximum(other.max_value);
        const new_self = self.rescale(global_max);
        const new_other = other.rescale(global_max);

        // Now that self and other are using the same scale, we can just add them:
        return .{
            .max_value = global_max,
            .values = new_self.values.add(new_other.values),
            .exp_sum = new_self.exp_sum.add(new_other.exp_sum),
        };
    }

    /// Update max_value and rescale attn and exp_sum accordingly.
    pub fn rescale(self: PartialSoftmax, max_value: Tensor) PartialSoftmax {
        const max_diff_exp = self.max_value.sub(max_value).exp();
        const sum_dtype = self.exp_sum.dtype();
        return .{
            .max_value = max_value,
            .values = self.values.mul(max_diff_exp.broad(self.values.shape())),
            .exp_sum = self.exp_sum.mul(max_diff_exp.convert(sum_dtype)),
        };
    }

    /// Divides the intermediary results by the exp_sum to get the proper attention values.
    pub fn finalize(self: PartialSoftmax) Tensor {
        return self.values.div(self.exp_sum.broad(self.values.shape()).convert(self.values.dtype()));
    }
};

/// Compute softmax over a chunk.
/// Returns intermediary results to allow aggregating later.
pub fn partialSoftmax(self: Tensor, axis: anytype) PartialSoftmax {
    const a = self.axis(axis);
    const max_val = self.max(a).maximum(Tensor.scalar(-1e16, self.dtype()));
    const out = self.sub(max_val.broad(self.shape())).exp();
    return .{
        .values = out,
        .exp_sum = out.convert(.f32).sum(a),
        .max_value = max_val,
    };
}

/// Compute sdpa on a chunk, and computes a partial softmax.
/// q: (B, H, Sq, H_dim) ⊙ k: (B, H, Sk, H_dim) -> qk: (B, H, Sq, Sk)
pub fn sdpaChunk(q_: Tensor, k_: Tensor, v_: Tensor, opts: SdpaOpts) PartialSoftmax {
    // this is a dupe of sdpa, but return the PartialSoftmax instead of true Attn.
    // Consider implementing sdpa from sdpaChunk.
    var q, var k, var v = .{ q_, k_, v_ };

    const err_template = "sdpa(q: {}, k: {}, v: {}, attn: {?}) is invalid ! ";
    const err_args = .{ q, k, v, opts.attn_mask };
    stdx.debug.assert(q.shape().hasTags(.{ .h, .q, .hd }), err_template ++ "q is missing tags {{.h, .q, .hd}}", err_args);
    stdx.debug.assert(k.shape().hasTags(.{ .h, .k, .hd }), err_template ++ "k is missing tags {{.h, .k, .hd}}", err_args);
    stdx.debug.assert(v.shape().hasTags(.{ .h, .k, .hd }), err_template ++ "v is missing tags {{.h, .k, .hd}}", err_args);

    if (q.dim(.h) != k.dim(.h)) {
        stdx.debug.assert(@mod(q.dim(.h), k.dim(.h)) == 0, err_template ++ "Different number of heads for keys and queries, but can't repeat keys.", err_args);
        // Note: we don't try to repeat queries.
        // Repeating keys is the interesting optimisation cause it reduces KV cache memory usage.
        const num_rep: u63 = @intCast(@divExact(q.dim(.h), k.dim(.h)));
        k, v = .{ k.repeat1d(.h, num_rep), v.repeat1d(.h, num_rep) };
    }
    const attn_mask = if (opts.attn_mask) |m| m else null;

    const dims = helpers.collectDims(.{ .h, .q, .k, .hd }, &.{ q, k, v, attn_mask }, .strict) catch {
        stdx.debug.panic(err_template ++ "Inputs have incompatible shapes.", err_args);
    };
    const sqrtHeadDim: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dims.hd)));
    const head_scaling = if (opts.scale) |s| s else Tensor.scalar(sqrtHeadDim, k.dtype());
    k = k.mul(head_scaling.convert(k.dtype()));

    var attn_weights = q.dot(k, .{.hd});
    // log.debug("attn_weights : {}", .{attn_weights});
    // log.debug("attn_mask : {?}", .{attn_mask});
    if (attn_mask) |mask| attn_weights = attn_weights.add(mask.broad(attn_weights.shape()));

    const partial = partialSoftmax(attn_weights, .k);
    const attn = partial.values.dot(v, .{.k}).transpose(q.shape());

    return .{
        .values = attn,
        // The renaming is because the above dot projected values.k into .hd,
        // do the same thing on the other tensors.
        // This work because dot is a linear operation, and commutes with `PartialSoftmax.finalize`
        .exp_sum = partial.exp_sum.rename(.{ .k = .hd }).transpose(attn.shape()),
        .max_value = partial.max_value.rename(.{ .k = .hd }).transpose(attn.shape()),
    };
}

test sdpaMemEfficient {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;

    // Note we use small input vectors to have the tests run reasonably fast,
    // but don't expect speed ups with this small sizes.
    const rng = try zml.compileFn(allocator, Tensor.Rng.normal, .{ Shape.init(.{ 1, 10, 512, 64 }, .f32), .{ .mean = 0, .stddev = 1 } }, platform);
    defer rng.deinit();

    const rng_mask = try zml.compileFn(allocator, Tensor.Rng.normal, .{ Shape.init(.{ 512, 512 }, .f32), .{ .mean = 0, .stddev = 1 } }, platform);
    defer rng_mask.deinit();

    // Note: it's fine to pass undefined here, cause the arguments have already been backed into the executable.
    const q = rng.call(undefined).withTags(.{ .b, .h, .q, .hd });
    const k = rng.call(undefined).withTags(.{ .b, .h, .k, .hd });
    const v = rng.call(undefined).withTags(.{ .b, .h, .k, .hd });
    const mask = rng_mask.call(undefined).withTags(.{ .q, .k });

    const ref_res = try zml.testing.compileAndCall(
        platform,
        sdpa,
        .{ q, k, v, .{ .attn_mask = mask, .scale = null } },
    );
    try std.testing.expectEqualSlices(i64, q.shape().dims(), ref_res.shape().dims());
    {
        // 4 k_chunks
        const res = try zml.testing.compileAndCall(
            platform,
            sdpaMemEfficient,
            .{
                q,
                k,
                v,
                .{ .attn_mask = mask, .scale = null },
                .{ .q_chunk_size = 256, .k_chunk_size = @divExact(512, 4) },
            },
        );

        try zml.testing.expectClose(ref_res, res, 2e-3);
    }
    {
        // 16 k_chunks
        const res = try zml.testing.compileAndCall(
            platform,
            sdpaMemEfficient,
            .{
                q,
                k,
                v,
                .{ .attn_mask = mask, .scale = null },
                .{ .q_chunk_size = 256, .k_chunk_size = @divExact(512, 16) },
            },
        );

        try zml.testing.expectClose(ref_res, res, 2e-3);
    }
}

test "sdpaMemEfficient transposed" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;

    // Note we use small input vectors to have the tests run reasonably fast,
    // but don't expect speed ups with this small sizes.
    const rng = try zml.compileFn(allocator, Tensor.Rng.normal, .{ Shape.init(.{ 1, 512, 10, 64 }, .f32), .{ .mean = 0, .stddev = 1 } }, platform);
    defer rng.deinit();

    const rng_mask = try zml.compileFn(allocator, Tensor.Rng.normal, .{ Shape.init(.{ 512, 512 }, .f32), .{ .mean = 0, .stddev = 1 } }, platform);
    defer rng_mask.deinit();

    // Note: it's fine to pass undefined here, cause the arguments have already been backed into the executable.
    const q = rng.call(undefined).withTags(.{ .b, .q, .h, .hd });
    const k = rng.call(undefined).withTags(.{ .b, .k, .h, .hd });
    const v = rng.call(undefined).withTags(.{ .b, .k, .h, .hd });
    const mask = rng_mask.call(undefined).withTags(.{ .q, .k });

    const ref_res = try zml.testing.compileAndCall(
        platform,
        sdpa,
        .{ q, k, v, .{ .attn_mask = mask, .scale = null } },
    );
    try std.testing.expectEqualSlices(i64, q.shape().dims(), ref_res.shape().dims());

    {
        const res = try zml.testing.compileAndCall(
            platform,
            sdpaMemEfficient,
            .{
                q,
                k,
                v,
                .{ .attn_mask = mask, .scale = null },
                .{ .q_chunk_size = @divExact(512, 2), .k_chunk_size = @divExact(512, 4) },
            },
        );

        try zml.testing.expectClose(ref_res, res, 1e-3);
    }

    {
        const res = try zml.testing.compileAndCall(
            platform,
            sdpaMemEfficient,
            .{
                q,
                k,
                v,
                .{ .attn_mask = mask, .scale = null },
                .{ .q_chunk_size = 512, .k_chunk_size = @divExact(512, 4) },
            },
        );

        try zml.testing.expectClose(ref_res, res, 1e-3);
    }
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
        const next_tokens = activations.argMax(.voc).indices.squeeze(.voc);
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
    const topk_idx = x.argMax(.topk).indices;

    // topk_idx is indices into topk.values ! so in the range [0, topk]
    // Convert for the original indices from the full [0, voc] range.
    const next_tokens = topk.indices.gatherValues(.voc, topk_idx.squeeze(.topk), .{});
    // log.debug("sampleTokens({}) -> {} -> {} -> {}", .{ activations, topk.indices, topk_idx, next_tokens });
    return .{ next_tokens, next_rng };
}

test sampleTokens {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;

    const inf = std.math.inf(f32);
    var rng_buff = try zml.Tensor.Rng.init(platform, 0xdeadbeef);
    defer rng_buff._state.deinit();

    const mod = try zml.compileFn(allocator, sampleTokens, .{ Shape.init(.{ .voc = 4 }, .f32), .{ .topk = 4, .temperature = 2.0 }, zml.Tensor.Rng.shape() }, platform);
    defer mod.deinit();

    inline for (.{
        .{ [_]f32{ inf, 3.0, 2.0, 1.0 }, 0 },
        .{ [_]f32{ -inf, 3.0, -inf, -inf }, 1 },
        .{ [_]f32{ 3.0, 2, inf, inf }, 2 },
    }) |logits_expected| {
        const logits, const expected: i32 = logits_expected;
        var logits_buff = try zml.Buffer.fromArray(platform, logits);
        defer logits_buff.deinit();
        var sampled, rng_buff = mod.call(.{ logits_buff, undefined, rng_buff });
        defer sampled.deinit();
        try zml.testing.expectEqual(expected, try sampled.getValue(i32));
    }
}

pub const DynamicSamplingStrategy = struct {
    max_top_k: u32,
    top_k: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    min_p: Tensor,

    pub const Opts = struct {
        top_k: u32,
        temperature: f32 = 1.0,
        top_p: f32 = 1.0,
        min_p: f32 = 0.0,
    };

    pub fn shapes(dtype: DataType, max_top_k: u32) zml.ShapeOf(DynamicSamplingStrategy) {
        const scalar_float = Shape.init(.{}, dtype);
        const scalar_i32 = Shape.init(.{}, .i32);
        return .{
            .max_top_k = max_top_k,
            .top_k = scalar_i32,
            .temperature = scalar_float,
            .top_p = scalar_float,
            .min_p = scalar_float,
        };
    }

    pub fn makeBuffers(
        platform: zml.Platform,
        dtype: zml.DataType,
        opts: Opts,
    ) !zml.Bufferized(DynamicSamplingStrategy) {
        return .{
            .max_top_k = 0,
            .top_k = try zml.Buffer.scalar(platform, opts.top_k, .i32),
            .temperature = try zml.Buffer.scalar(platform, opts.temperature, dtype),
            .top_p = try zml.Buffer.scalar(platform, opts.top_p, dtype),
            .min_p = try zml.Buffer.scalar(platform, opts.min_p, dtype),
        };
    }
};

/// Given the output of the last layer of a LM with a `.voc` axis,
/// Compute indices for the next tokens, following the given sampling strategy.
/// The dynamic sampling strategy is more expressive but top_p requires computing the softmax.
///
/// Options are:
///
/// * top_k: only sample among the k top scoring tokens,
/// * max_top_k: limit a compilation time what is the max possible runtime value for top_k, saving memory and compute by not having to fully sort the tokens.
/// * top_p: only sample among top scoring tokens whose probabilities sum up to top_p
/// * min_p: drop tokens whose probabilities are lower than a ratio of the most likely token
pub fn sampleTokensDynamic(logits: Tensor, opts: DynamicSamplingStrategy, rng: Tensor.Rng) struct { Tensor, Tensor.Rng } {
    var x, const topk_indices = fixupLogits(logits, opts);

    // the rest is similar to sampleTokens
    const next_rng, const gumbel_noise = rng.gumbel(x.shape());
    x = x.add(gumbel_noise);

    const topk_idx = x.argMax(.topk).indices;
    const next_tokens = topk_indices.gatherValues(.voc, topk_idx.squeeze(.topk), .{});
    return .{ next_tokens, next_rng };
}

fn fixupLogits(logits: Tensor, opts: DynamicSamplingStrategy) [2]Tensor {
    const min_inf = Tensor.constant(.{}, logits.dtype().minValue());

    // First reduce the vocab size to a reasonable sub set of candidate.
    const full_topk = if (opts.max_top_k > 0)
        logits.topK(opts.max_top_k, .voc, .{ .descending = true })
    else
        logits.sort(.voc, .{ .descending = true });

    // After the topk, we don't have .voc indices, anymore, only topk.
    var x = full_topk.values.rename(.{ .voc = .topk });
    // mask values above the dynamic top_k
    x = Tensor.iota(x.shape(), .topk).cmp(.GE, opts.top_k).select(min_inf, x);
    x = x.mul(opts.temperature);

    // if there are high values in x, softmax can overflow and will create nans in full probs
    // this propagate to probs_sum and probs_max.
    const probs = x.softmax(.topk);
    const probs_sum = probs.cumulativeSum(.topk);
    const probs_max = probs.slice1d(.topk, .{ .start = 0, .end = 1 });

    const top_p = opts.top_p.broad(x.shape());
    const min_p = probs_max.mul(opts.min_p).broad(x.shape());

    // * if first candidate has very high prob, then probs_sum is always greater than top_p and candidate is full false
    // * if first candidate score is even bigger, the probs become Nan because of the softmax,
    // then cmp is is full false, and candidate is full false too.
    const candidate = probs_sum.cmp(.LE, top_p).logical(.AND, probs.cmp(.GE, min_p));
    // * so we explicitly always accept first candidate.
    const first_token = Tensor.iota(x.shape(), .topk).cmp(.EQ, Tensor.scalar(0, .i32));
    x = candidate.logical(.OR, first_token).select(x, min_inf);

    return .{ x, full_topk.indices };
}

test sampleTokensDynamic {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;

    const ___ = -std.math.inf(f32);
    const logits = [_]f32{ @log(2.0), @log(1.0), @log(4.0), @log(3.0) };
    const top_k_indices = [_]i32{ 2, 3, 0, 1 };
    const logits_buff = try zml.Buffer.fromArray(platform, logits);
    const mod = try zml.compileFn(allocator, fixupLogits, .{ Shape.init(.{ .voc = logits.len }, .f32), DynamicSamplingStrategy.shapes(.f32, 0) }, platform);
    defer mod.deinit();

    const Args = struct { DynamicSamplingStrategy.Opts, [4]f32 };
    inline for ([_]Args{
        // top_k == logits.len -> just sort the input
        .{ .{ .top_k = 4 }, [_]f32{ @log(4.0), @log(3.0), @log(2.0), @log(1.0) } },
        .{ .{ .top_k = 2 }, [_]f32{ @log(4.0), @log(3.0), ___, ___ } },
        .{ .{ .top_k = 2, .temperature = 0.1 }, [_]f32{ @log(4.0) * 0.1, @log(3.0) * 0.1, ___, ___ } },
        // top_k == logits.len and small top_p  -> make sure at least one is returned
        .{ .{ .top_k = 4, .top_p = 0.1 }, [_]f32{ @log(4.0), ___, ___, ___ } },
        .{ .{ .top_k = 4, .top_p = 0.701 }, [_]f32{ @log(4.0), @log(3.0), ___, ___ } },
        .{ .{ .top_k = 4, .top_p = 0.901 }, [_]f32{ @log(4.0), @log(3.0), @log(2.0), ___ } },
        // Here top_p is computed on the top 3 items, so 0.701 isn't enougth anymore to allow @log(3.0)
        .{ .{ .top_k = 3, .top_p = 0.701 }, [_]f32{ @log(4.0), ___, ___, ___ } },
        // Here top_p allows the first 3 results, but min_p only accepts the first two.
        .{ .{ .top_k = 4, .top_p = 0.901, .min_p = 0.6 }, [_]f32{ @log(4.0), @log(3.0), ___, ___ } },
    }) |args_expected| {
        const args, const expected = args_expected;
        const new_logits, const indices = mod.call(.{ logits_buff, try DynamicSamplingStrategy.makeBuffers(platform, .f32, args) });
        try std.testing.expectEqual(top_k_indices, try indices.getValue(@TypeOf(top_k_indices)));
        try zml.testing.expectEqual(expected, try new_logits.getValue(@TypeOf(expected)));
    }

    {
        // Similar but use bf16, and uses infinity to trigger nans after the softmax.
        const bf16 = zml.floats.BFloat16;

        const mod_bf16 = try zml.compileFn(allocator, fixupLogits, .{ Shape.init(.{ .voc = logits.len }, .bf16), DynamicSamplingStrategy.shapes(.bf16, 0) }, platform);
        defer mod_bf16.deinit();
        const boost = bf16.inf();
        const nerf = bf16.minusInf();
        const logits_buff_2 = try zml.Buffer.fromArray(platform, [4]bf16{ boost, boost, bf16.fromF32(2), nerf });
        const new_logits, const indices = mod_bf16.call(.{ logits_buff_2, try DynamicSamplingStrategy.makeBuffers(platform, .bf16, .{ .top_k = 4, .top_p = 0.9, .min_p = 0.1 }) });
        try std.testing.expectEqual([_]i32{ 0, 1, 2, 3 }, try indices.getValue([4]i32));
        try zml.testing.expectEqual([_]bf16{ boost, nerf, nerf, nerf }, try new_logits.getValue([4]bf16));
    }
}
