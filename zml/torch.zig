const std = @import("std");
const log = std.log.scoped(.zml_torch);

const zml = @import("zml.zig");
const Tensor = zml.Tensor;
const meta = zml.meta;

/// Multiplies a matrix or a vector with a tensor,
/// following the semantic of pytorch `@` operator.
/// When both sides are matrices, it's the textbook matrix multiplication :
/// `matmul(.{ 8, 9 }, .{ 9, 10 }) -> .{ 8, 10 }`
/// When one of the input is a tensor, it assumes the first dimensions are batches,
/// and the last two ones are used for the regular matmul.
/// * `matmul(.{10}, .{10}) -> .{}`
/// * `matmul(.{10}, .{10}) -> .{}`
pub fn matmul(lhs: Tensor, rhs: Tensor) Tensor {
    meta.assert(lhs.rank() >= 1 and rhs.rank() >= 1, "Can't matmul({}, {}) ! The two tensors need to have at least rank 1.", .{ lhs, rhs });

    const contracting = [_][2]i8{.{ -1, if (rhs.rank() >= 2) rhs.rank() - 2 else 0 }};
    if (lhs.rank() == 1 or rhs.rank() <= 2) {
        // When lhs is a vector or rhs is small the torch semantics match the dot_general semantics and life is easy.
        return lhs.dotGeneral(rhs, &contracting, &.{});
    }

    meta.assert(lhs.rank() == 2, "Can't matmul({}, {}) ! One of the two tensors need to have a rank less than 2.", .{ lhs, rhs });

    // Pytorch treats the extra dimensions of rhs has batching dimensions,
    // and implicitly broadcast lhs along those.
    // We make this broadcasting explicit.
    var left_shape = rhs.shape();
    left_shape._dims.set(left_shape.axis(-2), lhs.dim(-2));
    left_shape._tags.set(left_shape.axis(-2), lhs.shape().tag(-2));
    left_shape._dims.set(left_shape.axis(-1), lhs.dim(-1));
    left_shape._tags.set(left_shape.axis(-1), lhs.shape().tag(-1));
    const lhs_broad = lhs.broadcastLeft(left_shape);

    const n_batching_axes = rhs.rank() - lhs.rank();
    var batching: [Tensor.MAX_RANK][2]i8 = undefined;
    for (0..n_batching_axes) |i| {
        batching[i] = .{ @intCast(i), @intCast(i) };
    }
    return lhs_broad.dotGeneral(rhs, &contracting, batching[0..n_batching_axes]);
}

test matmul {
    const platform = zml.testing.env();

    var comp = try zml.module.CompilationContext.init(std.heap.page_allocator, "test", platform);
    defer comp.deinit();

    comp.activate();
    defer comp.deactivate();

    // Generated with pytorch
    inline for (.{
        .{ .{20}, .{20}, .{} },
        .{ .{20}, .{ 20, 15 }, .{15} },
        .{ .{20}, .{ 11, 20, 15 }, .{ 11, 15 } },
        .{ .{20}, .{ 9, 11, 20, 15 }, .{ 9, 11, 15 } },
        .{ .{20}, .{ 7, 9, 11, 20, 15 }, .{ 7, 9, 11, 15 } },
        .{ .{20}, .{ 5, 7, 9, 11, 20, 15 }, .{ 5, 7, 9, 11, 15 } },
        .{ .{ 12, 20 }, .{20}, .{12} },
        .{ .{ 12, 20 }, .{ 20, 15 }, .{ 12, 15 } },
        .{ .{ 12, 20 }, .{ 11, 20, 15 }, .{ 11, 12, 15 } },
        .{ .{ 12, 20 }, .{ 9, 11, 20, 15 }, .{ 9, 11, 12, 15 } },
        .{ .{ 12, 20 }, .{ 7, 9, 11, 20, 15 }, .{ 7, 9, 11, 12, 15 } },
        .{ .{ 12, 20 }, .{ 5, 7, 9, 11, 20, 15 }, .{ 5, 7, 9, 11, 12, 15 } },
        .{ .{ 10, 12, 20 }, .{20}, .{ 10, 12 } },
        .{ .{ 10, 12, 20 }, .{ 20, 15 }, .{ 10, 12, 15 } },
        .{ .{ 8, 10, 12, 20 }, .{20}, .{ 8, 10, 12 } },
        .{ .{ 8, 10, 12, 20 }, .{ 20, 15 }, .{ 8, 10, 12, 15 } },
        .{ .{ 6, 8, 10, 12, 20 }, .{20}, .{ 6, 8, 10, 12 } },
        .{ .{ 6, 8, 10, 12, 20 }, .{ 20, 15 }, .{ 6, 8, 10, 12, 15 } },
        .{ .{ 4, 6, 8, 10, 12, 20 }, .{20}, .{ 4, 6, 8, 10, 12 } },
        .{ .{ 4, 6, 8, 10, 12, 20 }, .{ 20, 15 }, .{ 4, 6, 8, 10, 12, 15 } },
    }) |testcase| {
        const x_shape, const y_shape, const z_shape = testcase;
        const x = Tensor.constant(x_shape, .{ .f32 = 0.0 });
        const y = Tensor.constant(y_shape, .{ .f32 = 0.0 });
        const z = matmul(x, y);

        try std.testing.expectEqualSlices(i64, &z_shape, z.dims());
    }
}

/// Inserts a 1-dim axis at the given position.
/// Negative indexes are handled like pytorch, ie they are relative to the returned shaped:
/// - `.{5, 4}.unsqueeze(1)` returns `.{5, 1, 4}`
/// - `.{5, 4}.unsqueeze(-1)` returns `.{5, 4, 1}`
pub fn unsqueeze(
    self: Tensor,
    axis_: anytype,
) Tensor {
    meta.assert(self.rank() < Tensor.MAX_RANK - 1, "Can't unsqueeze {}, it's already at max rank.", .{self});
    const a = switch (@typeInfo(@TypeOf(axis_))) {
        .Int, .ComptimeInt => if (axis_ < 0)
            @as(i8, self.rank()) + 1 + axis_
        else
            self.axis(axis_),
        else => self.axis(axis_),
    };
    return self.insertAxes(a, .{._});
}

test unsqueeze {
    const UnsqueezeTest = struct {
        pub fn forward(x: Tensor) Tensor {
            var y = x;
            y = unsqueeze(y, 0);
            y = unsqueeze(y, -1);
            y = unsqueeze(y, -1);
            return y;
        }
    };
    const platform = zml.testing.env();

    const x = try zml.Buffer.fromArray(platform, @as([8]f16, undefined));
    const res = try zml.testing.compileAndCall(platform, UnsqueezeTest.forward, .{x});
    try zml.testing.expectEqualShapes(zml.Shape.init(.{ 1, 8, 1, 1 }, .f16), res.shape());
}

/// Given an input images with .{ .c, .w, .h } tags,
/// shuffle values between the channel (.c), width (.w) and height (.h) axes.
/// pixelShuffle(.{ .c, .w, .h }, u) -> .{ .c / u / u, .w * u, .h * u}
/// ref: https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html#pixelshuffle
pub fn pixelShuffle(tensor: Tensor, upscale_factor: u32) Tensor {
    const shape = tensor.shape();
    meta.assert(shape.hasTags(.{ .c, .w, .h }), "pixelShuffle({}) is invalide. Missing tags {{.c, .w, .h}}", .{tensor});

    meta.assert(@mod(shape.dim(.c), upscale_factor * upscale_factor) == 0, "pixelShuffle({}) is invalide. Number of channels {}, isn't divisible by upscale factor {}**2", .{ tensor, shape.dim(.c), upscale_factor });

    const s = tensor.splitAxis(.c, .{ .c = -1, .upscale_h = upscale_factor, .upscale_w = upscale_factor });
    const perm = s.shape().contiguousPerm(.{ .h, .upscale_h, .w, .upscale_w });
    const cont = s.transpose(perm.constSlice());
    return cont.merge(.{ .h = .{ .h, .upscale_h }, .w = .{ .w, .upscale_w } }).transpose(tensor.shape());
}

test pixelShuffle {
    const platform = zml.testing.env();

    const upscale_factor = 3;
    var digits: [9 * 4 * 4]i32 = undefined;
    for (&digits, 0..) |*d, i| d.* = @intCast(i);
    // TODO should we have tags in buffers ?
    const input = try zml.Buffer.fromSlice(platform, .{ 1, 9, 4, 4 }, &digits);
    const output = try zml.testing.compileAndCallWithTensors(
        platform,
        pixelShuffle,
        .{ zml.Shape.init(.{ .batch_size = 1, .c = 9, .h = 4, .w = 4 }, .i32), upscale_factor },
        .{ input, upscale_factor },
    );

    const exp = zml.HostBuffer.fromArray(&[1][1][12][12]i32{.{.{
        .{ 0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35 },
        .{ 48, 64, 80, 49, 65, 81, 50, 66, 82, 51, 67, 83 },
        .{ 96, 112, 128, 97, 113, 129, 98, 114, 130, 99, 115, 131 },
        .{ 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39 },
        .{ 52, 68, 84, 53, 69, 85, 54, 70, 86, 55, 71, 87 },
        .{ 100, 116, 132, 101, 117, 133, 102, 118, 134, 103, 119, 135 },
        .{ 8, 24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43 },
        .{ 56, 72, 88, 57, 73, 89, 58, 74, 90, 59, 75, 91 },
        .{ 104, 120, 136, 105, 121, 137, 106, 122, 138, 107, 123, 139 },
        .{ 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47 },
        .{ 60, 76, 92, 61, 77, 93, 62, 78, 94, 63, 79, 95 },
        .{ 108, 124, 140, 109, 125, 141, 110, 126, 142, 111, 127, 143 },
    }}});
    try zml.testing.expectClose(exp, output, 0);
}

/// Implementation of `torch.roll`.
///
/// Note: at the difference of Pytorch, shifts need to be explicitly repeated, even if they are the same for all axes.
/// ref: https://pytorch.org/docs/stable/generated/torch.roll.html
pub fn roll(self: Tensor, shifts: []const i64, axes_: []const u8) Tensor {
    // TODO(hugo) accept following syntax: x.roll(.{ .a = 5, .b = 8 })
    meta.assert(self.rank() > 0 and shifts.len == axes_.len, "Shifts length ({d}) and dims length ({d}) are not equal, we expect the same length.", .{ shifts.len, axes_.len });

    if (shifts.len != 1 or axes_.len != 1) {
        const tail_shifts = shifts[1..shifts.len];
        const tail_dims = axes_[1..axes_.len];
        const first_dim_rolled = roll(self, &.{shifts[0]}, &.{axes_[0]});
        return roll(first_dim_rolled, tail_shifts, tail_dims);
    }

    const a = axes_[0];
    const start = @mod(self.dim(a) - shifts[0], self.dim(a));
    const idx = Tensor.arange(.{ .start = start, .end = start + self.dim(a) }, .f32);
    const divisor: f32 = @floatFromInt(self.dim(a));
    return self.gather1d(a, idx.fmod(divisor).convert(.i32), .{});
}

test roll {
    const platform = zml.testing.env();

    const input = try zml.Buffer.fromSlice(platform, .{ 4, 2 }, &[_]f32{ 2, 2, 3, 4, 5, 6, 7, 8 });
    const res = try zml.testing.compileAndCall(
        platform,
        roll,
        .{ input, &[_]i64{ 2, 1 }, &[_]u8{ 0, 1 } },
    );

    const expectation = zml.HostBuffer.fromSlice(.{ 4, 2 }, &[_]f32{ 6, 5, 8, 7, 2, 1, 4, 3 });
    try zml.testing.expectClose(expectation, res, 1e0);
}

pub const MeshgridIndexing = enum { xy, ij };

/// Mimic Pytorch and Numpy api.
/// The .ij mode is just calling to `zml.nn.cartesianProduct`
/// and has simple semantics.
/// The .xy mode swap the role of the first two vectors, it's generally best
/// to rewrite the calling code to use .ij mode if possible.
/// See Numpy docs:
/// https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html#numpy.meshgrid
/// - In the 2-D case with vectors of length M and N:
///   * for ‘ij’ indexing, outputs are of shape (M, N)
///   * for ‘xy’ indexing, outputs are of shape (N, M)
/// - In the 3-D case with vectors of length M, N and P:
///   * for ‘ij’ indexing, outputs are of shape (M, N, P)
///   * for ‘xy’ indexing, outputs are of shape (N, M, P)
pub fn meshgrid(comptime N: u3, vectors: [N]Tensor, indexing: MeshgridIndexing) [N]Tensor {
    meta.assertComptime(vectors.len >= 1, "Invalid meshgrid. No input.", .{});
    meta.assertComptime(vectors.len <= Tensor.MAX_RANK, "Invalid meshgrid(...). Too many inputs: {}", .{vectors.len});

    if (vectors.len == 1) return vectors;

    return switch (indexing) {
        .ij => zml.Tensor.cartesianProduct(N, vectors),
        .xy => {
            const x, const y = vectors[0..2].*;
            var new_vectors = vectors;
            new_vectors[0..2].* = .{ y, x };
            var res = zml.Tensor.cartesianProduct(N, new_vectors);
            const y_res, const x_res = res[0..2].*;
            res[0..2].* = .{ x_res, y_res };
            return res;
        },
    };
}

test meshgrid {
    const platform = zml.testing.env();

    const x = try zml.Buffer.fromSlice(platform, .{6}, &[_]i32{ 0, 1, 2, 3, 4, 5 });
    const y = try zml.Buffer.fromSlice(platform, .{4}, &[_]i32{ 0, 1, 2, 3 });

    const Local = struct {
        pub fn meshgrid2(a: Tensor, b: Tensor, indexing: MeshgridIndexing) [2]Tensor {
            return meshgrid(2, .{ a, b }, indexing);
        }
    };

    // Only test .xy mode, sinc .ij is just calling cartesianProduct which
    // got its own tests.
    {
        const xs, const ys = try zml.testing.compileAndCall(platform, Local.meshgrid2, .{ x, y, .xy });
        try std.testing.expectEqualSlices(i64, &.{ 4, 6 }, xs.dims());
        try std.testing.expectEqualSlices(i64, &.{ 4, 6 }, ys.dims());
        try std.testing.expectEqualDeep(
            [4][6]i32{
                .{ 0, 1, 2, 3, 4, 5 },
                .{ 0, 1, 2, 3, 4, 5 },
                .{ 0, 1, 2, 3, 4, 5 },
                .{ 0, 1, 2, 3, 4, 5 },
            },
            try xs.getValue([4][6]i32),
        );
        try std.testing.expectEqualDeep(
            [4][6]i32{
                .{ 0, 0, 0, 0, 0, 0 },
                .{ 1, 1, 1, 1, 1, 1 },
                .{ 2, 2, 2, 2, 2, 2 },
                .{ 3, 3, 3, 3, 3, 3 },
            },
            try ys.getValue([4][6]i32),
        );
    }
}
