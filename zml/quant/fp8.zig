const kernel = @import("gemm_a16w8.zig");

const std = @import("std");
const zml = @import("../zml.zig");

const Shape = zml.Shape;
const Tensor = zml.Tensor;
const stdx = zml.stdx;

const group_k: i64 = 128;
const group_n: i64 = 128;
const block_m: i64 = 16;
const block_n: i64 = 64;
const block_k: i64 = 64;

pub fn gemm(x: zml.Tensor, y: zml.Tensor, y_scale: zml.Tensor) zml.Tensor {
    validateInputs(x, y, y_scale);

    const output_shape = outputShape(x, y);
    return zml.ops.manualComputation(
        .{ x, y, y_scale },
        output_shape,
        {},
        (struct {
            fn body(_: void, _: std.mem.Allocator, sharded_inputs: []const Tensor, sharded_output: Shape) Tensor {
                return gemmShard(sharded_inputs[0], sharded_inputs[1], sharded_inputs[2], sharded_output);
            }
        }).body,
    );
}

fn validateInputs(x: Tensor, y: Tensor, y_scale: Tensor) void {
    stdx.debug.assert(x.rank() == 2, "fp8.gemm expects a rank-2 activation matrix, got {f}", .{x.shape()});
    stdx.debug.assert(y.rank() == 2, "fp8.gemm expects a rank-2 weight matrix, got {f}", .{y.shape()});
    stdx.debug.assert(y_scale.rank() == 2, "fp8.gemm expects a rank-2 weight-scale matrix, got {f}", .{y_scale.shape()});
    stdx.debug.assert(x.dtype() == .bf16, "fp8.gemm expects bf16 activations, got {}", .{x.dtype()});
    stdx.debug.assert(y.dtype() == .f8e4m3fn, "fp8.gemm expects f8e4m3fn weights, got {}", .{y.dtype()});
    stdx.debug.assert(y_scale.dtype() == .f32, "fp8.gemm expects f32 weight scales, got {}", .{y_scale.dtype()});
    stdx.debug.assert(x.dim(1) == y.dim(0), "fp8.gemm expected activation K {} to match weight K {}", .{ x.dim(1), y.dim(0) });

    const k = x.dim(1);
    const n = y.dim(1);
    const scale_k = std.math.divCeil(i64, k, group_k) catch unreachable;
    const scale_n = std.math.divCeil(i64, n, group_n) catch unreachable;
    stdx.debug.assert(y_scale.dim(0) == scale_k and y_scale.dim(1) == scale_n, "fp8.gemm expected weight scales of shape [{}, {}], got {f}", .{ scale_k, scale_n, y_scale.shape() });
}

fn outputShape(x: Tensor, y: Tensor) Shape {
    var out = Shape.init(.{ x.dim(0), y.dim(1) }, .bf16)
        .setTag(0, x.shape().tag(0))
        .setTag(1, y.shape().tag(1));

    const m_partition = x.shape().partition(0);
    const n_partition = y.shape().partition(1);
    stdx.debug.assert(!samePartitionAxis(m_partition, n_partition), "fp8.gemm output axes cannot both be partitioned by the same mesh axis", .{});
    out._partitioning.set(0, m_partition);
    out._partitioning.set(1, n_partition);
    return out;
}

fn samePartitionAxis(a: Shape.PartitionSpec, b: Shape.PartitionSpec) bool {
    return switch (a) {
        .axis => |a_axis| switch (b) {
            .axis => |b_axis| std.mem.eql(u8, std.mem.span(a_axis), std.mem.span(b_axis)),
            else => false,
        },
        else => false,
    };
}

fn gemmShard(x: Tensor, y: Tensor, y_scale: Tensor, output_shape: Shape) Tensor {
    validateInputs(x, y, y_scale);
    stdx.debug.assert(output_shape.rank() == 2 and output_shape.dtype() == .bf16, "fp8.gemm expected local output shape to be rank-2 bf16, got {f}", .{output_shape});
    stdx.debug.assert(output_shape.dim(0) == x.dim(0) and output_shape.dim(1) == y.dim(1), "fp8.gemm expected local output shape {f} to match local GEMM [{}, {}]", .{ output_shape, x.dim(0), y.dim(1) });

    const m = output_shape.dim(0);
    const k = x.dim(1);
    const n = output_shape.dim(1);
    // The kernel loads one weight-scale row per split, so each split covers one K scale group.
    const num_ksplit = std.math.divCeil(i64, k, group_k) catch unreachable;
    const grid_m = std.math.divCeil(i64, m, block_m) catch unreachable;
    const grid_n = std.math.divCeil(i64, n, block_n) catch unreachable;
    const grid_mn = grid_m * grid_n;

    const x_strides = x.shape().computeElementStrides().constSlice();
    const y_strides = y.shape().computeElementStrides().constSlice();
    const y_scale_strides = y_scale.shape().computeElementStrides().constSlice();

    const kernel_output_shape = if (num_ksplit == 1)
        output_shape
    else
        Shape.init(.{ num_ksplit, m, n }, .bf16)
            .setTag(0, .ksplit)
            .setTag(1, x.shape().tag(0))
            .setTag(2, y.shape().tag(1));

    const partials = kernel.Kernel.call(
        .{
            .a_ptr = x,
            .b_ptr = y,
            .b_scale_ptr = y_scale,
            .M_ptr = scalarI64(m),
            .N_ptr = scalarI64(n),
            .K_ptr = scalarI64(k),
            .stride_am_ptr = scalarI64(x_strides[0]),
            .stride_ak_ptr = scalarI64(x_strides[1]),
            .stride_bk_ptr = scalarI64(y_strides[0]),
            .stride_bn_ptr = scalarI64(y_strides[1]),
            .stride_ck_ptr = scalarI64(if (num_ksplit == 1) 0 else m * n),
            .stride_cm_ptr = scalarI64(n),
            .stride_cn_ptr = scalarI64(1),
            .stride_bscale_k_ptr = scalarI64(y_scale_strides[0]),
            .stride_bscale_n_ptr = scalarI64(y_scale_strides[1]),
        },
        .{ .c = kernel_output_shape },
        .{
            .cfg = .{
                .a_dtype = .bf16,
                .b_dtype = .f8e4m3fn,
                .c_dtype = .bf16,
                .b_scale_dtype = .f32,
                .GROUP_K = @intCast(group_k),
                .GROUP_N = @intCast(group_n),
                .BLOCK_SIZE_M = @intCast(block_m),
                .BLOCK_SIZE_N = @intCast(block_n),
                .BLOCK_SIZE_K = @intCast(block_k),
                .NUM_KSPLIT = @intCast(num_ksplit),
                .SPLITK_BLOCK_SIZE = @intCast(group_k),
                .EVEN_K = @mod(k, group_k) == 0,
                .GRID_MN = @intCast(grid_mn),
            },
            .grid = .{ @intCast(grid_mn * num_ksplit), 1, 1 },
            .num_stages = 2,
            .num_warps = 4,
        },
    ).c;

    if (num_ksplit == 1) return partials;
    return partials.sum(.ksplit).reshape(.{ m, n }).withTags(output_shape);
}

test "gemm output shape preserves partitioning" {
    const x = Tensor.fromShape(Shape.init(.{ .m = 17, .k = 256 }, .bf16).withPartitioning(.{
        .m = .data,
        .k = .replicated,
    }));
    const y = Tensor.fromShape(Shape.init(.{ .k = 256, .n = 65 }, .f8e4m3fn).withPartitioning(.{
        .k = .replicated,
        .n = .model,
    }));

    const out = outputShape(x, y);
    try zml.testing.expectEqualShapes(Shape.init(.{ .m = 17, .n = 65 }, .bf16), out);
    try std.testing.expect(out.partition(0).eql(.init(.data)));
    try std.testing.expect(out.partition(1).eql(.init(.model)));
}

test "gemm output shape" {
    const platform = zml.testing.env();
    if (platform.target != .cuda and platform.target != .rocm and platform.target != .oneapi) return error.SkipZigTest;

    const Fwd = struct {
        fn forward(x: Tensor, y: Tensor, y_scale: Tensor) Tensor {
            return gemm(x, y, y_scale);
        }
    };

    const x = Tensor.init(.{ .m = 17, .k = 256 }, .bf16);
    const y = Tensor.init(.{ .k = 256, .n = 65 }, .f8e4m3fn);
    const y_scale = Tensor.init(.{ .ks = 2, .ns = 1 }, .f32);

    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Fwd.forward, .{ x, y, y_scale }, platform, .{});
    defer exe.deinit();

    try zml.testing.expectEqualShapes(Shape.init(.{ .m = 17, .n = 65 }, .bf16), exe.output_shapes[0]);
}

fn scalarI64(v: i64) Tensor {
    return Tensor.constant(.{ .i64 = v }).reshape(.{1});
}
