//! Native block-scaled FP8 operations.
//!
//! AMD CDNA3 matrix cores consume E4M3FNUZ, while NVIDIA matrix cores consume
//! OCP E4M3FN directly. Published OCP E4M3FN weights can use the same finite
//! nonzero encodings as FNUZ when their scale is doubled: the exponent-bias
//! difference makes every FNUZ value one half of the corresponding FN value.
//! Prepared ROCm checkpoints normalize the formats' differing zero and NaN
//! sentinel encodings and adjust scales offline. CUDA checkpoints retain the
//! original E4M3FN values and scales.

const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

const Compiler = @import("Compiler.zig");
const DataType = @import("dtype.zig").DataType;
const ops = @import("ops.zig");
const tri = @import("kernel.zig").triton;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const Builder = tri.Builder;
const DType = tri.DType;

const NativeBlockFp8 = struct {
    dtype: DataType,
    triton_dtype: DType,
};

fn nativeBlockFp8() NativeBlockFp8 {
    return switch (Compiler.current().platform.target) {
        .cuda => .{ .dtype = .f8e4m3fn, .triton_dtype = .f8e4m3fn },
        .rocm => .{ .dtype = .f8e4m3fnuz, .triton_dtype = .f8e4m3fnuz },
        else => |target| stdx.debug.panic("native block FP8 dot is unsupported on {s}", .{@tagName(target)}),
    };
}

fn blockFp8Max(dtype: DType) f32 {
    return switch (dtype) {
        .f8e4m3fn => 448.0,
        .f8e4m3fnuz => 240.0,
        else => std.debug.panic("unsupported block FP8 dtype: {s}", .{@tagName(dtype)}),
    };
}

/// Normalize the exceptional OCP E4M3FN encodings before a native-FNUZ
/// bitcast. OCP negative zero (`0x80`) is FNUZ NaN, while OCP's two NaNs are
/// ordinary finite FNUZ values. The returned tensor intentionally retains its
/// FN dtype; callers bitcast it only at the native kernel boundary.
pub fn normalizeOcpEncodingForFnuz(x: Tensor) Tensor {
    stdx.debug.assert(x.dtype() == .f8e4m3fn, "expected OCP E4M3FN, got {f}", .{x.shape()});

    const bits = x.bitCast(.u8);
    const zero = Tensor.scalar(0, .u8).broad(bits.shape());
    const fnuz_nan = Tensor.scalar(0x80, .u8).broad(bits.shape());
    const positive_ocp_nan = bits.cmp(.EQ, Tensor.scalar(0x7f, .u8));
    const negative_ocp_nan = bits.cmp(.EQ, Tensor.scalar(0xff, .u8));
    const ocp_nan = positive_ocp_nan.logical(.OR, negative_ocp_nan);
    const without_negative_zero = bits.cmp(.EQ, Tensor.scalar(0x80, .u8)).select(zero, bits);
    return ocp_nan.select(fnuz_nan, without_negative_zero).bitCast(.f8e4m3fn);
}

const QuantizeBlock128 = struct {
    const Cfg = struct {
        input_dtype: DType,
        fp8_dtype: DType,
        m: usize,
        k: usize,
    };

    const Kernel = tri.Kernel(Cfg, .{
        .name = "quantize_block128_fp8",
        .inputs = &.{"x"},
        .outputs = &.{ "q", "scale" },
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) tri.FinishError!void {
        const args = try b.declareArgs(.{
            .x_ptr = .{ .ptr = cfg.input_dtype },
            .q_ptr = .{ .ptr = cfg.fp8_dtype },
            .scale_ptr = .{ .ptr = .f32 },
        });

        const group_size: i64 = 128;
        const groups_per_row: i64 = @intCast(cfg.k / 128);
        const pid = b.programId(.x).to(.i64);
        const row = pid.div(groups_per_row);
        const group = pid.rem(groups_per_row);
        const group_offset = row.mul(@as(i64, @intCast(cfg.k))).add(group.mul(group_size));

        const offsets = b.arange(0, group_size, .i64);
        const x = b.load(args.x_ptr.addPtr(group_offset.add(offsets))).to(.f32);
        const absmax = b.max(b.absf(x)).maximum(@as(f32, 1e-6));
        const fp8_max = blockFp8Max(cfg.fp8_dtype);
        const scale = absmax.mul(@as(f32, 1.0) / fp8_max);
        const q = b.clampf(
            x.div(scale),
            b.splat(-fp8_max, &.{group_size}),
            b.splat(fp8_max, &.{group_size}),
        ).to(cfg.fp8_dtype);

        b.store(args.q_ptr.addPtr(group_offset.add(offsets)), q);
        b.store(args.scale_ptr.addPtr(pid), scale);
    }
};

const BlockScaledGemm = struct {
    const Cfg = struct {
        m: usize,
        n: usize,
        k: usize,
        block_m: usize,
        block_n: usize,
        fp8_dtype: DType,
    };

    const Kernel = tri.Kernel(Cfg, .{
        .name = "gemm_a8w8_blockscale",
        .inputs = &.{ "a", "b", "a_scale", "b_scale" },
        .outputs = &.{"c"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) tri.FinishError!void {
        return runBlockScaledGemm(b, cfg, false);
    }
};

const BlockScaledGemmSplitK = struct {
    const Cfg = struct {
        m: usize,
        n: usize,
        k: usize,
        block_m: usize,
        block_n: usize,
        split_k: usize,
        fp8_dtype: DType,
    };

    const Kernel = tri.Kernel(Cfg, .{
        .name = "gemm_a8w8_blockscale_splitk",
        .inputs = &.{ "a", "b", "a_scale", "b_scale" },
        .outputs = &.{"c"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) tri.FinishError!void {
        return runBlockScaledGemm(b, cfg, true);
    }
};

fn runBlockScaledGemm(b: *Builder, cfg: anytype, comptime split: bool) tri.FinishError!void {
    const args = try b.declareArgs(.{
        .a_ptr = .{ .ptr = cfg.fp8_dtype },
        .b_ptr = .{ .ptr = cfg.fp8_dtype },
        .a_scale_ptr = .{ .ptr = .f32 },
        .b_scale_ptr = .{ .ptr = .f32 },
        .c_ptr = .{ .ptr = if (split) .f32 else .bf16 },
    });

    const m: i64 = @intCast(cfg.m);
    const n: i64 = @intCast(cfg.n);
    const k: i64 = @intCast(cfg.k);
    const block_m: i64 = @intCast(cfg.block_m);
    const block_n: i64 = @intCast(cfg.block_n);
    const block_k: i64 = 128;
    const k_blocks: i64 = @intCast(cfg.k / 128);
    const split_k: i64 = if (split) @intCast(cfg.split_k) else 1;
    const blocks_per_split = @divExact(k_blocks, split_k);
    const n_programs: i64 = @intCast(std.math.divCeil(usize, cfg.n, cfg.block_n) catch unreachable);

    const pid = b.programId(.x).to(.i64);
    const pid_split = pid.rem(split_k);
    const pid_tile = pid.div(split_k);
    const pid_m = pid_tile.div(n_programs);
    const pid_n = pid_tile.rem(n_programs);
    const k_offset = pid_split.mul(blocks_per_split * block_k);
    const scale_k_offset = pid_split.mul(blocks_per_split);

    const offs_m = pid_m.mul(block_m).add(b.arange(0, block_m, .i64));
    const offs_n = pid_n.mul(block_n).add(b.arange(0, block_n, .i64));
    const offs_k = b.arange(0, block_k, .i64);
    const mask_m = offs_m.lt(m);
    const mask_n = offs_n.lt(n);

    const offs_m_col = b.expandDims(offs_m, 1);
    const offs_n_row = b.expandDims(offs_n, 0);
    const offs_k_row = b.expandDims(offs_k, 0);
    const offs_k_col = b.expandDims(offs_k, 1);

    const a_rows = b.broadcastTo(offs_m_col.mul(k), &.{ block_m, block_k });
    const a_cols = b.broadcastTo(offs_k_row.add(k_offset), &.{ block_m, block_k });
    const a_ptrs_init = args.a_ptr.addPtr(a_rows.add(a_cols));

    // B is stored [N, K], while the dot tile is loaded as [K, N].
    const b_rows = b.broadcastTo(offs_n_row.mul(k), &.{ block_k, block_n });
    const b_cols = b.broadcastTo(offs_k_col.add(k_offset), &.{ block_k, block_n });
    const b_ptrs_init = args.b_ptr.addPtr(b_rows.add(b_cols));

    const a_scale_ptrs_init = args.a_scale_ptr.addPtr(offs_m.mul(k_blocks).add(scale_k_offset));
    const b_scale_ptrs_init = args.b_scale_ptr.addPtr(offs_n.div(128).mul(k_blocks).add(scale_k_offset));

    const mask_a = b.broadcastTo(b.expandDims(mask_m, 1), &.{ block_m, block_k });
    const mask_b = b.broadcastTo(b.expandDims(mask_n, 0), &.{ block_k, block_n });
    const zero_a = b.zeros(&.{ block_m, block_k }, cfg.fp8_dtype);
    const zero_b = b.zeros(&.{ block_k, block_n }, cfg.fp8_dtype);

    var loop = b.openFor(@as(i64, 0), blocks_per_split, @as(i64, 1), .{
        a_ptrs_init,
        b_ptrs_init,
        a_scale_ptrs_init,
        b_scale_ptrs_init,
        b.zeros(&.{ block_m, block_n }, .f32),
    });
    {
        const a_val = b.loadOpts(loop.carried[0], .{ .mask = mask_a, .other = zero_a });
        const b_val = b.loadOpts(loop.carried[1], .{ .mask = mask_b, .other = zero_b });
        const a_scale = b.loadOpts(loop.carried[2], .{ .mask = mask_m, .other = b.zeros(&.{block_m}, .f32) });
        const b_scale = b.loadOpts(loop.carried[3], .{ .mask = mask_n, .other = b.zeros(&.{block_n}, .f32) });

        const dot = b.dotOpts(a_val, b_val, b.zeros(&.{ block_m, block_n }, .f32), .{
            .input_precision = .tf32,
            .max_num_imprecise_acc = 0,
        });
        const scaled = dot.mul(a_scale.expandDims(1)).mul(b_scale.expandDims(0));

        loop.yield(.{
            loop.carried[0].addPtr(b.splat(@as(i64, 128), &.{ block_m, block_k })),
            loop.carried[1].addPtr(b.splat(@as(i64, 128), &.{ block_k, block_n })),
            loop.carried[2].addPtr(1),
            loop.carried[3].addPtr(1),
            loop.carried[4].add(scaled),
        });
    }

    const offs_c_m = b.broadcastTo(offs_m_col.mul(n), &.{ block_m, block_n });
    const offs_c_n = b.broadcastTo(offs_n_row, &.{ block_m, block_n });
    const split_offset = pid_split.mul(m * n);
    const c_ptrs = args.c_ptr.addPtr(offs_c_m.add(offs_c_n).add(split_offset));
    const mask_c = b.broadcastTo(b.expandDims(mask_m, 1), &.{ block_m, block_n })
        .bitAnd(b.broadcastTo(b.expandDims(mask_n, 0), &.{ block_m, block_n }));
    b.storeOpts(c_ptrs, if (split) loop.results[4] else loop.results[4].to(.bf16), .{ .mask = mask_c });
}

test "block FP8 Triton kernels emit E4M3FN and E4M3FNUZ TTIR" {
    const allocator = std.testing.allocator;
    inline for (.{
        .{ .dtype = DType.f8e4m3fn, .needle = "f8E4M3FN" },
        .{ .dtype = DType.f8e4m3fnuz, .needle = "f8E4M3FNUZ" },
    }) |format| {
        const quantize_ir = try QuantizeBlock128.Kernel.emit(allocator, .{
            .input_dtype = .bf16,
            .fp8_dtype = format.dtype,
            .m = 16,
            .k = 256,
        });
        defer allocator.free(quantize_ir);
        try std.testing.expect(std.mem.indexOf(u8, quantize_ir, format.needle) != null);

        const direct_ir = try BlockScaledGemm.Kernel.emit(allocator, .{
            .m = 16,
            .n = 64,
            .k = 256,
            .block_m = 16,
            .block_n = 64,
            .fp8_dtype = format.dtype,
        });
        defer allocator.free(direct_ir);
        try std.testing.expect(std.mem.indexOf(u8, direct_ir, format.needle) != null);
        try std.testing.expect(std.mem.indexOf(u8, direct_ir, "tt.dot") != null);

        const split_ir = try BlockScaledGemmSplitK.Kernel.emit(allocator, .{
            .m = 16,
            .n = 64,
            .k = 2048,
            .block_m = 16,
            .block_n = 64,
            .split_k = 8,
            .fp8_dtype = format.dtype,
        });
        defer allocator.free(split_ir);
        try std.testing.expect(std.mem.indexOf(u8, split_ir, format.needle) != null);
        try std.testing.expect(std.mem.indexOf(u8, split_ir, "tt.dot") != null);
    }
}

const PreparedDot = struct {
    a: Tensor,
    a_scale: Tensor,
    b: Tensor,
    b_scale: Tensor,
    m: i64,
    n: i64,
    k: i64,
};

const QuantizedBlock128 = struct {
    q: Tensor,
    scale: Tensor,
};

/// Emit CUDA's online activation quantization as StableHLO so XLA can fuse
/// the value and scale outputs into one Triton kernel before consuming them in
/// its block-128 W8A8 scaled-dot arm.
fn quantizeBlock128Cuda(lhs_2d: Tensor) QuantizedBlock128 {
    const fp8_max = 448.0;
    const grouped = lhs_2d.convert(.f32).splitAxis(.fp8_k, .{
        .fp8_ks = -1,
        .fp8_block = 128,
    });
    const scale = grouped.abs().max(.fp8_block)
        .maximum(.scalar(1e-6, .f32))
        .scale(1.0 / fp8_max);
    const q = grouped.div(scale.broad(grouped.shape()))
        .clamp(.scalar(-fp8_max, .f32), .scalar(fp8_max, .f32))
        .convert(.f8e4m3fn)
        .reshape(lhs_2d.shape().withDtype(.f8e4m3fn));

    return .{
        .q = q,
        .scale = scale.squeeze(.fp8_block),
    };
}

fn prepareBlockScaledDot(lhs: Tensor, rhs_fn: Tensor, rhs_scale_fn: Tensor) PreparedDot {
    stdx.debug.assert(lhs.rank() >= 1 and rhs_fn.rank() == 2, "block FP8 GEMM expects lhs rank >= 1 and rhs rank 2, got {f} and {f}", .{ lhs.shape(), rhs_fn.shape() });
    stdx.debug.assert(rhs_fn.dtype() == .f8e4m3fn, "block FP8 GEMM expects E4M3FN weights, got {f}", .{rhs_fn.shape()});
    const k = rhs_fn.dim(1);
    const n = rhs_fn.dim(0);
    stdx.debug.assert(lhs.dim(-1) == k, "block FP8 GEMM contraction mismatch: {f} and {f}", .{ lhs.shape(), rhs_fn.shape() });
    stdx.debug.assert(@mod(k, 128) == 0, "block FP8 GEMM requires K divisible by 128, got {d}", .{k});
    stdx.debug.assert(
        (rhs_scale_fn.dtype() == .bf16 or rhs_scale_fn.dtype() == .f32) and rhs_scale_fn.rank() == 2,
        "block FP8 GEMM expects a rank-2 BF16 or F32 weight scale grid, got {f}",
        .{rhs_scale_fn.shape()},
    );
    stdx.debug.assert(
        rhs_scale_fn.dim(0) == std.math.divCeil(i64, n, 128) catch unreachable and rhs_scale_fn.dim(1) == @divExact(k, 128),
        "block FP8 GEMM scale grid mismatch for weights {f}: got {f}",
        .{ rhs_fn.shape(), rhs_scale_fn.shape() },
    );

    const m: i64 = @intCast(@divExact(lhs.shape().count(), @as(usize, @intCast(k))));
    const lhs_2d = lhs.reshape(.{ .fp8_m = m, .fp8_k = k });
    const groups_per_row = @divExact(k, 128);
    const native_fp8 = nativeBlockFp8();
    stdx.debug.assert(
        native_fp8.dtype != .f8e4m3fn or rhs_scale_fn.dtype() == .f32,
        "CUDA block FP8 GEMM requires F32 checkpoint scales, got {f}",
        .{rhs_scale_fn.shape()},
    );

    const quantized: QuantizedBlock128 = switch (Compiler.current().platform.target) {
        .cuda => quantizeBlock128Cuda(lhs_2d),
        .rocm => blk: {
            const out = QuantizeBlock128.Kernel.call(
                .{ .x = lhs_2d },
                .{
                    .q = Shape.init(.{ .fp8_m = m, .fp8_k = k }, native_fp8.dtype),
                    .scale = Shape.init(.{ .fp8_m = m, .fp8_ks = groups_per_row }, .f32),
                },
                .{
                    .cfg = .{
                        .input_dtype = tri.from(lhs.dtype()),
                        .fp8_dtype = native_fp8.triton_dtype,
                        .m = @intCast(m),
                        .k = @intCast(k),
                    },
                    .grid = .{ @intCast(m * groups_per_row), 1, 1 },
                    .num_stages = 1,
                    .num_warps = 1,
                },
            );
            break :blk .{ .q = out.q, .scale = out.scale };
        },
        else => unreachable,
    };

    // Prepared ROCm checkpoints contain normalized FNUZ bytes and adjusted
    // scales, so this bitcast has no inference-time cost. CUDA consumes the
    // checkpoint's native E4M3FN values and F32 scales without conversion.
    const rhs_native = if (native_fp8.dtype == .f8e4m3fnuz) rhs_fn.bitCast(.f8e4m3fnuz) else rhs_fn;
    const rhs_scale_native = rhs_scale_fn.convert(.f32);

    return .{
        .a = quantized.q,
        .a_scale = quantized.scale,
        .b = rhs_native,
        .b_scale = rhs_scale_native,
        .m = m,
        .n = n,
        .k = k,
    };
}

fn tritonBlockScaledDotLocal(lhs: Tensor, rhs_fn: Tensor, rhs_scale_fn: Tensor, output_shape: Shape) Tensor {
    const prepared = prepareBlockScaledDot(lhs, rhs_fn, rhs_scale_fn);
    const m = prepared.m;
    const n = prepared.n;
    const k = prepared.k;

    const block_m: usize = if (m <= 16) 16 else if (m <= 32) 32 else 64;
    const block_n: usize = 64;
    const grid_m = std.math.divCeil(i64, m, @intCast(block_m)) catch unreachable;
    const grid_n = std.math.divCeil(i64, n, @intCast(block_n)) catch unreachable;
    const split_k = blockScaledGemmSplitK(grid_m, grid_n, n, k);
    if (split_k > 1) {
        const partials = BlockScaledGemmSplitK.Kernel.call(
            .{ .a = prepared.a, .b = prepared.b, .a_scale = prepared.a_scale, .b_scale = prepared.b_scale },
            .{ .c = Shape.init(.{ .fp8_split = split_k, .fp8_m = m, .fp8_n = n }, .f32) },
            .{
                .cfg = .{
                    .m = @intCast(m),
                    .n = @intCast(n),
                    .k = @intCast(k),
                    .block_m = block_m,
                    .block_n = block_n,
                    .split_k = @intCast(split_k),
                    .fp8_dtype = tri.from(prepared.a.dtype()),
                },
                .grid = .{ @intCast(grid_m * grid_n * split_k), 1, 1 },
                .num_stages = 2,
                .num_warps = 4,
            },
        ).c;
        return partials.sum(.fp8_split).squeeze(.fp8_split).convert(.bf16).reshape(output_shape);
    }

    const out = BlockScaledGemm.Kernel.call(
        .{ .a = prepared.a, .b = prepared.b, .a_scale = prepared.a_scale, .b_scale = prepared.b_scale },
        .{ .c = Shape.init(.{ .fp8_m = m, .fp8_n = n }, .bf16) },
        .{
            .cfg = .{
                .m = @intCast(m),
                .n = @intCast(n),
                .k = @intCast(k),
                .block_m = block_m,
                .block_n = block_n,
                .fp8_dtype = tri.from(prepared.a.dtype()),
            },
            .grid = .{ @intCast(grid_m * grid_n), 1, 1 },
            .num_stages = 2,
            .num_warps = 4,
        },
    ).c;

    return out.reshape(output_shape);
}

fn scaledDotReference(inputs: []const Tensor, output_shape: Shape) Tensor {
    return ops.customCall(
        "zml$scaled_dot_unmatched",
        .{ inputs[0], inputs[1], inputs[2], inputs[3] },
        output_shape,
        {},
        .{ .has_side_effect = false },
    );
}

fn xlaBlockScaledDotLocal(lhs: Tensor, rhs_fn: Tensor, rhs_scale_fn: Tensor, output_shape: Shape) Tensor {
    const prepared = prepareBlockScaledDot(lhs, rhs_fn, rhs_scale_fn);
    stdx.debug.assert(@mod(prepared.n, 128) == 0, "XLA block-128 FP8 dot requires N divisible by 128, got {d}", .{prepared.n});

    const output_2d = Shape.init(.{ .fp8_m = prepared.m, .fp8_n = prepared.n }, .bf16);
    const mlir_ctx = Compiler.current().mlir_ctx;
    const dimension_numbers = mlir.Attribute.array(mlir_ctx, &.{
        .array(mlir_ctx, &.{
            .intArray(mlir_ctx, i64, &.{1}),
            .intArray(mlir_ctx, i64, &.{1}),
        }),
        .array(mlir_ctx, &.{
            .intArray(mlir_ctx, i64, &.{}),
            .intArray(mlir_ctx, i64, &.{}),
        }),
    });
    const result = ops.composite(
        "xla.scaled_dot",
        &.{ prepared.a, prepared.b, prepared.a_scale, prepared.b_scale },
        &.{output_2d},
        scaledDotReference,
        output_2d,
        .{ .composite_attributes = &.{.named(mlir_ctx, "dimension_numbers", dimension_numbers)} },
    )[0];
    return result.reshape(output_shape);
}

fn blockScaledGemmSplitK(grid_m: i64, grid_n: i64, n: i64, k: i64) i64 {
    if (k < 2048 or n >= 6144) return 1;

    const k_blocks = @divExact(k, 128);
    var split_k: i64 = if (n <= 1024) 8 else 4;
    const base_programs = grid_m * grid_n;
    while (split_k > 1 and base_programs * split_k > 256) split_k = @divExact(split_k, 2);
    while (split_k > 1 and (@mod(k_blocks, split_k) != 0 or @divFloor(k_blocks, split_k) < 2)) {
        split_k = @divExact(split_k, 2);
    }
    return split_k;
}

const BlockDotContext = struct {
    reduce_partials: bool,
};

fn blockScaledDot(
    lhs: Tensor,
    rhs_fn: Tensor,
    rhs_scale_fn: Tensor,
    output_shape: Shape,
) Tensor {
    const reduce_partials = switch (rhs_fn.shape().partition(-1)) {
        .axis => true,
        .open, .replicated, .unknown => false,
    };
    return ops.manualComputation(
        .{ lhs, rhs_fn, rhs_scale_fn },
        output_shape,
        BlockDotContext{ .reduce_partials = reduce_partials },
        (struct {
            fn body(ctx: BlockDotContext, _: std.mem.Allocator, inputs: []const Tensor, local_output: Shape) Tensor {
                const partial = switch (Compiler.current().platform.target) {
                    .cuda => xlaBlockScaledDotLocal(inputs[0], inputs[1], inputs[2], local_output),
                    .rocm => tritonBlockScaledDotLocal(inputs[0], inputs[1], inputs[2], local_output),
                    else => unreachable,
                };
                return if (ctx.reduce_partials) ops.allReduce(partial, Tensor.add) else partial;
            }
        }).body,
    );
}

pub fn nativeBlockScaledDot(lhs: Tensor, rhs_fn: Tensor, rhs_scale_fn: Tensor, output_shape: Shape) Tensor {
    return blockScaledDot(lhs, rhs_fn, rhs_scale_fn, output_shape);
}

/// Compatibility alias. CUDA now delegates the GEMM to XLA; ROCm still uses
/// the original Triton implementation.
pub const tritonBlockScaledDot = nativeBlockScaledDot;

const AbsorbedKeyDot = struct {
    const Cfg = struct {
        input_dtype: DType,
        m: usize,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        latent_dim: usize,
        block_m: usize,
    };

    const Kernel = tri.Kernel(Cfg, .{
        .name = "glm_absorbed_key_a8w8_blockscale_fnuz",
        .inputs = &.{ "x", "weight", "weight_scale" },
        .outputs = &.{"out"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) tri.FinishError!void {
        const args = try b.declareArgs(.{
            .x_ptr = .{ .ptr = cfg.input_dtype },
            .weight_ptr = .{ .ptr = .f8e4m3fnuz },
            .weight_scale_ptr = .{ .ptr = .f32 },
            .out_ptr = .{ .ptr = .bf16 },
        });

        const block_k: i64 = 64;
        const block_n: i64 = 64;
        const block_m: i64 = @intCast(cfg.block_m);
        const m: i64 = @intCast(cfg.m);
        const heads: i64 = @intCast(cfg.heads);
        const key_dim: i64 = @intCast(cfg.key_dim);
        const head_stride: i64 = @intCast(cfg.key_dim + cfg.value_dim);
        const latent_dim: i64 = @intCast(cfg.latent_dim);
        const latent_blocks: i64 = @intCast(cfg.latent_dim / 128);
        const n_tiles: i64 = @intCast(std.math.divCeil(usize, cfg.latent_dim, @intCast(block_n)) catch unreachable);

        const pid = b.programId(.x).to(.i64);
        const pid_n = pid.rem(n_tiles);
        const pid_mh = pid.div(n_tiles);
        const pid_h = pid_mh.rem(heads);
        const pid_m = pid_mh.div(heads);

        const offs_m = pid_m.mul(block_m).add(b.arange(0, block_m, .i64));
        const offs_n = pid_n.mul(block_n).add(b.arange(0, block_n, .i64));
        const offs_k = b.arange(0, block_k, .i64);
        const mask_m = offs_m.lt(m);
        const mask_n = offs_n.lt(latent_dim);

        const offs_m_col = b.expandDims(offs_m, 1);
        const offs_n_row = b.expandDims(offs_n, 0);
        const offs_k_row = b.expandDims(offs_k, 0);
        const mask_x = b.broadcastTo(b.expandDims(mask_m, 1), &.{ block_m, block_k });
        const mask_w = b.broadcastTo(b.expandDims(mask_n, 0), &.{ block_k, block_n });

        var acc = b.zeros(&.{ block_m, block_n }, .f32);
        for (0..@divExact(cfg.key_dim, @as(usize, @intCast(block_k)))) |q_block| {
            const q_offset: i64 = @intCast(q_block * @as(usize, @intCast(block_k)));

            const x_rows = b.broadcastTo(offs_m_col.mul(heads * key_dim), &.{ block_m, block_k });
            const x_cols = b.broadcastTo(
                offs_k_row.add(pid_h.mul(key_dim).add(q_offset)),
                &.{ block_m, block_k },
            );
            const x = b.loadOpts(args.x_ptr.addPtr(x_rows.add(x_cols)), .{
                .mask = mask_x,
                .other = b.zeros(&.{ block_m, block_k }, cfg.input_dtype),
            }).to(.f32);
            const x_scale = b.maxOpts(b.absf(x), .{ .axis = 1, .keep_dims = true }).maximum(@as(f32, 1e-6)).mul(@as(f32, 1.0 / 240.0));
            const x_q = b.clampf(
                x.div(x_scale),
                b.splat(@as(f32, -240.0), &.{ block_m, block_k }),
                b.splat(@as(f32, 240.0), &.{ block_m, block_k }),
            ).to(.f8e4m3fnuz);

            const weight_rows_1d = offs_k.add(pid_h.mul(head_stride).add(q_offset));
            const weight_rows = b.broadcastTo(b.expandDims(weight_rows_1d, 1).mul(latent_dim), &.{ block_k, block_n });
            const weight_cols = b.broadcastTo(offs_n_row, &.{ block_k, block_n });
            const weight = b.loadOpts(args.weight_ptr.addPtr(weight_rows.add(weight_cols)), .{
                .mask = mask_w,
                .other = b.zeros(&.{ block_k, block_n }, .f8e4m3fnuz),
            });

            const scale_row = pid_h.mul(head_stride).add(q_offset).div(128);
            const weight_scale_ptrs = args.weight_scale_ptr.addPtr(scale_row.mul(latent_blocks).add(offs_n.div(128)));
            const weight_scale = b.loadOpts(weight_scale_ptrs, .{
                .mask = mask_n,
                .other = b.zeros(&.{block_n}, .f32),
            });
            const dot = b.dotOpts(x_q, weight, b.zeros(&.{ block_m, block_n }, .f32), .{
                .input_precision = .tf32,
                .max_num_imprecise_acc = 0,
            });
            acc = acc.add(dot.mul(x_scale).mul(weight_scale.expandDims(0)));
        }

        const out_rows = b.broadcastTo(offs_m_col.mul(heads * latent_dim), &.{ block_m, block_n });
        const out_cols = b.broadcastTo(offs_n_row.add(pid_h.mul(latent_dim)), &.{ block_m, block_n });
        const out_mask = b.broadcastTo(b.expandDims(mask_m, 1), &.{ block_m, block_n })
            .bitAnd(b.broadcastTo(b.expandDims(mask_n, 0), &.{ block_m, block_n }));
        b.storeOpts(args.out_ptr.addPtr(out_rows.add(out_cols)), acc.to(.bf16), .{ .mask = out_mask });
    }
};

const AbsorbedValueDot = struct {
    const Cfg = struct {
        input_dtype: DType,
        m: usize,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        latent_dim: usize,
        block_m: usize,
    };

    const Kernel = tri.Kernel(Cfg, .{
        .name = "glm_absorbed_value_a8w8_blockscale_fnuz",
        .inputs = &.{ "x", "weight", "weight_scale" },
        .outputs = &.{"out"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) tri.FinishError!void {
        const args = try b.declareArgs(.{
            .x_ptr = .{ .ptr = cfg.input_dtype },
            .weight_ptr = .{ .ptr = .f8e4m3fnuz },
            .weight_scale_ptr = .{ .ptr = .f32 },
            .out_ptr = .{ .ptr = .bf16 },
        });

        const block_k: i64 = 128;
        const block_n: i64 = 64;
        const block_m: i64 = @intCast(cfg.block_m);
        const m: i64 = @intCast(cfg.m);
        const heads: i64 = @intCast(cfg.heads);
        const head_stride: i64 = @intCast(cfg.key_dim + cfg.value_dim);
        const key_dim: i64 = @intCast(cfg.key_dim);
        const value_dim: i64 = @intCast(cfg.value_dim);
        const latent_dim: i64 = @intCast(cfg.latent_dim);
        const latent_blocks: i64 = @intCast(cfg.latent_dim / 128);
        const n_tiles: i64 = @intCast(std.math.divCeil(usize, cfg.value_dim, @intCast(block_n)) catch unreachable);

        const pid = b.programId(.x).to(.i64);
        const pid_n = pid.rem(n_tiles);
        const pid_mh = pid.div(n_tiles);
        const pid_h = pid_mh.rem(heads);
        const pid_m = pid_mh.div(heads);

        const offs_m = pid_m.mul(block_m).add(b.arange(0, block_m, .i64));
        const offs_n = pid_n.mul(block_n).add(b.arange(0, block_n, .i64));
        const offs_k = b.arange(0, block_k, .i64);
        const mask_m = offs_m.lt(m);
        const mask_n = offs_n.lt(value_dim);

        const offs_m_col = b.expandDims(offs_m, 1);
        const offs_n_row = b.expandDims(offs_n, 0);
        const offs_k_row = b.expandDims(offs_k, 0);
        const offs_k_col = b.expandDims(offs_k, 1);
        const mask_x = b.broadcastTo(b.expandDims(mask_m, 1), &.{ block_m, block_k });
        const mask_w = b.broadcastTo(b.expandDims(mask_n, 0), &.{ block_k, block_n });

        var acc = b.zeros(&.{ block_m, block_n }, .f32);
        for (0..@divExact(cfg.latent_dim, @as(usize, @intCast(block_k)))) |k_block| {
            const k_offset: i64 = @intCast(k_block * @as(usize, @intCast(block_k)));

            const x_rows = b.broadcastTo(offs_m_col.mul(heads * latent_dim), &.{ block_m, block_k });
            const x_cols = b.broadcastTo(
                offs_k_row.add(pid_h.mul(latent_dim).add(k_offset)),
                &.{ block_m, block_k },
            );
            const x = b.loadOpts(args.x_ptr.addPtr(x_rows.add(x_cols)), .{
                .mask = mask_x,
                .other = b.zeros(&.{ block_m, block_k }, cfg.input_dtype),
            }).to(.f32);
            const x_scale = b.maxOpts(b.absf(x), .{ .axis = 1, .keep_dims = true }).maximum(@as(f32, 1e-6)).mul(@as(f32, 1.0 / 240.0));
            const x_q = b.clampf(
                x.div(x_scale),
                b.splat(@as(f32, -240.0), &.{ block_m, block_k }),
                b.splat(@as(f32, 240.0), &.{ block_m, block_k }),
            ).to(.f8e4m3fnuz);

            const weight_rows_1d = offs_n.add(pid_h.mul(head_stride).add(key_dim));
            const weight_rows = b.broadcastTo(b.expandDims(weight_rows_1d, 0).mul(latent_dim), &.{ block_k, block_n });
            const weight_cols = b.broadcastTo(offs_k_col.add(k_offset), &.{ block_k, block_n });
            const weight = b.loadOpts(args.weight_ptr.addPtr(weight_rows.add(weight_cols)), .{
                .mask = mask_w,
                .other = b.zeros(&.{ block_k, block_n }, .f8e4m3fnuz),
            });

            const weight_scale_rows = weight_rows_1d.div(128).mul(latent_blocks);
            const weight_scale = b.loadOpts(args.weight_scale_ptr.addPtr(weight_scale_rows.add(@as(i64, @intCast(k_block)))), .{
                .mask = mask_n,
                .other = b.zeros(&.{block_n}, .f32),
            });
            const dot = b.dotOpts(x_q, weight, b.zeros(&.{ block_m, block_n }, .f32), .{
                .input_precision = .tf32,
                .max_num_imprecise_acc = 0,
            });
            acc = acc.add(dot.mul(x_scale).mul(weight_scale.expandDims(0)));
        }

        const out_rows = b.broadcastTo(offs_m_col.mul(heads * value_dim), &.{ block_m, block_n });
        const out_cols = b.broadcastTo(offs_n_row.add(pid_h.mul(value_dim)), &.{ block_m, block_n });
        const out_mask = b.broadcastTo(b.expandDims(mask_m, 1), &.{ block_m, block_n })
            .bitAnd(b.broadcastTo(b.expandDims(mask_n, 0), &.{ block_m, block_n }));
        b.storeOpts(args.out_ptr.addPtr(out_rows.add(out_cols)), acc.to(.bf16), .{ .mask = out_mask });
    }
};

const AbsorbedDotContext = struct {
    key_dim: usize,
    value_dim: usize,
};

/// GLM's MLA query absorption contracts over rows of the published block-FP8
/// `kv_b_proj` matrix. The 64-wide contraction tiles are aligned to the
/// checkpoint's original 128-row scale blocks, including the alternating
/// half-block head offsets in GLM-5.2.
pub fn rocmAbsorbedKeyDot(
    lhs: Tensor,
    weight_fn: Tensor,
    weight_scale_fn: Tensor,
    key_dim: usize,
    value_dim: usize,
    output_shape: Shape,
) Tensor {
    stdx.debug.assert(@mod(key_dim, 64) == 0 and @mod(weight_fn.dim(1), 128) == 0, "unsupported absorbed-key geometry: {f}", .{weight_fn.shape()});
    return ops.manualComputation(
        .{ lhs, weight_fn.bitCast(.f8e4m3fnuz), weight_scale_fn.convert(.f32) },
        output_shape,
        AbsorbedDotContext{ .key_dim = key_dim, .value_dim = value_dim },
        (struct {
            fn body(ctx: AbsorbedDotContext, _: std.mem.Allocator, inputs: []const Tensor, local_output: Shape) Tensor {
                const x = inputs[0];
                const weight = inputs[1];
                const scale = inputs[2];
                const heads: usize = @intCast(x.dim(1));
                const m: usize = @intCast(@divExact(x.shape().count(), heads * ctx.key_dim));
                const latent_dim: usize = @intCast(weight.dim(1));
                stdx.debug.assert(weight.dim(0) == @as(i64, @intCast(heads * (ctx.key_dim + ctx.value_dim))), "absorbed-key sharding mismatch: {f} and {f}", .{ x.shape(), weight.shape() });
                const block_m: usize = if (m <= 16) 16 else if (m <= 32) 32 else 64;
                const grid_m = std.math.divCeil(usize, m, block_m) catch unreachable;
                const grid_n = std.math.divCeil(usize, latent_dim, 64) catch unreachable;
                return AbsorbedKeyDot.Kernel.call(
                    .{ .x = x, .weight = weight, .weight_scale = scale },
                    .{ .out = local_output },
                    .{
                        .cfg = .{
                            .input_dtype = tri.from(x.dtype()),
                            .m = m,
                            .heads = heads,
                            .key_dim = ctx.key_dim,
                            .value_dim = ctx.value_dim,
                            .latent_dim = latent_dim,
                            .block_m = block_m,
                        },
                        .grid = .{ @intCast(grid_m * heads * grid_n), 1, 1 },
                        .num_stages = 2,
                        .num_warps = 4,
                    },
                ).out;
            }
        }).body,
    );
}

/// GLM's MLA value expansion uses the same original `kv_b_proj` block scales
/// while keeping heads batched. This avoids materializing or dequantizing an
/// alternate value matrix at inference time.
pub fn rocmAbsorbedValueDot(
    lhs: Tensor,
    weight_fn: Tensor,
    weight_scale_fn: Tensor,
    key_dim: usize,
    value_dim: usize,
    output_shape: Shape,
) Tensor {
    stdx.debug.assert(@mod(value_dim, 64) == 0 and @mod(weight_fn.dim(1), 128) == 0, "unsupported absorbed-value geometry: {f}", .{weight_fn.shape()});
    return ops.manualComputation(
        .{ lhs, weight_fn.bitCast(.f8e4m3fnuz), weight_scale_fn.convert(.f32) },
        output_shape,
        AbsorbedDotContext{ .key_dim = key_dim, .value_dim = value_dim },
        (struct {
            fn body(ctx: AbsorbedDotContext, _: std.mem.Allocator, inputs: []const Tensor, local_output: Shape) Tensor {
                const x = inputs[0];
                const weight = inputs[1];
                const scale = inputs[2];
                const heads: usize = @intCast(x.dim(1));
                const latent_dim: usize = @intCast(x.dim(2));
                const m: usize = @intCast(@divExact(x.shape().count(), heads * latent_dim));
                stdx.debug.assert(weight.dim(0) == @as(i64, @intCast(heads * (ctx.key_dim + ctx.value_dim))) and weight.dim(1) == @as(i64, @intCast(latent_dim)), "absorbed-value sharding mismatch: {f} and {f}", .{ x.shape(), weight.shape() });
                const block_m: usize = if (m <= 16) 16 else if (m <= 32) 32 else 64;
                const grid_m = std.math.divCeil(usize, m, block_m) catch unreachable;
                const grid_n = std.math.divCeil(usize, ctx.value_dim, 64) catch unreachable;
                return AbsorbedValueDot.Kernel.call(
                    .{ .x = x, .weight = weight, .weight_scale = scale },
                    .{ .out = local_output },
                    .{
                        .cfg = .{
                            .input_dtype = tri.from(x.dtype()),
                            .m = m,
                            .heads = heads,
                            .key_dim = ctx.key_dim,
                            .value_dim = ctx.value_dim,
                            .latent_dim = latent_dim,
                            .block_m = block_m,
                        },
                        .grid = .{ @intCast(grid_m * heads * grid_n), 1, 1 },
                        .num_stages = 2,
                        .num_warps = 4,
                    },
                ).out;
            }
        }).body,
    );
}

test {
    std.testing.refAllDecls(@This());
}
