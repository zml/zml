const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;
const bf16 = zml.floats.BFloat16;
const ttir_compile_sandbox = @import("ttir_compile_sandbox");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const cfg = struct {
    const batch_size = 8;
    const token_count = 8;
    const num_heads = 32;
    const num_kv_heads = 8;
    const head_size = 128;
    const max_input_len = 1;
    const rcp_ln2 = 1.4426950216;
};

pub fn wrappedPrefillAttention(
    kernel_ttir: [:0]const u8,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    b_start_loc: Tensor,
    b_seq_len: Tensor,
    out: Tensor,
) Tensor {
    const q_strides = q.shape().computeElementStrides();
    const k_strides = k.shape().computeElementStrides();
    const v_strides = v.shape().computeElementStrides();
    const o_strides = out.shape().computeElementStrides();

    const sm_scale = (1.0 / @sqrt(@as(f32, cfg.head_size))) * cfg.rcp_ln2;
    // Keep shared memory usage under PJRT's 64KB limit.
    const block_m: i32 = 64;
    const max_input_len_i32: i32 = cfg.max_input_len;
    const head_i32: i32 = @intCast(cfg.num_heads);
    const grid: [3]i32 = .{
        @intCast(b_seq_len.dim(0)),
        head_i32,
        @intCast(@divFloor(max_input_len_i32 + block_m - 1, block_m)),
    };

    const target = zml.module.CompilationContext.current().platform.target;
    const num_warps: i32 = switch (target) {
        .rocm => 4,
        .cuda => 4,
        else => 4,
    };

    return zml.ops.triton(.{
        q,
        k,
        v,
        Tensor.scalar(sm_scale, .f32),
        b_start_loc,
        b_seq_len,
        Tensor.scalar(q_strides.get(0), .i64),
        Tensor.scalar(q_strides.get(1), .i64),
        Tensor.scalar(k_strides.get(0), .i64),
        Tensor.scalar(k_strides.get(1), .i64),
        Tensor.scalar(v_strides.get(0), .i64),
        Tensor.scalar(v_strides.get(1), .i64),
        Tensor.scalar(o_strides.get(0), .i64),
        Tensor.scalar(o_strides.get(1), .i64),
    }, .{out.shape()}, .{
        .name = "wrapped_fwd_kernel",
        .ir = kernel_ttir,
        .grid = grid,
        .num_stages = 1,
        .num_warps = num_warps,
        .debug = true,
        .output_operand_aliases = &.{},
    })[0];
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    log.info("\n{f}", .{platform.fmtVerbose()});

    if (platform.target != .cuda and platform.target != .rocm) {
        log.err("ttir_test requires CUDA/ROCm target, got {s}", .{@tagName(platform.target)});
        return;
    }

    const kernel_params =
        "{" ++
        "\"batch\":8," ++
        "\"seq_len\":1," ++
        "\"num_query_heads\":32," ++
        "\"num_kv_heads\":8," ++
        "\"head_size\":128" ++
        "}";
    const kernel_ttir = try ttir_compile_sandbox.getPrefillAttentionTtir(allocator, io, kernel_params);
    defer allocator.free(kernel_ttir);

    const q_shape: zml.Tensor = .init(.{ cfg.token_count, cfg.num_heads, cfg.head_size }, .bf16);
    const k_shape: zml.Tensor = .init(.{ cfg.token_count, cfg.num_kv_heads, cfg.head_size }, .bf16);
    const v_shape: zml.Tensor = .init(.{ cfg.token_count, cfg.num_kv_heads, cfg.head_size }, .bf16);
    const b_start_loc_shape: zml.Tensor = .init(.{cfg.batch_size}, .i32);
    const b_seq_len_shape: zml.Tensor = .init(.{cfg.batch_size}, .i32);
    const out_shape: zml.Tensor = q_shape;

    var exe = try platform.compileFn(allocator, io, wrappedPrefillAttention, .{
        kernel_ttir,
        q_shape,
        k_shape,
        v_shape,
        b_start_loc_shape,
        b_seq_len_shape,
        out_shape,
    });
    defer exe.deinit();

    var q = try zeroBuffer(allocator, io, platform, q_shape.shape());
    defer q.deinit();
    var k = try zeroBuffer(allocator, io, platform, k_shape.shape());
    defer k.deinit();
    var v = try zeroBuffer(allocator, io, platform, v_shape.shape());
    defer v.deinit();
    var b_start_loc = try zeroBuffer(allocator, io, platform, b_start_loc_shape.shape());
    defer b_start_loc.deinit();
    var b_seq_len = try zeroBuffer(allocator, io, platform, b_seq_len_shape.shape());
    defer b_seq_len.deinit();
    var out = try uninitBuffer(allocator, io, platform, out_shape.shape());
    defer out.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ q, k, v, b_start_loc, b_seq_len, out });

    exe.call(exe_args, &exe_results);
    var result: zml.Buffer = exe_results.get(zml.Buffer);
    defer result.deinit();

    var output = try result.toSliceAlloc(allocator, io);
    defer output.free(allocator);

    const out_items = output.constItems(bf16);
    const out_strides = result.shape().computeElementStrides();
    log.info("Output shape: {f}", .{result.shape()});
    std.debug.print("Output out[:,0,0] first 8 (bf16->f32):", .{});
    const d0: usize = @intCast(result.shape().dim(0));
    const n = @min(@as(usize, 8), d0);
    for (0..n) |i| {
        const f = loadOutBF16(out_items, out_strides, i, 0, 0);
        std.debug.print(" {d:.5}", .{f});
    }
    std.debug.print("\n", .{});
}

fn zeroBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    @memset(slice.data(), 0);
    return zml.Buffer.fromSlice(io, platform, slice);
}

fn uninitBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    return zml.Buffer.fromSlice(io, platform, slice);
}

fn loadOutBF16(items: []const bf16, strides: anytype, i: usize, j: usize, k: usize) f32 {
    const base = @as(usize, @intCast(strides.get(0))) * i +
        @as(usize, @intCast(strides.get(1))) * j +
        @as(usize, @intCast(strides.get(2))) * k;
    return items[base].toF32();
}
