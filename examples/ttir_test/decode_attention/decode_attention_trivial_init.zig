const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const cfg = struct {
    const batch_size = 8;
    const num_heads = 32;
    const num_kv_heads = 8;
    const head_size = 128;
    const head_size_padded = 128;
    const page_size = 16;
    const num_pages = 128;
    const max_num_pages_per_req = 64;
    const num_kv_splits = 8;
    const sm_scale = 0.08838834765;
};

pub fn wrappedDecodeAttentionStage1(
    q: Tensor,
    k_buffer: Tensor,
    v_buffer: Tensor,
    req_to_tokens: Tensor,
    b_seqlen: Tensor,
    out: Tensor,
) Tensor {
    const q_strides = q.shape().computeElementStrides();
    const k_strides = k_buffer.shape().computeElementStrides();
    const v_strides = v_buffer.shape().computeElementStrides();
    const o_strides = out.shape().computeElementStrides();

    const grid: [3]i32 = .{
        @intCast(q.dim(0)),
        @intCast(q.dim(1)),
        cfg.num_kv_splits,
    };

    const target = zml.module.CompilationContext.current().platform.target;
    const num_warps: i32 = switch (target) {
        .rocm => 2,
        .cuda => 2,
        else => 2,
    };

    return zml.ops.triton(.{
        q,
        k_buffer,
        v_buffer,
        Tensor.scalar(cfg.sm_scale, .f32),
        req_to_tokens,
        b_seqlen,
        Tensor.scalar(req_to_tokens.shape().computeElementStrides().get(0), .i64),
        Tensor.scalar(q_strides.get(0), .i64),
        Tensor.scalar(q_strides.get(1), .i64),
        Tensor.scalar(k_strides.get(1), .i64),
        Tensor.scalar(k_strides.get(2), .i64),
        Tensor.scalar(v_strides.get(1), .i64),
        Tensor.scalar(v_strides.get(2), .i64),
        Tensor.scalar(o_strides.get(0), .i64),
        Tensor.scalar(o_strides.get(1), .i64),
        Tensor.scalar(o_strides.get(2), .i64),
    }, .{out.shape()}, .{
        .name = "wrapped_fwd_kernel_stage1",
        .ir = @embedFile("decode_attention.ttir"),
        .grid = grid,
        .num_stages = 2,
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

    const q_shape: zml.Tensor = .init(.{ cfg.batch_size, cfg.num_heads, cfg.head_size }, .bf16);
    const k_shape: zml.Tensor = .init(.{ cfg.num_pages, cfg.page_size, cfg.num_kv_heads, cfg.head_size }, .bf16);
    const v_shape: zml.Tensor = .init(.{ cfg.num_pages, cfg.page_size, cfg.num_kv_heads, cfg.head_size }, .bf16);
    const req_to_tokens_shape: zml.Tensor = .init(.{ cfg.batch_size, cfg.max_num_pages_per_req }, .i32);
    const b_seqlen_shape: zml.Tensor = .init(.{cfg.batch_size}, .i32);
    const out_shape: zml.Tensor = .init(.{ cfg.batch_size, cfg.num_heads, cfg.num_kv_splits, cfg.head_size_padded + 1 }, .f32);

    var exe = try platform.compileFn(allocator, io, wrappedDecodeAttentionStage1, .{
        q_shape,
        k_shape,
        v_shape,
        req_to_tokens_shape,
        b_seqlen_shape,
        out_shape,
    });
    defer exe.deinit();

    var q = try zeroBuffer(allocator, io, platform, q_shape.shape());
    defer q.deinit();
    var k = try zeroBuffer(allocator, io, platform, k_shape.shape());
    defer k.deinit();
    var v = try zeroBuffer(allocator, io, platform, v_shape.shape());
    defer v.deinit();
    var req_to_tokens = try reqToTokensBuffer(allocator, io, platform);
    defer req_to_tokens.deinit();
    var b_seqlen = try onesI32Buffer(allocator, io, platform, b_seqlen_shape.shape());
    defer b_seqlen.deinit();
    var out = try uninitBuffer(allocator, io, platform, out_shape.shape());
    defer out.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ q, k, v, req_to_tokens, b_seqlen, out });

    exe.call(exe_args, &exe_results);
    var result: zml.Buffer = exe_results.get(zml.Buffer);
    defer result.deinit();

    var output = try result.toSliceAlloc(allocator, io);
    defer output.free(allocator);

    const out_items = output.constItems(f32);
    const out_strides = result.shape().computeElementStrides();
    log.info("Output shape: {f}", .{result.shape()});
    std.debug.print("Output att_out[:,0,0,0] first 8:", .{});
    const d0: usize = @intCast(result.shape().dim(0));
    const n = @min(@as(usize, 8), d0);
    for (0..n) |i| {
        const f = loadOutF32(out_items, out_strides, i, 0, 0, 0);
        std.debug.print(" {d:.5}", .{f});
    }
    std.debug.print("\n", .{});
}

fn reqToTokensBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{ cfg.batch_size, cfg.max_num_pages_per_req }, .i32));
    defer slice.free(allocator);

    const items = slice.items(i32);
    @memset(items, 0);
    for (0..cfg.batch_size) |b| {
        items[b * cfg.max_num_pages_per_req] = @intCast(b);
    }

    return zml.Buffer.fromSlice(io, platform, slice);
}

fn onesI32Buffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    for (slice.items(i32)) |*v| v.* = 1;
    return zml.Buffer.fromSlice(io, platform, slice);
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

fn loadOutF32(items: []const f32, strides: anytype, i: usize, j: usize, k: usize, l: usize) f32 {
    const base = @as(usize, @intCast(strides.get(0))) * i +
        @as(usize, @intCast(strides.get(1))) * j +
        @as(usize, @intCast(strides.get(2))) * k +
        @as(usize, @intCast(strides.get(3))) * l;
    return items[base];
}
