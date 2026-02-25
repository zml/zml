const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;
const bf16 = zml.floats.BFloat16;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const test_cfg = struct {
    const token_count = 8;
    const batch_size = 8;
    const num_heads = 32;
    const num_kv_heads = 8;
    const head_size = 128;
    const block_size = 16;
    const num_blocks = 4096;
    const max_input_len = 1;
    const max_seq_len = 8192;
    const scale = 0.08838834765;
    const k_scale = 1.0;
    const v_scale = 1.0;
};

pub fn wrappedUnifiedAttention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    block_table: Tensor,
    out: Tensor,
) Tensor {
    const q_strides = q.shape().computeElementStrides();
    const k_strides = k.shape().computeElementStrides();
    const v_strides = v.shape().computeElementStrides();
    const bt_strides = block_table.shape().computeElementStrides();

    // Kept for API parity with the torch-side call.
    const max_seqlen_q = test_cfg.max_input_len;
    const max_seqlen_k = test_cfg.max_seq_len;
    const causal = true;
    const window_size = [2]i32{ -1, -1 };
    const q_descale: ?f32 = null;
    _ = .{ max_seqlen_q, max_seqlen_k, causal, window_size, q_descale };

    const num_seqs = Tensor.scalar(block_table.dim(0), .i32);
    const num_query_heads: i32 = @intCast(test_cfg.num_heads);
    const num_kv_heads: i32 = @intCast(test_cfg.num_kv_heads);
    const num_queries_per_kv: i32 = @divExact(num_query_heads, num_kv_heads);
    const block_m: i32 = if (num_queries_per_kv <= 16) 16 else @intCast(std.math.ceilPowerOfTwo(i32, num_queries_per_kv));
    const block_q: i32 = @divExact(block_m, num_queries_per_kv);
    const q_len: i64 = @intCast(q.dim(0));
    const num_seqs_i64: i64 = @intCast(block_table.dim(0));
    const total_num_q_blocks: i64 = @divFloor(q_len, @as(i64, block_q)) + num_seqs_i64;
    const grid: [3]i32 = .{ @intCast(total_num_q_blocks), num_kv_heads, 1 };
    const target = zml.module.CompilationContext.current().platform.target;
    const num_warps: i32 = switch (target) {
        .rocm => 1,
        .cuda => 4,
        else => 4,
    };

    return zml.ops.triton(.{
        q,
        k,
        v,
        block_table,
        seqused_k,
        Tensor.scalar(test_cfg.scale, .f32),
        Tensor.scalar(test_cfg.k_scale, .f32),
        Tensor.scalar(test_cfg.v_scale, .f32),
        Tensor.scalar(1.0, .f32),
        Tensor.scalar(0.0, .f32),
        Tensor.scalar(bt_strides.get(0), .i64),
        Tensor.scalar(q_strides.get(0), .i64),
        Tensor.scalar(q_strides.get(1), .i64),
        Tensor.scalar(q_strides.get(0), .i64),
        Tensor.scalar(q_strides.get(1), .i64),
        Tensor.scalar(0, .i64),
        Tensor.scalar(k_strides.get(0), .i64),
        Tensor.scalar(k_strides.get(1), .i64),
        Tensor.scalar(k_strides.get(2), .i64),
        Tensor.scalar(v_strides.get(0), .i64),
        Tensor.scalar(v_strides.get(1), .i64),
        Tensor.scalar(v_strides.get(2), .i64),
        cu_seqlens_q,
        num_seqs,
    }, .{out.shape()}, .{
        .name = "wrapped_kernel_unified_attention_2d",
        .ir = @embedFile("2d_unified_attention_kernel.ttir"),
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

    const query_shape: zml.Tensor = .init(.{ test_cfg.token_count, test_cfg.num_heads, test_cfg.head_size }, .bf16);
    const key_cache_shape: zml.Tensor = .init(.{ test_cfg.num_blocks, test_cfg.block_size, test_cfg.num_kv_heads, test_cfg.head_size }, .bf16);
    const value_cache_shape: zml.Tensor = .init(.{ test_cfg.num_blocks, test_cfg.block_size, test_cfg.num_kv_heads, test_cfg.head_size }, .bf16);
    const out_shape: zml.Tensor = query_shape;
    const start_loc_shape: zml.Tensor = .init(.{test_cfg.batch_size + 1}, .i32);
    const context_seq_lens_shape: zml.Tensor = .init(.{test_cfg.batch_size}, .i32);
    const block_tables_shape: zml.Tensor = .init(.{ test_cfg.batch_size, @divFloor(test_cfg.max_seq_len, test_cfg.block_size) }, .i32);

    var exe = try platform.compileFn(allocator, io, wrappedUnifiedAttention, .{
        query_shape,
        key_cache_shape,
        value_cache_shape,
        start_loc_shape,
        context_seq_lens_shape,
        block_tables_shape,
        out_shape,
    });
    defer exe.deinit();

    const block_table_width: usize = @intCast(block_tables_shape.dim(1));

    var query = try tokenRampBuffer(allocator, io, platform, query_shape.shape());
    defer query.deinit();
    var key_cache = try zeroBuffer(allocator, io, platform, key_cache_shape.shape());
    defer key_cache.deinit();
    var value_cache = try valueCacheBlockRampBuffer(allocator, io, platform, value_cache_shape.shape());
    defer value_cache.deinit();
    var out = try uninitBuffer(allocator, io, platform, out_shape.shape());
    defer out.deinit();

    var start_loc = try startLocBuffer(allocator, io, platform);
    defer start_loc.deinit();
    var context_seq_lens = try contextSeqLensBuffer(allocator, io, platform);
    defer context_seq_lens.deinit();
    var block_tables = try blockTablesBuffer(allocator, io, platform, block_table_width);
    defer block_tables.deinit();

    // Debug: show query ramp at [token,0,0] for first few tokens.
    {
        var query_host = try query.toSliceAlloc(allocator, io);
        defer query_host.free(allocator);
        const bytes = query_host.constData();
        const d1: usize = @intCast(query_shape.dim(1));
        const d2: usize = @intCast(query_shape.dim(2));
        const stride0 = d1 * d2;
        const max_tokens = @min(@as(usize, 8), @as(usize, @intCast(query_shape.dim(0))));
        std.debug.print("query[token,0,0] first {d} tokens:", .{max_tokens});
        for (0..max_tokens) |t| {
            const idx = t * stride0;
            const byte_index = idx * 2;
            const lane = @as(u16, bytes[byte_index]) | (@as(u16, bytes[byte_index + 1]) << 8);
            const f = bf16ToF32(lane);
            std.debug.print(" {d}", .{f});
        }
        std.debug.print("\n", .{});
    }

    // Debug: show key_cache at [block,0,0,0] for first few blocks.
    {
        var key_cache_host = try key_cache.toSliceAlloc(allocator, io);
        defer key_cache_host.free(allocator);
        const bytes = key_cache_host.constData();
        const block_size: usize = @intCast(key_cache_shape.dim(1));
        const num_kv_heads: usize = @intCast(key_cache_shape.dim(2));
        const head_size: usize = @intCast(key_cache_shape.dim(3));
        const block_stride = block_size * num_kv_heads * head_size;
        const max_blocks = @min(@as(usize, 8), @as(usize, @intCast(key_cache_shape.dim(0))));
        std.debug.print("key_cache[block,0,0,0] first {d} blocks:", .{max_blocks});
        for (0..max_blocks) |b| {
            const idx = b * block_stride;
            const byte_index = idx * 2;
            const lane = @as(u16, bytes[byte_index]) | (@as(u16, bytes[byte_index + 1]) << 8);
            const f = bf16ToF32(lane);
            std.debug.print(" {d}", .{f});
        }
        std.debug.print("\n", .{});
    }

    // Debug: show start_loc, context_seq_lens, and block_tables first column.
    {
        var start_loc_host = try start_loc.toSliceAlloc(allocator, io);
        defer start_loc_host.free(allocator);
        const start_items = start_loc_host.items(i32);
        const max_start = @min(@as(usize, 9), start_items.len);
        std.debug.print("start_loc[0..{d}]:", .{max_start});
        for (0..max_start) |i| std.debug.print(" {d}", .{start_items[i]});
        std.debug.print("\n", .{});
    }
    {
        var ctx_host = try context_seq_lens.toSliceAlloc(allocator, io);
        defer ctx_host.free(allocator);
        const ctx_items = ctx_host.items(i32);
        const max_ctx = @min(@as(usize, 8), ctx_items.len);
        std.debug.print("context_seq_lens[0..{d}]:", .{max_ctx});
        for (0..max_ctx) |i| std.debug.print(" {d}", .{ctx_items[i]});
        std.debug.print("\n", .{});
    }
    {
        var bt_host = try block_tables.toSliceAlloc(allocator, io);
        defer bt_host.free(allocator);
        const bt_items = bt_host.items(i32);
        std.debug.print("block_tables[:,0] first 8:", .{});
        for (0..8) |b| {
            const idx = b * block_table_width;
            std.debug.print(" {d}", .{bt_items[idx]});
        }
        std.debug.print("\n", .{});
    }

    // Debug: show value_cache ramp at [block,0,0,0] for first few blocks.
    {
        var value_cache_host = try value_cache.toSliceAlloc(allocator, io);
        defer value_cache_host.free(allocator);
        const bytes = value_cache_host.constData();
        const block_size: usize = @intCast(value_cache_shape.dim(1));
        const num_kv_heads: usize = @intCast(value_cache_shape.dim(2));
        const head_size: usize = @intCast(value_cache_shape.dim(3));
        const block_stride = block_size * num_kv_heads * head_size;
        const max_blocks = @min(@as(usize, 8), @as(usize, @intCast(value_cache_shape.dim(0))));
        std.debug.print("value_cache[block,0,0,0] first {d} blocks:", .{max_blocks});
        for (0..max_blocks) |b| {
            const idx = b * block_stride;
            const byte_index = idx * 2;
            const lane = @as(u16, bytes[byte_index]) | (@as(u16, bytes[byte_index + 1]) << 8);
            const f = bf16ToF32(lane);
            std.debug.print(" {d}", .{f});
        }
        std.debug.print("\n", .{});
    }

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{
        query,
        key_cache,
        value_cache,
        start_loc,
        context_seq_lens,
        block_tables,
        out,
    });

    exe.call(exe_args, &exe_results);
    var result: zml.Buffer = exe_results.get(zml.Buffer);
    defer result.deinit();

    var output = try result.toSliceAlloc(allocator, io);
    defer output.free(allocator);

    const output_items = output.constItems(bf16);
    const n = @min(output_items.len, 8);
    log.info("Output shape: {f}", .{result.shape()});
    std.debug.print("Output sample (first {d} bf16 lanes as u16):", .{n});
    for (0..n) |i| {
        std.debug.print(" 0x{x:0>4}", .{@as(u16, @bitCast(output_items[i]))});
    }
    std.debug.print("\n", .{});

    const d0: usize = @intCast(result.shape().dim(0));
    const d1: usize = @intCast(result.shape().dim(1));
    const d2: usize = @intCast(result.shape().dim(2));
    const out_strides = result.shape().computeElementStrides();
    std.debug.print("Output strides (bytes): {d} {d} {d}\n", .{
        out_strides.get(0),
        out_strides.get(1),
        out_strides.get(2),
    });

    std.debug.print("Output o[:,0,0] (bf16->f32):", .{});
    for (0..d0) |i| {
        const f = loadOutBF16(output_items, out_strides, i, 0, 0);
        std.debug.print(" {d}", .{f});
    }
    std.debug.print("\n", .{});

    std.debug.print("Output o[0,:,0] (bf16->f32):", .{});
    for (0..d1) |j| {
        const f = loadOutBF16(output_items, out_strides, 0, j, 0);
        std.debug.print(" {d}", .{f});
    }
    std.debug.print("\n", .{});

    std.debug.print("Output o[0,0,:8] (bf16->f32):", .{});
    const n_d2 = @min(@as(usize, 8), d2);
    for (0..n_d2) |k| {
        const f = loadOutBF16(output_items, out_strides, 0, 0, k);
        std.debug.print(" {d}", .{f});
    }
    std.debug.print("\n", .{});
}

fn loadOutBF16(items: []const bf16, strides: anytype, i: usize, j: usize, k: usize) f32 {
    const base = @as(usize, @intCast(strides.get(0))) * i +
        @as(usize, @intCast(strides.get(1))) * j +
        @as(usize, @intCast(strides.get(2))) * k;
    return items[base].toF32();
}

fn zeroBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    @memset(slice.data(), 0);
    return zml.Buffer.fromSlice(io, platform, slice);
}

fn tokenRampBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    const d1: usize = @intCast(shape.dim(1));
    const d2: usize = @intCast(shape.dim(2));
    const stride0 = d1 * d2;

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .float => {
                    for (slice.items(ZigType), 0..) |*e, i| {
                        const token: f32 = @floatFromInt(i / stride0);
                        e.* = switch (ZigType) {
                            f64, f32 => @floatCast(token),
                            f16 => @floatCast(token),
                            inline else => |T| if (@hasDecl(T, "fromF32")) T.fromF32(token) else unreachable,
                        };
                    }
                },
                else => unreachable,
            }
        },
    }

    return zml.Buffer.fromSlice(io, platform, slice);
}

fn onesBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => {
                    for (slice.items(ZigType)) |*e| e.* = true;
                },
                .integer => {
                    for (slice.items(ZigType)) |*e| e.* = 1;
                },
                .float => {
                    for (slice.items(ZigType)) |*e| e.* = switch (ZigType) {
                        f64, f32 => 1.0,
                        f16 => 1.0,
                        inline else => |T| if (@hasDecl(T, "fromF32")) T.fromF32(1.0) else unreachable,
                    };
                },
                .complex => unreachable,
            }
        },
    }

    return zml.Buffer.fromSlice(io, platform, slice);
}

fn valueCacheBlockRampBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    const block_size: usize = @intCast(shape.dim(1));
    const num_kv_heads: usize = @intCast(shape.dim(2));
    const head_size: usize = @intCast(shape.dim(3));
    const block_stride = block_size * num_kv_heads * head_size;

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .float => {
                    for (slice.items(ZigType), 0..) |*e, i| {
                        const block_idx: f32 = @floatFromInt(i / block_stride);
                        e.* = switch (ZigType) {
                            f64, f32 => @floatCast(block_idx),
                            f16 => @floatCast(block_idx),
                            inline else => |T| if (@hasDecl(T, "fromF32")) T.fromF32(block_idx) else unreachable,
                        };
                    }
                },
                else => unreachable,
            }
        },
    }

    return zml.Buffer.fromSlice(io, platform, slice);
}

fn uninitBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    return zml.Buffer.fromSlice(io, platform, slice);
}

fn startLocBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{test_cfg.batch_size + 1}, .i32));
    defer slice.free(allocator);

    for (slice.items(i32), 0..) |*v, i| v.* = @intCast(i);
    return zml.Buffer.fromSlice(io, platform, slice);
}

fn contextSeqLensBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{test_cfg.batch_size}, .i32));
    defer slice.free(allocator);

    @memset(slice.items(i32), 1);
    return zml.Buffer.fromSlice(io, platform, slice);
}

fn blockTablesBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, width: usize) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{ test_cfg.batch_size, width }, .i32));
    defer slice.free(allocator);

    @memset(slice.items(i32), 0);
    const bt = slice.items(i32);
    for (0..test_cfg.batch_size) |b| {
        bt[b * width] = @intCast(b);
    }

    return zml.Buffer.fromSlice(io, platform, slice);
}
