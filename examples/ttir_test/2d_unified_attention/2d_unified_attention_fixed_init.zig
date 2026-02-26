const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;
const bf16 = zml.floats.BFloat16;
const testing = zml.testing;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const inputs_bytes = @embedFile("safetensors/2d_unified_attention_inputs.safetensors");
const outputs_bytes = @embedFile("safetensors/2d_unified_attention_output.safetensors");

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

    const inputs_path = try writeEmbeddedSafetensors(allocator, io, inputs_bytes, "2d_unified_attention_inputs.safetensors");
    defer allocator.free(inputs_path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, inputs_path);
    defer registry.deinit();

    const outputs_path = try writeEmbeddedSafetensors(allocator, io, outputs_bytes, "2d_unified_attention_output.safetensors");
    defer allocator.free(outputs_path);
    var outputs_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, outputs_path);
    defer outputs_registry.deinit();

    var query = try loadBufferFromRegistry(allocator, io, platform, &registry, "query");
    defer query.deinit();
    var key_cache = try loadBufferFromRegistry(allocator, io, platform, &registry, "key_cache");
    defer key_cache.deinit();
    var value_cache = try loadBufferFromRegistry(allocator, io, platform, &registry, "value_cache");
    defer value_cache.deinit();
    var block_tables = try loadBufferFromRegistry(allocator, io, platform, &registry, "block_tables");
    defer block_tables.deinit();
    var context_seq_lens = try loadBufferFromRegistry(allocator, io, platform, &registry, "context_seq_lens");
    defer context_seq_lens.deinit();
    var start_loc = try loadBufferFromRegistry(allocator, io, platform, &registry, "start_loc");
    defer start_loc.deinit();
    var out = try uninitBuffer(allocator, io, platform, out_shape.shape());
    defer out.deinit();
    var expected = try loadBufferFromRegistry(allocator, io, platform, &outputs_registry, "out");
    defer expected.deinit();

    log.info("query shape: {f}", .{query.shape()});
    log.info("key_cache shape: {f}", .{key_cache.shape()});
    log.info("value_cache shape: {f}", .{value_cache.shape()});
    log.info("block_tables shape: {f}", .{block_tables.shape()});
    log.info("context_seq_lens shape: {f}", .{context_seq_lens.shape()});
    log.info("start_loc shape: {f}", .{start_loc.shape()});


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

    const out_items = output.constItems(bf16);
    const out_strides = result.shape().computeElementStrides();
    log.info("Output shape: {f}", .{result.shape()});
    std.debug.print("Output o[:,0,0] first 8 (bf16->f32):", .{});
    const d0: usize = @intCast(result.shape().dim(0));
    const n = @min(@as(usize, 8), d0);
    for (0..n) |i| {
        const f = loadOutBF16(out_items, out_strides, i, 0, 0);
        std.debug.print(" {d:.5}", .{f});
    }
    std.debug.print("\n", .{});

    var matches = true;
    testing.expectClose(io, result, expected, .{}) catch {
        matches = false;
    };
    std.debug.print("\n\n", .{});
    if (matches) {
        std.debug.print("Output matches expected tensor\n", .{});
    } else {
        std.debug.print("Output does not match expected tensor\n", .{});
    }
}

fn writeEmbeddedSafetensors(allocator: std.mem.Allocator, io: std.Io, bytes: []const u8, filename: []const u8) ![]const u8 {
    const path = filename;
    const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer file.close(io);

    var writer = file.writer(io, &.{});
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();

    var real_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const real_len = try file.realPath(io, &real_buf);
    return try allocator.dupe(u8, real_buf[0..real_len]);
}

fn loadBufferFromRegistry(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    registry: *zml.safetensors.TensorRegistry,
    key: []const u8,
) !zml.Buffer {
    const tensor_desc = registry.tensors.get(key) orelse return error.NotFound;
    const shape = tensor_desc.shape;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try registry.reader(io, key, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, host_bytes);
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
