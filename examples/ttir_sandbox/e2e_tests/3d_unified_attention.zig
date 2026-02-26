const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;
const ttir_compile_sandbox = @import("ttir_compile_sandbox");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const cfg = struct {
    const token_count = 8;
    const batch_size = 8;
    const num_heads = 32;
    const num_kv_heads = 8;
    const head_size = 128;
    const head_size_padded = 128;
    const block_size = 16;
    const num_blocks = 4096;
    const max_input_len = 1;
    const max_seq_len = block_size;
    const num_segments = 8;
    const scale = 0.08838834765;
    const k_scale = 1.0;
    const v_scale = 1.0;
};

pub fn wrappedUnifiedAttention3D(
    kernel_ttir: [:0]const u8,
    reduce_ttir: [:0]const u8,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    block_table: Tensor,
    segm_output: Tensor,
    segm_max: Tensor,
    segm_expsum: Tensor,
    out: Tensor,
) Tensor {
    const q_strides = q.shape().computeElementStrides();
    const k_strides = k.shape().computeElementStrides();
    const v_strides = v.shape().computeElementStrides();
    const bt_strides = block_table.shape().computeElementStrides();
    const o_strides = out.shape().computeElementStrides();

    const num_seqs = Tensor.scalar(block_table.dim(0), .i32);
    const num_query_heads: i32 = @intCast(cfg.num_heads);
    const num_kv_heads: i32 = @intCast(cfg.num_kv_heads);
    const num_queries_per_kv: i32 = @divExact(num_query_heads, num_kv_heads);
    const block_m: i32 = if (num_queries_per_kv <= 16) 16 else @intCast(std.math.ceilPowerOfTwo(i32, num_queries_per_kv));
    const block_q: i32 = @divExact(block_m, num_queries_per_kv);
    const q_len: i64 = @intCast(q.dim(0));
    const num_seqs_i64: i64 = @intCast(block_table.dim(0));
    const total_num_q_blocks: i64 = @divFloor(q_len, @as(i64, block_q)) + num_seqs_i64;

    const grid_3d: [3]i32 = .{
        @intCast(total_num_q_blocks),
        num_kv_heads,
        cfg.num_segments,
    };

    const grid_reduce: [3]i32 = .{
        @intCast(q.dim(0)),
        num_query_heads,
        1,
    };

    const target = zml.module.CompilationContext.current().platform.target;
    const num_warps: i32 = switch (target) {
        .rocm => 1,
        .cuda => 4,
        else => 4,
    };

    const segm_results = zml.ops.triton(.{
        q,
        k,
        v,
        block_table,
        seqused_k,
        Tensor.scalar(cfg.scale, .f32),
        Tensor.scalar(cfg.k_scale, .f32),
        Tensor.scalar(cfg.v_scale, .f32),
        Tensor.scalar(0.0, .f32), // softcap
        Tensor.scalar(bt_strides.get(0), .i64),
        Tensor.scalar(q_strides.get(0), .i64),
        Tensor.scalar(q_strides.get(1), .i64),
        Tensor.scalar(0, .i64), // qq_bias_stride_0
        Tensor.scalar(k_strides.get(0), .i64),
        Tensor.scalar(k_strides.get(1), .i64),
        Tensor.scalar(k_strides.get(2), .i64),
        Tensor.scalar(v_strides.get(0), .i64),
        Tensor.scalar(v_strides.get(1), .i64),
        Tensor.scalar(v_strides.get(2), .i64),
        cu_seqlens_q,
        num_seqs,
    }, .{
        segm_max.shape(),
        segm_expsum.shape(),
        segm_output.shape(),
    }, .{
        .name = "wrapped_kernel_unified_attention_3d",
        .ir = kernel_ttir,
        .grid = grid_3d,
        .num_stages = 1,
        .num_warps = num_warps,
        .debug = true,
        .output_operand_aliases = &.{},
    });

    return zml.ops.triton(.{
        segm_results[2],
        segm_results[0],
        segm_results[1],
        seqused_k,
        num_seqs,
        Tensor.scalar(1.0, .f32),
        Tensor.scalar(o_strides.get(0), .i64),
        Tensor.scalar(o_strides.get(1), .i64),
        Tensor.scalar(bt_strides.get(0), .i64),
        cu_seqlens_q,
    }, .{out.shape()}, .{
        .name = "wrapped_reduce_segments",
        .ir = reduce_ttir,
        .grid = grid_reduce,
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
        "\"num_seqs\":8," ++
        "\"seq_len_q\":1," ++
        "\"seq_len_k\":16," ++
        "\"num_query_heads\":32," ++
        "\"num_kv_heads\":8," ++
        "\"head_size\":128," ++
        "\"block_size\":16," ++
        "\"max_blocks_per_seq\":1," ++
        "\"num_par_softmax_segments\":8" ++
        "}";
    const kernel_ttir = try ttir_compile_sandbox.get3dUnifiedAttentionTtir(allocator, io, kernel_params);
    defer allocator.free(kernel_ttir);
    const reduce_ttir = try ttir_compile_sandbox.get3dReduceSegmentsTtir(allocator, io, kernel_params);
    defer allocator.free(reduce_ttir);

    const query_shape: zml.Tensor = .init(.{ cfg.token_count, cfg.num_heads, cfg.head_size }, .bf16);
    const key_cache_shape: zml.Tensor = .init(.{ cfg.num_blocks, cfg.block_size, cfg.num_kv_heads, cfg.head_size }, .bf16);
    const value_cache_shape: zml.Tensor = .init(.{ cfg.num_blocks, cfg.block_size, cfg.num_kv_heads, cfg.head_size }, .bf16);
    const out_shape: zml.Tensor = query_shape;
    const start_loc_shape: zml.Tensor = .init(.{cfg.batch_size + 1}, .i32);
    const context_seq_lens_shape: zml.Tensor = .init(.{cfg.batch_size}, .i32);
    const block_tables_shape: zml.Tensor = .init(.{ cfg.batch_size, @divExact(cfg.max_seq_len, cfg.block_size) }, .i32);
    const segm_output_shape: zml.Tensor = .init(.{ cfg.token_count, cfg.num_heads, cfg.num_segments, cfg.head_size_padded }, .f32);
    const segm_max_shape: zml.Tensor = .init(.{ cfg.token_count, cfg.num_heads, cfg.num_segments }, .f32);
    const segm_expsum_shape: zml.Tensor = .init(.{ cfg.token_count, cfg.num_heads, cfg.num_segments }, .f32);

    var exe = try platform.compileFn(allocator, io, wrappedUnifiedAttention3D, .{
        kernel_ttir,
        reduce_ttir,
        query_shape,
        key_cache_shape,
        value_cache_shape,
        start_loc_shape,
        context_seq_lens_shape,
        block_tables_shape,
        segm_output_shape,
        segm_max_shape,
        segm_expsum_shape,
        out_shape,
    });
    defer exe.deinit();

    var query = try zeroBuffer(allocator, io, platform, query_shape.shape());
    defer query.deinit();
    var key_cache = try zeroBuffer(allocator, io, platform, key_cache_shape.shape());
    defer key_cache.deinit();
    var value_cache = try zeroBuffer(allocator, io, platform, value_cache_shape.shape());
    defer value_cache.deinit();
    var out = try uninitBuffer(allocator, io, platform, out_shape.shape());
    defer out.deinit();

    var start_loc = try startLocBuffer(allocator, io, platform);
    defer start_loc.deinit();
    var context_seq_lens = try contextSeqLensBuffer(allocator, io, platform);
    defer context_seq_lens.deinit();
    var block_tables = try blockTablesBuffer(allocator, io, platform, @intCast(block_tables_shape.dim(1)));
    defer block_tables.deinit();

    var segm_output = try uninitBuffer(allocator, io, platform, segm_output_shape.shape());
    defer segm_output.deinit();
    var segm_max = try uninitBuffer(allocator, io, platform, segm_max_shape.shape());
    defer segm_max.deinit();
    var segm_expsum = try uninitBuffer(allocator, io, platform, segm_expsum_shape.shape());
    defer segm_expsum.deinit();

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
        segm_output,
        segm_max,
        segm_expsum,
        out,
    });

    exe.call(exe_args, &exe_results);
    var result: zml.Buffer = exe_results.get(zml.Buffer);
    defer result.deinit();

    var output = try result.toSliceAlloc(allocator, io);
    defer output.free(allocator);

    const output_bytes = output.constData();
    const n = @min(output_bytes.len / 2, 8);
    log.info("Output shape: {f}", .{result.shape()});
    std.debug.print("Output sample (first {d} bf16 lanes as u16):", .{n});
    for (0..n) |i| {
        const lane = @as(u16, output_bytes[i * 2]) | (@as(u16, output_bytes[i * 2 + 1]) << 8);
        std.debug.print(" 0x{x:0>4}", .{lane});
    }
    std.debug.print("\n", .{});

    const d0: usize = @intCast(result.shape().dim(0));
    const d1: usize = @intCast(result.shape().dim(1));
    const d2: usize = @intCast(result.shape().dim(2));
    const stride0 = d1 * d2;
    std.debug.print("Output o[:,0,0] (bf16->f32):", .{});
    for (0..d0) |i| {
        const idx = i * stride0;
        const byte_index = idx * 2;
        if (byte_index + 1 >= output_bytes.len) break;
        const lane = @as(u16, output_bytes[byte_index]) | (@as(u16, output_bytes[byte_index + 1]) << 8);
        const f = bf16ToF32(lane);
        std.debug.print(" {d}", .{f});
    }
    std.debug.print("\n", .{});
}

fn bf16ToF32(bits: u16) f32 {
    const word = @as(u32, bits) << 16;
    return @bitCast(word);
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

fn startLocBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{cfg.batch_size + 1}, .i32));
    defer slice.free(allocator);

    const items = slice.items(i32);
    for (0..items.len) |i| items[i] = @intCast(i);
    return zml.Buffer.fromSlice(io, platform, slice);
}

fn contextSeqLensBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{cfg.batch_size}, .i32));
    defer slice.free(allocator);

    for (slice.items(i32)) |*v| v.* = cfg.max_input_len;
    return zml.Buffer.fromSlice(io, platform, slice);
}

fn blockTablesBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, width: usize) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{ cfg.batch_size, width }, .i32));
    defer slice.free(allocator);

    @memset(slice.items(i32), 0);
    return zml.Buffer.fromSlice(io, platform, slice);
}
