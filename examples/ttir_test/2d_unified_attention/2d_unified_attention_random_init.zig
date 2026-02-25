const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;

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
    const q_strides = q.shape().computeByteStrides();
    const k_strides = k.shape().computeByteStrides();
    const v_strides = v.shape().computeByteStrides();
    const bt_strides = block_table.shape().computeByteStrides();

    // Kept for API parity with the torch-side call.
    const max_seqlen_q = test_cfg.max_input_len;
    const max_seqlen_k = test_cfg.max_seq_len;
    const causal = true;
    const window_size = [2]i32{ -1, -1 };
    const q_descale: ?f32 = null;
    _ = .{ max_seqlen_q, max_seqlen_k, causal, window_size, q_descale };

    const num_seqs = Tensor.scalar(block_table.dim(0), .i32);
    const grid: [3]i32 = .{ @intCast(test_cfg.batch_size), @intCast(test_cfg.num_kv_heads), 1 };
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

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    var query = try randomNormalBuffer(allocator, io, platform, query_shape.shape(), random);
    defer query.deinit();
    var key_cache = try randomNormalBuffer(allocator, io, platform, key_cache_shape.shape(), random);
    defer key_cache.deinit();
    var value_cache = try randomNormalBuffer(allocator, io, platform, value_cache_shape.shape(), random);
    defer value_cache.deinit();
    var out = try uninitBuffer(allocator, io, platform, out_shape.shape());
    defer out.deinit();

    var start_loc = try startLocBuffer(allocator, io, platform);
    defer start_loc.deinit();
    var context_seq_lens = try contextSeqLensBuffer(allocator, io, platform);
    defer context_seq_lens.deinit();
    var block_tables = try blockTablesBuffer(allocator, io, platform, block_table_width);
    defer block_tables.deinit();

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

fn randomNormalBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, random: std.Random) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    for (slice.items(ZigType)) |*e| e.* = random.int(ZigType);
                },
                .float => {
                    for (slice.items(ZigType)) |*e| {
                        const value = randomNormal(random);
                        e.* = switch (ZigType) {
                            f64, f32 => @floatCast(value),
                            f16 => @floatCast(value),
                            inline else => |T| if (@hasDecl(T, "fromF32")) T.fromF32(value) else unreachable,
                        };
                    }
                },
                .complex => unreachable,
            }
        },
    }

    return zml.Buffer.fromSlice(io, platform, slice);
}

fn randomNormal(random: std.Random) f32 {
    const u1_raw = random.float(f32);
    const u1f = if (u1_raw == 0) std.math.floatEps(f32) else u1_raw;
    const u2f = random.float(f32);
    const r = @sqrt(-2.0 * @log(u1f));
    const theta = @as(f32, @floatCast(std.math.pi)) * 2.0 * u2f;
    return r * @cos(theta);
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

    const bt = slice.items(i32);
    for (bt) |*v| v.* = 65535;
    for (0..test_cfg.batch_size) |b| {
        bt[b * width] = @intCast(b);
    }

    return zml.Buffer.fromSlice(io, platform, slice);
}
