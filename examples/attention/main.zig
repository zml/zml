const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const ttir = @embedFile("kernel_unified_attention_2d_ptr.ttir");

//def select_2d_config(
//    block_size: int,
//    head_size: int,
//    all_decode: bool,
//    max_seqlen_q: int,
//    num_queries_per_kv: int,
//) -> dict:
//    block_m = 16 if num_queries_per_kv <= 16 else next_power_of_2(num_queries_per_kv)
//    tile_size = 64
//    max_num_stages_2d = 4 if head_size <= 128 else 2
//
//    if not all_decode:
//        num_stages_2d = 1
//        num_warps = 2
//    else:
//        num_stages_2d = 3
//        num_warps = 2
//        tile_size = block_size
//
//    if max_seqlen_q >= 256:
//        block_m = 128
//        num_stages_2d = 1
//        num_warps = 4
//
//    block_q = max(1, block_m // num_queries_per_kv)
//    return {
//        "BLOCK_M": block_m,
//        "BLOCK_Q": block_q,
//        "TILE_SIZE": tile_size,
//        "num_warps": num_warps,
//        "num_stages": min(max_num_stages_2d, num_stages_2d),
//    }

pub const Config2D = struct {
    block_m: usize,
    block_q: usize,
    tile_size: usize,
    num_warps: usize,
    num_stages: usize,
};

fn select2dConfig(block_size: usize, head_size: usize, all_decode: bool, max_seqlen_q: usize, num_queries_per_kv: usize) Config2D {
    var block_m: usize = if (num_queries_per_kv <= 16)
        16
    else
        std.math.ceilPowerOfTwoAssert(usize, num_queries_per_kv);

    const max_num_stages_2d: usize = if (head_size <= 128) 4 else 2;

    var num_stages_2d: usize, var num_warps: usize, const tile_size: usize = if (!all_decode) .{ 1, 2, 64 } else .{ 3, 2, block_size };

    if (max_seqlen_q >= 256) {
        block_m = 128;
        num_stages_2d = 1;
        num_warps = 4;
    }

    const block_q = @max(1, @divFloor(block_m, num_queries_per_kv));
    return .{
        .block_m = block_m,
        .block_q = block_q,
        .tile_size = tile_size,
        .num_warps = num_warps,
        .num_stages = @min(max_num_stages_2d, num_stages_2d),
    };
}

pub const Attention = struct {
    pub fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, block_table: zml.Tensor, seq_lens: zml.Tensor, query_start_len: zml.Tensor) zml.Tensor {
        const dummy = zml.Tensor.constant(zml.DataType.i8.zero());
        const block_table_strides = block_table.shape().computeElementStrides().constSlice();
        const block_table_strides_ptr = zml.Tensor.constant(zml.DataType.i64.constant(block_table_strides[0]));
        const q_strides = q.shape().computeElementStrides().constSlice();
        const q_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[0]));
        const q_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[1]));
        const k_strides = k.shape().computeElementStrides().constSlice();
        const k_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[0]));
        const k_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[1]));
        const k_strides_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[2]));
        const v_strides = k.shape().computeElementStrides().constSlice();
        const v_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[0]));
        const v_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[1]));
        const v_strides_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[2]));
        const num_seqs_ptr = zml.Tensor.constant(zml.DataType.i64.constant(block_table.dim(0)));
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
        const scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(scale));

        const num_queries_per_kv: usize = @intCast(@divExact(q.dim(.h), k.dim(.h)));
        const config = select2dConfig(@intCast(k.dim(.blocks)), @intCast(q.dim(.hd)), false, 128, num_queries_per_kv);

        const num_kv_heads: i32 = @intCast(k.dim(.h));
        const num_tokens: usize = @intCast(q.dim(0));
        const num_seqs: usize = @intCast(block_table.dim(0));
        const total_num_q_blocks: i32 = @intCast(num_tokens / config.block_q + num_seqs);

        const output = zml.ops.triton(.{
            q,
            k,
            v,
            dummy, // sink_ptr
            block_table,
            seq_lens,
            dummy, // alibi_slopes_ptr
            dummy, // qq_bias_ptr
            scale_ptr,
            dummy, // k_scale_ptr
            dummy, // v_scale_ptr
            dummy, // out_scale_ptr
            dummy, // softcap_ptr
            block_table_strides_ptr,
            q_strides_0_ptr,
            q_strides_1_ptr,
            q_strides_0_ptr,
            q_strides_1_ptr,
            dummy, // qq_bias_stride_0_ptr
            k_strides_0_ptr,
            k_strides_1_ptr,
            k_strides_2_ptr,
            v_strides_0_ptr,
            v_strides_1_ptr,
            v_strides_2_ptr,
            query_start_len,
            num_seqs_ptr,
        }, .{q.shape()}, .{
            .debug = false,
            .name = "kernel_unified_attention_2d_ptr",
            .ir = ttir,
            .grid = .{ num_kv_heads, total_num_q_blocks, 1 },
            .num_stages = @intCast(config.num_stages),
            .num_warps = @intCast(config.num_warps),
        });
        return output[0];
    }
};

pub fn main(init: std.process.Init) !void {
    //const CliArgs = struct {
    //    pub const help =
    //        \\ benchmark --size=4096 --dtype=f16
    //    ;
    //    size: usize = 4096,
    //    dtype: zml.DataType = .f16,
    //};

    const allocator = init.gpa;
    const io = init.io;

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const q: zml.Tensor = .init(.{ .s = 128, .h = 32, .hd = 128 }, .bf16);
    const k_cache: zml.Tensor = .init(.{ .pages = 64, .blocks = 16, .h = 8, .hd = 128 }, .bf16);
    const v_cache: zml.Tensor = .init(.{ .pages = 64, .blocks = 16, .h = 8, .hd = 128 }, .bf16);
    const block_table: zml.Tensor = .init(.{ .b = 1, .num_pages = 64 }, .i32);
    const seq_lens: zml.Tensor = .init(.{ .b = 1 }, .i32);
    const query_start_len: zml.Tensor = .init(.{ .b = 2 }, .i32);

    var exe = try platform.compileFn(allocator, io, Attention.forward, .{ q, k_cache, v_cache, block_table, seq_lens, query_start_len });
    defer exe.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    const batch_size: usize = @intCast(seq_lens.dim(.b));
    const query_len = @divExact(128, batch_size);
    const context_len = 0;
    const seqlen = context_len + query_len;

    var q_buffer = try createRandomBuffer(allocator, io, platform, q.shape(), random);
    defer q_buffer.deinit();
    var k_cache_buffer = try createRandomBuffer(allocator, io, platform, k_cache.shape(), random);
    defer k_cache_buffer.deinit();
    var v_cache_buffer = try createRandomBuffer(allocator, io, platform, v_cache.shape(), random);
    defer v_cache_buffer.deinit();
    var seq_lens_buffer = b: {
        const slice: zml.Slice = try .alloc(allocator, seq_lens.shape());
        defer slice.free(allocator);
        @memset(slice.items(i32), @intCast(seqlen));

        break :b try zml.Buffer.fromSlice(io, platform, slice);
    };
    defer seq_lens_buffer.deinit();

    var query_start_len_buffer = b: {
        const slice: zml.Slice = try .alloc(allocator, query_start_len.shape());
        defer slice.free(allocator);
        slice.items(i32)[0] = 0;
        for (1..batch_size) |i| {
            slice.items(i32)[i + 1] = slice.items(i32)[i] + @as(i32, @intCast(query_len));
        }
        break :b try zml.Buffer.fromSlice(io, platform, slice);
    };
    defer query_start_len_buffer.deinit();

    var block_table_buffer = b: {
        const slice: zml.Slice = try .alloc(allocator, block_table.shape());
        defer slice.free(allocator);

        for (0..batch_size) |i| {
            const max_num_pages: usize = @intCast(block_table.dim(.num_pages));
            const current_block_table = slice.items(i32)[i * max_num_pages ..][0..max_num_pages];
            @memset(current_block_table, std.math.maxInt(i32));
            const num_pages = (seqlen + 16 - 1) / 16;
            for (0..num_pages) |j| {
                current_block_table[j] = @intCast(j);
            }
        }

        break :b try zml.Buffer.fromSlice(io, platform, slice);
    };
    defer block_table_buffer.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ q_buffer, k_cache_buffer, v_cache_buffer, block_table_buffer, seq_lens_buffer, query_start_len_buffer });
    exe.call(args, &results);

    var output = results.get(zml.Buffer);
    defer output.deinit();

    const slice = try output.toSliceAlloc(allocator, io);
    defer slice.free(allocator);

    std.log.info("slice: {d}", .{slice});
}

fn createRandomBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, random: std.Random) !zml.Buffer {
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
                        const value = random.float(f32);
                        e.* = switch (ZigType) {
                            f64, f32 => value,
                            f16 => @floatCast(value),
                            inline else => |T| if (@hasDecl(T, "fromF32")) T.fromF32(value) else unreachable,
                        };
                    }
                },
                .complex => unreachable,
            }
        },
    }

    return .fromSlice(io, platform, slice);
}
