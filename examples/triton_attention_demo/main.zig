const std = @import("std");
const log = std.log;

const zml = @import("zml");
const attention = zml.attention.attention;
const triton_backend = zml.attention.triton;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const cfg = struct {
    const batch = 1;
    const token_count = 16;
    const num_heads = 8;
    const num_kv_heads = 8;
    const head_dim = 64;
};

fn simpleAttentionForward(kernel_ttir: [:0]const u8, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const metadata: attention.Metadata = .init(.{ .triton = .{} });
    const parameters: attention.Parameters = .init(.{ .triton = .{ .kernel_ttir = kernel_ttir } });
    return attention.attention(q, k, v, token_index, metadata, parameters);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    if (platform.target != .cuda and platform.target != .rocm) {
        log.err("triton_attention_demo requires CUDA/ROCm target, got {s}", .{@tagName(platform.target)});
        return;
    }

    const kernel_ttir = try triton_backend.compilePrefillAttentionTtir(allocator, io, .{
        .batch = cfg.batch,
        .seq_len = cfg.token_count,
        .num_query_heads = cfg.num_heads,
        .num_kv_heads = cfg.num_kv_heads,
        .head_size = cfg.head_dim,
        .causal = true,
    });
    defer allocator.free(kernel_ttir);

    const q_shape: zml.Tensor = .init(.{ .q = cfg.token_count, .h = cfg.num_heads, .hd = cfg.head_dim }, .f16);
    const k_shape: zml.Tensor = .init(.{ .k = cfg.token_count, .h = cfg.num_kv_heads, .hd = cfg.head_dim }, .f16);
    const v_shape: zml.Tensor = .init(.{ .k = cfg.token_count, .h = cfg.num_kv_heads, .hd = cfg.head_dim }, .f16);
    const token_index_shape: zml.Tensor = .init(.{}, .u32);

    var exe = try platform.compileFn(allocator, io, simpleAttentionForward, .{
        kernel_ttir,
        q_shape,
        k_shape,
        v_shape,
        token_index_shape,
    });
    defer exe.deinit();

    var q = try filledF16Buffer(allocator, io, platform, q_shape.shape(), 0.1);
    defer q.deinit();
    var k = try filledF16Buffer(allocator, io, platform, k_shape.shape(), 0.2);
    defer k.deinit();
    var v = try filledF16Buffer(allocator, io, platform, v_shape.shape(), 0.3);
    defer v.deinit();
    var token_index = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32);
    defer token_index.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ q, k, v, token_index });
    exe.call(exe_args, &exe_results);

    var result: zml.Buffer = exe_results.get(zml.Buffer);
    defer result.deinit();

    var host = try result.toSliceAlloc(allocator, io);
    defer host.free(allocator);

    const out = host.constItems(f16);
    const n = @min(@as(usize, 8), out.len);
    log.info("Output shape: {f}", .{result.shape()});
    std.debug.print("First {d} output values:", .{n});
    for (0..n) |i| {
        const v32: f32 = @floatCast(out[i]);
        std.debug.print(" {d:.5}", .{v32});
    }
    std.debug.print("\n", .{});
}

fn filledF16Buffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, scale: f32) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    const items = slice.items(f16);
    for (items, 0..) |*x, i| {
        const f: f32 = @as(f32, @floatFromInt(i % 23)) * scale;
        x.* = @floatCast(f);
    }

    return zml.Buffer.fromSlice(io, platform, slice);
}
