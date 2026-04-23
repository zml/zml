const std = @import("std");

const zml = @import("zml");
const bf16 = zml.floats.BFloat16;

const log = std.log.scoped(.neuron_attention);

const llama_num_heads = 32;
const llama_num_kv_heads = 8;
const llama_head_dim = 64;
const llama_prefill_seq_len = 128;

const default_decode_num_heads = 32;
const default_decode_num_kv_heads = 8;
const default_decode_head_dim = 64;
const default_decode_prior_len = 255;

const DecodeConfig = struct {
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    prior_len: i64,
};

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/module", .level = .info },
        .{ .scope = .@"zml/attention/neuron", .level = .info },
    },
};

const Attention = struct {
    fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
        return zml.attention.attention.attention(
            q,
            k,
            v,
            token_index,
            .{ .neuron = {} },
            .{ .neuron = {} },
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const decode_cfg = try readDecodeConfig(init.environ_map.*);

    const platform: *zml.Platform = try .init(allocator, io, .neuron, .{});
    defer platform.deinit(allocator, io);

    const sharding = try zml.sharding.replicatedSharding(platform);

    // try runPrefillSmoke(allocator, io, platform, sharding);
    try runDecodeSmoke(allocator, io, platform, sharding, decode_cfg);

    log.info("Neuron attention example completed on {s}", .{@tagName(platform.target)});
}

fn runPrefillSmoke(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
) !void {
    const q: zml.Tensor = zml.Tensor.init(.{
        .q = llama_prefill_seq_len,
        .h = llama_num_heads,
        .hd = llama_head_dim,
    }, .bf16);
    const k: zml.Tensor = zml.Tensor.init(.{
        .k = llama_prefill_seq_len,
        .h = llama_num_kv_heads,
        .hd = llama_head_dim,
    }, .bf16);
    const v: zml.Tensor = zml.Tensor.init(.{
        .k = llama_prefill_seq_len,
        .h = llama_num_kv_heads,
        .hd = llama_head_dim,
    }, .bf16);
    const token_index: zml.Tensor = .init(.{}, .u32);

    var exe = try platform.compileFn(allocator, io, Attention.forward, .{ q, k, v, token_index }, .{
        .shardings = &.{ sharding, sharding, sharding, sharding },
    });
    defer exe.deinit();

    var q_host: [llama_prefill_seq_len * llama_num_heads * llama_head_dim]bf16 = undefined;
    var k_host: [llama_prefill_seq_len * llama_num_kv_heads * llama_head_dim]bf16 = undefined;
    var v_host: [llama_prefill_seq_len * llama_num_kv_heads * llama_head_dim]bf16 = undefined;
    fillBf16(&q_host, 0.05);
    fillBf16(&k_host, 0.15);
    fillBf16(&v_host, 0.25);

    var q_buffer = try zml.Buffer.fromBytes(io, platform, q.shape(), sharding, std.mem.sliceAsBytes(&q_host));
    defer q_buffer.deinit();
    var k_buffer = try zml.Buffer.fromBytes(io, platform, k.shape(), sharding, std.mem.sliceAsBytes(&k_host));
    defer k_buffer.deinit();
    var v_buffer = try zml.Buffer.fromBytes(io, platform, v.shape(), sharding, std.mem.sliceAsBytes(&v_host));
    defer v_buffer.deinit();
    var token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32, sharding);
    defer token_index_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, Attention.forward, .{
        q_buffer,
        k_buffer,
        v_buffer,
        token_index_buffer,
    });
    defer output.deinit();

    log.info("Ran Neuron prefill attention path with Llama 3.2 1B dims q={} h={} kv_h={} hd={} output={f}", .{
        llama_prefill_seq_len,
        llama_num_heads,
        llama_num_kv_heads,
        llama_head_dim,
        output.shape(),
    });
}

fn runDecodeSmoke(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    cfg: DecodeConfig,
) !void {
    const q: zml.Tensor = zml.Tensor.init(.{
        .b = 1,
        .q = 1,
        .h = cfg.num_heads,
        .hd = cfg.head_dim,
    }, .bf16);
    const k: zml.Tensor = zml.Tensor.init(.{
        .b = 1,
        .k = cfg.prior_len + 1,
        .h = cfg.num_kv_heads,
        .hd = cfg.head_dim,
    }, .bf16);
    const v: zml.Tensor = zml.Tensor.init(.{
        .b = 1,
        .k = cfg.prior_len + 1,
        .h = cfg.num_kv_heads,
        .hd = cfg.head_dim,
    }, .bf16);
    const token_index: zml.Tensor = .init(.{}, .u32);

    var exe = try platform.compileFn(allocator, io, Attention.forward, .{ q, k, v, token_index }, .{
        .shardings = &.{ sharding, sharding, sharding, sharding },
    });
    defer exe.deinit();

    const q_host = try allocator.alloc(bf16, @intCast(cfg.num_heads * cfg.head_dim));
    defer allocator.free(q_host);
    const k_host = try allocator.alloc(bf16, @intCast((cfg.prior_len + 1) * cfg.num_kv_heads * cfg.head_dim));
    defer allocator.free(k_host);
    const v_host = try allocator.alloc(bf16, @intCast((cfg.prior_len + 1) * cfg.num_kv_heads * cfg.head_dim));
    defer allocator.free(v_host);
    fillBf16(q_host, 0.01);
    fillBf16(k_host, 0.02);
    fillBf16(v_host, 0.03);

    var q_buffer = try zml.Buffer.fromBytes(io, platform, q.shape(), sharding, std.mem.sliceAsBytes(q_host));
    defer q_buffer.deinit();
    var k_buffer = try zml.Buffer.fromBytes(io, platform, k.shape(), sharding, std.mem.sliceAsBytes(k_host));
    defer k_buffer.deinit();
    var v_buffer = try zml.Buffer.fromBytes(io, platform, v.shape(), sharding, std.mem.sliceAsBytes(v_host));
    defer v_buffer.deinit();
    var token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(cfg.prior_len)), .u32, sharding);
    defer token_index_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, Attention.forward, .{
        q_buffer,
        k_buffer,
        v_buffer,
        token_index_buffer,
    });
    defer output.deinit();

    log.info("Ran dense full-cache Neuron decode path prior_k={} full_k={} h={} kv_h={} hd={} output={f}", .{
        cfg.prior_len,
        cfg.prior_len + 1,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        output.shape(),
    });
}

fn readDecodeConfig(environ_map: std.process.Environ.Map) !DecodeConfig {
    return .{
        .num_heads = try readEnvInt(environ_map, "ZML_NEURON_DECODE_HEADS", default_decode_num_heads),
        .num_kv_heads = try readEnvInt(environ_map, "ZML_NEURON_DECODE_KV_HEADS", default_decode_num_kv_heads),
        .head_dim = try readEnvInt(environ_map, "ZML_NEURON_DECODE_HEAD_DIM", default_decode_head_dim),
        .prior_len = try readEnvInt(environ_map, "ZML_NEURON_DECODE_PRIOR_LEN", default_decode_prior_len),
    };
}

fn readEnvInt(environ_map: std.process.Environ.Map, name: []const u8, default_value: i64) !i64 {
    const value = environ_map.get(name) orelse return default_value;
    return try std.fmt.parseInt(i64, value, 10);
}

fn fillBf16(data: []bf16, offset: f32) void {
    for (data, 0..) |*elem, i| {
        const x: f32 = @as(f32, @floatFromInt(@mod(i, 23))) * 0.03125 + offset;
        elem.* = bf16.fromF32(x);
    }
}
