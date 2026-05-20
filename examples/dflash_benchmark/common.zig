const std = @import("std");

const zml = @import("zml");

pub const SessionOptions = struct {
    seqlen: u32,
    backend: zml.attention.attention.Backend,
    single: bool,
};

pub const GenerationOptions = struct {
    sampling_strategy: zml.nn.SamplingStrategy = .{},
};

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: ?u32,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(kv_shape: zml.Shape) KvCache {
        const sharded_shape = kv_shape.withPartitioning(.{ .h = .model });
        return .{
            .k = .fromShape(sharded_shape),
            .v = .fromShape(sharded_shape),
            .layer_index = null,
        };
    }

    pub fn deinitBuffer(kv: *Buffer) void {
        kv.k.deinit();
        kv.v.deinit();
    }

    pub fn replaceBuffers(dst: *Buffer, src: *Buffer) void {
        replaceBuffer(&dst.k, &src.k);
        replaceBuffer(&dst.v, &src.v);
    }

    pub fn initZeroBuffer(
        kv: KvCache,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        sharding: zml.Sharding,
    ) !Buffer {
        return .{
            .k = try zeroBuffer(allocator, io, platform, kv.k.shape(), sharding),
            .v = try zeroBuffer(allocator, io, platform, kv.v.shape(), sharding),
        };
    }

    pub fn keys(kv: KvCache) zml.Tensor {
        return kv.k.slice1d(.layer, .single(kv.layer_index orelse @panic("forgot to call atLayer")));
    }

    pub fn values(kv: KvCache) zml.Tensor {
        return kv.v.slice1d(.layer, .single(kv.layer_index orelse @panic("forgot to call atLayer")));
    }

    pub fn update(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor) KvCache {
        const k_shape = kv.k.shape().drop(.layer);
        const layer: zml.Tensor = .scalar(kv.layer_index orelse @panic("forgot to call atLayer"), .u32);

        return .{
            .k = kv.k.scatterSlices(.{ .layer = layer, .k = token_index }, new_k.convert(kv.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.k),
            .v = kv.v.scatterSlices(.{ .layer = layer, .k = token_index }, new_v.convert(kv.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.v),
            .layer_index = kv.layer_index,
        };
    }

    pub fn atLayer(kv: KvCache, layer_index: usize) KvCache {
        return .{
            .k = kv.k,
            .v = kv.v,
            .layer_index = @intCast(layer_index),
        };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .k = kv.k.reuseBuffer(other.k),
            .v = kv.v.reuseBuffer(other.v),
            .layer_index = null,
        };
    }
};

pub const Shardings = struct {
    model: zml.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        return .{
            .model = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth })),
        };
    }

    pub fn all(self: Shardings) [1]zml.Sharding {
        return .{self.model};
    }
};

pub fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}

fn zeroBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
) !zml.Buffer {
    const bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    return zml.Buffer.fromSlice(io, platform, zml.Slice.init(shape, bytes), sharding);
}

fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
