const std = @import("std");

const zml = @import("zml");

// =============================================================================
// recipe/weights.zig — TensorStore load helpers
// =============================================================================

/// Loader sized for one streamed transformer block (~768 MiB).
pub const loader_opts: zml.io.Loader.Opts = .{
    .dma_chunks = 8,
    .dma_chunk_size = 64 * zml.MiB,
    .parallelism = 8,
};

pub fn initLoader(allocator: std.mem.Allocator, platform: *const zml.Platform) !zml.io.Loader {
    return .init(allocator, platform, loader_opts);
}

pub fn populate(
    loader: *zml.io.Loader,
    io: std.Io,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    comptime T: type,
    model: *const T,
    buffers: *zml.Bufferized(T),
    progress: *std.Progress.Node,
) !void {
    try loader.load(io, T, model, buffers, store, shardings, .{ .progress = progress });
    try loader.await(io);
}

pub fn linear(
    store: zml.io.TensorStore.View,
    weight_name: []const u8,
    bias_name: ?[]const u8,
    partitions: anytype,
    bias_partitions: anytype,
) zml.nn.Linear {
    return .init(
        store.createTensor(weight_name, .{ .dout, .d }, partitions),
        if (bias_name) |name| store.maybeCreateTensor(name, .{.dout}, bias_partitions) else null,
        .d,
    );
}

pub fn layerNorm(store: zml.io.TensorStore.View, eps: f32) zml.nn.LayerNorm {
    return .{
        .weight = store.createTensor("weight", .{.d}, .replicated),
        .bias = store.maybeCreateTensor("bias", .{.d}, .replicated),
        .eps = eps,
    };
}

pub fn rmsNorm(store: zml.io.TensorStore.View, tagz: anytype, eps: f32) zml.nn.RmsNorm {
    return .{ .weight = store.createTensor("weight", tagz, .replicated), .eps = eps };
}

pub fn modelBytes(model: anytype) u64 {
    const Ctx = struct {
        n: u64 = 0,
        fn add(ctx: *@This(), t: *const zml.Tensor) void {
            ctx.n += t.shape().byteSize();
        }
    };
    var ctx: Ctx = .{};
    zml.meta.visit(Ctx.add, &ctx, model);
    return ctx.n;
}

pub fn load(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    comptime T: type,
    model: *const T,
    progress: *std.Progress.Node,
    loader: ?*zml.io.Loader,
) !zml.Bufferized(T) {
    var buffers = try zml.mem.bufferize(allocator, T, model);
    if (loader) |shared| {
        try populate(shared, io, store, shardings, T, model, &buffers, progress);
        return buffers;
    }
    var owned = try initLoader(allocator, platform);
    defer owned.deinit();
    try populate(&owned, io, store, shardings, T, model, &buffers, progress);
    return buffers;
}

pub fn fromItems(io: std.Io, platform: *const zml.Platform, shape: zml.Shape, items: anytype) !zml.Buffer {
    return fromItemsSharded(io, platform, shape, .replicated, items);
}

pub fn fromItemsSharded(
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
    items: anytype,
) !zml.Buffer {
    return zml.Buffer.fromBytes(io, platform, shape, sharding, std.mem.sliceAsBytes(items));
}

pub fn fromF32(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    values: []const f32,
) !zml.Buffer {
    return fromF32Sharded(allocator, io, platform, shape, .replicated, values);
}

pub fn fromF32Sharded(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
    values: []const f32,
) !zml.Buffer {
    switch (shape.dtype()) {
        .f32 => return fromItemsSharded(io, platform, shape, sharding, values),
        .bf16 => {
            const converted = try allocator.alloc(zml.floats.BFloat16, values.len);
            defer allocator.free(converted);
            for (converted, values) |*dst, src| dst.* = .fromF32(src);
            return fromItemsSharded(io, platform, shape, sharding, converted);
        },
        else => return error.UnsupportedEmbedDtype,
    }
}

pub fn toF32(allocator: std.mem.Allocator, io: std.Io, buffer: zml.Buffer) ![]f32 {
    const slice = try buffer.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    switch (buffer.shape().dtype()) {
        .f32 => return allocator.dupe(f32, slice.items(f32)),
        .bf16 => {
            const source = slice.items(zml.floats.BFloat16);
            const converted = try allocator.alloc(f32, source.len);
            for (converted, source) |*dst, value| dst.* = value.toF32();
            return converted;
        },
        else => return error.UnsupportedEmbedDtype,
    }
}
