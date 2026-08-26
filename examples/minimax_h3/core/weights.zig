const std = @import("std");

const zml = @import("zml");

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
    loader.load(io, T, model, buffers, store, shardings, .{ .progress = progress });
    try loader.await(io);
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

/// Rebind a compiled runner to the next streamed layer. `bake` is incremental;
/// reset the count or the previous layer stays bound.
pub fn rebake(runner: anytype, next: anytype) void {
    runner.args.baked_count = 0;
    runner.args.bake(next);
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
