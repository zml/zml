//! Shared generation context, DMA load, and weight constructors.
//! Model math lives in encoder / pack / dit / vae / audio.

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");

pub const loader_opts: zml.io.Loader.Opts = .{
    .dma_chunks = 8,
    .dma_chunk_size = 64 * zml.MiB,
    .parallelism = 8,
};

/// Allocator, IO, platform, mesh, and progress for one generation.
pub const Run = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shardings: config.Shardings,
    mesh: [1]zml.Sharding,
    progress: *std.Progress.Node,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        shardings: config.Shardings,
        progress: *std.Progress.Node,
    ) Run {
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .shardings = shardings,
            .mesh = .{shardings.model},
            .progress = progress,
        };
    }
};

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

pub fn rms(store: zml.io.TensorStore.View, tagz: anytype, eps: f32) zml.nn.RmsNorm {
    return .{ .weight = store.createTensor("weight", tagz, .replicated), .eps = eps };
}

pub fn ln(store: zml.io.TensorStore.View, eps: f32) zml.nn.LayerNorm {
    return .{
        .weight = store.createTensor("weight", .{.d}, .replicated),
        .bias = store.maybeCreateTensor("bias", .{.d}, .replicated),
        .eps = eps,
    };
}

/// Bufferize `m` and DMA weights from `store`. Pass a shared `loader` when
/// loading many layers in a loop so DMA stays pipelined.
pub fn load(
    run: *const Run,
    store: *zml.io.TensorStore,
    comptime T: type,
    m: *const T,
    loader: ?*zml.io.Loader,
) !zml.Bufferized(T) {
    var buffers = try zml.mem.bufferize(run.allocator, T, m);
    if (loader) |shared| {
        try shared.load(run.io, T, m, &buffers, store, &run.mesh, .{ .progress = run.progress });
        try shared.await(run.io);
        return buffers;
    }
    var owned: zml.io.Loader = try .init(run.allocator, run.platform, loader_opts);
    defer owned.deinit();
    try owned.load(run.io, T, m, &buffers, store, &run.mesh, .{ .progress = run.progress });
    try owned.await(run.io);
    return buffers;
}

/// `v ← v * std + mean` per latent channel.
pub fn applyLatentNorm(values: []f32, mean: []const f32, stddev: []const f32) void {
    const channels = mean.len;
    std.debug.assert(stddev.len == channels);
    std.debug.assert(values.len % channels == 0);
    for (0..values.len / channels) |row| {
        const pix = values[row * channels ..][0..channels];
        for (pix, mean, stddev) |*v, m, s| v.* = v.* * s + m;
    }
}

/// 3-axis MM-RoPE: concat t/h/w freqs, then duplicate.
pub fn ropeCat3(pos: zml.Tensor, inv: zml.Tensor) zml.Tensor {
    const parts = pos.convert(.f32).withPartialTags(.{ .s, .ax }).outer(inv).chunkExact(.ax, 3);
    const cat3 = zml.Tensor.concatenate(&.{ parts[0].squeeze(.ax), parts[1].squeeze(.ax), parts[2].squeeze(.ax) }, .f);
    return zml.Tensor.concatenate(&.{ cat3, cat3 }, .f);
}
