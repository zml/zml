//! Shared generation context, DMA load, and weight constructors.
//! Model math lives in encoder / pack / dit / vae / audio / vision.

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

/// `x * std + mean` broadcast on `.c`.
pub fn denorm(x: zml.Tensor, mean: []const f32, stddev: []const f32) zml.Tensor {
    const mean_t = zml.Tensor.constantTensor(.init(.{ .c = mean.len }, .f32), std.mem.sliceAsBytes(mean));
    const std_t = zml.Tensor.constantTensor(.init(.{ .c = stddev.len }, .f32), std.mem.sliceAsBytes(stddev));
    return x.mul(std_t.broad(x.shape())).add(mean_t.broad(x.shape()));
}

/// 3-axis MM-RoPE: concat t/h/w freqs, then duplicate.
pub fn ropeCat3(pos: zml.Tensor, inv: zml.Tensor) zml.Tensor {
    const parts = pos.convert(.f32).withPartialTags(.{ .s, .ax }).outer(inv).chunkExact(.ax, 3);
    const cat3 = zml.Tensor.concatenate(&.{ parts[0].squeeze(.ax), parts[1].squeeze(.ax), parts[2].squeeze(.ax) }, .f);
    return zml.Tensor.concatenate(&.{ cat3, cat3 }, .f);
}

pub const TilePlan = struct {
    starts: []u32,
    overlaps: []u32,

    pub fn deinit(self: TilePlan, allocator: std.mem.Allocator) void {
        allocator.free(self.starts);
        allocator.free(self.overlaps);
    }
};

/// Evenly spaced tile origins along one axis, overlaps aligned to `align_to`.
pub fn splitTiles(allocator: std.mem.Allocator, length: u32, tile_size: u32, min_overlap: u32, align_to: u32) !TilePlan {
    if (tile_size >= length) {
        const starts = try allocator.alloc(u32, 1);
        starts[0] = 0;
        return .{ .starts = starts, .overlaps = try allocator.alloc(u32, 0) };
    }
    var num_tiles = std.math.divCeil(u32, length, tile_size) catch unreachable;
    while (tile_size * num_tiles < min_overlap * (num_tiles - 1) + length) num_tiles += 1;
    const overlaps = try allocator.alloc(u32, num_tiles - 1);
    errdefer allocator.free(overlaps);
    @memset(overlaps, min_overlap);
    var remaining: i64 = @as(i64, tile_size) * num_tiles - @as(i64, min_overlap) * (num_tiles - 1) - length;
    var i: usize = 0;
    while (remaining >= align_to) : (i += 1) {
        overlaps[i % overlaps.len] += align_to;
        remaining -= align_to;
    }
    const starts = try allocator.alloc(u32, num_tiles);
    starts[0] = 0;
    for (1..num_tiles) |ti| starts[ti] = starts[ti - 1] + tile_size - overlaps[ti - 1];
    return .{ .starts = starts, .overlaps = overlaps };
}
