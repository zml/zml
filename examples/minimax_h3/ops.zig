//! Shared generation context, DMA loading, weight constructors, and model math helpers.

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
    mesh: [1]*const zml.Mesh,
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

fn placedTensor(store: zml.io.TensorStore.View, name: []const u8, tags: anytype, partitions: anytype) zml.Tensor {
    if (@TypeOf(partitions) == @EnumLiteral()) {
        return store.createTensor(name, tags, .replicated, .{});
    }
    return store.createTensor(name, tags, .model, partitions);
}

fn maybePlacedTensor(store: zml.io.TensorStore.View, name: []const u8, tags: anytype, partitions: anytype) ?zml.Tensor {
    if (@TypeOf(partitions) == @EnumLiteral()) {
        return store.maybeCreateTensor(name, tags, .replicated, .{});
    }
    return store.maybeCreateTensor(name, tags, .model, partitions);
}

pub fn linear(
    store: zml.io.TensorStore.View,
    weight_name: []const u8,
    bias_name: ?[]const u8,
    partitions: anytype,
    bias_partitions: anytype,
) zml.nn.Linear {
    return .init(
        placedTensor(store, weight_name, .{ .dout, .d }, partitions),
        if (bias_name) |name| maybePlacedTensor(store, name, .{.dout}, bias_partitions) else null,
        .d,
    );
}

pub fn rms(store: zml.io.TensorStore.View, tagz: anytype, eps: f32) RmsNorm {
    return .{ .weight = store.createTensor("weight", tagz, .replicated, .{}), .eps = eps };
}

pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        // Weight sole tag selects the axis to normalize over (either .d or .hd)
        const axis = self.weight.shape().tag(0);
        return zml.nn.rmsNorm(x, axis, self.eps).mul(
            self.weight.convert(x.dtype()).broad(x.shape()),
        );
    }
};

// Split half prefix RoPE. x has an even width final .hd axis
// cos and sin have identical shapes and duplicated coeficcient halfs
pub fn applyRotary(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
    const rotaryDim = cos.dim(-1);
    const half = @divExact(rotaryDim, 2);

    // Split the first rotaryDim channel of each head into real and imaginary halfs. Keep the remaining channels unchanged.
    const xReal, const xImag, const xPass = zml.nn.splitRealImgPass(x, .real_im_pass, @intCast(rotaryDim));

    // cos and sin repeat the same values in both halfs. Keep only the first half.
    const cosHalf = cos.slice(-1, .{ .end = half }).renameTag(-1, .hd).broad(xReal.shape());
    const sinHalf = sin.slice(-1, .{ .end = half }).renameTag(-1, .hd).broad(xReal.shape());

    const yReal = xReal.mul(cosHalf).add(xImag.negate().mul(sinHalf));
    const yImag = xImag.mul(cosHalf).add(xReal.mul(sinHalf));
    return zml.nn.mergeRealImgPass(yReal, yImag, xPass, .real_im_pass);
}

pub fn ln(store: zml.io.TensorStore.View, eps: f32) zml.nn.LayerNorm {
    return .{
        .weight = store.createTensor("weight", .{.d}, .replicated, .{}),
        .bias = store.maybeCreateTensor("bias", .{.d}, .replicated, .{}),
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
        try shared.load(run.io, T, m, &buffers, store, .{ .progress = run.progress });
        try shared.await(run.io);
        return buffers;
    }
    var owned: zml.io.Loader = try .init(run.allocator, run.platform, loader_opts);
    defer owned.deinit();
    try owned.load(run.io, T, m, &buffers, store, .{ .progress = run.progress });
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
