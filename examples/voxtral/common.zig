const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;
const Shape = zml.Shape;

pub fn linear(store: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        store.createTensorWithTags("weight", .{ .dout, .d }),
        store.maybeCreateTensorWithTags("bias", .{.dout}),
        .d,
    );
}

pub fn deinitLinear(l: *zml.Bufferized(zml.nn.Linear)) void {
    l.weight.deinit();
    if (l.bias) |*b| b.deinit();
}

pub fn rmsNorm(x: Tensor, weight: Tensor, eps: f32) Tensor {
    return zml.nn.rmsNorm(x, .d, eps).mul(weight.broad(x.shape()));
}

pub fn loadModel(
    comptime T: type,
    self: *const T,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    comptime label: []const u8,
) !zml.Bufferized(T) {
    progress.increaseEstimatedTotalItems(store.view().count());

    var timer: std.time.Timer = try .start();
    var total_bytes: usize = 0;
    defer {
        const took = timer.read();
        log.info("Loaded " ++ label ++ " weights [{Bi:.2}, {D}, {Bi:.2}/s]", .{
            total_bytes,
            took,
            total_bytes / took * std.time.ns_per_s,
        });
    }

    return zml.io.load(T, self, allocator, io, platform, .{
        .dma_chunks = 32,
        .dma_chunk_size = 128 * zml.MiB,
        .progress = progress,
        .store = store,
        .parallelism = 16,
        .total_bytes = &total_bytes,
    });
}

/// KV cache for all layers of a transformer.
/// Stores K/V tensors with shape {layer, k=max_seq_len, h, hd}.
pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: Shape) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
            .layer_index = .init(.{}, .u32),
        };
    }

    pub fn initShape(kv_shape: Shape) zml.ShapeOf(KvCache) {
        return .{
            .k = kv_shape,
            .v = kv_shape,
            .layer_index = Shape.init(.{}, .u32),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), .{}),
            .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), .{}),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
        self.layer_index.deinit();
    }

    pub fn keys(self: KvCache) Tensor {
        return self.k.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache) Tensor {
        return self.v.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: Tensor, new_v: Tensor, token_index: ?Tensor) KvCache {
        return .{
            .k = scatterCache(self.k, new_k, self.layer_index, token_index),
            .v = scatterCache(self.v, new_v, self.layer_index, token_index),
            .layer_index = self.layer_index,
        };
    }

    fn scatterCache(cache: Tensor, new: Tensor, layer_index: Tensor, token_index: ?Tensor) Tensor {
        const k_shape = cache.shape().drop(.layer);
        const converted = new.convert(cache.dtype()).transpose(k_shape);
        const scatter_opts: Tensor.ScatterOpts = .{ .indices_are_sorted = true, .update_fn = Tensor.ScatterOpts.override };

        return if (token_index) |idx|
            cache.scatterSlices(.{ .layer = layer_index.broad(idx.shape()), .k = idx }, converted, scatter_opts).reuseBuffer(cache)
        else
            cache.scatterSlices(.{ .layer = layer_index }, converted, scatter_opts).reuseBuffer(cache);
    }

    pub fn atLayer(self: KvCache, layer_index: usize) KvCache {
        return .{
            .k = self.k,
            .v = self.v,
            .layer_index = Tensor.scalar(layer_index, .u32),
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
            .layer_index = self.layer_index.reuseBuffer(other.layer_index),
        };
    }
};
