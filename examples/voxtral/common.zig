const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;

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
