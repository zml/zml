const std = @import("std");

const zml = @import("zml");
const layer = @import("kimi_k3/layer.zig");
const runtime_weights = @import("kimi_k3/runtime_weights.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    weights: []const u8,

    pub const help =
        \\Use kimi_k3_runtime_weights_tests --weights=<S4-directory>
        \\
        \\Stage every Kimi K3 runtime weight family sequentially on NVIDIA CUDA.
        \\
    ;
};

fn elapsedUs(io: std.Io, started: i96) i96 {
    return @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - started, 1000);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.90 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.weights);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const loader: runtime_weights.Loader = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .store = &store,
        .sharding = platform.replicated_sharding,
    };
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});

    var started = std.Io.Clock.now(.real, io).toNanoseconds();
    var head = try loader.loadHead();
    defer zml.Buffer.deinitAll(runtime_weights.HeadTensors, &head);
    if (head.embedding.shape().dim(.voc) != 163840 or head.embedding.shape().dim(.d) != 7168) {
        return error.InvalidKimiK3HeadShape;
    }
    try stdout_file.interface.print("KIMI_K3_RUNTIME_LOAD_PASS family=head load_us={}\n", .{elapsedUs(io, started)});
    try stdout_file.interface.flush();

    started = std.Io.Clock.now(.real, io).toNanoseconds();
    var layer0 = try loader.loadLayer0();
    defer zml.Buffer.deinitAll(layer.Layer0Weights, &layer0);
    if (layer0.input_norm.shape().dim(.d) != 7168) return error.InvalidKimiK3Layer0Shape;
    try stdout_file.interface.print("KIMI_K3_RUNTIME_LOAD_PASS family=kda_dense layer=0 load_us={}\n", .{elapsedUs(io, started)});
    try stdout_file.interface.flush();

    started = std.Io.Clock.now(.real, io).toNanoseconds();
    var kda_moe = try loader.loadKdaMoe(1);
    if (kda_moe.common.moe.experts.w1.values.shape().dim(.expert) != runtime_weights.expert_count) {
        return error.InvalidKimiK3KdaExpertBankShape;
    }
    const kda_load_us = elapsedUs(io, started);
    zml.Buffer.deinitAll(layer.KdaMoeWeights, &kda_moe);
    try stdout_file.interface.print(
        "KIMI_K3_RUNTIME_LOAD_PASS family=kda_moe layer=1 experts={} load_us={}\n",
        .{ runtime_weights.expert_count, kda_load_us },
    );
    try stdout_file.interface.flush();

    started = std.Io.Clock.now(.real, io).toNanoseconds();
    var mla_moe = try loader.loadMlaMoe(3);
    if (mla_moe.common.moe.experts.w2.values.shape().dim(.expert) != runtime_weights.expert_count) {
        return error.InvalidKimiK3MlaExpertBankShape;
    }
    const mla_load_us = elapsedUs(io, started);
    zml.Buffer.deinitAll(layer.MlaMoeWeights, &mla_moe);
    try stdout_file.interface.print(
        "KIMI_K3_RUNTIME_LOAD_PASS family=mla_moe layer=3 experts={} load_us={}\n",
        .{ runtime_weights.expert_count, mla_load_us },
    );
    try stdout_file.interface.writeAll("KIMI_K3_RUNTIME_LOAD_ALL_PASS backend=cuda resident=head+layer0 staged_banks=2\n");
    try stdout_file.interface.flush();
}
