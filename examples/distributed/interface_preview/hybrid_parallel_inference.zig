//! Data-parallel hosts plus tensor-parallel local GPUs interface preview.
//! This file intentionally has no Bazel target.

const std = @import("std");

const common = @import("common.zig");
const zml = @import("zml");

const Projection = struct {
    pub fn forward(
        hidden: zml.Tensor,
        vocabulary_weights: zml.Tensor,
    ) zml.Tensor {
        const local_logits = hidden.dot(
            vocabulary_weights,
            .hidden,
        ).withPartitioning(.{
            .batch = .data,
            .vocabulary = .model,
        });

        // Gather only across the two GPUs within each host. Batches remain
        // divided between hosts, while each host obtains complete logits for
        // its own prompts.
        return zml.ops.allGatherAxes(
            local_logits,
            .{.model},
            .vocabulary,
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try common.Job.parse(init);

    const platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    const sharding = try common.hybridSharding(platform);

    const hidden_shape = zml.Shape.init(
        .{ .batch = 32, .hidden = 4096 },
        .f16,
    ).withPartitioning(.{
        .batch = .data,
        .hidden = .replicated,
    });
    const weight_shape = zml.Shape.init(
        .{ .hidden = 4096, .vocabulary = 128_000 },
        .f16,
    ).withPartitioning(.{
        .hidden = .replicated,
        .vocabulary = .model,
    });

    var executable = try platform.compileFn(
        allocator,
        io,
        Projection.forward,
        .{
            zml.Tensor.fromShape(hidden_shape),
            zml.Tensor.fromShape(weight_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer executable.deinit();

    // The future range-aware loader reads only the vocabulary-weight ranges
    // owned by this process's two addressable GPUs.
    var weights = try zml.distributed.loadShardedBuffer(
        allocator,
        io,
        platform,
        "model.safetensors",
        "output_projection.weight",
        weight_shape,
        sharding,
    );
    defer weights.deinit();

    // Each process tokenizes and loads a different part of the global batch.
    var hidden = try zml.Buffer.fromProcessLocalSlice(
        io,
        platform,
        try zml.distributed.nextInferenceBatch(allocator, job.process_index),
        hidden_shape,
        sharding,
    );
    defer hidden.deinit();

    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{ hidden, weights });
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var local_logits = results.get(zml.Buffer);
    defer local_logits.deinit();
    std.debug.print(
        "process={d} owns {d}/{d} output shards\n",
        .{
            platform.processIndex(),
            local_logits.numLocalShards(),
            local_logits.numGlobalShards(),
        },
    );
}
