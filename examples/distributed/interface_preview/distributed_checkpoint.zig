//! Atomic distributed-checkpoint interface preview.
//! This file intentionally has no Bazel target.

const std = @import("std");

const common = @import("common.zig");
const zml = @import("zml");

const TrainingState = struct {
    step: u64,
    weights: zml.Buffer,
    optimizer_moments: zml.Buffer,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try common.Job.parse(init);

    const platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    const data_sharding = try common.dataSharding(platform);

    var state = try zml.distributed.exampleTrainingState(
        allocator,
        io,
        platform,
        data_sharding,
    );
    defer state.weights.deinit();
    defer state.optimizer_moments.deinit();

    var checkpoints = try zml.distributed.CheckpointManager.init(
        allocator,
        io,
        .{
            .directory = "/shared/checkpoints/demo",
            .namespace = job.namespace,
            .keep_last = 3,
        },
    );
    defer checkpoints.deinit(allocator, io);

    // Each process writes one temporary file containing only its local
    // shards. The manager synchronizes ranks, atomically renames the shard
    // files, and lets process 0 publish the manifest last.
    try checkpoints.save(
        io,
        platform,
        TrainingState{
            .step = state.step,
            .weights = state.weights,
            .optimizer_moments = state.optimizer_moments,
        },
    );

    const restored = try checkpoints.restoreLatest(
        io,
        platform,
        TrainingState,
    );
    defer restored.weights.deinit();
    defer restored.optimizer_moments.deinit();

    std.debug.print(
        "process={d} restored step={d}, local weight shards={d}\n",
        .{
            platform.processIndex(),
            restored.step,
            restored.weights.numLocalShards(),
        },
    );
}
