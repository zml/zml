//! Synchronous data-parallel training-step interface preview.
//! This file intentionally has no Bazel target.

const std = @import("std");

const common = @import("common.zig");
const zml = @import("zml");

const StepOutput = struct {
    weights: zml.Tensor,
    loss: zml.Tensor,
};

const TrainStep = struct {
    learning_rate: f32,
    replica_scale: f32,

    pub fn forward(
        self: TrainStep,
        weights: zml.Tensor,
        features: zml.Tensor,
        labels: zml.Tensor,
    ) StepOutput {
        const predictions = features.dot(weights, .feature);
        const residual = predictions.sub(labels);

        const local_loss = residual.mul(residual)
            .mean(.batch)
            .mean(.output);
        const local_gradient = features
            .transpose(.{ .feature, .batch })
            .dot(residual, .batch);

        const global_loss = zml.ops.allReduceAxes(
            local_loss,
            .{.data},
            zml.Tensor.add,
        ).scale(self.replica_scale);
        const global_gradient = zml.ops.allReduceAxes(
            local_gradient,
            .{.data},
            zml.Tensor.add,
        ).scale(self.replica_scale);

        return .{
            .weights = weights.sub(
                global_gradient.scale(self.learning_rate),
            ),
            .loss = global_loss,
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try common.Job.parse(init);

    const platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    const data_sharding = try common.dataSharding(platform);

    const features_shape = zml.Shape.init(
        .{ .batch = 64, .feature = 16 },
        .f32,
    ).withPartitioning(.{
        .batch = .data,
        .feature = .replicated,
    });
    const labels_shape = zml.Shape.init(
        .{ .batch = 64, .output = 8 },
        .f32,
    ).withPartitioning(.{
        .batch = .data,
        .output = .replicated,
    });
    const weights_shape = zml.Shape.init(
        .{ .feature = 16, .output = 8 },
        .f32,
    ).withPartitioning(.{
        .feature = .replicated,
        .output = .replicated,
    });

    const replica_count: f32 = @floatFromInt(
        platform.globalDevices().len,
    );
    const step: TrainStep = .{
        .learning_rate = 0.001,
        .replica_scale = 1.0 / replica_count,
    };
    var executable = try platform.compile(
        allocator,
        io,
        step,
        .forward,
        .{
            zml.Tensor.fromShape(weights_shape),
            zml.Tensor.fromShape(features_shape),
            zml.Tensor.fromShape(labels_shape),
        },
        .{ .shardings = &.{data_sharding} },
    );
    defer executable.deinit();

    // Proposed convenience API, similar to JAX's
    // make_array_from_process_local_data. Each rank supplies only its local
    // sub-batch while the Buffer retains the declared global shape.
    var batch = try zml.distributed.ProcessLocalBatch.allocate(
        allocator,
        job.process_index,
        features_shape,
        labels_shape,
    );
    defer batch.deinit(allocator);

    var features = try zml.Buffer.fromProcessLocalSlice(
        io,
        platform,
        batch.features,
        features_shape,
        data_sharding,
    );
    defer features.deinit();
    var labels = try zml.Buffer.fromProcessLocalSlice(
        io,
        platform,
        batch.labels,
        labels_shape,
        data_sharding,
    );
    defer labels.deinit();

    // Replicated parameters have one equivalent local copy on every GPU.
    var weights = try zml.Buffer.fromSlice(
        io,
        platform,
        batch.initial_weights,
        zml.Sharding.replicated,
    );
    defer weights.deinit();

    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{ weights, features, labels });
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var updated_weights = results.get(.weights);
    defer updated_weights.deinit();
    var global_loss = results.get(.loss);
    defer global_loss.deinit();

    if (platform.processIndex() == 0) {
        const value = try global_loss.toSliceAlloc(allocator, io);
        defer value.free(allocator);
        std.debug.print("global loss={any}\n", .{value.items(f32)});
    }
}
