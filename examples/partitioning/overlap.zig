const std = @import("std");
const zml = @import("zml");
const stdx = @import("stdx");
const asynk = @import("async");
const testing = @import("std").testing;

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

const log = std.log.scoped(.@"examples/overlap");

/// This module simulates a compute-heavy workload.
const ComputeIntensiveModule = struct {
    // By making this a compile-time constant, we allow the compiler to fully
    // unroll the loop, creating a large, static computation graph that is ideal
    // for optimization and overlapping.
    comptime num_iterations: u32 = 300,

    /// Runs a loop of computations for a fixed number of iterations.
    pub fn forward(self: @This(), work_init: zml.Tensor) zml.Tensor {
        var work = work_init.withTags(.{ .rows, .cols });

        // A simple Zig `for` loop unrolls into a sequence of MLIR operations,
        // which can be efficiently scheduled by the XLA compiler.
        for (0..self.num_iterations) |_| {
            const prev_work = work;

            // Standard matrix multiplication: work @ work
            // Contracts axis 1 (.cols) of `work` with axis 0 (.rows) of `work`.
            const work_squared = work.dotGeneral(work, &.{.{ 1, 0 }}, &.{});

            // The output of dotGeneral is untagged, so we must add the tags back.
            const work_squared_tagged = work_squared.withTags(.{ .rows, .cols });

            const work_activated = work_squared_tagged.tanh();
            work = work_activated.add(prev_work);
        }
        return work;
    }
};

const OverlapModel = struct {
    // We simulate a model with multiple layers to create a pipeline.
    var num_layers: u32 = 4;

    pub fn forward(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        const mesh = a.mesh();
        const compute_module: ComputeIntensiveModule = .{};

        var a_intermediate = a;
        // The result of the b computation is carried over.
        // It starts as the original `b` input tensor.
        var b_result = b.convert(.f32);

        for (0..OverlapModel.num_layers) |i| {
            // Stage 1: Communication on the 'a' data path.
            // The result of the previous layer's communication is reduced.
            const a_reduced = zml.ops.allReduce(a_intermediate, .x, mesh, i + 1);

            // Stage 2: Independent Heavy Computation on the 'b' data path.
            // This is the heavy part that does not depend on `a_reduced`.
            // The XLA compiler can schedule this to run while `allReduce` is in flight.
            // The key fix is that the computation is now based on the *result* of the
            // previous layer's `b` computation, maintaining a logical data flow.
            const b_squared = b_result.dotGeneral(b_result.transpose(.{ .n, .m }), &.{.{ 1, 0 }}, &.{});
            const b_squared_tagged = b_squared.withTags(.{ .rows, .cols });
            const b_processed = zml.call(compute_module, .forward, .{b_squared_tagged});

            // Stage 3: Dependent Computation (Synchronization Point)
            // Use the result of the all-reduce. This forces a wait on the communication.
            const a_scalar = a_reduced.sum(0).sum(1).asScalar();

            // The result of the communication is the input for the next layer's communication.
            a_intermediate = a_reduced;
            // The result of this layer's computation is the input for the next layer's computation.
            // We give it back the original tags so the next iteration's transpose works.
            b_result = b_processed.add(a_scalar.convert(.f32).broad(b_processed.shape())).withTags(.{ .m, .n });
        }

        // Return the final result from the 'b' data path, converted to the original type.
        return b_result.convert(.i32);
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{ .cpu = .{ .cpu_device_count = 8 } }).withCompilationOptions(.{
        .sharding_enabled = true,
        .xla_dump_to = "/tmp/zml/collectives_overlap",
        .xla_dump_fusion_visualization = true,
    });

    const num_devices = platform.getDevices().len;
    if (num_devices < 2) {
        log.warn("This example requires at least 2 devices to demonstrate collectives. Exiting.", .{});
        return;
    }
    context.printAvailablePlatforms(platform);

    const mesh = zml.Mesh.init(.{ .x = num_devices });
    log.info("Using mesh: {}", .{mesh});

    const M: i64 = 8192;
    const N: i64 = 8192;
    const K: i64 = 8192;

    const shape_a = zml.Shape.init(.{ .m = M, .n = K }, .i32).withPartitioning(.{ .n = .x });
    const shape_b = zml.Shape.init(.{ .m = M, .n = N }, .i32).withPartitioning(.{ .n = .x });

    log.info("Compiling the OverlapModel...", .{});
    var compile_timer = try std.time.Timer.start();
    const mod = try zml.compileFn(allocator, OverlapModel.forward, .{ shape_a, shape_b }, mesh, platform);
    defer mod.deinit();
    log.info("Compilation complete in {d:.3}s.", .{@as(f32, @floatFromInt(compile_timer.read())) / 1e9});

    log.info("Preparing input data...", .{});
    const a_data_slice = try zml.slice.arange(allocator, shape_a, .{});
    defer allocator.free(a_data_slice);
    const a_buffer = try zml.Buffer.from(platform, .init(mesh, shape_a), a_data_slice, .{});
    defer a_buffer.deinit();

    const b_data_slice = try zml.slice.arange(allocator, shape_b, .{});
    defer allocator.free(b_data_slice);
    const b_buffer = try zml.Buffer.from(platform, .init(mesh, shape_b), b_data_slice, .{});
    defer b_buffer.deinit();
    log.info("Input data prepared and transferred to device.", .{});

    log.info("Executing the model...", .{});
    var exec_timer = try std.time.Timer.start();
    const result_buffer = mod.call(.{ a_buffer, b_buffer });
    defer result_buffer.deinit();
    const result_data = try result_buffer.toHost(allocator); // Await completion for timing
    defer allocator.free(result_data);
    log.info("✅ Execution finished in {d:.3}ms.", .{@as(f32, @floatFromInt(exec_timer.read())) / 1e6});

    log.info("result: {}", .{zml.slice.pretty(result_buffer.shape(), result_data)});
}
