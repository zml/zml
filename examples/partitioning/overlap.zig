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
    iterations: zml.Tensor,

    const LoopContext = struct {
        iterations: zml.Tensor,
    };

    fn loopCond(ctx: LoopContext, work: zml.Tensor, i: zml.Tensor) zml.Tensor {
        _ = work; // The `work` tensor is just a loop-carried variable.
        return i.cmp(.LT, ctx.iterations);
    }

    fn loopBody(_: LoopContext, work: zml.Tensor, i: zml.Tensor) [2]zml.Tensor {
        const prev_work = work; // Shape: {rows, cols}, {M, M}

        // Standard matrix multiplication: work @ work
        // Contracts axis 1 (.cols) of `work` with axis 0 (.rows) of `work`.
        const work_squared = work.dotGeneral(work, &.{.{ 1, 0 }}, &.{});

        // The output of dotGeneral is untagged, so we must add the tags back.
        const work_squared_tagged = work_squared.withTags(.{ .rows, .cols });

        const work_activated = work_squared_tagged.tanh();
        const work_residual = work_activated.add(prev_work);

        const next_i = i.addConstant(1);
        return .{ work_residual, next_i };
    }

    /// Runs a loop of computations for a dynamic number of iterations.
    pub fn forward(self: @This(), work_init: zml.Tensor) zml.Tensor {
        const i_init = zml.Tensor.scalar(0, .i32);
        const loop_context = LoopContext{ .iterations = self.iterations };

        // We tag `work_init` here to ensure the loop variable has the correct tags
        // for the first iteration. The loop body is responsible for maintaining the tags.
        const tagged_work_init = work_init.withTags(.{ .rows, .cols });

        const final_loop_vars = zml.ops.while_(loopCond, loopBody, loop_context, .{ tagged_work_init, i_init });

        // The result of the while loop is a tuple. We only care about the first
        // element, which is the processed tensor.
        return final_loop_vars[0];
    }
};

const OverlapModel = struct {
    pub fn forward(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        const mesh = a.mesh();
        const a_reduced = zml.ops.allReduce(a, .x, mesh, null);

        const b_f32 = b.convert(.f32);

        const b_squared = b_f32.dotGeneral(b_f32.transpose(.{ .n, .m }), &.{.{ 1, 0 }}, &.{});

        const num_iterations = a.choose(.{ .m = 0, .n = 0 }).convert(.i32);
        const compute_module: ComputeIntensiveModule = .{ .iterations = num_iterations };
        const b_processed = zml.call(compute_module, .forward, .{b_squared});

        const a_scalar = a_reduced.sum(0).sum(1).asScalar();
        const result = b_processed.add(a_scalar.convert(.f32).broad(b_processed.shape()));

        return result;
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

    const M: i64 = 16384;
    const N: i64 = 16384;
    const K: i64 = 16384;

    const shape_a = zml.Shape.init(.{ .m = M, .n = K }, .i32).withPartitioning(.{ .n = .x });
    const shape_b = zml.Shape.init(.{ .m = M, .n = N }, .i32).withPartitioning(.{ .n = .x });

    log.info("Compiling the OverlapModel...", .{});
    const mod = try zml.compileFn(allocator, OverlapModel.forward, .{ shape_a, shape_b }, mesh, platform);
    defer mod.deinit();
    log.info("Compilation complete.", .{});

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
    var timer = try std.time.Timer.start();
    const result_buffer = mod.call(.{ a_buffer, b_buffer });
    defer result_buffer.deinit();
    const result_data = try result_buffer.toHost(allocator); // Await completion for timing
    defer allocator.free(result_data);
    log.info("Execution finished in {d:.3}ms.", .{@as(f32, @floatFromInt(timer.read())) / 1e6});
    log.info("Result (first 8 elements): {any}", .{result_data[0..@min(result_data.len, 8)]});
}
