const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const stage2_checkpoint_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:ff_forward_staged_h2_validation_check -- <stage2_checkpoint.safetensors> <ff_fixture.safetensors> [h2_fixture.safetensors] [token_limit]", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:ff_forward_staged_h2_validation_check -- <stage2_checkpoint.safetensors> <ff_fixture.safetensors> [h2_fixture.safetensors] [token_limit]", .{});
        return error.InvalidArgs;
    };

    var h2_fixture_path: ?[]const u8 = null;
    var token_limit: ?usize = null;

    if (it.next()) |arg3| {
        token_limit = std.fmt.parseInt(usize, arg3, 10) catch null;
        if (token_limit == null) {
            h2_fixture_path = arg3;
            if (it.next()) |arg4| {
                token_limit = std.fmt.parseInt(usize, arg4, 10) catch {
                    std.log.err("Invalid token_limit: {s}", .{arg4});
                    return error.InvalidArgs;
                };
            }
        }
    }

    var stage2_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, stage2_checkpoint_path) catch |err| {
        std.log.err("Failed to open stage-2 checkpoint: {s}", .{stage2_checkpoint_path});
        return err;
    };
    defer stage2_registry.deinit();

    var stage2_store: zml.io.TensorStore = .fromRegistry(allocator, &stage2_registry);
    defer stage2_store.deinit();

    var fixture_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
        std.log.err("Failed to open FF fixture: {s}", .{fixture_path});
        return err;
    };
    defer fixture_registry.deinit();

    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    var ff_input = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, "ff.input0", replicated_sharding);
    defer ff_input.deinit();

    var ff_expected = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, "ff.output0", replicated_sharding);
    defer ff_expected.deinit();

    if (token_limit) |limit| {
        ff_input = try check_utils.sliceTokenPrefix(io, platform, ff_input, replicated_sharding, limit);
        ff_expected = try check_utils.sliceTokenPrefix(io, platform, ff_expected, replicated_sharding, limit);
        std.log.info("Using token_limit={d}; sliced fixture tensors", .{limit});
    }

    var h2_expected: ?zml.Buffer = null;
    defer if (h2_expected) |*b| b.deinit();

    if (h2_fixture_path) |path| {
        var h2_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, path) catch |err| {
            std.log.err("Failed to open h2 fixture: {s}", .{path});
            return err;
        };
        defer h2_registry.deinit();

        var h2_store: zml.io.TensorStore = .fromRegistry(allocator, &h2_registry);
        defer h2_store.deinit();

        h2_expected = try check_utils.loadBufferFromStore(allocator, io, platform, &h2_store, "ff.input0", replicated_sharding);
        if (token_limit) |limit| {
            h2_expected = try check_utils.sliceTokenPrefix(io, platform, h2_expected.?, replicated_sharding, limit);
        }
        std.log.info("Loaded h2 fixture for intermediate validation: {s}", .{path});
    }

    var ff_params_shape = model.initBlock0FFParams(stage2_store.view());

    const in_shape = ff_input.shape();
    const in_tensor = zml.Tensor.fromShape(in_shape);
    const h2_shape = zml.Shape.init(
        .{ in_shape.dims()[0], in_shape.dims()[1], ff_params_shape.proj.weight.dim(.d_ff) },
        in_shape.dtype(),
    );

    std.log.info("Compiling staged FF graph (linear1+gelu[f32 math] then linear2)...", .{});
    var exe_stage1 = try platform.compileFn(
        allocator,
        io,
        model.forwardFFLinear1GeluF32,
        .{ in_tensor, ff_params_shape },
        .{ .shardings = &.{replicated_sharding} },
    );
    defer exe_stage1.deinit();

    var exe_stage2 = try platform.compileFn(
        allocator,
        io,
        model.forwardFFLinear2,
        .{ zml.Tensor.fromShape(h2_shape).withTags(.{ .b, .t, .d_ff }), ff_params_shape },
        .{ .shardings = &.{replicated_sharding} },
    );
    defer exe_stage2.deinit();
    std.log.info("Compile completed", .{});

    std.log.info("Loading FF parameters from checkpoint...", .{});
    var ff_params_buffers = try zml.io.load(model.FeedForward.Params, &ff_params_shape, allocator, io, platform, .{
        .store = &stage2_store,
        .shardings = &.{replicated_sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0FFBuffers(&ff_params_buffers);
    std.log.info("Parameter load completed", .{});

    // Stage 1: Execute to get h2 from actual computation
    var args1 = try exe_stage1.args(allocator);
    defer args1.deinit(allocator);
    var res1 = try exe_stage1.results(allocator);
    defer res1.deinit(allocator);

    args1.set(.{ ff_input, ff_params_buffers });
    std.log.info("Executing stage1 (linear1+gelu)...", .{});
    exe_stage1.call(args1, &res1);
    var h2_computed = res1.get(zml.Buffer);
    defer h2_computed.deinit();

    if (h2_expected) |h2_ref| {
        std.log.info("Validating stage1 output against h2 fixture...", .{});
        const stage1_metrics = try check_utils.compareBuffers(io, h2_computed, h2_ref, 0.2, 0.01);
        const stage1_loose_pass = stage1_metrics.close_fraction >= 0.999;
        const stage1_strict_pass = stage1_metrics.close_fraction >= 0.99999 and stage1_metrics.max_abs_error <= 0.05;
        std.log.info(
            "Stage1 metrics: max_abs={d:.6}, mean_abs={d:.6}, close_fraction={d:.6}, loose_pass={any}, strict_pass={any}",
            .{ stage1_metrics.max_abs_error, stage1_metrics.mean_abs_error, stage1_metrics.close_fraction, stage1_loose_pass, stage1_strict_pass },
        );
        if (!stage1_loose_pass) return error.TestUnexpectedResult;
    }

    // Stage 2: Execute with h2 from stage1
    var args2 = try exe_stage2.args(allocator);
    defer args2.deinit(allocator);
    var res2 = try exe_stage2.results(allocator);
    defer res2.deinit(allocator);

    args2.set(.{ h2_computed, ff_params_buffers });
    std.log.info("Executing stage2 (linear2)...", .{});
    exe_stage2.call(args2, &res2);
    std.log.info("Execution completed", .{});

    var ff_output = res2.get(zml.Buffer);
    defer ff_output.deinit();

    const abs_tol: f32 = 0.2;
    const rel_tol: f32 = 0.01;
    const min_close_fraction: f64 = 0.999;

    const metrics_stage2_computed = try check_utils.compareBuffers(io, ff_output, ff_expected, abs_tol, rel_tol);
    const stage2_computed_pass = metrics_stage2_computed.close_fraction >= min_close_fraction;
    std.log.info(
        "Stage2(computed h2) metrics: max_abs={d:.6}, mean_abs={d:.6}, close_fraction={d:.6}, pass={any}",
        .{ metrics_stage2_computed.max_abs_error, metrics_stage2_computed.mean_abs_error, metrics_stage2_computed.close_fraction, stage2_computed_pass },
    );

    var stage2_fixture_pass = true;
    if (h2_expected) |h2_ref| {
        var args2_ref = try exe_stage2.args(allocator);
        defer args2_ref.deinit(allocator);
        var res2_ref = try exe_stage2.results(allocator);
        defer res2_ref.deinit(allocator);

        args2_ref.set(.{ h2_ref, ff_params_buffers });
        std.log.info("Executing stage2 (linear2) with fixture h2...", .{});
        exe_stage2.call(args2_ref, &res2_ref);

        var ff_output_ref = res2_ref.get(zml.Buffer);
        defer ff_output_ref.deinit();

        const metrics_stage2_fixture = try check_utils.compareBuffers(io, ff_output_ref, ff_expected, abs_tol, rel_tol);
        stage2_fixture_pass = metrics_stage2_fixture.close_fraction >= min_close_fraction;
        std.log.info(
            "Stage2(fixture h2) metrics: max_abs={d:.6}, mean_abs={d:.6}, close_fraction={d:.6}, pass={any}",
            .{ metrics_stage2_fixture.max_abs_error, metrics_stage2_fixture.mean_abs_error, metrics_stage2_fixture.close_fraction, stage2_fixture_pass },
        );
    }

    if (!stage2_computed_pass or !stage2_fixture_pass) {
        std.log.err("FF staged h2 validation FAILED", .{});
        return error.TestUnexpectedResult;
    }

    std.log.info("FF staged h2 validation PASSED", .{});
}
