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
        std.log.err("Usage: bazel run //examples/ltx:ff_forward_staged_check -- <stage2_checkpoint.safetensors> <ff_fixture.safetensors> [token_limit]", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:ff_forward_staged_check -- <stage2_checkpoint.safetensors> <ff_fixture.safetensors> [token_limit]", .{});
        return error.InvalidArgs;
    };

    const token_limit: ?usize = if (it.next()) |v|
        std.fmt.parseInt(usize, v, 10) catch {
            std.log.err("Invalid token_limit: {s}", .{v});
            return error.InvalidArgs;
        }
    else
        null;

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

    var ff_params_shape = model.initBlock0FFParams(stage2_store.view());

    const in_shape = ff_input.shape();
    const in_tensor = zml.Tensor.fromShape(in_shape);
    const h2_shape = zml.Shape.init(
        .{ in_shape.dims()[0], in_shape.dims()[1], ff_params_shape.proj.weight.dim(.d_ff) },
        in_shape.dtype(),
    );

    std.log.info("Compiling staged FF graph (linear1+gelu then linear2)...", .{});
    var exe_stage1 = try platform.compileFn(
        allocator,
        io,
        model.forwardFFLinear1Gelu,
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

    var args1 = try exe_stage1.args(allocator);
    defer args1.deinit(allocator);
    var res1 = try exe_stage1.results(allocator);
    defer res1.deinit(allocator);

    args1.set(.{ ff_input, ff_params_buffers });
    std.log.info("Executing stage1 (linear1+gelu)...", .{});
    exe_stage1.call(args1, &res1);
    var h2_buf = res1.get(zml.Buffer);
    defer h2_buf.deinit();

    var args2 = try exe_stage2.args(allocator);
    defer args2.deinit(allocator);
    var res2 = try exe_stage2.results(allocator);
    defer res2.deinit(allocator);

    args2.set(.{ h2_buf, ff_params_buffers });
    std.log.info("Executing stage2 (linear2)...", .{});
    exe_stage2.call(args2, &res2);
    std.log.info("Execution completed", .{});

    var ff_output = res2.get(zml.Buffer);
    defer ff_output.deinit();

    try zml.testing.expectClose(io, ff_output, ff_expected, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("FF staged parity PASSED", .{});
}
