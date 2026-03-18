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
        std.log.err("Usage: bazel run //examples/ltx:ff_linear1_gelu_check -- <stage2_checkpoint.safetensors> <ff_net0_fixture.safetensors> <ff_net2_fixture.safetensors> [token_limit]", .{});
        return error.InvalidArgs;
    };

    const net0_fixture_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:ff_linear1_gelu_check -- <stage2_checkpoint.safetensors> <ff_net0_fixture.safetensors> <ff_net2_fixture.safetensors> [token_limit]", .{});
        return error.InvalidArgs;
    };

    const net2_fixture_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:ff_linear1_gelu_check -- <stage2_checkpoint.safetensors> <ff_net0_fixture.safetensors> <ff_net2_fixture.safetensors> [token_limit]", .{});
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

    var net0_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, net0_fixture_path) catch |err| {
        std.log.err("Failed to open net0 fixture: {s}", .{net0_fixture_path});
        return err;
    };
    defer net0_registry.deinit();

    var net2_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, net2_fixture_path) catch |err| {
        std.log.err("Failed to open net2 fixture: {s}", .{net2_fixture_path});
        return err;
    };
    defer net2_registry.deinit();

    var net0_store: zml.io.TensorStore = .fromRegistry(allocator, &net0_registry);
    defer net0_store.deinit();

    var net2_store: zml.io.TensorStore = .fromRegistry(allocator, &net2_registry);
    defer net2_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    // Input to ff.net.0
    var in_buf = try check_utils.loadBufferFromStore(allocator, io, platform, &net0_store, "ff.input0", replicated_sharding);
    defer in_buf.deinit();

    // Expected output after linear1+gelu is input to ff.net.2
    var expected_buf = try check_utils.loadBufferFromStore(allocator, io, platform, &net2_store, "ff.input0", replicated_sharding);
    defer expected_buf.deinit();

    if (token_limit) |limit| {
        in_buf = try check_utils.sliceTokenPrefix(io, platform, in_buf, replicated_sharding, limit);
        expected_buf = try check_utils.sliceTokenPrefix(io, platform, expected_buf, replicated_sharding, limit);
        std.log.info("Using token_limit={d}; sliced fixture tensors", .{limit});
    }

    var ff_params_shape = model.initBlock0FFParams(stage2_store.view());

    const input_tensor = zml.Tensor.fromShape(in_buf.shape());
    std.log.info("Compiling FF linear1+gelu graph...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        model.forwardFFLinear1Gelu,
        .{ input_tensor, ff_params_shape },
        .{ .shardings = &.{replicated_sharding} },
    );
    defer exe.deinit();
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

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ in_buf, ff_params_buffers });
    std.log.info("Executing FF linear1+gelu forward...", .{});
    exe.call(args, &results);
    std.log.info("Execution completed", .{});

    var out_buf = results.get(zml.Buffer);
    defer out_buf.deinit();

    std.log.info("Starting strict comparison...", .{});
    try zml.testing.expectClose(io, out_buf, expected_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("FF linear1+gelu parity PASSED", .{});
}

