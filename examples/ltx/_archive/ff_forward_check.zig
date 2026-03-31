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
        std.log.err("Usage: bazel run //examples/ltx:ff_forward_check -- <stage2_checkpoint.safetensors> <ff_fixture.safetensors>", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:ff_forward_check -- <stage2_checkpoint.safetensors> <ff_fixture.safetensors> [token_limit]", .{});
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

    const discovered_blocks = countTransformerBlocks(&stage2_registry);
    if (discovered_blocks == 0) {
        std.log.err("No transformer blocks found in checkpoint", .{});
        return error.NoTransformerBlocks;
    }
    std.log.info("Detected transformer blocks: {d}", .{discovered_blocks});

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

    const input_tensor = zml.Tensor.fromShape(ff_input.shape());
    std.log.info("Compiling FF graph...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        model.forwardFF,
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

    args.set(.{ ff_input, ff_params_buffers });
    std.log.info("Executing FF forward...", .{});
    exe.call(args, &results);
    std.log.info("Execution completed", .{});

    var ff_output = results.get(zml.Buffer);
    defer ff_output.deinit();

    try zml.testing.expectClose(io, ff_output, ff_expected, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("FF block0 parity PASSED", .{});
}

fn countTransformerBlocks(registry: *const zml.safetensors.TensorRegistry) usize {
    const prefixes = [_][]const u8{
        "model.velocity_model.transformer_blocks.",
        "model.diffusion_model.transformer_blocks.",
        "velocity_model.transformer_blocks.",
        "diffusion_model.transformer_blocks.",
    };

    var max_block_index: usize = 0;
    var seen_any = false;

    const keys = registry.tensors.keys();
    for (keys) |k| {
        var matched_prefix: ?[]const u8 = null;
        for (prefixes) |p| {
            if (std.mem.startsWith(u8, k, p)) {
                matched_prefix = p;
                break;
            }
        }
        const prefix = matched_prefix orelse continue;

        const tail = k[prefix.len..];
        const dot_pos = std.mem.indexOfScalar(u8, tail, '.') orelse continue;
        const idx_txt = tail[0..dot_pos];

        const idx = std.fmt.parseInt(usize, idx_txt, 10) catch continue;
        if (!seen_any or idx > max_block_index) {
            max_block_index = idx;
            seen_any = true;
        }
    }

    if (!seen_any) return 0;
    return max_block_index + 1;
}
