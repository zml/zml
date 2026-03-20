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
        std.log.err("Usage: bazel run //examples/ltx:audio_ff_forward_check -- <stage2_checkpoint.safetensors> <audio_ff_fixture.safetensors>", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:audio_ff_forward_check -- <stage2_checkpoint.safetensors> <audio_ff_fixture.safetensors> [token_limit]", .{});
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
        std.log.err("Failed to open audio_ff fixture: {s}", .{fixture_path});
        return err;
    };
    defer fixture_registry.deinit();

    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    var audio_ff_input = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, "audio_ff.input0", replicated_sharding);
    defer audio_ff_input.deinit();

    var audio_ff_expected = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, "audio_ff.output0", replicated_sharding);
    defer audio_ff_expected.deinit();

    if (token_limit) |limit| {
        audio_ff_input = try check_utils.sliceTokenPrefix(io, platform, audio_ff_input, replicated_sharding, limit);
        audio_ff_expected = try check_utils.sliceTokenPrefix(io, platform, audio_ff_expected, replicated_sharding, limit);
        std.log.info("Using token_limit={d}; sliced fixture tensors", .{limit});
    }

    var audio_ff_params_shape = model.initBlock0AudioFFParams(stage2_store.view());

    const input_tensor = zml.Tensor.fromShape(audio_ff_input.shape());
    std.log.info("Compiling audio_ff graph...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        model.forwardFF,
        .{ input_tensor, audio_ff_params_shape },
        .{ .shardings = &.{replicated_sharding} },
    );
    defer exe.deinit();
    std.log.info("Compile completed", .{});

    std.log.info("Loading audio_ff parameters from checkpoint...", .{});
    var audio_ff_params_buffers = try zml.io.load(model.FeedForward.Params, &audio_ff_params_shape, allocator, io, platform, .{
        .store = &stage2_store,
        .shardings = &.{replicated_sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0AudioFFBuffers(&audio_ff_params_buffers);
    std.log.info("Parameter load completed", .{});

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ audio_ff_input, audio_ff_params_buffers });
    std.log.info("Executing audio_ff forward...", .{});
    exe.call(args, &results);
    std.log.info("Execution completed", .{});

    var audio_ff_output = results.get(zml.Buffer);
    defer audio_ff_output.deinit();

    try zml.testing.expectClose(io, audio_ff_output, audio_ff_expected, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("audio_ff block0 parity PASSED", .{});
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
        if (idx > max_block_index) {
            max_block_index = idx;
            seen_any = true;
        }
    }

    if (seen_any) {
        return max_block_index + 1;
    }

    return 0;
}
