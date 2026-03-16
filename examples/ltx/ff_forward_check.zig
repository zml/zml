const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");

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
        std.log.err("Usage: bazel run //examples/ltx:ff_forward_check -- <stage2_checkpoint.safetensors> <ff_fixture.safetensors>", .{});
        return error.InvalidArgs;
    };

    var stage2_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, stage2_checkpoint_path);
    defer stage2_registry.deinit();

    const discovered_blocks = countTransformerBlocks(&stage2_registry);
    if (discovered_blocks == 0) {
        std.log.err("No transformer blocks found in checkpoint", .{});
        return error.NoTransformerBlocks;
    }
    std.log.info("Detected transformer blocks: {d}", .{discovered_blocks});

    var stage2_store: zml.io.TensorStore = .fromRegistry(allocator, &stage2_registry);
    defer stage2_store.deinit();

    var fixture_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer fixture_registry.deinit();

    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    var ff_input = try loadBufferFromStore(allocator, io, platform, &fixture_store, "ff.input0", replicated_sharding);
    defer ff_input.deinit();

    var ff_expected = try loadBufferFromStore(allocator, io, platform, &fixture_store, "ff.output0", replicated_sharding);
    defer ff_expected.deinit();

    var ff_params_shape = model.initBlock0FFParams(stage2_store.view());

    const input_tensor = zml.Tensor.fromShape(ff_input.shape());
    var exe = try platform.compileFn(
        allocator,
        io,
        model.forwardFF,
        .{ input_tensor, ff_params_shape },
        .{ .shardings = &.{replicated_sharding} },
    );
    defer exe.deinit();

    var ff_params_buffers = try zml.io.load(model.FeedForward.Params, &ff_params_shape, allocator, io, platform, .{
        .store = &stage2_store,
        .shardings = &.{replicated_sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0FFBuffers(&ff_params_buffers);

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ ff_input, ff_params_buffers });
    exe.call(args, &results);

    var ff_output = results.get(zml.Buffer);
    defer ff_output.deinit();

    try zml.testing.expectClose(io, ff_output, ff_expected, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("FF block0 parity PASSED", .{});
}

fn loadBufferFromStore(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const shape = store.view().getShape(key) orelse {
        std.log.err("Tensor not found in fixture: {s}", .{key});
        return error.NotFound;
    };

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();

    _ = try reader.interface.readSliceAll(host_bytes);
    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
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
