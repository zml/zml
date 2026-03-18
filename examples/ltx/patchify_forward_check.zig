const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Mode = enum {
    video,
    audio,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const stage2_checkpoint_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:patchify_forward_check -- <stage2_checkpoint.safetensors> <patchify_fixture.safetensors>", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:patchify_forward_check -- <stage2_checkpoint.safetensors> <patchify_fixture.safetensors> [video|audio] [token_limit]", .{});
        return error.InvalidArgs;
    };

    var mode: Mode = .video;
    var token_limit: ?usize = null;

    if (it.next()) |arg3| {
        token_limit = std.fmt.parseInt(usize, arg3, 10) catch null;
        if (token_limit == null) {
            mode = try parseMode(arg3);
        }
    }

    if (it.next()) |arg4| {
        token_limit = std.fmt.parseInt(usize, arg4, 10) catch {
            std.log.err("Invalid token_limit: {s}", .{arg4});
            return error.InvalidArgs;
        };
    }

    if (it.next() != null) {
        std.log.err("Too many arguments", .{});
        return error.InvalidArgs;
    }

    var stage2_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, stage2_checkpoint_path) catch |err| {
        std.log.err("Failed to open stage-2 checkpoint: {s}", .{stage2_checkpoint_path});
        return err;
    };
    defer stage2_registry.deinit();

    var stage2_store: zml.io.TensorStore = .fromRegistry(allocator, &stage2_registry);
    defer stage2_store.deinit();

    var fixture_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
        std.log.err("Failed to open patchify fixture: {s}", .{fixture_path});
        return err;
    };
    defer fixture_registry.deinit();

    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    const input_key = switch (mode) {
        .video => "patchify.input0",
        .audio => "audio_patchify.input0",
    };
    const output_key = switch (mode) {
        .video => "patchify.output0",
        .audio => "audio_patchify.output0",
    };

    var patchify_input = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, input_key, replicated_sharding);
    defer patchify_input.deinit();

    var patchify_expected = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, output_key, replicated_sharding);
    defer patchify_expected.deinit();

    if (token_limit) |limit| {
        patchify_input = try check_utils.sliceTokenPrefix(io, platform, patchify_input, replicated_sharding, limit);
        patchify_expected = try check_utils.sliceTokenPrefix(io, platform, patchify_expected, replicated_sharding, limit);
        std.log.info("Using token_limit={d}; sliced fixture tensors", .{limit});
    }

    var patchify_params_shape = switch (mode) {
        .video => model.initPatchifyParams(stage2_store.view()),
        .audio => model.initAudioPatchifyParams(stage2_store.view()),
    };

    const input_tensor = zml.Tensor.fromShape(patchify_input.shape());
    std.log.info("Compiling {s} patchify graph...", .{@tagName(mode)});
    var exe = try platform.compileFn(
        allocator,
        io,
        model.forwardPatchify,
        .{ input_tensor, patchify_params_shape },
        .{ .shardings = &.{replicated_sharding} },
    );
    defer exe.deinit();
    std.log.info("Compile completed", .{});

    std.log.info("Loading {s} patchify parameters from checkpoint...", .{@tagName(mode)});
    var patchify_params_buffers = try zml.io.load(model.Patchify.Params, &patchify_params_shape, allocator, io, platform, .{
        .store = &stage2_store,
        .shardings = &.{replicated_sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadPatchifyBuffers(&patchify_params_buffers);
    std.log.info("Parameter load completed", .{});

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ patchify_input, patchify_params_buffers });
    std.log.info("Executing {s} patchify forward...", .{@tagName(mode)});
    exe.call(args, &results);
    std.log.info("Execution completed", .{});

    var patchify_output = results.get(zml.Buffer);
    defer patchify_output.deinit();

    try zml.testing.expectClose(io, patchify_output, patchify_expected, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("{s} patchify parity PASSED", .{@tagName(mode)});
}

fn parseMode(v: []const u8) !Mode {
    if (std.mem.eql(u8, v, "video")) return .video;
    if (std.mem.eql(u8, v, "audio")) return .audio;

    std.log.err("Invalid mode: {s}. Expected one of: video, audio", .{v});
    return error.InvalidArgs;
}
