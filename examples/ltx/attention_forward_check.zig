const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

comptime {
    // The PJRT call wrappers introspect large generated type names at comptime.
    // Raise the default quota so this checker can compile with attention param structs.
    @setEvalBranchQuota(20000);
}

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Mode = enum {
    attn1,
    attn2,
    audio_attn1,
    audio_attn2,
    audio_to_video_attn,
    video_to_audio_attn,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const stage2_checkpoint_path = it.next() orelse {
        std.log.err(
            "Usage: bazel run //examples/ltx:attention_forward_check -- <stage2_checkpoint.safetensors> <attention_fixture.safetensors> <mode> [token_limit]",
            .{},
        );
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err(
            "Usage: bazel run //examples/ltx:attention_forward_check -- <stage2_checkpoint.safetensors> <attention_fixture.safetensors> <mode> [token_limit]",
            .{},
        );
        return error.InvalidArgs;
    };
    const mode_txt = it.next() orelse {
        std.log.err("Missing mode. Expected one of: attn1, attn2, audio_attn1, audio_attn2, audio_to_video_attn, video_to_audio_attn", .{});
        return error.InvalidArgs;
    };

    const mode = try parseMode(mode_txt);

    const token_limit: ?usize = if (it.next()) |v|
        std.fmt.parseInt(usize, v, 10) catch {
            std.log.err("Invalid token_limit: {s}", .{v});
            return error.InvalidArgs;
        }
    else
        null;

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
        std.log.err("Failed to open attention fixture: {s}", .{fixture_path});
        return err;
    };
    defer fixture_registry.deinit();

    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    const input_key = try std.fmt.allocPrint(allocator, "{s}.input0", .{@tagName(mode)});
    defer allocator.free(input_key);
    const output_key = try std.fmt.allocPrint(allocator, "{s}.output0", .{@tagName(mode)});
    defer allocator.free(output_key);

    var attn_input = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, input_key, replicated_sharding);
    defer attn_input.deinit();

    var attn_expected = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, output_key, replicated_sharding);
    defer attn_expected.deinit();

    if (token_limit) |limit| {
        attn_input = try check_utils.sliceTokenPrefix(io, platform, attn_input, replicated_sharding, limit);
        attn_expected = try check_utils.sliceTokenPrefix(io, platform, attn_expected, replicated_sharding, limit);
        std.log.info("Using token_limit={d}; sliced fixture tensors", .{limit});
    }

    const kind = modeToKind(mode);
    var attn_params_shape = model.initBlock0AttentionParams(stage2_store.view(), kind);

    const input_tensor = zml.Tensor.fromShape(attn_input.shape());
    std.log.info("Compiling attention graph for mode={s}...", .{@tagName(mode)});
    var exe = switch (mode) {
        .attn1 => try platform.compileFn(allocator, io, model.forwardBlock0Attn1, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} }),
        .attn2 => try platform.compileFn(allocator, io, model.forwardBlock0Attn2, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} }),
        .audio_attn1 => try platform.compileFn(allocator, io, model.forwardBlock0AudioAttn1, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} }),
        .audio_attn2 => try platform.compileFn(allocator, io, model.forwardBlock0AudioAttn2, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} }),
        .audio_to_video_attn => try platform.compileFn(allocator, io, model.forwardBlock0AudioToVideoAttn, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} }),
        .video_to_audio_attn => try platform.compileFn(allocator, io, model.forwardBlock0VideoToAudioAttn, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} }),
    };
    defer exe.deinit();
    std.log.info("Compile completed", .{});

    std.log.info("Loading attention parameters from checkpoint...", .{});
    var attn_params_buffers = try zml.io.load(model.Attention.Params, &attn_params_shape, allocator, io, platform, .{
        .store = &stage2_store,
        .shardings = &.{replicated_sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0AttentionBuffers(&attn_params_buffers);
    std.log.info("Parameter load completed", .{});

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ attn_input, attn_params_buffers });
    std.log.info("Executing attention forward for mode={s}...", .{@tagName(mode)});
    exe.call(args, &results);
    std.log.info("Execution completed", .{});

    var attn_output = results.get(zml.Buffer);
    defer attn_output.deinit();

    try zml.testing.expectClose(io, attn_output, attn_expected, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("attention parity PASSED for mode={s}", .{@tagName(mode)});
}

fn parseMode(v: []const u8) !Mode {
    inline for (std.meta.fields(Mode)) |field| {
        if (std.mem.eql(u8, v, field.name)) {
            return @enumFromInt(field.value);
        }
    }

    std.log.err("Invalid mode: {s}. Expected one of: attn1, attn2, audio_attn1, audio_attn2, audio_to_video_attn, video_to_audio_attn", .{v});
    return error.InvalidArgs;
}

fn modeToKind(mode: Mode) model.AttentionKind {
    return switch (mode) {
        .attn1 => .attn1,
        .attn2 => .attn2,
        .audio_attn1 => .audio_attn1,
        .audio_attn2 => .audio_attn2,
        .audio_to_video_attn => .audio_to_video_attn,
        .video_to_audio_attn => .video_to_audio_attn,
    };
}
