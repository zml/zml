/// M3 parity checker: block-0 video FF residual.
///
/// Two check modes depending on fixture contents:
///   1. FF-only (required):
///        block0_ff_residual.vx_scaled
///        block0_ff_residual.ff_out
///      verifies ff(vx_scaled) == ff_out
///
///   2. Full residual algebra (optional additional keys):
///        block0_ff_residual.vgate_mlp
///        block0_ff_residual.vx_in
///        block0_ff_residual.vx_out
///      verifies vx_in + ff_out * vgate_mlp == vx_out

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

    std.log.info("M3 block0 video FF residual parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const stage2_checkpoint_path = it.next() orelse {
        std.log.err(
            "Usage: bazel run //examples/ltx:block0_ff_residual_check -- " ++
                "<stage2_checkpoint.safetensors> <fixture.safetensors> [token_limit]",
            .{},
        );
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err(
            "Usage: bazel run //examples/ltx:block0_ff_residual_check -- " ++
                "<stage2_checkpoint.safetensors> <fixture.safetensors> [token_limit]",
            .{},
        );
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
        std.log.err("Failed to open fixture: {s}", .{fixture_path});
        return err;
    };
    defer fixture_registry.deinit();

    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    var vx_scaled = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, "block0_ff_residual.vx_scaled", replicated_sharding);
    defer vx_scaled.deinit();

    var ff_expected = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, "block0_ff_residual.ff_out", replicated_sharding);
    defer ff_expected.deinit();

    if (token_limit) |limit| {
        vx_scaled = try check_utils.sliceTokenPrefix(io, platform, vx_scaled, replicated_sharding, limit);
        ff_expected = try check_utils.sliceTokenPrefix(io, platform, ff_expected, replicated_sharding, limit);
    }

    var ff_params_shape = model.initBlock0FFParams(stage2_store.view());

    std.log.info("Compiling FF graph...", .{});
    var ff_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardFF,
        .{ zml.Tensor.fromShape(vx_scaled.shape()), ff_params_shape },
        .{ .shardings = &.{replicated_sharding} },
    );
    defer ff_exe.deinit();
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

    var ff_args = try ff_exe.args(allocator);
    defer ff_args.deinit(allocator);

    var ff_results = try ff_exe.results(allocator);
    defer ff_results.deinit(allocator);

    ff_args.set(.{ vx_scaled, ff_params_buffers });
    std.log.info("Executing FF forward...", .{});
    ff_exe.call(ff_args, &ff_results);

    var ff_computed = ff_results.get(zml.Buffer);
    defer ff_computed.deinit();

    try zml.testing.expectClose(io, ff_computed, ff_expected, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M3 ff(vx_scaled) parity PASSED", .{});

    const has_residual_keys =
        fixture_store.view().hasKey("block0_ff_residual.vgate_mlp") and
        fixture_store.view().hasKey("block0_ff_residual.vx_in") and
        fixture_store.view().hasKey("block0_ff_residual.vx_out");

    if (!has_residual_keys) {
        std.log.info("Skipping residual test (missing vgate_mlp/vx_in/vx_out keys).", .{});
        std.log.info("M3 block0 video FF parity PASSED (ff-only mode)", .{});
        return;
    }

    var vgate = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, "block0_ff_residual.vgate_mlp", replicated_sharding);
    defer vgate.deinit();

    var vx_in = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, "block0_ff_residual.vx_in", replicated_sharding);
    defer vx_in.deinit();

    var vx_out_expected = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, "block0_ff_residual.vx_out", replicated_sharding);
    defer vx_out_expected.deinit();

    if (token_limit) |limit| {
        vx_in = try check_utils.sliceTokenPrefix(io, platform, vx_in, replicated_sharding, limit);
        vx_out_expected = try check_utils.sliceTokenPrefix(io, platform, vx_out_expected, replicated_sharding, limit);
    }

    std.log.info("Compiling FF residual algebra graph...", .{});
    var res_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0VideoFFResidualFromFFOut,
        .{
            zml.Tensor.fromShape(vx_in.shape()),
            zml.Tensor.fromShape(ff_expected.shape()),
            zml.Tensor.fromShape(vgate.shape()),
        },
        .{ .shardings = &.{replicated_sharding} },
    );
    defer res_exe.deinit();

    var res_args = try res_exe.args(allocator);
    defer res_args.deinit(allocator);

    var res_results = try res_exe.results(allocator);
    defer res_results.deinit(allocator);

    res_args.set(.{ vx_in, ff_expected, vgate });
    std.log.info("Executing FF residual algebra...", .{});
    res_exe.call(res_args, &res_results);

    var vx_out_computed = res_results.get(zml.Buffer);
    defer vx_out_computed.deinit();

    try zml.testing.expectClose(io, vx_out_computed, vx_out_expected, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("✓ M3 vx + ff_out * vgate_mlp residual parity PASSED", .{});
    std.log.info("M3 block0 video FF parity PASSED (full residual mode)", .{});
}
