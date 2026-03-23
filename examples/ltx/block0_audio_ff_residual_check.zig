/// M4-C parity checker: block-0 audio FF residual.
///
/// Two check modes:
///   1. FF-only (required keys):
///        block0_audio_ff_residual.ax_scaled, .ff_out
///      → verifies audio_ff(ax_scaled) == ff_out
///
///   2. Full residual (optional additional keys):
///        block0_audio_ff_residual.agate_mlp, .ax_in, .ax_out
///      → additionally verifies ax_in + ff_out * agate_mlp == ax_out
///
/// Generate fixture with:
///   python examples/ltx/export_block0_audio_ff_residual_fixture.py <trace.pt> <fixture.safetensors>
///
/// Run:
///   bazel run //examples/ltx:block0_audio_ff_residual_check -- \
///       <stage2_checkpoint.safetensors> <fixture.safetensors> [token_limit]

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

comptime {
    @setEvalBranchQuota(20000);
}

pub const std_options: std.Options = .{
    .log_level = .info,
};

fn loadBuf(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    return check_utils.loadBufferFromStore(allocator, io, platform, store, key, sharding) catch |err| {
        std.log.err("Fixture missing key: {s}", .{key});
        return err;
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("M4-C block0 audio FF residual parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next();

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: block0_audio_ff_residual_check <checkpoint.safetensors> <fixture.safetensors> [token_limit]", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: block0_audio_ff_residual_check <checkpoint.safetensors> <fixture.safetensors> [token_limit]", .{});
        return error.InvalidArgs;
    };
    const token_limit: ?usize = if (it.next()) |v|
        std.fmt.parseInt(usize, v, 10) catch {
            std.log.err("Invalid token_limit: {s}", .{v});
            return error.InvalidArgs;
        }
    else
        null;

    var ckpt_reg: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    var fix_reg: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
        std.log.err("Failed to open fixture: {s}", .{fixture_path});
        return err;
    };
    defer fix_reg.deinit();
    var fix_store: zml.io.TensorStore = .fromRegistry(allocator, &fix_reg);
    defer fix_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    var ax_scaled_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_ff_residual.ax_scaled", sharding);
    defer ax_scaled_buf.deinit();
    var ff_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_ff_residual.ff_out", sharding);
    defer ff_ref_buf.deinit();

    if (token_limit) |lim| {
        ax_scaled_buf = try check_utils.sliceTokenPrefix(io, platform, ax_scaled_buf, sharding, lim);
        ff_ref_buf = try check_utils.sliceTokenPrefix(io, platform, ff_ref_buf, sharding, lim);
        std.log.info("Using token_limit={d}", .{lim});
    }

    var ff_params_shape = model.initBlock0AudioFFParams(ckpt_store.view());

    std.log.info("Compiling audio FF graph...", .{});
    var ff_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0AudioFF,
        .{ zml.Tensor.fromShape(ax_scaled_buf.shape()), ff_params_shape },
        .{ .shardings = &.{sharding} },
    );
    defer ff_exe.deinit();
    std.log.info("Compile completed", .{});

    var ff_params = try zml.io.load(model.FeedForward.Params, &ff_params_shape, allocator, io, platform, .{
        .store = &ckpt_store,
        .shardings = &.{sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0AudioFFBuffers(&ff_params);

    var ff_args = try ff_exe.args(allocator);
    defer ff_args.deinit(allocator);
    var ff_res = try ff_exe.results(allocator);
    defer ff_res.deinit(allocator);

    ff_args.set(.{ ax_scaled_buf, ff_params });
    std.log.info("Executing audio FF forward...", .{});
    ff_exe.call(ff_args, &ff_res);

    var ff_computed = ff_res.get(zml.Buffer);
    defer ff_computed.deinit();

    try zml.testing.expectClose(io, ff_computed, ff_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M4-C audio_ff(ax_scaled) parity PASSED", .{});

    // ── Optional full-residual test ─────────────────────────────────────────
    const has_residual =
        fix_store.view().hasKey("block0_audio_ff_residual.agate_mlp") and
        fix_store.view().hasKey("block0_audio_ff_residual.ax_in") and
        fix_store.view().hasKey("block0_audio_ff_residual.ax_out");

    if (!has_residual) {
        std.log.info("Skipping residual test (agate_mlp/ax_in/ax_out absent).", .{});
        std.log.info("M4-C block0 audio FF parity PASSED (ff-only mode)", .{});
        return;
    }

    var agate_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_ff_residual.agate_mlp", sharding);
    defer agate_buf.deinit();
    var ax_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_ff_residual.ax_in", sharding);
    defer ax_in_buf.deinit();
    var ax_out_ref = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_ff_residual.ax_out", sharding);
    defer ax_out_ref.deinit();

    if (token_limit) |lim| {
        ax_in_buf = try check_utils.sliceTokenPrefix(io, platform, ax_in_buf, sharding, lim);
        ax_out_ref = try check_utils.sliceTokenPrefix(io, platform, ax_out_ref, sharding, lim);
    }

    std.log.info("Compiling audio FF residual algebra graph...", .{});
    var res_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0AudioFFResidualFromFFOut,
        .{
            zml.Tensor.fromShape(ax_in_buf.shape()),
            zml.Tensor.fromShape(ff_ref_buf.shape()),
            zml.Tensor.fromShape(agate_buf.shape()),
        },
        .{ .shardings = &.{sharding} },
    );
    defer res_exe.deinit();

    var res_args = try res_exe.args(allocator);
    defer res_args.deinit(allocator);
    var res_results = try res_exe.results(allocator);
    defer res_results.deinit(allocator);

    res_args.set(.{ ax_in_buf, ff_ref_buf, agate_buf });
    std.log.info("Executing audio FF residual algebra...", .{});
    res_exe.call(res_args, &res_results);

    var ax_out_computed = res_results.get(zml.Buffer);
    defer ax_out_computed.deinit();

    try zml.testing.expectClose(io, ax_out_computed, ax_out_ref, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M4-C ax + audio_ff_out * agate_mlp residual parity PASSED", .{});
    std.log.info("M4-C block0 audio FF parity PASSED (full residual mode)", .{});
}
