/// M4-B parity checker: block-0 audio text cross-attention residual.
///
/// Two check modes:
///   1. Attn2-only (required keys):
///        block0_audio_text_ca.attn2_x, .context, .attn2_out
///      → verifies audio_attn2(attn2_x, context) == attn2_out
///
///   2. Full residual (optional additional keys):
///        block0_audio_text_ca.ax_in, .text_ca_out, .ax_out
///      → additionally verifies ax_in + text_ca_out == ax_out
///
/// Generate fixture with:
///   python examples/ltx/export_block0_audio_text_ca_fixture.py <trace.pt> <fixture.safetensors>
///
/// Run:
///   bazel run //examples/ltx:block0_audio_text_ca_check -- \
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

    std.log.info("M4-B block0 audio text cross-attn residual parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next();

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: block0_audio_text_ca_check <checkpoint.safetensors> <fixture.safetensors> [token_limit]", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: block0_audio_text_ca_check <checkpoint.safetensors> <fixture.safetensors> [token_limit]", .{});
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

    var attn2_x_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_text_ca.attn2_x", sharding);
    defer attn2_x_buf.deinit();
    var context_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_text_ca.context", sharding);
    defer context_buf.deinit();
    var attn2_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_text_ca.attn2_out", sharding);
    defer attn2_ref_buf.deinit();

    if (token_limit) |lim| {
        attn2_x_buf = try check_utils.sliceTokenPrefix(io, platform, attn2_x_buf, sharding, lim);
        context_buf = try check_utils.sliceTokenPrefix(io, platform, context_buf, sharding, lim);
        attn2_ref_buf = try check_utils.sliceTokenPrefix(io, platform, attn2_ref_buf, sharding, lim);
        std.log.info("Using token_limit={d}", .{lim});
    }

    var attn2_shape = model.initBlock0AttentionParams(ckpt_store.view(), .audio_attn2);

    std.log.info("Compiling block0 audio text cross-attn graph...", .{});
    var attn2_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0AudioAttn2WithContext,
        .{
            zml.Tensor.fromShape(attn2_x_buf.shape()),
            zml.Tensor.fromShape(context_buf.shape()),
            attn2_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer attn2_exe.deinit();
    std.log.info("Compile completed", .{});

    var attn2_params = try zml.io.load(model.Attention.Params, &attn2_shape, allocator, io, platform, .{
        .store = &ckpt_store,
        .shardings = &.{sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0AttentionBuffers(&attn2_params);

    var attn2_args = try attn2_exe.args(allocator);
    defer attn2_args.deinit(allocator);
    var attn2_res = try attn2_exe.results(allocator);
    defer attn2_res.deinit(allocator);

    attn2_args.set(.{ attn2_x_buf, context_buf, attn2_params });
    std.log.info("Executing audio attn2 forward...", .{});
    attn2_exe.call(attn2_args, &attn2_res);

    var attn2_computed = attn2_res.get(zml.Buffer);
    defer attn2_computed.deinit();

    try zml.testing.expectClose(io, attn2_computed, attn2_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M4-B audio_attn2(attn2_x, context) parity PASSED", .{});

    // ── Optional full-residual test ─────────────────────────────────────────
    const has_residual =
        fix_store.view().hasKey("block0_audio_text_ca.ax_in") and
        fix_store.view().hasKey("block0_audio_text_ca.text_ca_out") and
        fix_store.view().hasKey("block0_audio_text_ca.ax_out");

    if (!has_residual) {
        std.log.info("Skipping residual test (ax_in/text_ca_out/ax_out absent).", .{});
        std.log.info("M4-B block0 audio text cross-attn parity PASSED (attn-only mode)", .{});
        return;
    }

    var ax_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_text_ca.ax_in", sharding);
    defer ax_in_buf.deinit();
    var ca_out_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_text_ca.text_ca_out", sharding);
    defer ca_out_buf.deinit();
    var ax_out_ref = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_text_ca.ax_out", sharding);
    defer ax_out_ref.deinit();

    if (token_limit) |lim| {
        ax_in_buf = try check_utils.sliceTokenPrefix(io, platform, ax_in_buf, sharding, lim);
        ca_out_buf = try check_utils.sliceTokenPrefix(io, platform, ca_out_buf, sharding, lim);
        ax_out_ref = try check_utils.sliceTokenPrefix(io, platform, ax_out_ref, sharding, lim);
    }

    std.log.info("Compiling audio text-ca residual algebra graph...", .{});
    var res_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0AudioTextCaResidualFromDelta,
        .{
            zml.Tensor.fromShape(ax_in_buf.shape()),
            zml.Tensor.fromShape(ca_out_buf.shape()),
        },
        .{ .shardings = &.{sharding} },
    );
    defer res_exe.deinit();

    var res_args = try res_exe.args(allocator);
    defer res_args.deinit(allocator);
    var res_results = try res_exe.results(allocator);
    defer res_results.deinit(allocator);

    res_args.set(.{ ax_in_buf, ca_out_buf });
    std.log.info("Executing audio text-ca residual...", .{});
    res_exe.call(res_args, &res_results);

    var ax_out_computed = res_results.get(zml.Buffer);
    defer ax_out_computed.deinit();

    try zml.testing.expectClose(io, ax_out_computed, ax_out_ref, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M4-B ax + audio_text_ca_out residual parity PASSED", .{});
    std.log.info("M4-B block0 audio text cross-attn parity PASSED (full residual mode)", .{});
}
