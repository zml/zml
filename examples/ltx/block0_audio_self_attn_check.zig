/// M4-A parity checker: block-0 audio self-attn residual.
///
/// Two check modes:
///   1. Attn1-only (required keys):
///        block0_audio_self_attn.norm_ax, .pe_cos, .pe_sin, .attn1_out
///      → verifies  audio_attn1(norm_ax, pe) == attn1_out
///
///   2. Full residual (optional additional keys):
///        block0_audio_self_attn.ax_in, .agate_msa, .ax_out
///      → additionally verifies  ax + audio_attn1(norm_ax, pe) * agate_msa == ax_out
///
/// Generate fixture with:
///   python examples/ltx/export_block0_audio_self_attn_fixture.py <trace.pt> <fixture.safetensors>
///
/// Run:
///   bazel run //examples/ltx:block0_audio_self_attn_check -- \
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

    std.log.info("M4-A block0 audio self-attn residual parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next();

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: block0_audio_self_attn_check <checkpoint.safetensors> <fixture.safetensors> [token_limit]", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: block0_audio_self_attn_check <checkpoint.safetensors> <fixture.safetensors> [token_limit]", .{});
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

    // ── Required fixture tensors ────────────────────────────────────────────
    var norm_ax_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_self_attn.norm_ax", sharding);
    defer norm_ax_buf.deinit();

    var attn1_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_self_attn.attn1_out", sharding);
    defer attn1_ref_buf.deinit();

    const has_pe = fix_store.view().hasKey("block0_audio_self_attn.pe_cos") and
        fix_store.view().hasKey("block0_audio_self_attn.pe_sin");

    var pe_cos_buf: ?zml.Buffer = null;
    var pe_sin_buf: ?zml.Buffer = null;
    if (has_pe) {
        pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_self_attn.pe_cos", sharding);
        pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_self_attn.pe_sin", sharding);
    }
    defer if (pe_cos_buf) |*b| b.deinit();
    defer if (pe_sin_buf) |*b| b.deinit();

    if (token_limit) |lim| {
        norm_ax_buf = try check_utils.sliceTokenPrefix(io, platform, norm_ax_buf, sharding, lim);
        attn1_ref_buf = try check_utils.sliceTokenPrefix(io, platform, attn1_ref_buf, sharding, lim);
        if (pe_cos_buf) |pe| {
            pe_cos_buf = if (pe.shape().rank() == 4)
                try check_utils.sliceTokenPrefixBHTD(io, platform, pe, sharding, lim)
            else
                try check_utils.sliceTokenPrefix(io, platform, pe, sharding, lim);
        }
        if (pe_sin_buf) |pe| {
            pe_sin_buf = if (pe.shape().rank() == 4)
                try check_utils.sliceTokenPrefixBHTD(io, platform, pe, sharding, lim)
            else
                try check_utils.sliceTokenPrefix(io, platform, pe, sharding, lim);
        }
        std.log.info("Using token_limit={d}", .{lim});
    }

    // ── Load audio_attn1 params ─────────────────────────────────────────────
    var attn1_shape = model.initBlock0AttentionParams(ckpt_store.view(), .audio_attn1);

    std.log.info("Compiling block0 audio self-attn graph...", .{});
    var attn1_exe = if (has_pe)
        try platform.compileFn(
            allocator,
            io,
            model.forwardBlock0AudioSelfAttn,
            .{
                zml.Tensor.fromShape(norm_ax_buf.shape()),
                zml.Tensor.fromShape(pe_cos_buf.?.shape()),
                zml.Tensor.fromShape(pe_sin_buf.?.shape()),
                attn1_shape,
            },
            .{ .shardings = &.{sharding} },
        )
    else
        try platform.compileFn(
            allocator,
            io,
            model.forwardBlock0AudioAttn1,
            .{ zml.Tensor.fromShape(norm_ax_buf.shape()), attn1_shape },
            .{ .shardings = &.{sharding} },
        );
    defer attn1_exe.deinit();
    std.log.info("Compile completed", .{});

    var attn1_params = try zml.io.load(model.Attention.Params, &attn1_shape, allocator, io, platform, .{
        .store = &ckpt_store,
        .shardings = &.{sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0AttentionBuffers(&attn1_params);

    var attn1_args = try attn1_exe.args(allocator);
    defer attn1_args.deinit(allocator);
    var attn1_res = try attn1_exe.results(allocator);
    defer attn1_res.deinit(allocator);

    if (has_pe) {
        attn1_args.set(.{ norm_ax_buf, pe_cos_buf.?, pe_sin_buf.?, attn1_params });
    } else {
        attn1_args.set(.{ norm_ax_buf, attn1_params });
    }
    std.log.info("Executing audio attn1 forward...", .{});
    attn1_exe.call(attn1_args, &attn1_res);

    var attn1_computed = attn1_res.get(zml.Buffer);
    defer attn1_computed.deinit();

    try zml.testing.expectClose(io, attn1_computed, attn1_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M4-A audio_attn1(norm_ax, pe) parity PASSED", .{});

    // ── Optional full-residual test ─────────────────────────────────────────
    const has_residual =
        fix_store.view().hasKey("block0_audio_self_attn.ax_in") and
        fix_store.view().hasKey("block0_audio_self_attn.agate_msa") and
        fix_store.view().hasKey("block0_audio_self_attn.ax_out");

    if (!has_residual) {
        std.log.info("Skipping residual test (ax_in/agate_msa/ax_out absent).", .{});
        std.log.info("M4-A block0 audio self-attn parity PASSED (attn-only mode)", .{});
        return;
    }

    var ax_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_self_attn.ax_in", sharding);
    defer ax_in_buf.deinit();
    var agate_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_self_attn.agate_msa", sharding);
    defer agate_buf.deinit();
    var ax_out_ref = try loadBuf(allocator, io, platform, &fix_store, "block0_audio_self_attn.ax_out", sharding);
    defer ax_out_ref.deinit();

    if (token_limit) |lim| {
        ax_in_buf = try check_utils.sliceTokenPrefix(io, platform, ax_in_buf, sharding, lim);
        ax_out_ref = try check_utils.sliceTokenPrefix(io, platform, ax_out_ref, sharding, lim);
    }

    std.log.info("Compiling audio self-attn residual algebra graph...", .{});
    var res_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0AudioSelfAttnResidualFromAttnOut,
        .{
            zml.Tensor.fromShape(ax_in_buf.shape()),
            zml.Tensor.fromShape(attn1_ref_buf.shape()),
            zml.Tensor.fromShape(agate_buf.shape()),
        },
        .{ .shardings = &.{sharding} },
    );
    defer res_exe.deinit();

    var res_args = try res_exe.args(allocator);
    defer res_args.deinit(allocator);
    var res_results = try res_exe.results(allocator);
    defer res_results.deinit(allocator);

    res_args.set(.{ ax_in_buf, attn1_ref_buf, agate_buf });
    std.log.info("Executing audio self-attn residual...", .{});
    res_exe.call(res_args, &res_results);

    var ax_out_computed = res_results.get(zml.Buffer);
    defer ax_out_computed.deinit();

    try zml.testing.expectClose(io, ax_out_computed, ax_out_ref, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M4-A ax + audio_attn1(norm_ax, pe) * agate_msa residual parity PASSED", .{});
    std.log.info("M4-A block0 audio self-attn parity PASSED (full residual mode)", .{});
}
