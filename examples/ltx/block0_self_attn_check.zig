/// M1 parity checker: video self-attn residual for block 0.
///
/// Tests the equation at the heart of BasicAVTransformerBlock.forward (video path):
///
///   vx_out = vx + attn1(norm_vx, pe) * vgate_msa
///
/// where norm_vx and vgate_msa are pre-computed AdaLN values captured from
/// Python and provided directly as fixture tensors (so this checker does NOT
/// need the full AdaLN pipeline in Zig).
///
/// Two check modes depending on fixture contents:
///   1. Attn1-only (required keys):
///        block0_self_attn.norm_vx, .pe_cos, .pe_sin, .attn1_out
///      → verifies  attn1(norm_vx, pe) == attn1_out
///
///   2. Full residual (optional additional keys):
///        block0_self_attn.vx_in, .vgate_msa, .vx_out
///      → additionally verifies  vx + attn1(norm_vx, pe) * vgate_msa == vx_out
///
/// Generate the fixture with:
///   python examples/ltx/export_block0_self_attn_fixture.py \
///       <replay_trace.pt> <output.safetensors> [--checkpoint-path <ckpt>]
///
/// Run:
///   bazel run //examples/ltx:block0_self_attn_check -- \
///       <stage2_checkpoint.safetensors> <fixture.safetensors>

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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("M1 block0 video self-attn residual parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const stage2_checkpoint_path = it.next() orelse {
        std.log.err(
            "Usage: bazel run //examples/ltx:block0_self_attn_check -- " ++
                "<stage2_checkpoint.safetensors> <fixture.safetensors> [token_limit]",
            .{},
        );
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err(
            "Usage: bazel run //examples/ltx:block0_self_attn_check -- " ++
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

    // ── Load registry / stores ──────────────────────────────────────────────
    var ckpt_registry: zml.safetensors.TensorRegistry =
        zml.safetensors.TensorRegistry.fromPath(allocator, io, stage2_checkpoint_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{stage2_checkpoint_path});
        return err;
    };
    defer ckpt_registry.deinit();

    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_registry);
    defer ckpt_store.deinit();

    var fix_registry: zml.safetensors.TensorRegistry =
        zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
        std.log.err("Failed to open fixture: {s}", .{fixture_path});
        return err;
    };
    defer fix_registry.deinit();

    var fix_store: zml.io.TensorStore = .fromRegistry(allocator, &fix_registry);
    defer fix_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const sharding = try zml.sharding.replicatedSharding(platform);

    // ── Load required fixture tensors ───────────────────────────────────────
    var norm_vx_buf = loadFixtureBuf(allocator, io, platform, &fix_store, "block0_self_attn.norm_vx", sharding) catch |err| {
        std.log.err("Fixture missing key: block0_self_attn.norm_vx", .{});
        return err;
    };
    defer norm_vx_buf.deinit();

    var pe_cos_buf = loadFixtureBuf(allocator, io, platform, &fix_store, "block0_self_attn.pe_cos", sharding) catch |err| {
        std.log.err("Fixture missing key: block0_self_attn.pe_cos", .{});
        return err;
    };
    defer pe_cos_buf.deinit();

    var pe_sin_buf = loadFixtureBuf(allocator, io, platform, &fix_store, "block0_self_attn.pe_sin", sharding) catch |err| {
        std.log.err("Fixture missing key: block0_self_attn.pe_sin", .{});
        return err;
    };
    defer pe_sin_buf.deinit();

    var attn1_out_ref_buf = loadFixtureBuf(allocator, io, platform, &fix_store, "block0_self_attn.attn1_out", sharding) catch |err| {
        std.log.err("Fixture missing key: block0_self_attn.attn1_out", .{});
        return err;
    };
    defer attn1_out_ref_buf.deinit();

    // ── Apply optional token_limit to token-major tensors ───────────────────
    if (token_limit) |lim| {
        norm_vx_buf = try check_utils.sliceTokenPrefix(io, platform, norm_vx_buf, sharding, lim);
        attn1_out_ref_buf = try check_utils.sliceTokenPrefix(io, platform, attn1_out_ref_buf, sharding, lim);
        // PE slicing depends on rank (BHTD vs TH*HD); use the BHTD helper when rank==4.
        if (pe_cos_buf.shape().rank() == 4) {
            pe_cos_buf = try check_utils.sliceTokenPrefixBHTD(io, platform, pe_cos_buf, sharding, lim);
            pe_sin_buf = try check_utils.sliceTokenPrefixBHTD(io, platform, pe_sin_buf, sharding, lim);
        } else {
            pe_cos_buf = try check_utils.sliceTokenPrefix(io, platform, pe_cos_buf, sharding, lim);
            pe_sin_buf = try check_utils.sliceTokenPrefix(io, platform, pe_sin_buf, sharding, lim);
        }
        std.log.info("Using token_limit={d}; sliced fixture tensors", .{lim});
    }

    // ── Load attn1 parameters ───────────────────────────────────────────────
    var attn1_params_shape = model.initBlock0AttentionParams(ckpt_store.view(), .attn1);

    std.log.info("Compiling block0 video self-attn graph...", .{});
    var attn1_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0VideoSelfAttn,
        .{
            zml.Tensor.fromShape(norm_vx_buf.shape()),
            zml.Tensor.fromShape(pe_cos_buf.shape()),
            zml.Tensor.fromShape(pe_sin_buf.shape()),
            attn1_params_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer attn1_exe.deinit();
    std.log.info("Compile completed", .{});

    std.log.info("Loading attn1 parameters from checkpoint...", .{});
    var attn1_params_buffers = try zml.io.load(model.Attention.Params, &attn1_params_shape, allocator, io, platform, .{
        .store = &ckpt_store,
        .shardings = &.{sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0AttentionBuffers(&attn1_params_buffers);
    std.log.info("Parameter load completed", .{});

    var attn1_args = try attn1_exe.args(allocator);
    defer attn1_args.deinit(allocator);

    var attn1_results = try attn1_exe.results(allocator);
    defer attn1_results.deinit(allocator);

    attn1_args.set(.{ norm_vx_buf, pe_cos_buf, pe_sin_buf, attn1_params_buffers });
    std.log.info("Executing block0 attn1 forward...", .{});
    attn1_exe.call(attn1_args, &attn1_results);
    std.log.info("Execution completed", .{});

    var attn1_out_computed = attn1_results.get(zml.Buffer);
    defer attn1_out_computed.deinit();

    try zml.testing.expectClose(io, attn1_out_computed, attn1_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M1 attn1(norm_vx, pe) parity PASSED", .{});

    // ── Optional full-residual test ─────────────────────────────────────────
    const has_residual_keys =
        fix_store.view().hasKey("block0_self_attn.vx_in") and
        fix_store.view().hasKey("block0_self_attn.vgate_msa") and
        fix_store.view().hasKey("block0_self_attn.vx_out");

    if (!has_residual_keys) {
        std.log.info(
            "Skipping residual test (keys block0_self_attn.vx_in/.vgate_msa/.vx_out absent). " ++
                "Re-generate fixture with --checkpoint-path to enable the full M1 residual check.",
            .{},
        );
        std.log.info("M1 block0 video self-attn parity PASSED (attn1-only mode)", .{});
        return;
    }

    var vx_in_buf = try loadFixtureBuf(allocator, io, platform, &fix_store, "block0_self_attn.vx_in", sharding);
    defer vx_in_buf.deinit();

    var vgate_buf = try loadFixtureBuf(allocator, io, platform, &fix_store, "block0_self_attn.vgate_msa", sharding);
    defer vgate_buf.deinit();

    var vx_out_ref_buf = try loadFixtureBuf(allocator, io, platform, &fix_store, "block0_self_attn.vx_out", sharding);
    defer vx_out_ref_buf.deinit();

    if (token_limit) |lim| {
        vx_in_buf = try check_utils.sliceTokenPrefix(io, platform, vx_in_buf, sharding, lim);
        vx_out_ref_buf = try check_utils.sliceTokenPrefix(io, platform, vx_out_ref_buf, sharding, lim);
        // vgate_msa is [B, 1, D] — token dim is already 1, no slicing needed.
    }

    std.log.info("Compiling block0 full self-attn residual graph...", .{});
    var residual_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0VideoSelfAttnResidualFromAttnOut,
        .{
            zml.Tensor.fromShape(vx_in_buf.shape()),
            zml.Tensor.fromShape(attn1_out_ref_buf.shape()),
            zml.Tensor.fromShape(vgate_buf.shape()),
        },
        .{ .shardings = &.{sharding} },
    );
    defer residual_exe.deinit();

    var residual_args = try residual_exe.args(allocator);
    defer residual_args.deinit(allocator);

    var residual_results = try residual_exe.results(allocator);
    defer residual_results.deinit(allocator);

    residual_args.set(.{ vx_in_buf, attn1_out_ref_buf, vgate_buf });
    std.log.info("Executing block0 video self-attn residual...", .{});
    residual_exe.call(residual_args, &residual_results);

    var vx_out_computed = residual_results.get(zml.Buffer);
    defer vx_out_computed.deinit();

    try zml.testing.expectClose(io, vx_out_computed, vx_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("✓ M1 vx + attn1(norm_vx, pe) * vgate_msa residual parity PASSED", .{});
    std.log.info("M1 block0 video self-attn parity PASSED (full residual mode)", .{});
}

fn loadFixtureBuf(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    fix_store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    return check_utils.loadBufferFromStore(allocator, io, platform, fix_store, key, sharding);
}
