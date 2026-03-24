/// M6 parity checker: full block-0 video + audio stream composition.
///
/// Verifies:
///   1) forwardBlock0VideoStream(vx_in, norm_vx, ...) == vx_out
///   2) forwardBlock0AudioStream(ax_in, norm_ax, ...) == ax_out
///
/// Both streams use AdaLN-pre-normalised inputs from fixture (norm_vx, norm_ax,
/// vx_scaled, ax_scaled) and cross-stream contexts (a2v_ctx, v2a_ctx) captured
/// at the exact execution point from the Python reference.

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

    std.log.info("M6 block0 full stream parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next();

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: block0_full_check <checkpoint.safetensors> <fixture.safetensors>", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: block0_full_check <checkpoint.safetensors> <fixture.safetensors>", .{});
        return error.InvalidArgs;
    };

    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    var fix_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
        std.log.err("Failed to open fixture: {s}", .{fixture_path});
        return err;
    };
    defer fix_reg.deinit();
    var fix_store: zml.io.TensorStore = .fromRegistry(allocator, &fix_reg);
    defer fix_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ── Load video stream fixture tensors ───────────────────────────────────
    var vx_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.vx_in", sharding);
    defer vx_in_buf.deinit();
    var norm_vx_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.norm_vx", sharding);
    defer norm_vx_buf.deinit();
    var v_text_x_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v_text_x", sharding);
    defer v_text_x_buf.deinit();
    var v_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v_pe_cos", sharding);
    defer v_pe_cos_buf.deinit();
    var v_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v_pe_sin", sharding);
    defer v_pe_sin_buf.deinit();
    var vgate_msa_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.vgate_msa", sharding);
    defer vgate_msa_buf.deinit();
    var vgate_text_ca_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.vgate_text_ca", sharding);
    defer vgate_text_ca_buf.deinit();
    var v_text_ctx_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v_text_ctx", sharding);
    defer v_text_ctx_buf.deinit();
    var a2v_x_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a2v_x", sharding);
    defer a2v_x_buf.deinit();
    var a2v_ctx_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a2v_ctx", sharding);
    defer a2v_ctx_buf.deinit();
    var a2v_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a2v_pe_cos", sharding);
    defer a2v_pe_cos_buf.deinit();
    var a2v_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a2v_pe_sin", sharding);
    defer a2v_pe_sin_buf.deinit();
    var a2v_k_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a2v_k_pe_cos", sharding);
    defer a2v_k_pe_cos_buf.deinit();
    var a2v_k_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a2v_k_pe_sin", sharding);
    defer a2v_k_pe_sin_buf.deinit();
    var a2v_gate_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a2v_gate", sharding);
    defer a2v_gate_buf.deinit();
    var a2v_mask_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a2v_mask", sharding);
    defer a2v_mask_buf.deinit();
    var vx_scaled_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.vx_scaled", sharding);
    defer vx_scaled_buf.deinit();
    var vgate_mlp_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.vgate_mlp", sharding);
    defer vgate_mlp_buf.deinit();
    var vx_out_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.vx_out", sharding);
    defer vx_out_ref_buf.deinit();

    // ── Load audio stream fixture tensors ───────────────────────────────────
    var ax_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.ax_in", sharding);
    defer ax_in_buf.deinit();
    var norm_ax_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.norm_ax", sharding);
    defer norm_ax_buf.deinit();
    var a_text_x_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a_text_x", sharding);
    defer a_text_x_buf.deinit();
    var a_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a_pe_cos", sharding);
    defer a_pe_cos_buf.deinit();
    var a_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a_pe_sin", sharding);
    defer a_pe_sin_buf.deinit();
    var agate_msa_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.agate_msa", sharding);
    defer agate_msa_buf.deinit();
    var agate_text_ca_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.agate_text_ca", sharding);
    defer agate_text_ca_buf.deinit();
    var a_text_ctx_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.a_text_ctx", sharding);
    defer a_text_ctx_buf.deinit();
    var v2a_x_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v2a_x", sharding);
    defer v2a_x_buf.deinit();
    var v2a_ctx_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v2a_ctx", sharding);
    defer v2a_ctx_buf.deinit();
    var v2a_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v2a_pe_cos", sharding);
    defer v2a_pe_cos_buf.deinit();
    var v2a_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v2a_pe_sin", sharding);
    defer v2a_pe_sin_buf.deinit();
    var v2a_k_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v2a_k_pe_cos", sharding);
    defer v2a_k_pe_cos_buf.deinit();
    var v2a_k_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v2a_k_pe_sin", sharding);
    defer v2a_k_pe_sin_buf.deinit();
    var v2a_gate_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v2a_gate", sharding);
    defer v2a_gate_buf.deinit();
    var v2a_mask_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.v2a_mask", sharding);
    defer v2a_mask_buf.deinit();
    var ax_scaled_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.ax_scaled", sharding);
    defer ax_scaled_buf.deinit();
    var agate_mlp_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.agate_mlp", sharding);
    defer agate_mlp_buf.deinit();
    var ax_out_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_full.ax_out", sharding);
    defer ax_out_ref_buf.deinit();

    // ── Load and compile Block0FullParams ───────────────────────────────────
    var params_shape = model.initBlock0FullParams(ckpt_store.view());

    std.log.info("Compiling video stream graph...", .{});
    var video_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0VideoStream,
        .{
            zml.Tensor.fromShape(vx_in_buf.shape()),
            zml.Tensor.fromShape(norm_vx_buf.shape()),
            zml.Tensor.fromShape(v_text_x_buf.shape()),
            zml.Tensor.fromShape(v_pe_cos_buf.shape()),
            zml.Tensor.fromShape(v_pe_sin_buf.shape()),
            zml.Tensor.fromShape(vgate_msa_buf.shape()),
            zml.Tensor.fromShape(vgate_text_ca_buf.shape()),
            zml.Tensor.fromShape(v_text_ctx_buf.shape()),
            zml.Tensor.fromShape(a2v_x_buf.shape()),
            zml.Tensor.fromShape(a2v_ctx_buf.shape()),
            zml.Tensor.fromShape(a2v_pe_cos_buf.shape()),
            zml.Tensor.fromShape(a2v_pe_sin_buf.shape()),
            zml.Tensor.fromShape(a2v_k_pe_cos_buf.shape()),
            zml.Tensor.fromShape(a2v_k_pe_sin_buf.shape()),
            zml.Tensor.fromShape(a2v_gate_buf.shape()),
            zml.Tensor.fromShape(a2v_mask_buf.shape()),
            zml.Tensor.fromShape(vx_scaled_buf.shape()),
            zml.Tensor.fromShape(vgate_mlp_buf.shape()),
            params_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer video_exe.deinit();

    var params_bufs = try zml.io.load(model.Block0FullParams, &params_shape, allocator, io, platform, .{
        .store = &ckpt_store,
        .shardings = &.{sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0FullBuffers(&params_bufs);

    var video_args = try video_exe.args(allocator);
    defer video_args.deinit(allocator);
    var video_res = try video_exe.results(allocator);
    defer video_res.deinit(allocator);

    video_args.set(.{
        vx_in_buf, norm_vx_buf, v_text_x_buf, v_pe_cos_buf, v_pe_sin_buf, vgate_msa_buf,
        vgate_text_ca_buf, v_text_ctx_buf, a2v_x_buf, a2v_ctx_buf, a2v_pe_cos_buf, a2v_pe_sin_buf,
        a2v_k_pe_cos_buf, a2v_k_pe_sin_buf, a2v_gate_buf, a2v_mask_buf,
        vx_scaled_buf, vgate_mlp_buf, params_bufs,
    });
    std.log.info("Executing video stream...", .{});
    video_exe.call(video_args, &video_res);

    const vx_out_buf = video_res.get(zml.Buffer);
    try zml.testing.expectClose(io, vx_out_buf, vx_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M6 video stream parity PASSED", .{});

    // ── Audio stream ────────────────────────────────────────────────────────
    std.log.info("Compiling audio stream graph...", .{});
    var audio_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0AudioStream,
        .{
            zml.Tensor.fromShape(ax_in_buf.shape()),
            zml.Tensor.fromShape(norm_ax_buf.shape()),
            zml.Tensor.fromShape(a_text_x_buf.shape()),
            zml.Tensor.fromShape(a_pe_cos_buf.shape()),
            zml.Tensor.fromShape(a_pe_sin_buf.shape()),
            zml.Tensor.fromShape(agate_msa_buf.shape()),
            zml.Tensor.fromShape(agate_text_ca_buf.shape()),
            zml.Tensor.fromShape(a_text_ctx_buf.shape()),
            zml.Tensor.fromShape(v2a_x_buf.shape()),
            zml.Tensor.fromShape(v2a_ctx_buf.shape()),
            zml.Tensor.fromShape(v2a_pe_cos_buf.shape()),
            zml.Tensor.fromShape(v2a_pe_sin_buf.shape()),
            zml.Tensor.fromShape(v2a_k_pe_cos_buf.shape()),
            zml.Tensor.fromShape(v2a_k_pe_sin_buf.shape()),
            zml.Tensor.fromShape(v2a_gate_buf.shape()),
            zml.Tensor.fromShape(v2a_mask_buf.shape()),
            zml.Tensor.fromShape(ax_scaled_buf.shape()),
            zml.Tensor.fromShape(agate_mlp_buf.shape()),
            params_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer audio_exe.deinit();

    var audio_args = try audio_exe.args(allocator);
    defer audio_args.deinit(allocator);
    var audio_res = try audio_exe.results(allocator);
    defer audio_res.deinit(allocator);

    audio_args.set(.{
        ax_in_buf, norm_ax_buf, a_text_x_buf, a_pe_cos_buf, a_pe_sin_buf, agate_msa_buf,
        agate_text_ca_buf, a_text_ctx_buf, v2a_x_buf, v2a_ctx_buf, v2a_pe_cos_buf, v2a_pe_sin_buf,
        v2a_k_pe_cos_buf, v2a_k_pe_sin_buf, v2a_gate_buf, v2a_mask_buf,
        ax_scaled_buf, agate_mlp_buf, params_bufs,
    });
    std.log.info("Executing audio stream...", .{});
    audio_exe.call(audio_args, &audio_res);

    const ax_out_buf = audio_res.get(zml.Buffer);
    try zml.testing.expectClose(io, ax_out_buf, ax_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M6 audio stream parity PASSED", .{});

    std.log.info("M6 block0 full stream parity PASSED", .{});
}
