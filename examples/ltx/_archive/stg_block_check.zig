/// STG block parity checker: V-passthrough self-attention variant.
///
/// Verifies that forwardBlock0NativeSTG(inputs) produces outputs matching
/// the Python reference where self-attention is replaced by to_out(to_v(x)).
///
/// Usage: stg_block_check <checkpoint.safetensors> <stg_fixture.safetensors>
///
/// The fixture must contain:
///   - All block0_native.* inputs (reused from the normal block fixture)
///   - stg_block.vx_out, stg_block.ax_out (Python STG reference outputs)

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

    std.log.info("STG block (V-passthrough) parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next();

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: stg_block_check <checkpoint.safetensors> <stg_fixture.safetensors>", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: stg_block_check <checkpoint.safetensors> <stg_fixture.safetensors>", .{});
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

    // ── Load shared block inputs from fixture ─────────────────────────────
    var vx_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.vx_in", sharding);
    defer vx_in_buf.deinit();
    var ax_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.ax_in", sharding);
    defer ax_in_buf.deinit();

    var video_timesteps_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.video_timesteps", sharding);
    defer video_timesteps_buf.deinit();
    var audio_timesteps_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.audio_timesteps", sharding);
    defer audio_timesteps_buf.deinit();
    var v_prompt_timestep_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.v_prompt_timestep", sharding);
    defer v_prompt_timestep_buf.deinit();
    var a_prompt_timestep_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.a_prompt_timestep", sharding);
    defer a_prompt_timestep_buf.deinit();

    var v_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.v_pe_cos", sharding);
    defer v_pe_cos_buf.deinit();
    var v_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.v_pe_sin", sharding);
    defer v_pe_sin_buf.deinit();
    var a_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.a_pe_cos", sharding);
    defer a_pe_cos_buf.deinit();
    var a_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.a_pe_sin", sharding);
    defer a_pe_sin_buf.deinit();

    var v_text_ctx_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.v_text_ctx", sharding);
    defer v_text_ctx_buf.deinit();
    var a_text_ctx_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.a_text_ctx", sharding);
    defer a_text_ctx_buf.deinit();

    var v_cross_ss_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.v_cross_ss_ts", sharding);
    defer v_cross_ss_ts_buf.deinit();
    var v_cross_gate_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.v_cross_gate_ts", sharding);
    defer v_cross_gate_ts_buf.deinit();
    var a_cross_ss_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.a_cross_ss_ts", sharding);
    defer a_cross_ss_ts_buf.deinit();
    var a_cross_gate_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.a_cross_gate_ts", sharding);
    defer a_cross_gate_ts_buf.deinit();

    var a2v_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.a2v_pe_cos", sharding);
    defer a2v_pe_cos_buf.deinit();
    var a2v_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.a2v_pe_sin", sharding);
    defer a2v_pe_sin_buf.deinit();
    var a2v_k_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.a2v_k_pe_cos", sharding);
    defer a2v_k_pe_cos_buf.deinit();
    var a2v_k_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.a2v_k_pe_sin", sharding);
    defer a2v_k_pe_sin_buf.deinit();

    var v2a_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.v2a_pe_cos", sharding);
    defer v2a_pe_cos_buf.deinit();
    var v2a_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.v2a_pe_sin", sharding);
    defer v2a_pe_sin_buf.deinit();
    var v2a_k_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.v2a_k_pe_cos", sharding);
    defer v2a_k_pe_cos_buf.deinit();
    var v2a_k_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.v2a_k_pe_sin", sharding);
    defer v2a_k_pe_sin_buf.deinit();

    // ── Load STG reference outputs ────────────────────────────────────────
    var vx_out_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "stg_block.vx_out", sharding);
    defer vx_out_ref_buf.deinit();
    var ax_out_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "stg_block.ax_out", sharding);
    defer ax_out_ref_buf.deinit();

    // Also load normal reference for diagnostic (should differ from STG)
    var vx_out_normal_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.vx_out", sharding);
    defer vx_out_normal_buf.deinit();
    var ax_out_normal_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.ax_out", sharding);
    defer ax_out_normal_buf.deinit();

    // ── Load model weights ────────────────────────────────────────────────
    var params_shape = model.initBlock0FullParams(ckpt_store.view());
    var params_bufs = try zml.io.load(model.Block0FullParams, &params_shape, allocator, io, platform, .{
        .store = &ckpt_store,
        .shardings = &.{sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0FullBuffers(&params_bufs);

    // ── Compile and run forwardBlock0NativeSTG ────────────────────────────
    std.log.info("Compiling STG (V-passthrough) block graph...", .{});
    var stg_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0NativeSTG,
        .{
            zml.Tensor.fromShape(vx_in_buf.shape()),
            zml.Tensor.fromShape(ax_in_buf.shape()),
            zml.Tensor.fromShape(video_timesteps_buf.shape()),
            zml.Tensor.fromShape(audio_timesteps_buf.shape()),
            zml.Tensor.fromShape(v_prompt_timestep_buf.shape()),
            zml.Tensor.fromShape(a_prompt_timestep_buf.shape()),
            zml.Tensor.fromShape(v_pe_cos_buf.shape()),
            zml.Tensor.fromShape(v_pe_sin_buf.shape()),
            zml.Tensor.fromShape(a_pe_cos_buf.shape()),
            zml.Tensor.fromShape(a_pe_sin_buf.shape()),
            zml.Tensor.fromShape(v_text_ctx_buf.shape()),
            zml.Tensor.fromShape(a_text_ctx_buf.shape()),
            zml.Tensor.fromShape(v_cross_ss_ts_buf.shape()),
            zml.Tensor.fromShape(v_cross_gate_ts_buf.shape()),
            zml.Tensor.fromShape(a_cross_ss_ts_buf.shape()),
            zml.Tensor.fromShape(a_cross_gate_ts_buf.shape()),
            zml.Tensor.fromShape(a2v_pe_cos_buf.shape()),
            zml.Tensor.fromShape(a2v_pe_sin_buf.shape()),
            zml.Tensor.fromShape(a2v_k_pe_cos_buf.shape()),
            zml.Tensor.fromShape(a2v_k_pe_sin_buf.shape()),
            zml.Tensor.fromShape(v2a_pe_cos_buf.shape()),
            zml.Tensor.fromShape(v2a_pe_sin_buf.shape()),
            zml.Tensor.fromShape(v2a_k_pe_cos_buf.shape()),
            zml.Tensor.fromShape(v2a_k_pe_sin_buf.shape()),
            params_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer stg_exe.deinit();

    var stg_args = try stg_exe.args(allocator);
    defer stg_args.deinit(allocator);
    var stg_res = try stg_exe.results(allocator);
    defer stg_res.deinit(allocator);

    stg_args.set(.{
        vx_in_buf, ax_in_buf, video_timesteps_buf, audio_timesteps_buf,
        v_prompt_timestep_buf, a_prompt_timestep_buf,
        v_pe_cos_buf, v_pe_sin_buf, a_pe_cos_buf, a_pe_sin_buf,
        v_text_ctx_buf, a_text_ctx_buf,
        v_cross_ss_ts_buf, v_cross_gate_ts_buf, a_cross_ss_ts_buf, a_cross_gate_ts_buf,
        a2v_pe_cos_buf, a2v_pe_sin_buf, a2v_k_pe_cos_buf, a2v_k_pe_sin_buf,
        v2a_pe_cos_buf, v2a_pe_sin_buf, v2a_k_pe_cos_buf, v2a_k_pe_sin_buf,
        params_bufs,
    });

    std.log.info("Running STG forward...", .{});
    stg_exe.call(stg_args, &stg_res);

    const out = stg_res.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));

    // ── Sanity check: STG outputs should differ from normal ───────────────
    std.log.info("Sanity: STG vs Normal outputs (should differ)...", .{});
    const v_sanity = try check_utils.compareBuffersExtended(io, out.vx_out, vx_out_normal_buf, 1e-3, 1e-2);
    std.log.info("  Video STG vs Normal: cos_sim={d:.6} close={d:.6} max_abs={d:.4}", .{
        v_sanity.cosine_similarity, v_sanity.close_fraction, v_sanity.max_abs_error,
    });
    const a_sanity = try check_utils.compareBuffersExtended(io, out.ax_out, ax_out_normal_buf, 1e-3, 1e-2);
    std.log.info("  Audio STG vs Normal: cos_sim={d:.6} close={d:.6} max_abs={d:.4}", .{
        a_sanity.cosine_similarity, a_sanity.close_fraction, a_sanity.max_abs_error,
    });

    if (v_sanity.cosine_similarity > 0.9999 and a_sanity.cosine_similarity > 0.9999) {
        std.log.warn("STG outputs nearly identical to normal — V-passthrough may not be active!", .{});
    }

    // ── Parity check: STG outputs vs Python reference ─────────────────────
    std.log.info("Checking STG parity: Zig vs Python reference...", .{});

    const v_metrics = try check_utils.compareBuffersExtended(io, out.vx_out, vx_out_ref_buf, 1e-3, 1e-2);
    std.log.info("  Video: cos_sim={d:.6} close={d:.6} max_abs={d:.4} mean_abs={d:.6} rmse={d:.6}", .{
        v_metrics.cosine_similarity, v_metrics.close_fraction, v_metrics.max_abs_error,
        v_metrics.mean_abs_error,    v_metrics.rmse_error,
    });

    const a_metrics = try check_utils.compareBuffersExtended(io, out.ax_out, ax_out_ref_buf, 1e-3, 1e-2);
    std.log.info("  Audio: cos_sim={d:.6} close={d:.6} max_abs={d:.4} mean_abs={d:.6} rmse={d:.6}", .{
        a_metrics.cosine_similarity, a_metrics.close_fraction, a_metrics.max_abs_error,
        a_metrics.mean_abs_error,    a_metrics.rmse_error,
    });

    // Apply same tolerance as block0_native_check
    try zml.testing.expectClose(io, out.vx_out, vx_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("STG video parity PASSED", .{});

    try zml.testing.expectClose(io, out.ax_out, ax_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("STG audio parity PASSED", .{});

    std.log.info("STG block (V-passthrough) parity PASSED", .{});
}
