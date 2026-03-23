/// Native block-0 parity checker: inline-AdaLN full stream composition.
///
/// Verifies:
///   1) forwardBlock0NativeVideo(...) == block0_native.vx_out
///   2) forwardBlock0NativeAudio(...) == block0_native.ax_out
///
/// Unlike block0_full_check, this checker does not consume precomputed module
/// inputs such as norm_vx/vx_scaled/gates. It validates inline AdaLN wiring from
/// scale-shift tables plus timestep embeddings.

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

    std.log.info("Native block0 full stream parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next();

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: block0_native_check <checkpoint.safetensors> <fixture.safetensors>", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: block0_native_check <checkpoint.safetensors> <fixture.safetensors>", .{});
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

    // Shared native inputs.
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

    var vx_out_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.vx_out", sharding);
    defer vx_out_ref_buf.deinit();
    var ax_out_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_native.ax_out", sharding);
    defer ax_out_ref_buf.deinit();

    var params_shape = model.initBlock0FullParams(ckpt_store.view());
    var params_bufs = try zml.io.load(model.Block0FullParams, &params_shape, allocator, io, platform, .{
        .store = &ckpt_store,
        .shardings = &.{sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0FullBuffers(&params_bufs);

    std.log.info("Compiling native video graph...", .{});
    var video_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0NativeVideo,
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
    defer video_exe.deinit();

    var video_args = try video_exe.args(allocator);
    defer video_args.deinit(allocator);
    var video_res = try video_exe.results(allocator);
    defer video_res.deinit(allocator);

    video_args.set(.{
        vx_in_buf, ax_in_buf, video_timesteps_buf, audio_timesteps_buf,
        v_prompt_timestep_buf, a_prompt_timestep_buf,
        v_pe_cos_buf, v_pe_sin_buf, a_pe_cos_buf, a_pe_sin_buf,
        v_text_ctx_buf, a_text_ctx_buf,
        v_cross_ss_ts_buf, v_cross_gate_ts_buf, a_cross_ss_ts_buf, a_cross_gate_ts_buf,
        a2v_pe_cos_buf, a2v_pe_sin_buf, a2v_k_pe_cos_buf, a2v_k_pe_sin_buf,
        v2a_pe_cos_buf, v2a_pe_sin_buf, v2a_k_pe_cos_buf, v2a_k_pe_sin_buf,
        params_bufs,
    });
    video_exe.call(video_args, &video_res);

    const vx_out_buf = video_res.get(zml.Buffer);
    try zml.testing.expectClose(io, vx_out_buf, vx_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("Native video parity PASSED", .{});

    std.log.info("Compiling native audio graph...", .{});
    var audio_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0NativeAudio,
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
    defer audio_exe.deinit();

    var audio_args = try audio_exe.args(allocator);
    defer audio_args.deinit(allocator);
    var audio_res = try audio_exe.results(allocator);
    defer audio_res.deinit(allocator);

    audio_args.set(.{
        vx_in_buf, ax_in_buf, video_timesteps_buf, audio_timesteps_buf,
        v_prompt_timestep_buf, a_prompt_timestep_buf,
        v_pe_cos_buf, v_pe_sin_buf, a_pe_cos_buf, a_pe_sin_buf,
        v_text_ctx_buf, a_text_ctx_buf,
        v_cross_ss_ts_buf, v_cross_gate_ts_buf, a_cross_ss_ts_buf, a_cross_gate_ts_buf,
        a2v_pe_cos_buf, a2v_pe_sin_buf, a2v_k_pe_cos_buf, a2v_k_pe_sin_buf,
        v2a_pe_cos_buf, v2a_pe_sin_buf, v2a_k_pe_cos_buf, v2a_k_pe_sin_buf,
        params_bufs,
    });
    audio_exe.call(audio_args, &audio_res);

    const ax_out_buf = audio_res.get(zml.Buffer);
    try zml.testing.expectClose(io, ax_out_buf, ax_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("Native audio parity PASSED", .{});

    std.log.info("Native block0 full stream parity PASSED", .{});
}
