/// Native block-slice chain-trend checker (48 contiguous blocks).
///
/// Answers the key question: does audio error continue to accumulate linearly across
/// blocks 0..47, or does it saturate / explode?
///
/// Strategy: compile a SINGLE block executable and run it 48 times, feeding each
/// block's computed output into the next block's input (free-running chain).
/// Per-block extended stats are printed so the accumulation slope is visible.
/// The final computed outputs are compared against the fixture's reference outputs.
///
/// Checkpoint expectations:
///   transformer_blocks are re-indexed locally as transformer_blocks.0..47.
///
/// Usage:
///   ulimit -s unlimited
///   bazel run --@zml//platforms:cuda=true //examples/ltx:block_slice_48_check -- \
///     <checkpoint.safetensors> <fixture.safetensors> [--extended-error-stats]

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

comptime {
    @setEvalBranchQuota(500000);
}

pub const std_options: std.Options = .{
    .log_level = .info,
};

const N = 48;

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

const CheckContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    params_shape: model.BlockSlice48FullParams,
    params_bufs: [N]zml.Bufferized(model.Block0FullParams),

    vx_in_buf: zml.Buffer,
    ax_in_buf: zml.Buffer,
    video_timesteps_buf: zml.Buffer,
    audio_timesteps_buf: zml.Buffer,
    v_prompt_timestep_buf: zml.Buffer,
    a_prompt_timestep_buf: zml.Buffer,
    v_pe_cos_buf: zml.Buffer,
    v_pe_sin_buf: zml.Buffer,
    a_pe_cos_buf: zml.Buffer,
    a_pe_sin_buf: zml.Buffer,
    v_text_ctx_buf: zml.Buffer,
    a_text_ctx_buf: zml.Buffer,
    v_cross_ss_ts_buf: zml.Buffer,
    v_cross_gate_ts_buf: zml.Buffer,
    a_cross_ss_ts_buf: zml.Buffer,
    a_cross_gate_ts_buf: zml.Buffer,
    a2v_pe_cos_buf: zml.Buffer,
    a2v_pe_sin_buf: zml.Buffer,
    a2v_k_pe_cos_buf: zml.Buffer,
    a2v_k_pe_sin_buf: zml.Buffer,
    a2v_mask_block_bufs: [N]?zml.Buffer,
    v2a_pe_cos_buf: zml.Buffer,
    v2a_pe_sin_buf: zml.Buffer,
    v2a_k_pe_cos_buf: zml.Buffer,
    v2a_k_pe_sin_buf: zml.Buffer,
    v2a_mask_block_bufs: [N]?zml.Buffer,
    vx_out_ref_buf: zml.Buffer,
    ax_out_ref_buf: zml.Buffer,
    vx_block_in_ref_bufs: [N]?zml.Buffer,
    ax_block_in_ref_bufs: [N]?zml.Buffer,
    vx_block_out_ref_bufs: [N]?zml.Buffer,
    ax_block_out_ref_bufs: [N]?zml.Buffer,

    fn deinit(self: *CheckContext) void {
        self.vx_in_buf.deinit();
        self.ax_in_buf.deinit();
        self.video_timesteps_buf.deinit();
        self.audio_timesteps_buf.deinit();
        self.v_prompt_timestep_buf.deinit();
        self.a_prompt_timestep_buf.deinit();
        self.v_pe_cos_buf.deinit();
        self.v_pe_sin_buf.deinit();
        self.a_pe_cos_buf.deinit();
        self.a_pe_sin_buf.deinit();
        self.v_text_ctx_buf.deinit();
        self.a_text_ctx_buf.deinit();
        self.v_cross_ss_ts_buf.deinit();
        self.v_cross_gate_ts_buf.deinit();
        self.a_cross_ss_ts_buf.deinit();
        self.a_cross_gate_ts_buf.deinit();
        self.a2v_pe_cos_buf.deinit();
        self.a2v_pe_sin_buf.deinit();
        self.a2v_k_pe_cos_buf.deinit();
        self.a2v_k_pe_sin_buf.deinit();
        for (&self.a2v_mask_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        self.v2a_pe_cos_buf.deinit();
        self.v2a_pe_sin_buf.deinit();
        self.v2a_k_pe_cos_buf.deinit();
        self.v2a_k_pe_sin_buf.deinit();
        for (&self.v2a_mask_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        self.vx_out_ref_buf.deinit();
        self.ax_out_ref_buf.deinit();
        for (&self.vx_block_in_ref_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.ax_block_in_ref_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.vx_block_out_ref_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.ax_block_out_ref_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        inline for (0..N) |i| {
            model.unloadBlock0FullBuffers(&self.params_bufs[i]);
        }
        self.platform.deinit(self.allocator);
    }

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        ckpt_path: []const u8,
        fixture_path: []const u8,
    ) !CheckContext {
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
        errdefer platform.deinit(allocator);
        const sharding = try zml.sharding.replicatedSharding(platform);

        var vx_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.vx_in", sharding);
        errdefer vx_in_buf.deinit();
        var ax_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.ax_in", sharding);
        errdefer ax_in_buf.deinit();

        var video_timesteps_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.video_timesteps", sharding);
        errdefer video_timesteps_buf.deinit();
        var audio_timesteps_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.audio_timesteps", sharding);
        errdefer audio_timesteps_buf.deinit();
        var v_prompt_timestep_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_prompt_timestep", sharding);
        errdefer v_prompt_timestep_buf.deinit();
        var a_prompt_timestep_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_prompt_timestep", sharding);
        errdefer a_prompt_timestep_buf.deinit();

        var v_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_pe_cos", sharding);
        errdefer v_pe_cos_buf.deinit();
        var v_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_pe_sin", sharding);
        errdefer v_pe_sin_buf.deinit();
        var a_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_pe_cos", sharding);
        errdefer a_pe_cos_buf.deinit();
        var a_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_pe_sin", sharding);
        errdefer a_pe_sin_buf.deinit();

        var v_text_ctx_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_text_ctx", sharding);
        errdefer v_text_ctx_buf.deinit();
        var a_text_ctx_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_text_ctx", sharding);
        errdefer a_text_ctx_buf.deinit();

        var v_cross_ss_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_cross_ss_ts", sharding);
        errdefer v_cross_ss_ts_buf.deinit();
        var v_cross_gate_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_cross_gate_ts", sharding);
        errdefer v_cross_gate_ts_buf.deinit();
        var a_cross_ss_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_cross_ss_ts", sharding);
        errdefer a_cross_ss_ts_buf.deinit();
        var a_cross_gate_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_cross_gate_ts", sharding);
        errdefer a_cross_gate_ts_buf.deinit();

        var a2v_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a2v_pe_cos", sharding);
        errdefer a2v_pe_cos_buf.deinit();
        var a2v_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a2v_pe_sin", sharding);
        errdefer a2v_pe_sin_buf.deinit();
        var a2v_k_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a2v_k_pe_cos", sharding);
        errdefer a2v_k_pe_cos_buf.deinit();
        var a2v_k_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a2v_k_pe_sin", sharding);
        errdefer a2v_k_pe_sin_buf.deinit();

        var v2a_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v2a_pe_cos", sharding);
        errdefer v2a_pe_cos_buf.deinit();
        var v2a_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v2a_pe_sin", sharding);
        errdefer v2a_pe_sin_buf.deinit();
        var v2a_k_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v2a_k_pe_cos", sharding);
        errdefer v2a_k_pe_cos_buf.deinit();
        var v2a_k_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v2a_k_pe_sin", sharding);
        errdefer v2a_k_pe_sin_buf.deinit();

        var a2v_mask_block_bufs: [N]?zml.Buffer = [_]?zml.Buffer{null} ** N;
        errdefer for (&a2v_mask_block_bufs) |*buf| { if (buf.*) |*b| b.deinit(); };
        var v2a_mask_block_bufs: [N]?zml.Buffer = [_]?zml.Buffer{null} ** N;
        errdefer for (&v2a_mask_block_bufs) |*buf| { if (buf.*) |*b| b.deinit(); };
        for (0..N) |i| {
            const a2v_key = try std.fmt.allocPrint(allocator, "block_slice_native.a2v_mask_block_{d}", .{i});
            defer allocator.free(a2v_key);
            if (fix_store.view().hasKey(a2v_key)) {
                a2v_mask_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, a2v_key, sharding);
            }
            const v2a_key = try std.fmt.allocPrint(allocator, "block_slice_native.v2a_mask_block_{d}", .{i});
            defer allocator.free(v2a_key);
            if (fix_store.view().hasKey(v2a_key)) {
                v2a_mask_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, v2a_key, sharding);
            }
        }

        var vx_out_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.vx_out", sharding);
        errdefer vx_out_ref_buf.deinit();
        var ax_out_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.ax_out", sharding);
        errdefer ax_out_ref_buf.deinit();

        var vx_block_out_ref_bufs: [N]?zml.Buffer = [_]?zml.Buffer{null} ** N;
        errdefer for (&vx_block_out_ref_bufs) |*buf| { if (buf.*) |*b| b.deinit(); };
        var ax_block_out_ref_bufs: [N]?zml.Buffer = [_]?zml.Buffer{null} ** N;
        errdefer for (&ax_block_out_ref_bufs) |*buf| { if (buf.*) |*b| b.deinit(); };
        var vx_block_in_ref_bufs: [N]?zml.Buffer = [_]?zml.Buffer{null} ** N;
        errdefer for (&vx_block_in_ref_bufs) |*buf| { if (buf.*) |*b| b.deinit(); };
        var ax_block_in_ref_bufs: [N]?zml.Buffer = [_]?zml.Buffer{null} ** N;
        errdefer for (&ax_block_in_ref_bufs) |*buf| { if (buf.*) |*b| b.deinit(); };
        for (0..N) |i| {
            const vx_in_key = try std.fmt.allocPrint(allocator, "block_slice_native.vx_in_block_{d}", .{i});
            defer allocator.free(vx_in_key);
            if (fix_store.view().hasKey(vx_in_key)) {
                vx_block_in_ref_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, vx_in_key, sharding);
            }
            const ax_in_key = try std.fmt.allocPrint(allocator, "block_slice_native.ax_in_block_{d}", .{i});
            defer allocator.free(ax_in_key);
            if (fix_store.view().hasKey(ax_in_key)) {
                ax_block_in_ref_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, ax_in_key, sharding);
            }

            const vx_key = try std.fmt.allocPrint(allocator, "block_slice_native.vx_out_block_{d}", .{i});
            defer allocator.free(vx_key);
            if (fix_store.view().hasKey(vx_key)) {
                vx_block_out_ref_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, vx_key, sharding);
            }
            const ax_key = try std.fmt.allocPrint(allocator, "block_slice_native.ax_out_block_{d}", .{i});
            defer allocator.free(ax_key);
            if (fix_store.view().hasKey(ax_key)) {
                ax_block_out_ref_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, ax_key, sharding);
            }
        }

        var params_shape = model.initBlockSlice48FullParams(ckpt_store.view());
        var params_bufs: [N]zml.Bufferized(model.Block0FullParams) = undefined;
        inline for (0..N) |i| {
            params_bufs[i] = try zml.io.load(model.Block0FullParams, &params_shape.blocks[i], allocator, io, platform, .{
                .store = &ckpt_store,
                .shardings = &.{sharding},
                .parallelism = 16,
                .dma_chunks = 8,
                .dma_chunk_size = 64 * zml.MiB,
            });
        }
        errdefer {
            inline for (0..N) |i| {
                model.unloadBlock0FullBuffers(&params_bufs[i]);
            }
        }

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .sharding = sharding,
            .params_shape = params_shape,
            .params_bufs = params_bufs,
            .vx_in_buf = vx_in_buf,
            .ax_in_buf = ax_in_buf,
            .video_timesteps_buf = video_timesteps_buf,
            .audio_timesteps_buf = audio_timesteps_buf,
            .v_prompt_timestep_buf = v_prompt_timestep_buf,
            .a_prompt_timestep_buf = a_prompt_timestep_buf,
            .v_pe_cos_buf = v_pe_cos_buf,
            .v_pe_sin_buf = v_pe_sin_buf,
            .a_pe_cos_buf = a_pe_cos_buf,
            .a_pe_sin_buf = a_pe_sin_buf,
            .v_text_ctx_buf = v_text_ctx_buf,
            .a_text_ctx_buf = a_text_ctx_buf,
            .v_cross_ss_ts_buf = v_cross_ss_ts_buf,
            .v_cross_gate_ts_buf = v_cross_gate_ts_buf,
            .a_cross_ss_ts_buf = a_cross_ss_ts_buf,
            .a_cross_gate_ts_buf = a_cross_gate_ts_buf,
            .a2v_pe_cos_buf = a2v_pe_cos_buf,
            .a2v_pe_sin_buf = a2v_pe_sin_buf,
            .a2v_k_pe_cos_buf = a2v_k_pe_cos_buf,
            .a2v_k_pe_sin_buf = a2v_k_pe_sin_buf,
            .a2v_mask_block_bufs = a2v_mask_block_bufs,
            .v2a_pe_cos_buf = v2a_pe_cos_buf,
            .v2a_pe_sin_buf = v2a_pe_sin_buf,
            .v2a_k_pe_cos_buf = v2a_k_pe_cos_buf,
            .v2a_k_pe_sin_buf = v2a_k_pe_sin_buf,
            .v2a_mask_block_bufs = v2a_mask_block_bufs,
            .vx_out_ref_buf = vx_out_ref_buf,
            .ax_out_ref_buf = ax_out_ref_buf,
            .vx_block_in_ref_bufs = vx_block_in_ref_bufs,
            .ax_block_in_ref_bufs = ax_block_in_ref_bufs,
            .vx_block_out_ref_bufs = vx_block_out_ref_bufs,
            .ax_block_out_ref_bufs = ax_block_out_ref_bufs,
        };
    }
};

fn runCheck(
    ctx: *CheckContext,
    use_extended_error_stats: bool,
    teacher_forced: bool,
    use_ref_video_inputs: bool,
    use_ref_audio_inputs: bool,
    use_video_all_residuals_f32: bool,
    use_f32_carry: bool,
) !void {
    std.log.info("Compiling single-block exe for {d}-block chain run...", .{N});

    const has_masks = ctx.a2v_mask_block_bufs[0] != null;

    // For f32-carry: compile with f32 activation shapes so the chain stays in f32
    // throughout. The VideoAllResidualsF32 variant is used because it explicitly
    // converts weights (linearForwardF32 / forwardAudioFFPrecise) which is required
    // when the carry dtype is f32 but weights remain bf16.
    const vx_compile_shape = if (use_f32_carry) ctx.vx_in_buf.shape().withDtype(.f32) else ctx.vx_in_buf.shape();
    const ax_compile_shape = if (use_f32_carry) ctx.ax_in_buf.shape().withDtype(.f32) else ctx.ax_in_buf.shape();

    var single_block_exe = if (has_masks and (use_video_all_residuals_f32 or use_f32_carry))
        try ctx.platform.compileFn(
            ctx.allocator,
            ctx.io,
            model.forwardBlock0NativeWithAVMasksVideoAllResidualsF32,
            .{
                zml.Tensor.fromShape(vx_compile_shape),
                zml.Tensor.fromShape(ax_compile_shape),
                zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_mask_block_bufs[0].?.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_mask_block_bufs[0].?.shape()),
                ctx.params_shape.blocks[0],
            },
            .{ .shardings = &.{ctx.sharding} },
        )
    else if (has_masks)
        try ctx.platform.compileFn(
            ctx.allocator,
            ctx.io,
            model.forwardBlock0NativeWithAVMasks,
            .{
                zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
                zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
                zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_mask_block_bufs[0].?.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_mask_block_bufs[0].?.shape()),
                ctx.params_shape.blocks[0],
            },
            .{ .shardings = &.{ctx.sharding} },
        )
    else
        try ctx.platform.compileFn(
            ctx.allocator,
            ctx.io,
            model.forwardBlock0Native,
            .{
                zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
                zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
                zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                ctx.params_shape.blocks[0],
            },
            .{ .shardings = &.{ctx.sharding} },
        );
    defer single_block_exe.deinit();

    // Free-running chain: each block's output feeds the next block's input.
    // For f32-carry: convert initial activations to f32 so the chain never
    // rounds back to bf16 at block boundaries.
    var h_v: zml.Buffer = blk: {
        if (use_f32_carry) {
            std.log.info("F32-carry: casting vx_in to f32 for chain init...", .{});
            var cast_v = try ctx.platform.compileFn(
                ctx.allocator, ctx.io, model.castToF32,
                .{zml.Tensor.fromShape(ctx.vx_in_buf.shape())},
                .{ .shardings = &.{ctx.sharding} },
            );
            defer cast_v.deinit();
            var cv_args = try cast_v.args(ctx.allocator);
            defer cv_args.deinit(ctx.allocator);
            var cv_res = try cast_v.results(ctx.allocator);
            defer cv_res.deinit(ctx.allocator);
            cv_args.set(.{ctx.vx_in_buf});
            cast_v.call(cv_args, &cv_res);
            break :blk cv_res.get(zml.Buffer);
        }
        break :blk try check_utils.copyBuffer(ctx.io, ctx.platform, ctx.vx_in_buf, ctx.sharding);
    };
    defer h_v.deinit();
    var h_a: zml.Buffer = blk: {
        if (use_f32_carry) {
            std.log.info("F32-carry: casting ax_in to f32 for chain init...", .{});
            var cast_a = try ctx.platform.compileFn(
                ctx.allocator, ctx.io, model.castToF32,
                .{zml.Tensor.fromShape(ctx.ax_in_buf.shape())},
                .{ .shardings = &.{ctx.sharding} },
            );
            defer cast_a.deinit();
            var ca_args = try cast_a.args(ctx.allocator);
            defer ca_args.deinit(ctx.allocator);
            var ca_res = try cast_a.results(ctx.allocator);
            defer ca_res.deinit(ctx.allocator);
            ca_args.set(.{ctx.ax_in_buf});
            cast_a.call(ca_args, &ca_res);
            break :blk ca_res.get(zml.Buffer);
        }
        break :blk try check_utils.copyBuffer(ctx.io, ctx.platform, ctx.ax_in_buf, ctx.sharding);
    };
    defer h_a.deinit();

    if (teacher_forced) {
        std.log.info("Running {d}-block teacher-forced per-block check...", .{N});
    } else if (use_ref_video_inputs or use_ref_audio_inputs) {
        std.log.info(
            "Running {d}-block hybrid check (ref_video_inputs={any}, ref_audio_inputs={any})...",
            .{ N, use_ref_video_inputs, use_ref_audio_inputs },
        );
    } else {
        std.log.info("Running {d}-block free-running chain...", .{N});
    }
    for (0..N) |i| {
        var args = try single_block_exe.args(ctx.allocator);
        defer args.deinit(ctx.allocator);
        var results = try single_block_exe.results(ctx.allocator);
        defer results.deinit(ctx.allocator);

        const force_v_ref = teacher_forced or use_ref_video_inputs;
        const force_a_ref = teacher_forced or use_ref_audio_inputs;
        const in_v = if (force_v_ref) (ctx.vx_block_in_ref_bufs[i] orelse h_v) else h_v;
        const in_a = if (force_a_ref) (ctx.ax_block_in_ref_bufs[i] orelse h_a) else h_a;

        var vin_rel_l2: ?f64 = null;
        var ain_rel_l2: ?f64 = null;
        if (!teacher_forced and !force_v_ref) {
            if (ctx.vx_block_in_ref_bufs[i]) |vref_in| {
                if (use_extended_error_stats) {
                    const vin_ext = try check_utils.compareBuffersExtended(ctx.io, in_v, vref_in, 0.2, 0.01);
                    vin_rel_l2 = vin_ext.rel_l2_error;
                    std.log.info(
                        "Video block {d:>2} input drift: close={d:.8} rel_l2={d:.6} p90={d:.6}",
                        .{ i, vin_ext.close_fraction, vin_ext.rel_l2_error, vin_ext.abs_err_p90 },
                    );
                } else {
                    const vin_m = try check_utils.compareBuffers(ctx.io, in_v, vref_in, 0.2, 0.01);
                    std.log.info(
                        "Video block {d:>2} input drift: close={d:.8} max_abs={d:.6}",
                        .{ i, vin_m.close_fraction, vin_m.max_abs_error },
                    );
                }
            }
        }
        if (!teacher_forced and !force_a_ref) {
            if (ctx.ax_block_in_ref_bufs[i]) |aref_in| {
                if (use_extended_error_stats) {
                    const ain_ext = try check_utils.compareBuffersExtended(ctx.io, in_a, aref_in, 0.2, 0.01);
                    ain_rel_l2 = ain_ext.rel_l2_error;
                    std.log.info(
                        "Audio block {d:>2} input drift: close={d:.8} rel_l2={d:.6} p90={d:.6}",
                        .{ i, ain_ext.close_fraction, ain_ext.rel_l2_error, ain_ext.abs_err_p90 },
                    );
                } else {
                    const ain_m = try check_utils.compareBuffers(ctx.io, in_a, aref_in, 0.2, 0.01);
                    std.log.info(
                        "Audio block {d:>2} input drift: close={d:.8} max_abs={d:.6}",
                        .{ i, ain_m.close_fraction, ain_m.max_abs_error },
                    );
                }
            }
        }

        if (has_masks) {
            args.set(.{
            in_v, in_a,
                ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
                ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
                ctx.v_pe_cos_buf, ctx.v_pe_sin_buf,
                ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
                ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
                ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf,
                ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
                ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf,
                ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf,
                ctx.a2v_mask_block_bufs[i].?,
                ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf,
                ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf,
                ctx.v2a_mask_block_bufs[i].?,
                ctx.params_bufs[i],
            });
        } else {
            args.set(.{
                in_v, in_a,
                ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
                ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
                ctx.v_pe_cos_buf, ctx.v_pe_sin_buf,
                ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
                ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
                ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf,
                ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
                ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf,
                ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf,
                ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf,
                ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf,
                ctx.params_bufs[i],
            });
        }
        single_block_exe.call(args, &results);

        const out = results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));

        if (ctx.ax_block_out_ref_bufs[i]) |ref| {
            const m = try check_utils.compareBuffers(ctx.io, out.ax_out, ref, 0.2, 0.01);
            if (use_extended_error_stats) {
                const ext = try check_utils.compareBuffersExtended(ctx.io, out.ax_out, ref, 0.2, 0.01);
                std.log.info(
                    "Audio block {d:>2} chain: close={d:.8} p50={d:.6} p90={d:.6} p99={d:.6} frac>1e-2={d:.6} rel_l2={d:.6} cos={d:.8}",
                    .{ i, m.close_fraction, ext.abs_err_p50, ext.abs_err_p90, ext.abs_err_p99, ext.frac_abs_err_gt_1e2, ext.rel_l2_error, ext.cosine_similarity },
                );
                if (!teacher_forced and !force_a_ref) {
                    if (ain_rel_l2) |in_rel| {
                        const gain = if (in_rel > 0.0) ext.rel_l2_error / in_rel else 0.0;
                        std.log.info("Audio block {d:>2} amplification: rel_l2_out/rel_l2_in={d:.3}", .{ i, gain });
                    }
                }
            } else {
                std.log.info(
                    "Audio block {d:>2} chain: close={d:.8} max_abs={d:.6} mean_abs={d:.6}",
                    .{ i, m.close_fraction, m.max_abs_error, m.mean_abs_error },
                );
            }
        }

        if (ctx.vx_block_out_ref_bufs[i]) |ref| {
            const m = try check_utils.compareBuffers(ctx.io, out.vx_out, ref, 0.2, 0.01);
            if (use_extended_error_stats) {
                const ext = try check_utils.compareBuffersExtended(ctx.io, out.vx_out, ref, 0.2, 0.01);
                std.log.info(
                    "Video block {d:>2} chain: close={d:.8} p50={d:.6} p90={d:.6} p99={d:.6} frac>1e-2={d:.6} rel_l2={d:.6} cos={d:.8}",
                    .{ i, m.close_fraction, ext.abs_err_p50, ext.abs_err_p90, ext.abs_err_p99, ext.frac_abs_err_gt_1e2, ext.rel_l2_error, ext.cosine_similarity },
                );
                if (!teacher_forced and !force_v_ref) {
                    if (vin_rel_l2) |in_rel| {
                        const gain = if (in_rel > 0.0) ext.rel_l2_error / in_rel else 0.0;
                        std.log.info("Video block {d:>2} amplification: rel_l2_out/rel_l2_in={d:.3}", .{ i, gain });
                    }
                }
            } else {
                std.log.info(
                    "Video block {d:>2} chain: close={d:.8} max_abs={d:.6} mean_abs={d:.6}",
                    .{ i, m.close_fraction, m.max_abs_error, m.mean_abs_error },
                );
            }
        }

        const next_v = out.vx_out;
        const next_a = out.ax_out;
        h_v.deinit();
        h_a.deinit();
        h_v = next_v;
        h_a = next_a;
    }

    if (teacher_forced or use_ref_video_inputs or use_ref_audio_inputs) {
        std.log.info("Non-free-running mode complete. Skipping final chain expectClose assertions.", .{});
        return;
    }

    // Final outputs: compare computed chain result vs Python reference.
    // Always print extended metrics in f32-carry mode (that's the point of the run).
    if (use_extended_error_stats or use_f32_carry) {
        const vext = try check_utils.compareBuffersExtended(ctx.io, h_v, ctx.vx_out_ref_buf, 0.2, 0.01);
        std.log.info(
            "Video final: close={d:.8} max_abs={d:.6} mean_abs={d:.6} rmse={d:.6} rel_l2={d:.6} cos={d:.8}",
            .{ vext.close_fraction, vext.max_abs_error, vext.mean_abs_error, vext.rmse_error, vext.rel_l2_error, vext.cosine_similarity },
        );
        std.log.info(
            "Video final: abs_err quantiles p50={d:.6} p90={d:.6} p99={d:.6} p99.9={d:.6}",
            .{ vext.abs_err_p50, vext.abs_err_p90, vext.abs_err_p99, vext.abs_err_p999 },
        );
        const aext = try check_utils.compareBuffersExtended(ctx.io, h_a, ctx.ax_out_ref_buf, 0.2, 0.01);
        std.log.info(
            "Audio final: close={d:.8} max_abs={d:.6} mean_abs={d:.6} rmse={d:.6} rel_l2={d:.6} cos={d:.8}",
            .{ aext.close_fraction, aext.max_abs_error, aext.mean_abs_error, aext.rmse_error, aext.rel_l2_error, aext.cosine_similarity },
        );
        std.log.info(
            "Audio final: abs_err quantiles p50={d:.6} p90={d:.6} p99={d:.6} p99.9={d:.6}",
            .{ aext.abs_err_p50, aext.abs_err_p90, aext.abs_err_p99, aext.abs_err_p999 },
        );
        std.log.info(
            "Audio final: sign fractions pos={d:.6} neg={d:.6} zero={d:.6}",
            .{ aext.positive_diff_fraction, aext.negative_diff_fraction, aext.zero_diff_fraction },
        );
        std.log.info(
            "Audio final: frac(|err|>1e-3)={d:.6} frac(|err|>1e-2)={d:.6} frac(|err|>1e-1)={d:.6}",
            .{ aext.frac_abs_err_gt_1e3, aext.frac_abs_err_gt_1e2, aext.frac_abs_err_gt_1e1 },
        );
    }

    // In f32-carry mode the outputs are f32 while references are bf16; do not hard-assert
    // with expectClose (which may not support cross-dtype) — just report the metrics above.
    if (use_f32_carry) {
        std.log.info("F32-carry mode complete. Skipping expectClose (cross-dtype f32 vs bf16).", .{});
        return;
    }
    try zml.testing.expectClose(ctx.io, h_v, ctx.vx_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("{d}-block video parity PASSED", .{N});

    try zml.testing.expectClose(ctx.io, h_a, ctx.ax_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("{d}-block audio parity PASSED", .{N});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("{d}-block chain trend checker", .{N});

    var it = init.minimal.args.iterate();
    _ = it.next();

    const usage = "Usage: block_slice_48_check <checkpoint.safetensors> <fixture.safetensors> [--extended-error-stats] [--teacher-forced] [--ref-video-inputs] [--ref-audio-inputs] [--video-all-residuals-f32] [--f32-carry]";

    const ckpt_path = it.next() orelse {
        std.log.err(usage, .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err(usage, .{});
        return error.InvalidArgs;
    };

    var use_extended_error_stats = false;
    var teacher_forced = false;
    var use_ref_video_inputs = false;
    var use_ref_audio_inputs = false;
    var use_video_all_residuals_f32 = false;
    var use_f32_carry = false;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--extended-error-stats")) {
            use_extended_error_stats = true;
        } else if (std.mem.eql(u8, arg, "--teacher-forced")) {
            teacher_forced = true;
        } else if (std.mem.eql(u8, arg, "--ref-video-inputs")) {
            use_ref_video_inputs = true;
        } else if (std.mem.eql(u8, arg, "--ref-audio-inputs")) {
            use_ref_audio_inputs = true;
        } else if (std.mem.eql(u8, arg, "--video-all-residuals-f32")) {
            use_video_all_residuals_f32 = true;
        } else if (std.mem.eql(u8, arg, "--f32-carry")) {
            use_f32_carry = true;
        } else {
            std.log.err("Unknown arg: {s}", .{arg});
            std.log.err(usage, .{});
            return error.InvalidArgs;
        }
    }

    var ctx = try CheckContext.init(allocator, io, ckpt_path, fixture_path);
    defer ctx.deinit();

    try runCheck(&ctx, use_extended_error_stats, teacher_forced, use_ref_video_inputs, use_ref_audio_inputs, use_video_all_residuals_f32, use_f32_carry);
}
