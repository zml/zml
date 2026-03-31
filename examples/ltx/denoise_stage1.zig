/// Stage 1 denoiser — full denoising loop with guidance.
///
/// Phase 1: single-pass (conditional only), no guidance. (DONE)
/// Phase 2 (current): 4-pass structure (CFG + STG + modality isolation)
///                     with forwardGuiderCombine.
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:denoise_stage1 -- \
///       /root/models/ltx-2.3/ltx-2.3-22b.safetensors \
///       /root/e2e_demo/stage1_inputs.safetensors \
///       /root/e2e_demo/stage1_out/

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");

comptime {
    @setEvalBranchQuota(200000);
}

pub const std_options: std.Options = .{ .log_level = .info };

/// STG perturbation is applied at this 0-based block index.
/// Matches LTX_2_3_PARAMS.stg_blocks=[28] in Python (also 0-based).
const STG_BLOCK_IDX: usize = 28;

/// Number of denoising steps.
const NUM_STEPS: usize = model.stage1_default_schedule.num_steps;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("Stage 1 denoiser", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: denoise_stage1 <base_checkpoint.safetensors> <stage1_inputs.safetensors> <output_dir/> [reset_latents.safetensors]", .{});
        return error.InvalidArgs;
    };
    const inputs_path = it.next() orelse {
        std.log.err("Usage: denoise_stage1 <base_checkpoint.safetensors> <stage1_inputs.safetensors> <output_dir/> [reset_latents.safetensors]", .{});
        return error.InvalidArgs;
    };
    const output_dir = it.next() orelse {
        std.log.err("Usage: denoise_stage1 <base_checkpoint.safetensors> <stage1_inputs.safetensors> <output_dir/> [reset_latents.safetensors]", .{});
        return error.InvalidArgs;
    };
    // Remaining positional/optional args
    var reset_path: ?[]const u8 = null;
    var use_bf16_attn = false;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--bf16-attn")) {
            use_bf16_attn = true;
        } else if (reset_path == null) {
            reset_path = arg;
        }
    }
    const is_reset = reset_path != null;
    if (is_reset) {
        std.log.info("Reset mode: will load Python latents at each step from {s}", .{reset_path.?});
    }
    if (use_bf16_attn) {
        std.log.info("bf16 attention mode enabled — all block exes use bf16-native attention.", .{});
    }

    // ========================================================================
    // Section A: CLI argument parsing (above) / Section B: Sigma schedule
    // ========================================================================
    const sigmas = model.computeSigmaSchedule(
        NUM_STEPS,
        NUM_STEPS,
        model.stage1_default_schedule.default_num_tokens,
        model.stage1_default_schedule.max_shift,
        model.stage1_default_schedule.base_shift,
        model.stage1_default_schedule.terminal,
    );
    std.log.info("Sigma schedule ({d} steps): [{d:.6} ... {d:.6}]", .{
        NUM_STEPS, sigmas[0], sigmas[NUM_STEPS],
    });

    // ========================================================================
    // Section C: Open stores + load inputs
    // ========================================================================
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    var inputs_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, inputs_path) catch |err| {
        std.log.err("Failed to open inputs: {s}", .{inputs_path});
        return err;
    };
    defer inputs_reg.deinit();
    var inputs_store: zml.io.TensorStore = .fromRegistry(allocator, &inputs_reg);
    defer inputs_store.deinit();

    // Optional: reset latents for per-step reset test
    var reset_reg: zml.safetensors.TensorRegistry = undefined;
    var reset_store: zml.io.TensorStore = undefined;
    if (reset_path) |rp| {
        reset_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, rp) catch |err| {
            std.log.err("Failed to open reset latents: {s}", .{rp});
            return err;
        };
        reset_store = .fromRegistry(allocator, &reset_reg);
    }
    defer if (is_reset) {
        reset_store.deinit();
        reset_reg.deinit();
    };

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ========================================================================
    // (continued) Load inputs
    // ========================================================================
    std.log.info("Loading inputs...", .{});

    // The exported latents are already noised (clean + noise * sigma) by
    // the Python pipeline, so we skip forwardNoiseInit entirely.
    var v_latent_buf = try loadBuf(allocator, io, platform, &inputs_store, "video_latent", sharding);
    var a_latent_buf = try loadBuf(allocator, io, platform, &inputs_store, "audio_latent", sharding);
    var v_mask_buf = try loadBuf(allocator, io, platform, &inputs_store, "video_denoise_mask", sharding);
    defer v_mask_buf.deinit();
    var a_mask_buf = try loadBuf(allocator, io, platform, &inputs_store, "audio_denoise_mask", sharding);
    defer a_mask_buf.deinit();
    var v_clean_buf = try loadBuf(allocator, io, platform, &inputs_store, "video_clean_latent", sharding);
    defer v_clean_buf.deinit();
    var a_clean_buf = try loadBuf(allocator, io, platform, &inputs_store, "audio_clean_latent", sharding);
    defer a_clean_buf.deinit();
    var v_positions_buf = try loadBuf(allocator, io, platform, &inputs_store, "video_positions", sharding);
    defer v_positions_buf.deinit();
    var a_positions_buf = try loadBuf(allocator, io, platform, &inputs_store, "audio_positions", sharding);
    defer a_positions_buf.deinit();

    // Stage 1 needs both positive and negative contexts.
    var v_context_pos_buf = try loadBuf(allocator, io, platform, &inputs_store, "v_context_pos", sharding);
    defer v_context_pos_buf.deinit();
    var a_context_pos_buf = try loadBuf(allocator, io, platform, &inputs_store, "a_context_pos", sharding);
    defer a_context_pos_buf.deinit();
    var v_context_neg_buf = try loadBuf(allocator, io, platform, &inputs_store, "v_context_neg", sharding);
    defer v_context_neg_buf.deinit();
    var a_context_neg_buf = try loadBuf(allocator, io, platform, &inputs_store, "a_context_neg", sharding);
    defer a_context_neg_buf.deinit();

    std.log.info("  video_latent (noised): {any}", .{v_latent_buf.shape()});
    std.log.info("  video_clean:           {any}", .{v_clean_buf.shape()});
    std.log.info("  audio_latent (noised): {any}", .{a_latent_buf.shape()});
    std.log.info("  audio_clean:           {any}", .{a_clean_buf.shape()});

    // ========================================================================
    // Section D: Compile executables
    // ========================================================================
    const sigma_scalar_shape = zml.Shape.init(.{}, .f32);
    std.log.info("Compiling preprocessing exe...", .{});

    const preprocess_shape = model.initPreprocessParams(ckpt_store.view());
    var preprocess_exe = try platform.compileFn(
        allocator, io,
        model.forwardPreprocess,
        .{
            zml.Tensor.fromShape(v_latent_buf.shape()),
            zml.Tensor.fromShape(a_latent_buf.shape()),
            zml.Tensor.fromShape(v_mask_buf.shape()),
            zml.Tensor.fromShape(a_mask_buf.shape()),
            zml.Tensor.fromShape(zml.Shape.init(.{1}, .f32)),
            zml.Tensor.fromShape(zml.Shape.init(.{1}, .f32)),
            zml.Tensor.fromShape(v_positions_buf.shape()),
            zml.Tensor.fromShape(a_positions_buf.shape()),
            zml.Tensor.fromShape(v_context_pos_buf.shape()),
            zml.Tensor.fromShape(a_context_pos_buf.shape()),
            preprocess_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer preprocess_exe.deinit();
    std.log.info("Preprocessing exe compiled.", .{});

    // Load preprocessing weights and run once for shape discovery
    std.log.info("Loading preprocessing weights...", .{});
    var preprocess_bufs = try zml.io.load(
        model.PreprocessParams,
        &preprocess_shape,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer model.PreprocessParams.unloadBuffers(&preprocess_bufs);

    std.log.info("Running initial preprocessing for shape discovery...", .{});
    var sigma_1d_init = try sigma1dBuffer(io, platform, sigmas[0], sharding);
    defer sigma_1d_init.deinit();

    var pre_args_init = try preprocess_exe.args(allocator);
    defer pre_args_init.deinit(allocator);
    var pre_results_init = try preprocess_exe.results(allocator);
    defer pre_results_init.deinit(allocator);
    pre_args_init.set(.{
        v_latent_buf, a_latent_buf,
        v_mask_buf, a_mask_buf,
        sigma_1d_init, sigma_1d_init,
        v_positions_buf, a_positions_buf,
        v_context_pos_buf, a_context_pos_buf,
        preprocess_bufs,
    });
    preprocess_exe.call(pre_args_init, &pre_results_init);
    const init_pre_out = pre_results_init.get(zml.Bufferized(model.PreprocessOutput));

    // ---- Shared block compile args (24 tensors + params) ----
    var block_params_shape = try allocator.create(model.FullStepParams);
    defer allocator.destroy(block_params_shape);
    block_params_shape.* = model.initFullStepParams(ckpt_store.view());

    const block_compile_args = .{
        zml.Tensor.fromShape(init_pre_out.vx.shape()),
        zml.Tensor.fromShape(init_pre_out.ax.shape()),
        zml.Tensor.fromShape(init_pre_out.video_timesteps.shape()),
        zml.Tensor.fromShape(init_pre_out.audio_timesteps.shape()),
        zml.Tensor.fromShape(init_pre_out.v_prompt_timestep.shape()),
        zml.Tensor.fromShape(init_pre_out.a_prompt_timestep.shape()),
        zml.Tensor.fromShape(init_pre_out.v_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.v_pe_sin.shape()),
        zml.Tensor.fromShape(init_pre_out.a_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.a_pe_sin.shape()),
        zml.Tensor.fromShape(init_pre_out.v_text_ctx.shape()),
        zml.Tensor.fromShape(init_pre_out.a_text_ctx.shape()),
        zml.Tensor.fromShape(init_pre_out.v_cross_ss_ts.shape()),
        zml.Tensor.fromShape(init_pre_out.v_cross_gate_ts.shape()),
        zml.Tensor.fromShape(init_pre_out.a_cross_ss_ts.shape()),
        zml.Tensor.fromShape(init_pre_out.a_cross_gate_ts.shape()),
        zml.Tensor.fromShape(init_pre_out.a2v_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.a2v_pe_sin.shape()),
        zml.Tensor.fromShape(init_pre_out.a2v_k_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.a2v_k_pe_sin.shape()),
        zml.Tensor.fromShape(init_pre_out.v2a_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.v2a_pe_sin.shape()),
        zml.Tensor.fromShape(init_pre_out.v2a_k_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.v2a_k_pe_sin.shape()),
        block_params_shape.blocks[0],
    };
    const compile_opts: zml.module.CompilationOptions = .{ .shardings = &.{sharding} };

    // ---- Compile block exe (normal — Passes 1 & 2) ----
    std.log.info("Compiling block exe (normal, bf16_attn={})...", .{use_bf16_attn});
    var block_normal_exe = if (use_bf16_attn)
        try platform.compileFn(allocator, io, model.forwardBlock0NativeBf16Attn, block_compile_args, compile_opts)
    else
        try platform.compileFn(allocator, io, model.forwardBlock0Native, block_compile_args, compile_opts);
    defer block_normal_exe.deinit();
    std.log.info("Block exe (normal) compiled.", .{});

    // ---- Compile block exe (STG — V-passthrough at self-attn, Pass 3 block 28) ----
    std.log.info("Compiling block exe (STG, bf16_attn={})...", .{use_bf16_attn});
    var block_stg_exe = if (use_bf16_attn)
        try platform.compileFn(allocator, io, model.forwardBlock0NativeSTGBf16Attn, block_compile_args, compile_opts)
    else
        try platform.compileFn(allocator, io, model.forwardBlock0NativeSTG, block_compile_args, compile_opts);
    defer block_stg_exe.deinit();
    std.log.info("Block exe (STG) compiled.", .{});

    // ---- Compile block exe (isolated — WithAVMasks, zero masks for Pass 4) ----
    std.log.info("Compiling block exe (isolated, bf16_attn={})...", .{use_bf16_attn});
    const mask_scalar_shape = zml.Shape.init(.{}, .bf16);
    const iso_compile_args = .{
        zml.Tensor.fromShape(init_pre_out.vx.shape()),
        zml.Tensor.fromShape(init_pre_out.ax.shape()),
        zml.Tensor.fromShape(init_pre_out.video_timesteps.shape()),
        zml.Tensor.fromShape(init_pre_out.audio_timesteps.shape()),
        zml.Tensor.fromShape(init_pre_out.v_prompt_timestep.shape()),
        zml.Tensor.fromShape(init_pre_out.a_prompt_timestep.shape()),
        zml.Tensor.fromShape(init_pre_out.v_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.v_pe_sin.shape()),
        zml.Tensor.fromShape(init_pre_out.a_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.a_pe_sin.shape()),
        zml.Tensor.fromShape(init_pre_out.v_text_ctx.shape()),
        zml.Tensor.fromShape(init_pre_out.a_text_ctx.shape()),
        zml.Tensor.fromShape(init_pre_out.v_cross_ss_ts.shape()),
        zml.Tensor.fromShape(init_pre_out.v_cross_gate_ts.shape()),
        zml.Tensor.fromShape(init_pre_out.a_cross_ss_ts.shape()),
        zml.Tensor.fromShape(init_pre_out.a_cross_gate_ts.shape()),
        zml.Tensor.fromShape(init_pre_out.a2v_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.a2v_pe_sin.shape()),
        zml.Tensor.fromShape(init_pre_out.a2v_k_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.a2v_k_pe_sin.shape()),
        zml.Tensor.fromShape(mask_scalar_shape), // a2v_mask
        zml.Tensor.fromShape(init_pre_out.v2a_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.v2a_pe_sin.shape()),
        zml.Tensor.fromShape(init_pre_out.v2a_k_pe_cos.shape()),
        zml.Tensor.fromShape(init_pre_out.v2a_k_pe_sin.shape()),
        zml.Tensor.fromShape(mask_scalar_shape), // v2a_mask
        block_params_shape.blocks[0],
    };
    var block_iso_exe = if (use_bf16_attn)
        try platform.compileFn(allocator, io, model.forwardBlock0NativeWithAVMasksBf16Attn, iso_compile_args, compile_opts)
    else
        try platform.compileFn(allocator, io, model.forwardBlock0NativeWithAVMasks, iso_compile_args, compile_opts);
    defer block_iso_exe.deinit();
    std.log.info("Block exe (isolated) compiled.", .{});

    // ---- Compile output projection exes ----
    std.log.info("Compiling output projection exes...", .{});
    const v_emb_2d_shape = init_pre_out.v_embedded_timestep.shape();
    const a_emb_2d_shape = init_pre_out.a_embedded_timestep.shape();

    var proj_v_exe = try platform.compileFn(
        allocator, io,
        model.forwardOutputProjection,
        .{
            zml.Tensor.fromShape(init_pre_out.vx.shape()).withPartialTags(.{ .b, .t, .d }),
            zml.Tensor.fromShape(v_emb_2d_shape).withPartialTags(.{ .b, .d_emb }),
            block_params_shape.norm_proj_out,
        },
        .{ .shardings = &.{sharding} },
    );
    defer proj_v_exe.deinit();

    var proj_a_exe = try platform.compileFn(
        allocator, io,
        model.forwardOutputProjection,
        .{
            zml.Tensor.fromShape(init_pre_out.ax.shape()).withPartialTags(.{ .b, .t, .d }),
            zml.Tensor.fromShape(a_emb_2d_shape).withPartialTags(.{ .b, .d_emb }),
            block_params_shape.audio_norm_proj_out,
        },
        .{ .shardings = &.{sharding} },
    );
    defer proj_a_exe.deinit();
    std.log.info("Output projection exes compiled.", .{});

    // ---- Compile denoising step exes ----
    // ---- Compile vel→x0 exes (forwardToDenoised) ----
    std.log.info("Compiling vel→x0 (toDenoised) exes...", .{});

    var to_denoised_v_exe = try platform.compileFn(
        allocator, io,
        model.forwardToDenoised,
        .{
            zml.Tensor.fromShape(v_latent_buf.shape()),
            zml.Tensor.fromShape(v_latent_buf.shape()),
            zml.Tensor.fromShape(v_mask_buf.shape()),
            zml.Tensor.fromShape(sigma_scalar_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer to_denoised_v_exe.deinit();

    var to_denoised_a_exe = try platform.compileFn(
        allocator, io,
        model.forwardToDenoised,
        .{
            zml.Tensor.fromShape(a_latent_buf.shape()),
            zml.Tensor.fromShape(a_latent_buf.shape()),
            zml.Tensor.fromShape(a_mask_buf.shape()),
            zml.Tensor.fromShape(sigma_scalar_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer to_denoised_a_exe.deinit();
    std.log.info("vel→x0 exes compiled.", .{});

    // ---- Compile denoising step exes (from x0, not velocity) ----
    std.log.info("Compiling denoising step (from x0) exes...", .{});

    var denoise_v_exe = try platform.compileFn(
        allocator, io,
        model.forwardDenoisingStepFromX0,
        .{
            zml.Tensor.fromShape(v_latent_buf.shape()),
            zml.Tensor.fromShape(v_latent_buf.shape()),
            zml.Tensor.fromShape(v_mask_buf.shape()),
            zml.Tensor.fromShape(v_clean_buf.shape()),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer denoise_v_exe.deinit();

    var denoise_a_exe = try platform.compileFn(
        allocator, io,
        model.forwardDenoisingStepFromX0,
        .{
            zml.Tensor.fromShape(a_latent_buf.shape()),
            zml.Tensor.fromShape(a_latent_buf.shape()),
            zml.Tensor.fromShape(a_mask_buf.shape()),
            zml.Tensor.fromShape(a_clean_buf.shape()),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer denoise_a_exe.deinit();
    std.log.info("Denoising step (from x0) exes compiled.", .{});

    // ---- Compile guider combine exe ----
    std.log.info("Compiling guider combine exe...", .{});
    const vel_v_shape = v_latent_buf.shape(); // [B, T_v, 128] bf16
    const vel_a_shape = a_latent_buf.shape(); // [B, T_a, 128] bf16
    var guider_combine_exe = try platform.compileFn(
        allocator, io,
        model.forwardGuiderCombine,
        .{
            // 4 video velocities (cond, neg, ptb, iso)
            zml.Tensor.fromShape(vel_v_shape),
            zml.Tensor.fromShape(vel_v_shape),
            zml.Tensor.fromShape(vel_v_shape),
            zml.Tensor.fromShape(vel_v_shape),
            // 4 audio velocities (cond, neg, ptb, iso)
            zml.Tensor.fromShape(vel_a_shape),
            zml.Tensor.fromShape(vel_a_shape),
            zml.Tensor.fromShape(vel_a_shape),
            zml.Tensor.fromShape(vel_a_shape),
            // 4 video guidance scalars (cfg, stg, mod, rescale) — f32
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            // 4 audio guidance scalars (cfg, stg, mod, rescale) — f32
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer guider_combine_exe.deinit();
    std.log.info("Guider combine exe compiled.", .{});

    // ========================================================================
    // Section E: Load weights
    // ========================================================================
    std.log.info("Loading 48 block weights...", .{});
    var block_params_bufs = try allocator.create([48]zml.Bufferized(model.Block0FullParams));
    defer allocator.destroy(block_params_bufs);
    for (0..48) |i| {
        block_params_bufs[i] = try zml.io.load(
            model.Block0FullParams,
            &block_params_shape.blocks[i],
            allocator, io, platform,
            .{
                .store = &ckpt_store,
                .shardings = &.{sharding},
                .parallelism = 4,
                .dma_chunks = 4,
                .dma_chunk_size = 16 * zml.MiB,
            },
        );
    }
    defer for (&block_params_bufs.*) |*bp| model.unloadBlock0FullBuffers(bp);

    std.log.info("Loading output projection weights...", .{});
    var proj_v_bufs = try zml.io.load(
        model.OutputProjection.Params,
        &block_params_shape.norm_proj_out,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer model.OutputProjection.Params.unloadBuffers(&proj_v_bufs);

    var proj_a_bufs = try zml.io.load(
        model.OutputProjection.Params,
        &block_params_shape.audio_norm_proj_out,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer model.OutputProjection.Params.unloadBuffers(&proj_a_bufs);
    std.log.info("All weights loaded.", .{});

    // ========================================================================
    // (continued) Create constant buffers for guidance
    // ========================================================================
    // Zero masks for isolation pass (scalar bf16, broadcast to all elements)
    var zero_mask_buf = try zml.Buffer.scalar(io, platform, @as(f32, 0.0), .bf16, sharding);
    defer zero_mask_buf.deinit();

    // Guidance scalars (f32)
    var cfg_v_buf = try zml.Buffer.scalar(io, platform, @as(f32, 3.0), .f32, sharding);
    defer cfg_v_buf.deinit();
    var stg_v_buf = try zml.Buffer.scalar(io, platform, @as(f32, 1.0), .f32, sharding);
    defer stg_v_buf.deinit();
    var mod_v_buf = try zml.Buffer.scalar(io, platform, @as(f32, 3.0), .f32, sharding);
    defer mod_v_buf.deinit();
    var rescale_v_buf = try zml.Buffer.scalar(io, platform, @as(f32, 0.7), .f32, sharding);
    defer rescale_v_buf.deinit();
    var cfg_a_buf = try zml.Buffer.scalar(io, platform, @as(f32, 7.0), .f32, sharding);
    defer cfg_a_buf.deinit();
    var stg_a_buf = try zml.Buffer.scalar(io, platform, @as(f32, 1.0), .f32, sharding);
    defer stg_a_buf.deinit();
    var mod_a_buf = try zml.Buffer.scalar(io, platform, @as(f32, 3.0), .f32, sharding);
    defer mod_a_buf.deinit();
    var rescale_a_buf = try zml.Buffer.scalar(io, platform, @as(f32, 0.7), .f32, sharding);
    defer rescale_a_buf.deinit();

    // ========================================================================
    // Section F: Denoising loop — 4-pass with guidance
    // ========================================================================
    std.log.info("Starting {d}-step denoising loop (Phase 2: 4-pass guidance)...", .{NUM_STEPS});

    for (0..NUM_STEPS) |step_idx| {
        const sigma = sigmas[step_idx];
        const sigma_next = sigmas[step_idx + 1];

        std.log.info("", .{});
        std.log.info("===== Step {d}/{d}: sigma={d:.6} -> {d:.6} =====", .{
            step_idx + 1, NUM_STEPS, sigma, sigma_next,
        });

        // Optionally reset to Python's reference latent for per-step isolation
        if (is_reset) {
            const v_key = try std.fmt.allocPrint(allocator, "v_lat_{d}", .{step_idx});
            defer allocator.free(v_key);
            const a_key = try std.fmt.allocPrint(allocator, "a_lat_{d}", .{step_idx});
            defer allocator.free(a_key);
            v_latent_buf.deinit();
            v_latent_buf = try loadBuf(allocator, io, platform, &reset_store, v_key, sharding);
            a_latent_buf.deinit();
            a_latent_buf = try loadBuf(allocator, io, platform, &reset_store, a_key, sharding);
            std.log.info("  Reset to Python latent for step {d}", .{step_idx});
        }

        var sigma_1d = try sigma1dBuffer(io, platform, sigma, sharding);
        defer sigma_1d.deinit();
        var sigma_buf = try zml.Buffer.scalar(io, platform, sigma, .f32, sharding);
        defer sigma_buf.deinit();
        var sigma_next_buf = try zml.Buffer.scalar(io, platform, sigma_next, .f32, sharding);
        defer sigma_next_buf.deinit();

        // ---- F.1: Preprocessing (once per step, with positive context) ----
        std.log.info("  Preprocessing...", .{});
        var pre_args = try preprocess_exe.args(allocator);
        defer pre_args.deinit(allocator);
        var pre_results = try preprocess_exe.results(allocator);
        defer pre_results.deinit(allocator);

        pre_args.set(.{
            v_latent_buf, a_latent_buf,
            v_mask_buf, a_mask_buf,
            sigma_1d, sigma_1d,
            v_positions_buf, a_positions_buf,
            v_context_pos_buf, a_context_pos_buf,
            preprocess_bufs,
        });
        preprocess_exe.call(pre_args, &pre_results);
        const pre_out = pre_results.get(zml.Bufferized(model.PreprocessOutput));

        // ---- F.2: Pass 1 — Conditional (positive context, normal blocks) ----
        std.log.info("  Pass 1 (conditional): 48-block chain...", .{});
        var cond_h_v = pre_out.vx;
        var cond_h_a = pre_out.ax;

        for (0..48) |i| {
            var args = try block_normal_exe.args(allocator);
            defer args.deinit(allocator);
            var results = try block_normal_exe.results(allocator);
            defer results.deinit(allocator);

            args.set(.{
                cond_h_v, cond_h_a,
                pre_out.video_timesteps, pre_out.audio_timesteps,
                pre_out.v_prompt_timestep, pre_out.a_prompt_timestep,
                pre_out.v_pe_cos, pre_out.v_pe_sin,
                pre_out.a_pe_cos, pre_out.a_pe_sin,
                pre_out.v_text_ctx, pre_out.a_text_ctx,
                pre_out.v_cross_ss_ts, pre_out.v_cross_gate_ts,
                pre_out.a_cross_ss_ts, pre_out.a_cross_gate_ts,
                pre_out.a2v_pe_cos, pre_out.a2v_pe_sin,
                pre_out.a2v_k_pe_cos, pre_out.a2v_k_pe_sin,
                pre_out.v2a_pe_cos, pre_out.v2a_pe_sin,
                pre_out.v2a_k_pe_cos, pre_out.v2a_k_pe_sin,
                block_params_bufs[i],
            });
            block_normal_exe.call(args, &results);
            const out = results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
            if (i > 0) { cond_h_v.deinit(); cond_h_a.deinit(); }
            cond_h_v = out.vx_out;
            cond_h_a = out.ax_out;
            if (i % 16 == 15 or i == 47) std.log.info("    block {d:>2} done", .{i});
        }

        // Output projection → cond velocities
        var cond_v_vel = try runOutputProjection(allocator, &proj_v_exe, cond_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var cond_a_vel = try runOutputProjection(allocator, &proj_a_exe, cond_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        cond_h_v.deinit();
        cond_h_a.deinit();

        // Dump step-0 conditional velocity for validation
        if (step_idx == 0) {
            std.log.info("  Dumping step-0 velocity for validation...", .{});
            try writeBuffer(allocator, io, cond_v_vel, output_dir, "step0_video_vel.bin");
            try writeBuffer(allocator, io, cond_a_vel, output_dir, "step0_audio_vel.bin");
            // Also dump the patchified latent for re-use check
            try writeBuffer(allocator, io, pre_out.vx, output_dir, "step0_pre_out_vx.bin");
        }

        // Convert cond velocities → x0 predictions (matching Python's X0Model output)
        var cond_v_x0 = try runToDenoised(allocator, &to_denoised_v_exe, v_latent_buf, cond_v_vel, v_mask_buf, sigma_buf);
        var cond_a_x0 = try runToDenoised(allocator, &to_denoised_a_exe, a_latent_buf, cond_a_vel, a_mask_buf, sigma_buf);
        cond_v_vel.deinit();
        cond_a_vel.deinit();

        // ---- F.3: Pass 2 — Negative/CFG (negative context, normal blocks) ----
        std.log.info("  Pass 2 (negative/CFG): 48-block chain...", .{});
        var neg_h_v = pre_out.vx;
        var neg_h_a = pre_out.ax;

        for (0..48) |i| {
            var args = try block_normal_exe.args(allocator);
            defer args.deinit(allocator);
            var results = try block_normal_exe.results(allocator);
            defer results.deinit(allocator);

            args.set(.{
                neg_h_v, neg_h_a,
                pre_out.video_timesteps, pre_out.audio_timesteps,
                pre_out.v_prompt_timestep, pre_out.a_prompt_timestep,
                pre_out.v_pe_cos, pre_out.v_pe_sin,
                pre_out.a_pe_cos, pre_out.a_pe_sin,
                v_context_neg_buf, a_context_neg_buf, // negative text context
                pre_out.v_cross_ss_ts, pre_out.v_cross_gate_ts,
                pre_out.a_cross_ss_ts, pre_out.a_cross_gate_ts,
                pre_out.a2v_pe_cos, pre_out.a2v_pe_sin,
                pre_out.a2v_k_pe_cos, pre_out.a2v_k_pe_sin,
                pre_out.v2a_pe_cos, pre_out.v2a_pe_sin,
                pre_out.v2a_k_pe_cos, pre_out.v2a_k_pe_sin,
                block_params_bufs[i],
            });
            block_normal_exe.call(args, &results);
            const out = results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
            if (i > 0) { neg_h_v.deinit(); neg_h_a.deinit(); }
            neg_h_v = out.vx_out;
            neg_h_a = out.ax_out;
            if (i % 16 == 15 or i == 47) std.log.info("    block {d:>2} done", .{i});
        }

        var neg_v_vel = try runOutputProjection(allocator, &proj_v_exe, neg_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var neg_a_vel = try runOutputProjection(allocator, &proj_a_exe, neg_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        neg_h_v.deinit();
        neg_h_a.deinit();

        // Dump step-0 negative velocity + patchified latent state (donation check)
        if (step_idx == 0) {
            std.log.info("  Dumping step-0 neg velocity...", .{});
            try writeBuffer(allocator, io, neg_v_vel, output_dir, "step0_neg_video_vel.bin");
            try writeBuffer(allocator, io, neg_a_vel, output_dir, "step0_neg_audio_vel.bin");
            try writeBuffer(allocator, io, pre_out.vx, output_dir, "step0_pre_out_vx_after_pass2.bin");
        }

        // Convert neg velocities → x0 predictions
        var neg_v_x0 = try runToDenoised(allocator, &to_denoised_v_exe, v_latent_buf, neg_v_vel, v_mask_buf, sigma_buf);
        var neg_a_x0 = try runToDenoised(allocator, &to_denoised_a_exe, a_latent_buf, neg_a_vel, a_mask_buf, sigma_buf);
        neg_v_vel.deinit();
        neg_a_vel.deinit();

        // ---- F.4: Pass 3 — STG (positive context, V-passthrough at block 28) ----
        std.log.info("  Pass 3 (STG): 48-block chain (STG at block {d})...", .{STG_BLOCK_IDX});
        var ptb_h_v = pre_out.vx;
        var ptb_h_a = pre_out.ax;

        for (0..48) |i| {
            if (i == STG_BLOCK_IDX) {
                // STG block: V-passthrough at self-attention
                var args = try block_stg_exe.args(allocator);
                defer args.deinit(allocator);
                var results = try block_stg_exe.results(allocator);
                defer results.deinit(allocator);

                args.set(.{
                    ptb_h_v, ptb_h_a,
                    pre_out.video_timesteps, pre_out.audio_timesteps,
                    pre_out.v_prompt_timestep, pre_out.a_prompt_timestep,
                    pre_out.v_pe_cos, pre_out.v_pe_sin,
                    pre_out.a_pe_cos, pre_out.a_pe_sin,
                    pre_out.v_text_ctx, pre_out.a_text_ctx,
                    pre_out.v_cross_ss_ts, pre_out.v_cross_gate_ts,
                    pre_out.a_cross_ss_ts, pre_out.a_cross_gate_ts,
                    pre_out.a2v_pe_cos, pre_out.a2v_pe_sin,
                    pre_out.a2v_k_pe_cos, pre_out.a2v_k_pe_sin,
                    pre_out.v2a_pe_cos, pre_out.v2a_pe_sin,
                    pre_out.v2a_k_pe_cos, pre_out.v2a_k_pe_sin,
                    block_params_bufs[i],
                });
                block_stg_exe.call(args, &results);
                const out = results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
                if (i > 0) { ptb_h_v.deinit(); ptb_h_a.deinit(); }
                ptb_h_v = out.vx_out;
                ptb_h_a = out.ax_out;
                std.log.info("    block {d:>2} done (STG)", .{i});
            } else {
                var args = try block_normal_exe.args(allocator);
                defer args.deinit(allocator);
                var results = try block_normal_exe.results(allocator);
                defer results.deinit(allocator);

                args.set(.{
                    ptb_h_v, ptb_h_a,
                    pre_out.video_timesteps, pre_out.audio_timesteps,
                    pre_out.v_prompt_timestep, pre_out.a_prompt_timestep,
                    pre_out.v_pe_cos, pre_out.v_pe_sin,
                    pre_out.a_pe_cos, pre_out.a_pe_sin,
                    pre_out.v_text_ctx, pre_out.a_text_ctx,
                    pre_out.v_cross_ss_ts, pre_out.v_cross_gate_ts,
                    pre_out.a_cross_ss_ts, pre_out.a_cross_gate_ts,
                    pre_out.a2v_pe_cos, pre_out.a2v_pe_sin,
                    pre_out.a2v_k_pe_cos, pre_out.a2v_k_pe_sin,
                    pre_out.v2a_pe_cos, pre_out.v2a_pe_sin,
                    pre_out.v2a_k_pe_cos, pre_out.v2a_k_pe_sin,
                    block_params_bufs[i],
                });
                block_normal_exe.call(args, &results);
                const out = results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
                if (i > 0) { ptb_h_v.deinit(); ptb_h_a.deinit(); }
                ptb_h_v = out.vx_out;
                ptb_h_a = out.ax_out;
                if (i % 16 == 15 or i == 47) std.log.info("    block {d:>2} done", .{i});
            }
        }

        var ptb_v_vel = try runOutputProjection(allocator, &proj_v_exe, ptb_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var ptb_a_vel = try runOutputProjection(allocator, &proj_a_exe, ptb_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        ptb_h_v.deinit();
        ptb_h_a.deinit();

        // Dump step-0 STG velocity
        if (step_idx == 0) {
            std.log.info("  Dumping step-0 ptb velocity...", .{});
            try writeBuffer(allocator, io, ptb_v_vel, output_dir, "step0_ptb_video_vel.bin");
            try writeBuffer(allocator, io, ptb_a_vel, output_dir, "step0_ptb_audio_vel.bin");
        }

        // Convert ptb velocities → x0 predictions
        var ptb_v_x0 = try runToDenoised(allocator, &to_denoised_v_exe, v_latent_buf, ptb_v_vel, v_mask_buf, sigma_buf);
        var ptb_a_x0 = try runToDenoised(allocator, &to_denoised_a_exe, a_latent_buf, ptb_a_vel, a_mask_buf, sigma_buf);
        ptb_v_vel.deinit();
        ptb_a_vel.deinit();

        // ---- F.5: Pass 4 — Isolated (positive context, zero AV masks) ----
        std.log.info("  Pass 4 (isolated): 48-block chain...", .{});
        var iso_h_v = pre_out.vx;
        var iso_h_a = pre_out.ax;

        for (0..48) |i| {
            var args = try block_iso_exe.args(allocator);
            defer args.deinit(allocator);
            var results = try block_iso_exe.results(allocator);
            defer results.deinit(allocator);

            args.set(.{
                iso_h_v, iso_h_a,
                pre_out.video_timesteps, pre_out.audio_timesteps,
                pre_out.v_prompt_timestep, pre_out.a_prompt_timestep,
                pre_out.v_pe_cos, pre_out.v_pe_sin,
                pre_out.a_pe_cos, pre_out.a_pe_sin,
                pre_out.v_text_ctx, pre_out.a_text_ctx,
                pre_out.v_cross_ss_ts, pre_out.v_cross_gate_ts,
                pre_out.a_cross_ss_ts, pre_out.a_cross_gate_ts,
                pre_out.a2v_pe_cos, pre_out.a2v_pe_sin,
                pre_out.a2v_k_pe_cos, pre_out.a2v_k_pe_sin,
                zero_mask_buf, // a2v_mask = 0 (no audio→video cross-attn)
                pre_out.v2a_pe_cos, pre_out.v2a_pe_sin,
                pre_out.v2a_k_pe_cos, pre_out.v2a_k_pe_sin,
                zero_mask_buf, // v2a_mask = 0 (no video→audio cross-attn)
                block_params_bufs[i],
            });
            block_iso_exe.call(args, &results);
            const out = results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
            if (i > 0) { iso_h_v.deinit(); iso_h_a.deinit(); }
            iso_h_v = out.vx_out;
            iso_h_a = out.ax_out;
            if (i % 16 == 15 or i == 47) std.log.info("    block {d:>2} done", .{i});
        }

        var iso_v_vel = try runOutputProjection(allocator, &proj_v_exe, iso_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var iso_a_vel = try runOutputProjection(allocator, &proj_a_exe, iso_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        iso_h_v.deinit();
        iso_h_a.deinit();

        // Dump step-0 iso velocity + guided velocity
        if (step_idx == 0) {
            std.log.info("  Dumping step-0 iso velocity...", .{});
            try writeBuffer(allocator, io, iso_v_vel, output_dir, "step0_iso_video_vel.bin");
            try writeBuffer(allocator, io, iso_a_vel, output_dir, "step0_iso_audio_vel.bin");
        }

        // Convert iso velocities → x0 predictions
        var iso_v_x0 = try runToDenoised(allocator, &to_denoised_v_exe, v_latent_buf, iso_v_vel, v_mask_buf, sigma_buf);
        var iso_a_x0 = try runToDenoised(allocator, &to_denoised_a_exe, a_latent_buf, iso_a_vel, a_mask_buf, sigma_buf);
        iso_v_vel.deinit();
        iso_a_vel.deinit();

        // ---- F.6: Guider combine ----
        std.log.info("  Guider combine...", .{});
        var gc_args = try guider_combine_exe.args(allocator);
        defer gc_args.deinit(allocator);
        var gc_results = try guider_combine_exe.results(allocator);
        defer gc_results.deinit(allocator);

        gc_args.set(.{
            cond_v_x0, neg_v_x0, ptb_v_x0, iso_v_x0,
            cond_a_x0, neg_a_x0, ptb_a_x0, iso_a_x0,
            cfg_v_buf, stg_v_buf, mod_v_buf, rescale_v_buf,
            cfg_a_buf, stg_a_buf, mod_a_buf, rescale_a_buf,
        });
        guider_combine_exe.call(gc_args, &gc_results);
        const gc_out = gc_results.get(zml.Bufferized(model.GuiderCombineResult));

        var guided_v_x0 = gc_out.guided_v;
        var guided_a_x0 = gc_out.guided_a;

        // Dump step-0 guided x0 (after guider combine)
        if (step_idx == 0) {
            std.log.info("  Dumping step-0 guided x0...", .{});
            try writeBuffer(allocator, io, guided_v_x0, output_dir, "step0_guided_video_vel.bin");
            try writeBuffer(allocator, io, guided_a_x0, output_dir, "step0_guided_audio_vel.bin");
        }

        // Free the 8 per-pass x0 predictions
        cond_v_x0.deinit();
        neg_v_x0.deinit();
        ptb_v_x0.deinit();
        iso_v_x0.deinit();
        cond_a_x0.deinit();
        neg_a_x0.deinit();
        ptb_a_x0.deinit();
        iso_a_x0.deinit();

        // ---- F.7: Denoising step from guided x0 (post_process + Euler) ----
        std.log.info("  Euler step (from x0)...", .{});

        var dv_args = try denoise_v_exe.args(allocator);
        defer dv_args.deinit(allocator);
        var dv_results = try denoise_v_exe.results(allocator);
        defer dv_results.deinit(allocator);
        dv_args.set(.{ v_latent_buf, guided_v_x0, v_mask_buf, v_clean_buf, sigma_buf, sigma_next_buf });
        denoise_v_exe.call(dv_args, &dv_results);
        const dv_out = dv_results.get(zml.Bufferized(model.DenoisingStepResult));

        var da_args = try denoise_a_exe.args(allocator);
        defer da_args.deinit(allocator);
        var da_results = try denoise_a_exe.results(allocator);
        defer da_results.deinit(allocator);
        da_args.set(.{ a_latent_buf, guided_a_x0, a_mask_buf, a_clean_buf, sigma_buf, sigma_next_buf });
        denoise_a_exe.call(da_args, &da_results);
        const da_out = da_results.get(zml.Bufferized(model.DenoisingStepResult));

        guided_v_x0.deinit();
        guided_a_x0.deinit();

        // Update latents for next iteration
        v_latent_buf.deinit();
        a_latent_buf.deinit();
        v_latent_buf = dv_out.next_latent;
        a_latent_buf = da_out.next_latent;

        // Dump per-step latent output
        if (is_reset or step_idx == 0 or step_idx == 4 or step_idx == 14) {
            const v_name = try std.fmt.allocPrint(allocator, "step{d}_video_latent.bin", .{step_idx + 1});
            defer allocator.free(v_name);
            const a_name = try std.fmt.allocPrint(allocator, "step{d}_audio_latent.bin", .{step_idx + 1});
            defer allocator.free(a_name);
            std.log.info("  Dumping step-{d} latent...", .{step_idx + 1});
            try writeBuffer(allocator, io, v_latent_buf, output_dir, v_name);
            try writeBuffer(allocator, io, a_latent_buf, output_dir, a_name);
        }

        std.log.info("  Step {d} complete.", .{step_idx + 1});
    }

    std.log.info("", .{});
    std.log.info("Denoising complete. Writing output...", .{});

    // ========================================================================
    // Section G: Write output
    // ========================================================================
    try writeBuffer(allocator, io, v_latent_buf, output_dir, "video_latent.bin");
    try writeBuffer(allocator, io, a_latent_buf, output_dir, "audio_latent.bin");

    v_latent_buf.deinit();
    a_latent_buf.deinit();

    std.log.info("Done. Output written to {s}", .{output_dir});
}

// ============================================================================
// Helpers
// ============================================================================

fn runOutputProjection(
    allocator: std.mem.Allocator,
    exe: anytype,
    h: zml.Buffer,
    emb_ts: zml.Buffer,
    bufs: anytype,
) !zml.Buffer {
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ h, emb_ts, bufs });
    exe.call(args, &results);
    return results.get(zml.Buffer);
}

/// Convert velocity → x0 prediction: x0 = sample - vel * (mask * sigma)
fn runToDenoised(
    allocator: std.mem.Allocator,
    exe: anytype,
    sample: zml.Buffer,
    velocity: zml.Buffer,
    mask: zml.Buffer,
    sigma: zml.Buffer,
) !zml.Buffer {
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ sample, velocity, mask, sigma });
    exe.call(args, &results);
    return results.get(zml.Buffer);
}

fn loadBuf(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    name: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const shape = store.view().getShape(name) orelse {
        std.log.err("Tensor not found in store: {s}", .{name});
        return error.TensorNotFound;
    };
    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);
    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(name, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);
    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}

fn sigma1dBuffer(
    io: std.Io,
    platform: *zml.Platform,
    sigma: f32,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const shape = zml.Shape.init(.{ .b = 1 }, .f32);
    return zml.Buffer.fromBytes(io, platform, shape, sharding, std.mem.asBytes(&sigma));
}

fn writeBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    buf: zml.Buffer,
    dir: []const u8,
    filename: []const u8,
) !void {
    const slice = try buf.toSliceAlloc(allocator, io);
    defer slice.free(allocator);

    const path = try std.fs.path.join(allocator, &.{ dir, filename });
    defer allocator.free(path);

    const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer file.close(io);

    var write_buf: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &write_buf);
    try writer.interface.writeAll(slice.constData());
    try writer.interface.flush();

    std.log.info("  Wrote {s} ({d} bytes, shape {any})", .{ path, slice.constData().len, buf.shape() });
}
