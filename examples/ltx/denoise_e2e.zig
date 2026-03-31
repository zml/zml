/// End-to-end Stage 2 denoiser.
///
/// Loads patchified noised latent states + text contexts from a safetensors
/// file (produced by export_stage2_inputs.py), runs the 3-step Euler
/// denoising loop using the velocity model, and writes the denoised
/// patchified latents as raw binary files for decode_latents.py.
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:denoise_e2e -- \
///       /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
///       /root/e2e_demo/stage2_inputs.safetensors \
///       /root/e2e_demo/

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");

comptime {
    @setEvalBranchQuota(200000);
}

pub const std_options: std.Options = .{ .log_level = .info };

/// Sigma schedule for stage-2 distilled (3 steps).
const SIGMAS = [4]f32{ 0.909375, 0.725, 0.421875, 0.0 };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("Stage 2 denoiser (e2e demo)", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: denoise_e2e <checkpoint.safetensors> <stage2_inputs.safetensors> <output_dir/> [--bf16-attn]", .{});
        return error.InvalidArgs;
    };
    const inputs_path = it.next() orelse {
        std.log.err("Usage: denoise_e2e <checkpoint.safetensors> <stage2_inputs.safetensors> <output_dir/> [--bf16-attn]", .{});
        return error.InvalidArgs;
    };
    const output_dir = it.next() orelse {
        std.log.err("Usage: denoise_e2e <checkpoint.safetensors> <stage2_inputs.safetensors> <output_dir/> [--bf16-attn]", .{});
        return error.InvalidArgs;
    };

    // Optional: --bf16-attn to use bf16-native attention (needed for large resolutions
    // to avoid 72+ GiB f32 attention matrix OOM).
    var use_bf16_attn = false;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--bf16-attn")) {
            use_bf16_attn = true;
        }
    }

    // ========================================================================
    // Section A: CLI argument parsing (above) / Section B: Open stores
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

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ========================================================================
    // (continued) Load inputs
    // ========================================================================
    std.log.info("Loading inputs...", .{});

    var v_noise_buf = try loadBuf(allocator, io, platform, &inputs_store, "video_noise", sharding);
    defer v_noise_buf.deinit();
    var a_noise_buf = try loadBuf(allocator, io, platform, &inputs_store, "audio_noise", sharding);
    defer a_noise_buf.deinit();
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
    var v_context_buf = try loadBuf(allocator, io, platform, &inputs_store, "v_context", sharding);
    defer v_context_buf.deinit();
    var a_context_buf = try loadBuf(allocator, io, platform, &inputs_store, "a_context", sharding);
    defer a_context_buf.deinit();

    std.log.info("  video_noise: {any}", .{v_noise_buf.shape()});
    std.log.info("  video_clean: {any}", .{v_clean_buf.shape()});
    std.log.info("  audio_noise: {any}", .{a_noise_buf.shape()});
    std.log.info("  audio_clean: {any}", .{a_clean_buf.shape()});

    // ========================================================================
    // Section C: Compile executables (noise init + preprocessing + blocks + projection + denoising)
    // ========================================================================
    std.log.info("Compiling noise init...", .{});
    const sigma_scalar_shape = zml.Shape.init(.{}, .f32);

    var noise_init_v_exe = try platform.compileFn(
        allocator, io,
        model.forwardNoiseInit,
        .{
            zml.Tensor.fromShape(v_clean_buf.shape()),
            zml.Tensor.fromShape(v_noise_buf.shape()),
            zml.Tensor.fromShape(v_mask_buf.shape()),
            zml.Tensor.fromShape(sigma_scalar_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer noise_init_v_exe.deinit();

    var noise_init_a_exe = try platform.compileFn(
        allocator, io,
        model.forwardNoiseInit,
        .{
            zml.Tensor.fromShape(a_clean_buf.shape()),
            zml.Tensor.fromShape(a_noise_buf.shape()),
            zml.Tensor.fromShape(a_mask_buf.shape()),
            zml.Tensor.fromShape(sigma_scalar_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer noise_init_a_exe.deinit();

    std.log.info("Running noise init...", .{});
    var sigma0_buf = try zml.Buffer.scalar(io, platform, SIGMAS[0], .f32, sharding);
    defer sigma0_buf.deinit();

    // Video noise init
    var ni_v_args = try noise_init_v_exe.args(allocator);
    defer ni_v_args.deinit(allocator);
    var ni_v_results = try noise_init_v_exe.results(allocator);
    defer ni_v_results.deinit(allocator);
    ni_v_args.set(.{ v_clean_buf, v_noise_buf, v_mask_buf, sigma0_buf });
    noise_init_v_exe.call(ni_v_args, &ni_v_results);
    var v_latent_buf = ni_v_results.get(zml.Buffer);

    // Audio noise init
    var ni_a_args = try noise_init_a_exe.args(allocator);
    defer ni_a_args.deinit(allocator);
    var ni_a_results = try noise_init_a_exe.results(allocator);
    defer ni_a_results.deinit(allocator);
    ni_a_args.set(.{ a_clean_buf, a_noise_buf, a_mask_buf, sigma0_buf });
    noise_init_a_exe.call(ni_a_args, &ni_a_results);
    var a_latent_buf = ni_a_results.get(zml.Buffer);

    std.log.info("  video_latent (noised): {any}", .{v_latent_buf.shape()});
    std.log.info("  audio_latent (noised): {any}", .{a_latent_buf.shape()});

    // ========================================================================
    // (continued) Compile remaining executables
    // ========================================================================
    std.log.info("Compiling preprocessing exe...", .{});

    // Run initial preprocessing to discover output shapes
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
            zml.Tensor.fromShape(v_context_buf.shape()),
            zml.Tensor.fromShape(a_context_buf.shape()),
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
    var sigma_1d_init = try sigma1dBuffer(io, platform, SIGMAS[0], sharding);
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
        v_context_buf, a_context_buf,
        preprocess_bufs,
    });
    preprocess_exe.call(pre_args_init, &pre_results_init);
    const init_pre_out = pre_results_init.get(zml.Bufferized(model.PreprocessOutput));

    // Compile block exe
    std.log.info("Compiling block exe (bf16_attn={})...", .{use_bf16_attn});
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

    var block_exe = if (use_bf16_attn)
        try platform.compileFn(allocator, io, model.forwardBlock0NativeBf16Attn, block_compile_args, compile_opts)
    else
        try platform.compileFn(allocator, io, model.forwardBlock0Native, block_compile_args, compile_opts);
    defer block_exe.deinit();
    std.log.info("Block exe compiled.", .{});

    // Compile output projection exes
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

    // Compile denoising step exes
    std.log.info("Compiling denoising step exes...", .{});

    var denoise_v_exe = try platform.compileFn(
        allocator, io,
        model.forwardDenoisingStep,
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
        model.forwardDenoisingStep,
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
    std.log.info("Denoising step exes compiled.", .{});

    // ========================================================================
    // Section D: Load weights
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
    // Section E: Denoising loop (3-step Euler)
    // ========================================================================
    std.log.info("Starting 3-step denoising loop...", .{});

    for (0..3) |step_idx| {
        const sigma = SIGMAS[step_idx];
        const sigma_next = SIGMAS[step_idx + 1];

        std.log.info("", .{});
        std.log.info("===== Step {d}: sigma={d:.6} → {d:.6} =====", .{ step_idx, sigma, sigma_next });

        var sigma_1d = try sigma1dBuffer(io, platform, sigma, sharding);
        defer sigma_1d.deinit();
        var sigma_buf = try zml.Buffer.scalar(io, platform, sigma, .f32, sharding);
        defer sigma_buf.deinit();
        var sigma_next_buf = try zml.Buffer.scalar(io, platform, sigma_next, .f32, sharding);
        defer sigma_next_buf.deinit();

        // ---- 1. Preprocessing ----
        std.log.info("  Running preprocessing...", .{});
        var pre_args = try preprocess_exe.args(allocator);
        defer pre_args.deinit(allocator);
        var pre_results = try preprocess_exe.results(allocator);
        defer pre_results.deinit(allocator);

        pre_args.set(.{
            v_latent_buf, a_latent_buf,
            v_mask_buf, a_mask_buf,
            sigma_1d, sigma_1d,
            v_positions_buf, a_positions_buf,
            v_context_buf, a_context_buf,
            preprocess_bufs,
        });
        preprocess_exe.call(pre_args, &pre_results);

        const pre_out = pre_results.get(zml.Bufferized(model.PreprocessOutput));

        // ---- 2. 48-block chain ----
        std.log.info("  Running 48-block chain...", .{});
        var h_v = pre_out.vx;
        var h_a = pre_out.ax;

        for (0..48) |i| {
            var args = try block_exe.args(allocator);
            defer args.deinit(allocator);
            var results = try block_exe.results(allocator);
            defer results.deinit(allocator);

            args.set(.{
                h_v, h_a,
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
            block_exe.call(args, &results);

            const out = results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));

            if (i > 0) {
                h_v.deinit();
                h_a.deinit();
            }
            h_v = out.vx_out;
            h_a = out.ax_out;

            if (i % 16 == 15 or i == 47) {
                std.log.info("    block {d:>2} done", .{i});
            }
        }

        // ---- 3. Output projection ----
        std.log.info("  Running output projection...", .{});

        var proj_v_args = try proj_v_exe.args(allocator);
        defer proj_v_args.deinit(allocator);
        var proj_v_results = try proj_v_exe.results(allocator);
        defer proj_v_results.deinit(allocator);
        proj_v_args.set(.{ h_v, pre_out.v_embedded_timestep, proj_v_bufs });
        proj_v_exe.call(proj_v_args, &proj_v_results);
        var video_vel = proj_v_results.get(zml.Buffer);

        var proj_a_args = try proj_a_exe.args(allocator);
        defer proj_a_args.deinit(allocator);
        var proj_a_results = try proj_a_exe.results(allocator);
        defer proj_a_results.deinit(allocator);
        proj_a_args.set(.{ h_a, pre_out.a_embedded_timestep, proj_a_bufs });
        proj_a_exe.call(proj_a_args, &proj_a_results);
        var audio_vel = proj_a_results.get(zml.Buffer);

        h_v.deinit();
        h_a.deinit();

        // ---- 4. Denoising step ----
        std.log.info("  Running denoising step...", .{});

        var dv_args = try denoise_v_exe.args(allocator);
        defer dv_args.deinit(allocator);
        var dv_results = try denoise_v_exe.results(allocator);
        defer dv_results.deinit(allocator);
        dv_args.set(.{ v_latent_buf, video_vel, v_mask_buf, v_clean_buf, sigma_buf, sigma_next_buf });
        denoise_v_exe.call(dv_args, &dv_results);
        const dv_out = dv_results.get(zml.Bufferized(model.DenoisingStepResult));

        var da_args = try denoise_a_exe.args(allocator);
        defer da_args.deinit(allocator);
        var da_results = try denoise_a_exe.results(allocator);
        defer da_results.deinit(allocator);
        da_args.set(.{ a_latent_buf, audio_vel, a_mask_buf, a_clean_buf, sigma_buf, sigma_next_buf });
        denoise_a_exe.call(da_args, &da_results);
        const da_out = da_results.get(zml.Bufferized(model.DenoisingStepResult));

        video_vel.deinit();
        audio_vel.deinit();

        // Update latent for next iteration
        v_latent_buf.deinit();
        a_latent_buf.deinit();
        v_latent_buf = dv_out.next_latent;
        a_latent_buf = da_out.next_latent;

        std.log.info("  Step {d} complete.", .{step_idx});
    }

    std.log.info("", .{});
    std.log.info("Denoising complete. Writing output...", .{});

    // ========================================================================
    // Section F: Write output
    // ========================================================================
    // Video: bf16 [B, T_v, 128] — patchified
    // Audio: bf16 [B, T_a, 128] — patchified
    try writeBuffer(allocator, io, v_latent_buf, output_dir, "video_latent.bin");
    try writeBuffer(allocator, io, a_latent_buf, output_dir, "audio_latent.bin");

    v_latent_buf.deinit();
    a_latent_buf.deinit();

    std.log.info("Done. Output written to {s}", .{output_dir});
}

// ============================================================================
// Helpers
// ============================================================================

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
