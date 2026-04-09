/// Unified end-to-end pipeline: Stage 1 → Bridge → Stage 2
///
/// Runs the full LTX-2.3 denoising pipeline in a single binary:
///   1. Stage 1: 30-step guided denoising (4-pass per step × 48 blocks)
///   2. Bridge:  unpatchify → upsample → patchify → noise init
///   3. Stage 2: 3-step distilled denoising (1-pass × 48 blocks)
///
/// GPU buffers are passed directly between phases — no intermediate files.
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
///       --stage1-ckpt /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
///       --stage2-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
///       --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
///       --stage1-inputs /root/e2e_demo/stage1_inputs.safetensors \
///       --stage2-noise /root/e2e_demo/stage2_noise.safetensors \
///       --meta /root/e2e_demo/pipeline_meta.json \
///       --output-dir /root/e2e_demo/unified_out/ \
///       --bf16-attn-stage2

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const conv_ops = @import("conv_ops.zig");
const image_loading = @import("image_loading.zig");
const upsampler = @import("upsampler.zig");
const video_vae = @import("video_vae.zig");
const video_vae_encoder = @import("video_vae_encoder.zig");
const audio_vae = @import("audio_vae.zig");
const vocoder = @import("vocoder.zig");

comptime {
    @setEvalBranchQuota(200000);
}

pub const std_options: std.Options = .{ .log_level = .info };

// ============================================================================
// Pipeline metadata (parsed from pipeline_meta.json)
// ============================================================================

const StageMeta = struct {
    h_lat: i64,
    w_lat: i64,
    f_lat: i64,
    t_audio: i64,
};

const PipelineMeta = struct {
    frame_rate: f64,
    stage1: StageMeta,
    stage2: StageMeta,
};

// ============================================================================
// CLI argument parsing
// ============================================================================

const CliArgs = struct {
    stage1_ckpt: []const u8,
    stage2_ckpt: []const u8,
    upsampler_ckpt: []const u8,
    stage1_inputs: []const u8,
    stage2_noise: []const u8,
    meta: []const u8,
    output_dir: []const u8,
    bf16_attn_stage1: bool,
    bf16_attn_stage2: bool,
    dump_intermediates: bool,
    image: ?[]const u8,
};

fn parseArgs(it: anytype) !CliArgs {
    var args: CliArgs = .{
        .stage1_ckpt = undefined,
        .stage2_ckpt = undefined,
        .upsampler_ckpt = undefined,
        .stage1_inputs = undefined,
        .stage2_noise = undefined,
        .meta = undefined,
        .output_dir = undefined,
        .bf16_attn_stage1 = false,
        .bf16_attn_stage2 = false,
        .dump_intermediates = false,
        .image = null,
    };
    var have = [_]bool{false} ** 7;

    _ = it.next(); // exe name

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--stage1-ckpt")) {
            args.stage1_ckpt = it.next() orelse return error.InvalidArgs;
            have[0] = true;
        } else if (std.mem.eql(u8, arg, "--stage2-ckpt")) {
            args.stage2_ckpt = it.next() orelse return error.InvalidArgs;
            have[1] = true;
        } else if (std.mem.eql(u8, arg, "--upsampler-ckpt")) {
            args.upsampler_ckpt = it.next() orelse return error.InvalidArgs;
            have[2] = true;
        } else if (std.mem.eql(u8, arg, "--stage1-inputs")) {
            args.stage1_inputs = it.next() orelse return error.InvalidArgs;
            have[3] = true;
        } else if (std.mem.eql(u8, arg, "--stage2-noise")) {
            args.stage2_noise = it.next() orelse return error.InvalidArgs;
            have[4] = true;
        } else if (std.mem.eql(u8, arg, "--meta")) {
            args.meta = it.next() orelse return error.InvalidArgs;
            have[5] = true;
        } else if (std.mem.eql(u8, arg, "--output-dir")) {
            args.output_dir = it.next() orelse return error.InvalidArgs;
            have[6] = true;
        } else if (std.mem.eql(u8, arg, "--bf16-attn-stage1")) {
            args.bf16_attn_stage1 = true;
        } else if (std.mem.eql(u8, arg, "--bf16-attn-stage2")) {
            args.bf16_attn_stage2 = true;
        } else if (std.mem.eql(u8, arg, "--dump-intermediates")) {
            args.dump_intermediates = true;
        } else if (std.mem.eql(u8, arg, "--image")) {
            args.image = it.next() orelse return error.InvalidArgs;
        }
    }

    for (have) |h| {
        if (!h) {
            std.log.err(
                "Usage: inference --stage1-ckpt <path> --stage2-ckpt <path> " ++
                    "--upsampler-ckpt <path> --stage1-inputs <path> --stage2-noise <path> " ++
                    "--meta <path> --output-dir <path> [--bf16-attn-stage1] [--bf16-attn-stage2] [--dump-intermediates]",
                .{},
            );
            return error.InvalidArgs;
        }
    }

    return args;
}

// ============================================================================
// Stage 1 result — buffers kept alive for bridge / Stage 2
// ============================================================================

const Stage1Result = struct {
    v_latent: zml.Buffer, // [1, T_v1, 128] bf16
    a_latent: zml.Buffer, // [1, T_a,  128] bf16
    v_context_pos: zml.Buffer, // [1, S, 4096] bf16 — kept for Stage 2
    a_context_pos: zml.Buffer, // [1, S, 2048] bf16 — kept for Stage 2

    fn deinit(self: *Stage1Result) void {
        self.v_latent.deinit();
        self.a_latent.deinit();
        self.v_context_pos.deinit();
        self.a_context_pos.deinit();
    }
};

// ============================================================================
// Bridge result — all buffers needed by Stage 2
// ============================================================================

const BridgeResult = struct {
    v_latent: zml.Buffer, // [1, T_v2, 128] bf16 (noised)
    a_latent: zml.Buffer, // [1, T_a,  128] bf16 (noised)
    v_positions: zml.Buffer, // [1, 3, T_v2, 2] bf16
    a_positions: zml.Buffer, // [1, 1, T_a,  2] f32
    v_mask: zml.Buffer, // [1, T_v2, 1] f32
    a_mask: zml.Buffer, // [1, T_a,  1] f32
    v_context: zml.Buffer, // [1, S, 4096] bf16
    a_context: zml.Buffer, // [1, S, 2048] bf16
    v_clean: zml.Buffer, // [1, T_v2, 128] bf16 (clean latent for Euler step)
    a_clean: zml.Buffer, // [1, T_a,  128] bf16 (clean audio for Euler step)

    fn deinit(self: *BridgeResult) void {
        self.v_latent.deinit();
        self.a_latent.deinit();
        self.v_positions.deinit();
        self.a_positions.deinit();
        self.v_mask.deinit();
        self.a_mask.deinit();
        self.v_context.deinit();
        self.a_context.deinit();
        self.v_clean.deinit();
        self.a_clean.deinit();
    }
};

// ============================================================================
// Stage 2 result — final denoised latents
// ============================================================================

const Stage2Result = struct {
    v_latent: zml.Buffer, // [1, T_v2, 128] bf16
    a_latent: zml.Buffer, // [1, T_a,  128] bf16

    fn deinit(self: *Stage2Result) void {
        self.v_latent.deinit();
        self.a_latent.deinit();
    }
};

const VideoFrames = struct {
    data: []u8,
    width: usize,
    height: usize,
    num_frames: usize,
    allocator: std.mem.Allocator,

    fn deinit(self: *VideoFrames) void {
        self.allocator.free(self.data);
    }
};

// ============================================================================
// Constants
// ============================================================================

/// STG perturbation block index (0-based). Matches LTX_2_3_PARAMS.stg_blocks=[28].
const STG_BLOCK_IDX: usize = 28;

/// Number of Stage 1 denoising steps.
const NUM_STAGE1_STEPS: usize = model.stage1_default_schedule.num_steps;

// ============================================================================
// Image conditioning: graph functions + runtime helper
// ============================================================================

/// Result of applying image conditioning to the initial state.
const ConditioningResult = struct {
    latent: zml.Tensor,
    clean_latent: zml.Tensor,
    denoise_mask: zml.Tensor,
};

/// Graph function: Apply image conditioning to initial denoising state.
///
/// Replaces the first `n_img_tokens` positions (= first frame) in the video
/// latent, clean latent, and denoise mask with the encoded image tokens.
///
/// Python ref: VideoConditionByLatentIndex.apply_to() + GaussianNoiser.__call__()
///
/// latent:        [1, T_video, 128] bf16 (noised initial state)
/// clean_latent:  [1, T_video, 128] bf16
/// denoise_mask:  [1, T_video, 1]   f32  (1.0 = denoise, 0.0 = keep)
/// img_tokens:    [1, n_img, 128]   bf16 (encoded image, patchified)
///
/// For conditioned positions (first n_img tokens):
///   latent       = img_tokens  (clean — no noise)
///   clean_latent = img_tokens
///   denoise_mask = 0.0  (strength=1.0 → 1 - 1 = 0)
fn forwardApplyConditioning(
    latent: zml.Tensor,
    clean_latent: zml.Tensor,
    denoise_mask: zml.Tensor,
    img_tokens: zml.Tensor,
) ConditioningResult {
    const n_img = img_tokens.dim(1);

    // Build replacement tensors at full sequence length by concatenating

    // latent: [img_tokens ; latent[n_img:]]
    const latent_rest = latent.slice1d(1, .{ .start = n_img });
    const new_latent = zml.Tensor.concatenate(&.{ img_tokens, latent_rest }, 1);

    // clean_latent: [img_tokens ; clean_latent[n_img:]]
    const clean_rest = clean_latent.slice1d(1, .{ .start = n_img });
    const new_clean = zml.Tensor.concatenate(&.{ img_tokens, clean_rest }, 1);

    // denoise_mask: [zeros(n_img, 1) ; mask[n_img:]]
    const zero_mask = zml.Tensor.zeroes(
        zml.Shape.init(.{ 1, n_img, 1 }, .f32),
    );
    const mask_rest = denoise_mask.slice1d(1, .{ .start = n_img });
    const new_mask = zml.Tensor.concatenate(&.{ zero_mask, mask_rest }, 1);

    return .{
        .latent = new_latent,
        .clean_latent = new_clean,
        .denoise_mask = new_mask,
    };
}

/// Encode image and patchify: pixel image → latent tokens.
/// [B, 3, 1, H, W] bf16 → [B, 128, 1, H/32, W/32] → [B, n_img, 128]
fn forwardEncodeAndPatchify(
    pixel_input: zml.Tensor,
    stats: conv_ops.PerChannelStats,
    encoder_params: video_vae_encoder.VideoVaeEncoderParams,
) zml.Tensor {
    // VAE encode: [B, 3, 1, H, W] → [B, 128, 1, H/32, W/32]
    const encoded = video_vae_encoder.forwardVideoVaeEncode(pixel_input, stats, encoder_params);
    // Patchify to token space: [B, 128, 1, H', W'] → [B, H'*W', 128]
    return upsampler.forwardPatchifyVideo(encoded);
}

/// Runtime helper: compile VAE encoder, run it on an image, return patchified
/// tokens as a Buffer. Caller owns the returned buffer.
fn encodeImageToTokens(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    ckpt_path: []const u8,
    image_buf: zml.Buffer,
) !zml.Buffer {
    // Open checkpoint for encoder weights
    var ckpt_reg = try zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path);
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    const encoder_shape = video_vae_encoder.initVideoVaeEncoderParams(ckpt_store.view());
    const stats_shape = conv_ops.initPerChannelStats(ckpt_store.view());

    std.log.info("  Compiling VAE encoder...", .{});
    var exe = try platform.compileFn(
        allocator, io,
        forwardEncodeAndPatchify,
        .{
            zml.Tensor.fromShape(image_buf.shape()),
            stats_shape,
            encoder_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();

    std.log.info("  Loading encoder weights...", .{});
    const encoder_bufs = try zml.io.load(
        video_vae_encoder.VideoVaeEncoderParams, &encoder_shape,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    const stats_bufs = try zml.io.load(
        conv_ops.PerChannelStats, &stats_shape,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );

    std.log.info("  Running encoder...", .{});
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ image_buf, stats_bufs, encoder_bufs });
    exe.call(args, &results);
    return results.get(zml.Buffer);
}

/// Runtime helper: apply image conditioning to existing latent/clean/mask buffers.
/// Returns new buffers; caller must deinit the old ones.
fn applyConditioning(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    v_latent: zml.Buffer,
    v_clean: zml.Buffer,
    v_mask: zml.Buffer,
    img_tokens: zml.Buffer,
) !struct { latent: zml.Buffer, clean: zml.Buffer, mask: zml.Buffer } {
    var exe = try platform.compileFn(
        allocator, io,
        forwardApplyConditioning,
        .{
            zml.Tensor.fromShape(v_latent.shape()),
            zml.Tensor.fromShape(v_clean.shape()),
            zml.Tensor.fromShape(v_mask.shape()),
            zml.Tensor.fromShape(img_tokens.shape()),
        },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ v_latent, v_clean, v_mask, img_tokens });
    exe.call(args, &results);
    const out = results.get(zml.Bufferized(ConditioningResult));
    return .{ .latent = out.latent, .clean = out.clean_latent, .mask = out.denoise_mask };
}



// ============================================================================
// Main
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("LTX-2.3 Unified Pipeline (Stage 1 → Bridge → Stage 2)", .{});

    var it = init.minimal.args.iterate();
    const args = try parseArgs(&it);

    std.log.info("  stage1-ckpt:     {s}", .{args.stage1_ckpt});
    std.log.info("  stage2-ckpt:     {s}", .{args.stage2_ckpt});
    std.log.info("  upsampler-ckpt:  {s}", .{args.upsampler_ckpt});
    std.log.info("  stage1-inputs:   {s}", .{args.stage1_inputs});
    std.log.info("  stage2-noise:    {s}", .{args.stage2_noise});
    std.log.info("  meta:            {s}", .{args.meta});
    std.log.info("  output-dir:      {s}", .{args.output_dir});
    std.log.info("  bf16-attn-stage1:  {}", .{args.bf16_attn_stage1});
    std.log.info("  bf16-attn-stage2:  {}", .{args.bf16_attn_stage2});
    std.log.info("  dump-intermediates: {}", .{args.dump_intermediates});
    std.log.info("  image:             {s}", .{args.image orelse "(none)"});

    // ---- Ensure output directory exists ----
    try std.Io.Dir.createDirPath(.cwd(), io, args.output_dir);

    // ---- Load pipeline metadata ----
    const pipe_meta = try loadPipelineMeta(allocator, io, args.meta);
    const stage2_sigmas: []const f32 = &model.stage2_distilled_sigmas;
    std.log.info("  Stage 1: F={d} H={d} W={d} T_a={d}", .{
        pipe_meta.stage1.f_lat, pipe_meta.stage1.h_lat, pipe_meta.stage1.w_lat, pipe_meta.stage1.t_audio,
    });
    std.log.info("  Stage 2: F={d} H={d} W={d} T_a={d} sigmas_len={d}", .{
        pipe_meta.stage2.f_lat, pipe_meta.stage2.h_lat, pipe_meta.stage2.w_lat, pipe_meta.stage2.t_audio, stage2_sigmas.len,
    });
    std.log.info("  fps={d:.1}", .{pipe_meta.frame_rate});

    // ---- Platform init ----
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ========================================================================
    // Phase 1: Stage 1
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Phase 1: Stage 1 — {d}-step guided denoising ===", .{NUM_STAGE1_STEPS});

    var s1 = try runStage1(allocator, io, platform, sharding, args.stage1_ckpt, args.stage1_inputs, args.bf16_attn_stage1, args.image, pipe_meta, args.dump_intermediates, args.output_dir);

    if (args.dump_intermediates) {
        try writeBuffer(allocator, io, s1.v_latent, args.output_dir, "stage1_video_latent.bin");
        try writeBuffer(allocator, io, s1.a_latent, args.output_dir, "stage1_audio_latent.bin");
    }

    // ========================================================================
    // Phase 2: Bridge
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Phase 2: Bridge — upsample + prepare Stage 2 inputs ===", .{});

    var bridge = try runBridge(
        allocator,
        io,
        platform,
        sharding,
        s1.v_latent,
        s1.a_latent,
        s1.v_context_pos,
        s1.a_context_pos,
        args.upsampler_ckpt,
        args.stage2_ckpt,
        args.stage2_noise,
        pipe_meta,
        args.image,
    );
    // Stage 1 latent/context buffers are consumed by bridge (ownership transferred)

    // ========================================================================
    // Phase 3: Stage 2
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Phase 3: Stage 2 — {d}-step distilled denoising ===", .{stage2_sigmas.len - 1});

    var s2 = try runStage2(allocator, io, platform, sharding, args.stage2_ckpt, &bridge, args.bf16_attn_stage2, stage2_sigmas);

    // Free bridge-only buffers (positions, masks, clean latents, contexts).
    // The noised latents (v_latent, a_latent) were consumed by Stage 2's loop
    // and replaced with final outputs — runStage2 set them to undefined.
    bridge.v_positions.deinit();
    bridge.a_positions.deinit();
    bridge.v_mask.deinit();
    bridge.a_mask.deinit();
    bridge.v_context.deinit();
    bridge.a_context.deinit();
    bridge.v_clean.deinit();
    bridge.a_clean.deinit();

    // ========================================================================
    // Write denoised latents (optional, for debugging)
    // ========================================================================
    if (args.dump_intermediates) {
        std.log.info("", .{});
        std.log.info("Writing denoised latents...", .{});
        try writeBuffer(allocator, io, s2.v_latent, args.output_dir, "video_latent.bin");
        try writeBuffer(allocator, io, s2.a_latent, args.output_dir, "audio_latent.bin");
    }

    // ========================================================================
    // Phase 4: Video VAE Decode
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Phase 4: Video VAE Decode ===", .{});

    var video_frames = try runVideoVaeDecode(allocator, io, platform, sharding, args.stage2_ckpt, s2.v_latent, pipe_meta);
    defer video_frames.deinit();

    if (args.dump_intermediates) {
        try writeRawBytes(allocator, io, video_frames.data, args.output_dir, "frames.bin");
    }

    // ========================================================================
    // Phase 5: Audio VAE Decode
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Phase 5: Audio VAE Decode ===", .{});

    var audio_mel = try runAudioVaeDecode(allocator, io, platform, sharding, args.stage2_ckpt, s2.a_latent, pipe_meta);

    if (args.dump_intermediates) {
        try writeBuffer(allocator, io, audio_mel, args.output_dir, "audio_mel.bin");
    }

    // ========================================================================
    // Phase 6: Vocoder + BWE (mel → 48kHz waveform)
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Phase 6: Vocoder + BWE ===", .{});

    var waveform_buf = try runVocoderWithBWE(allocator, io, platform, sharding, args.stage2_ckpt, audio_mel);
    defer waveform_buf.deinit();
    audio_mel.deinit();

    s2.deinit();

    // ========================================================================
    // Final: Encode video + audio → output.mp4
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Encoding output.mp4 (video + audio) ===", .{});

    const waveform_slice = try waveform_buf.toSliceAlloc(allocator, io);
    defer waveform_slice.free(allocator);
    const waveform_bytes = waveform_slice.constData();

    // Waveform shape is [1, 2, N] f32 (planar: all left then all right).
    // Interleave to [N, 2] f32 (L R L R...) for ffmpeg.
    const num_channels: usize = @intCast(waveform_buf.shape().dim(1));
    const num_samples: usize = @intCast(waveform_buf.shape().dim(2));
    const planar_f32: [*]const f32 = @alignCast(@ptrCast(waveform_bytes.ptr));
    const interleaved = try allocator.alloc(f32, num_channels * num_samples);
    defer allocator.free(interleaved);
    for (0..num_samples) |i| {
        for (0..num_channels) |ch| {
            interleaved[i * num_channels + ch] = planar_f32[ch * num_samples + i];
        }
    }
    const interleaved_bytes = std.mem.sliceAsBytes(interleaved);

    // Also write interleaved waveform.bin for debugging
    try writeRawBytes(allocator, io, interleaved_bytes, args.output_dir, "waveform.bin");

    try encodeOutputMp4(allocator, io, video_frames, interleaved_bytes, num_channels, pipe_meta.frame_rate, args.output_dir);

    std.log.info("Done.", .{});
}

// ============================================================================
// Phase 1: Stage 1 — 30-step guided denoising
// ============================================================================

fn runStage1(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    ckpt_path: []const u8,
    inputs_path: []const u8,
    use_bf16_attn: bool,
    image_path: ?[]const u8,
    pipe_meta: PipelineMeta,
    dump_intermediates: bool,
    output_dir: []const u8,
) !Stage1Result {
    // ---- Sigma schedule ----
    const sigmas = model.computeSigmaSchedule(
        NUM_STAGE1_STEPS,
        NUM_STAGE1_STEPS,
        model.stage1_default_schedule.default_num_tokens,
        model.stage1_default_schedule.max_shift,
        model.stage1_default_schedule.base_shift,
        model.stage1_default_schedule.terminal,
    );
    std.log.info("Sigma schedule ({d} steps): [{d:.6} ... {d:.6}]", .{
        NUM_STAGE1_STEPS, sigmas[0], sigmas[NUM_STAGE1_STEPS],
    });

    // ---- Open stores ----
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

    // ---- Load inputs ----
    std.log.info("Loading Stage 1 inputs...", .{});

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

    // Text contexts — positive AND negative for Stage 1 guidance.
    // Positive contexts are kept alive and returned for Stage 2.
    var v_context_pos_buf = try loadBuf(allocator, io, platform, &inputs_store, "v_context_pos", sharding);
    var a_context_pos_buf = try loadBuf(allocator, io, platform, &inputs_store, "a_context_pos", sharding);
    var v_context_neg_buf = try loadBuf(allocator, io, platform, &inputs_store, "v_context_neg", sharding);
    defer v_context_neg_buf.deinit();
    var a_context_neg_buf = try loadBuf(allocator, io, platform, &inputs_store, "a_context_neg", sharding);
    defer a_context_neg_buf.deinit();

    std.log.info("  video_latent (noised)", .{});
    std.log.info("  audio_latent (noised)", .{});

    // ---- Image conditioning (optional) ----
    if (image_path) |img_path| {
        std.log.info("Applying image conditioning to Stage 1...", .{});

        // Load image from disk, resize to Stage 1 pixel resolution, normalize to bf16 [-1,1].
        // Stage 1 uses half resolution: pixel dims = h_lat * 32, w_lat * 32.
        const s1_pixel_h: u32 = @intCast(pipe_meta.stage1.h_lat * 32);
        const s1_pixel_w: u32 = @intCast(pipe_meta.stage1.w_lat * 32);
        var image_s1_buf = try image_loading.loadAndPreprocess(allocator, io, platform, sharding, img_path, s1_pixel_h, s1_pixel_w);
        defer image_s1_buf.deinit();
        std.log.info("  image_s1: {any}", .{image_s1_buf.shape().dims()});

        // Dump preprocessed image for comparison
        if (dump_intermediates) {
            try writeBuffer(allocator, io, image_s1_buf, output_dir, "s1_image_preprocessed.bin");
        }

        // Encode image → patchified tokens
        var img_tokens = try encodeImageToTokens(allocator, io, platform, sharding, ckpt_path, image_s1_buf);
        defer img_tokens.deinit();
        std.log.info("  image tokens: {any}", .{img_tokens.shape().dims()});

        // Apply conditioning: replace first-frame tokens in latent/clean/mask
        const cond = try applyConditioning(allocator, io, platform, sharding, v_latent_buf, v_clean_buf, v_mask_buf, img_tokens);

        // Replace old buffers with conditioned ones
        v_latent_buf.deinit();
        v_latent_buf = cond.latent;
        v_clean_buf.deinit();
        v_clean_buf = cond.clean;
        v_mask_buf.deinit();
        v_mask_buf = cond.mask;

        std.log.info("  Conditioning applied (n_img={d})", .{img_tokens.shape().dim(1)});

        // Debug dumps: img_tokens and conditioned state
        if (dump_intermediates) {
            try writeBuffer(allocator, io, img_tokens, output_dir, "s1_img_tokens.bin");
            try writeBuffer(allocator, io, v_latent_buf, output_dir, "s1_conditioned_latent.bin");
            try writeBuffer(allocator, io, v_clean_buf, output_dir, "s1_conditioned_clean.bin");
            try writeBuffer(allocator, io, v_mask_buf, output_dir, "s1_conditioned_mask.bin");
        }
    }

    // ---- Compile executables ----
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

    // Load preprocessing weights + run once for shape discovery
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

    // ---- Compile block executables ----
    var block_params_shape = try allocator.create(model.FullStepParams);
    defer allocator.destroy(block_params_shape);
    block_params_shape.* = model.initFullStepParams(ckpt_store.view());

    const block_compile_args = .{
        zml.Tensor.fromShape(init_pre_out.vx.shape()),
        zml.Tensor.fromShape(init_pre_out.ax.shape()),
        zml.Tensor.fromShape(init_pre_out.video_timesteps.shape()),
        zml.Tensor.fromShape(init_pre_out.audio_timesteps.shape()),
        zml.Tensor.fromShape(init_pre_out.video_timesteps_zero.shape()),
        zml.Tensor.fromShape(init_pre_out.audio_timesteps_zero.shape()),
        zml.Tensor.fromShape(init_pre_out.v_denoise_mask.shape()),
        zml.Tensor.fromShape(init_pre_out.a_denoise_mask.shape()),
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

    std.log.info("Compiling block exe (normal, bf16_attn={})...", .{use_bf16_attn});
    var block_normal_exe = if (use_bf16_attn)
        try platform.compileFn(allocator, io, model.forwardBlock0NativeBf16Attn, block_compile_args, compile_opts)
    else
        try platform.compileFn(allocator, io, model.forwardBlock0Native, block_compile_args, compile_opts);
    defer block_normal_exe.deinit();

    std.log.info("Compiling block exe (STG, bf16_attn={})...", .{use_bf16_attn});
    var block_stg_exe = if (use_bf16_attn)
        try platform.compileFn(allocator, io, model.forwardBlock0NativeSTGBf16Attn, block_compile_args, compile_opts)
    else
        try platform.compileFn(allocator, io, model.forwardBlock0NativeSTG, block_compile_args, compile_opts);
    defer block_stg_exe.deinit();

    std.log.info("Compiling block exe (isolated, bf16_attn={})...", .{use_bf16_attn});
    const mask_scalar_shape = zml.Shape.init(.{}, .bf16);
    const iso_compile_args = .{
        zml.Tensor.fromShape(init_pre_out.vx.shape()),
        zml.Tensor.fromShape(init_pre_out.ax.shape()),
        zml.Tensor.fromShape(init_pre_out.video_timesteps.shape()),
        zml.Tensor.fromShape(init_pre_out.audio_timesteps.shape()),
        zml.Tensor.fromShape(init_pre_out.video_timesteps_zero.shape()),
        zml.Tensor.fromShape(init_pre_out.audio_timesteps_zero.shape()),
        zml.Tensor.fromShape(init_pre_out.v_denoise_mask.shape()),
        zml.Tensor.fromShape(init_pre_out.a_denoise_mask.shape()),
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

    // ---- Compile output projection exes ----
    std.log.info("Compiling output projection exes...", .{});
    const v_emb_shape = init_pre_out.v_embedded_timestep.shape();
    const a_emb_shape = init_pre_out.a_embedded_timestep.shape();

    var proj_v_exe = try platform.compileFn(
        allocator, io,
        model.forwardOutputProjection,
        .{
            zml.Tensor.fromShape(init_pre_out.vx.shape()).withPartialTags(.{ .b, .t, .d }),
            zml.Tensor.fromShape(v_emb_shape).withPartialTags(.{ .b, .t, .d_emb }),
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
            zml.Tensor.fromShape(a_emb_shape).withPartialTags(.{ .b, .t, .d_emb }),
            block_params_shape.audio_norm_proj_out,
        },
        .{ .shardings = &.{sharding} },
    );
    defer proj_a_exe.deinit();

    // ---- Compile vel→x0 exes ----
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

    // ---- Compile denoising step exes (from x0) ----
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

    // ---- Compile guider combine exe ----
    std.log.info("Compiling guider combine exe...", .{});
    const vel_v_shape = v_latent_buf.shape();
    const vel_a_shape = a_latent_buf.shape();
    var guider_combine_exe = try platform.compileFn(
        allocator, io,
        model.forwardGuiderCombine,
        .{
            zml.Tensor.fromShape(vel_v_shape),
            zml.Tensor.fromShape(vel_v_shape),
            zml.Tensor.fromShape(vel_v_shape),
            zml.Tensor.fromShape(vel_v_shape),
            zml.Tensor.fromShape(vel_a_shape),
            zml.Tensor.fromShape(vel_a_shape),
            zml.Tensor.fromShape(vel_a_shape),
            zml.Tensor.fromShape(vel_a_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer guider_combine_exe.deinit();
    std.log.info("All Stage 1 exes compiled.", .{});

    // ---- Load weights ----
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
    std.log.info("All Stage 1 weights loaded.", .{});

    // ---- Guidance scalars ----
    var zero_mask_buf = try zml.Buffer.scalar(io, platform, @as(f32, 0.0), .bf16, sharding);
    defer zero_mask_buf.deinit();
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

    // ---- Denoising loop: 30 steps × 4 passes ----
    std.log.info("Starting {d}-step denoising loop (4-pass guidance)...", .{NUM_STAGE1_STEPS});

    for (0..NUM_STAGE1_STEPS) |step_idx| {
        const sigma = sigmas[step_idx];
        const sigma_next = sigmas[step_idx + 1];

        std.log.info("", .{});
        std.log.info("===== Step {d}/{d}: sigma={d:.6} -> {d:.6} =====", .{
            step_idx + 1, NUM_STAGE1_STEPS, sigma, sigma_next,
        });

        var sigma_1d = try sigma1dBuffer(io, platform, sigma, sharding);
        defer sigma_1d.deinit();
        var sigma_buf = try zml.Buffer.scalar(io, platform, sigma, .f32, sharding);
        defer sigma_buf.deinit();
        var sigma_next_buf = try zml.Buffer.scalar(io, platform, sigma_next, .f32, sharding);
        defer sigma_next_buf.deinit();

        // ---- Preprocessing (once per step, with positive context) ----
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

        // ---- Pass 1: Conditional (positive context, normal blocks) ----
        std.log.info("  Pass 1 (conditional): 48-block chain...", .{});
        var cond_h_v = pre_out.vx;
        var cond_h_a = pre_out.ax;

        for (0..48) |i| {
            var blk_args = try block_normal_exe.args(allocator);
            defer blk_args.deinit(allocator);
            var blk_results = try block_normal_exe.results(allocator);
            defer blk_results.deinit(allocator);

            blk_args.set(.{
                cond_h_v, cond_h_a,
                pre_out.video_timesteps, pre_out.audio_timesteps,
                pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                pre_out.v_denoise_mask, pre_out.a_denoise_mask,
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
            block_normal_exe.call(blk_args, &blk_results);
            const out = blk_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
            if (i > 0) { cond_h_v.deinit(); cond_h_a.deinit(); }
            cond_h_v = out.vx_out;
            cond_h_a = out.ax_out;
        }

        var cond_v_vel = try runOutputProjection(allocator, &proj_v_exe, cond_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var cond_a_vel = try runOutputProjection(allocator, &proj_a_exe, cond_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        cond_h_v.deinit();
        cond_h_a.deinit();

        var cond_v_x0 = try runToDenoised(allocator, &to_denoised_v_exe, v_latent_buf, cond_v_vel, v_mask_buf, sigma_buf);
        var cond_a_x0 = try runToDenoised(allocator, &to_denoised_a_exe, a_latent_buf, cond_a_vel, a_mask_buf, sigma_buf);
        cond_v_vel.deinit();
        cond_a_vel.deinit();

        // ---- Pass 2: Negative/CFG (negative context, normal blocks) ----
        std.log.info("  Pass 2 (negative/CFG): 48-block chain...", .{});
        var neg_h_v = pre_out.vx;
        var neg_h_a = pre_out.ax;

        for (0..48) |i| {
            var blk_args = try block_normal_exe.args(allocator);
            defer blk_args.deinit(allocator);
            var blk_results = try block_normal_exe.results(allocator);
            defer blk_results.deinit(allocator);

            blk_args.set(.{
                neg_h_v, neg_h_a,
                pre_out.video_timesteps, pre_out.audio_timesteps,
                pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                pre_out.v_denoise_mask, pre_out.a_denoise_mask,
                pre_out.v_prompt_timestep, pre_out.a_prompt_timestep,
                pre_out.v_pe_cos, pre_out.v_pe_sin,
                pre_out.a_pe_cos, pre_out.a_pe_sin,
                v_context_neg_buf, a_context_neg_buf,
                pre_out.v_cross_ss_ts, pre_out.v_cross_gate_ts,
                pre_out.a_cross_ss_ts, pre_out.a_cross_gate_ts,
                pre_out.a2v_pe_cos, pre_out.a2v_pe_sin,
                pre_out.a2v_k_pe_cos, pre_out.a2v_k_pe_sin,
                pre_out.v2a_pe_cos, pre_out.v2a_pe_sin,
                pre_out.v2a_k_pe_cos, pre_out.v2a_k_pe_sin,
                block_params_bufs[i],
            });
            block_normal_exe.call(blk_args, &blk_results);
            const out = blk_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
            if (i > 0) { neg_h_v.deinit(); neg_h_a.deinit(); }
            neg_h_v = out.vx_out;
            neg_h_a = out.ax_out;
        }

        var neg_v_vel = try runOutputProjection(allocator, &proj_v_exe, neg_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var neg_a_vel = try runOutputProjection(allocator, &proj_a_exe, neg_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        neg_h_v.deinit();
        neg_h_a.deinit();

        var neg_v_x0 = try runToDenoised(allocator, &to_denoised_v_exe, v_latent_buf, neg_v_vel, v_mask_buf, sigma_buf);
        var neg_a_x0 = try runToDenoised(allocator, &to_denoised_a_exe, a_latent_buf, neg_a_vel, a_mask_buf, sigma_buf);
        neg_v_vel.deinit();
        neg_a_vel.deinit();

        // ---- Pass 3: STG (positive context, V-passthrough at block 28) ----
        std.log.info("  Pass 3 (STG): 48-block chain (STG at block {d})...", .{STG_BLOCK_IDX});
        var ptb_h_v = pre_out.vx;
        var ptb_h_a = pre_out.ax;

        for (0..48) |i| {
            if (i == STG_BLOCK_IDX) {
                var blk_args = try block_stg_exe.args(allocator);
                defer blk_args.deinit(allocator);
                var blk_results = try block_stg_exe.results(allocator);
                defer blk_results.deinit(allocator);

                blk_args.set(.{
                    ptb_h_v, ptb_h_a,
                    pre_out.video_timesteps, pre_out.audio_timesteps,
                    pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                    pre_out.v_denoise_mask, pre_out.a_denoise_mask,
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
                block_stg_exe.call(blk_args, &blk_results);
                const out = blk_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
                if (i > 0) { ptb_h_v.deinit(); ptb_h_a.deinit(); }
                ptb_h_v = out.vx_out;
                ptb_h_a = out.ax_out;
            } else {
                var blk_args = try block_normal_exe.args(allocator);
                defer blk_args.deinit(allocator);
                var blk_results = try block_normal_exe.results(allocator);
                defer blk_results.deinit(allocator);

                blk_args.set(.{
                    ptb_h_v, ptb_h_a,
                    pre_out.video_timesteps, pre_out.audio_timesteps,
                    pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                    pre_out.v_denoise_mask, pre_out.a_denoise_mask,
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
                block_normal_exe.call(blk_args, &blk_results);
                const out = blk_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
                if (i > 0) { ptb_h_v.deinit(); ptb_h_a.deinit(); }
                ptb_h_v = out.vx_out;
                ptb_h_a = out.ax_out;
            }
        }

        var ptb_v_vel = try runOutputProjection(allocator, &proj_v_exe, ptb_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var ptb_a_vel = try runOutputProjection(allocator, &proj_a_exe, ptb_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        ptb_h_v.deinit();
        ptb_h_a.deinit();

        var ptb_v_x0 = try runToDenoised(allocator, &to_denoised_v_exe, v_latent_buf, ptb_v_vel, v_mask_buf, sigma_buf);
        var ptb_a_x0 = try runToDenoised(allocator, &to_denoised_a_exe, a_latent_buf, ptb_a_vel, a_mask_buf, sigma_buf);
        ptb_v_vel.deinit();
        ptb_a_vel.deinit();

        // ---- Pass 4: Isolated (positive context, zero AV masks) ----
        std.log.info("  Pass 4 (isolated): 48-block chain...", .{});
        var iso_h_v = pre_out.vx;
        var iso_h_a = pre_out.ax;

        for (0..48) |i| {
            var blk_args = try block_iso_exe.args(allocator);
            defer blk_args.deinit(allocator);
            var blk_results = try block_iso_exe.results(allocator);
            defer blk_results.deinit(allocator);

            blk_args.set(.{
                iso_h_v, iso_h_a,
                pre_out.video_timesteps, pre_out.audio_timesteps,
                pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                pre_out.v_denoise_mask, pre_out.a_denoise_mask,
                pre_out.v_prompt_timestep, pre_out.a_prompt_timestep,
                pre_out.v_pe_cos, pre_out.v_pe_sin,
                pre_out.a_pe_cos, pre_out.a_pe_sin,
                pre_out.v_text_ctx, pre_out.a_text_ctx,
                pre_out.v_cross_ss_ts, pre_out.v_cross_gate_ts,
                pre_out.a_cross_ss_ts, pre_out.a_cross_gate_ts,
                pre_out.a2v_pe_cos, pre_out.a2v_pe_sin,
                pre_out.a2v_k_pe_cos, pre_out.a2v_k_pe_sin,
                zero_mask_buf,
                pre_out.v2a_pe_cos, pre_out.v2a_pe_sin,
                pre_out.v2a_k_pe_cos, pre_out.v2a_k_pe_sin,
                zero_mask_buf,
                block_params_bufs[i],
            });
            block_iso_exe.call(blk_args, &blk_results);
            const out = blk_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
            if (i > 0) { iso_h_v.deinit(); iso_h_a.deinit(); }
            iso_h_v = out.vx_out;
            iso_h_a = out.ax_out;
        }

        var iso_v_vel = try runOutputProjection(allocator, &proj_v_exe, iso_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var iso_a_vel = try runOutputProjection(allocator, &proj_a_exe, iso_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        iso_h_v.deinit();
        iso_h_a.deinit();

        var iso_v_x0 = try runToDenoised(allocator, &to_denoised_v_exe, v_latent_buf, iso_v_vel, v_mask_buf, sigma_buf);
        var iso_a_x0 = try runToDenoised(allocator, &to_denoised_a_exe, a_latent_buf, iso_a_vel, a_mask_buf, sigma_buf);
        iso_v_vel.deinit();
        iso_a_vel.deinit();

        // ---- Guider combine ----
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

        cond_v_x0.deinit();
        neg_v_x0.deinit();
        ptb_v_x0.deinit();
        iso_v_x0.deinit();
        cond_a_x0.deinit();
        neg_a_x0.deinit();
        ptb_a_x0.deinit();
        iso_a_x0.deinit();

        // ---- Euler step from guided x0 ----
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

        v_latent_buf.deinit();
        a_latent_buf.deinit();
        v_latent_buf = dv_out.next_latent;
        a_latent_buf = da_out.next_latent;

        std.log.info("  Step {d} complete.", .{step_idx + 1});
    }

    std.log.info("Stage 1 denoising complete.", .{});

    return .{
        .v_latent = v_latent_buf,
        .a_latent = a_latent_buf,
        .v_context_pos = v_context_pos_buf,
        .a_context_pos = a_context_pos_buf,
    };
}

// ============================================================================
// Phase 2: Bridge — unpatchify → upsample → patchify → noise init
// ============================================================================

fn runBridge(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    s1_video: zml.Buffer,
    s1_audio: zml.Buffer,
    v_context: zml.Buffer,
    a_context: zml.Buffer,
    upsampler_ckpt_path: []const u8,
    main_ckpt_path: []const u8,
    noise_path: []const u8,
    pipe_meta: PipelineMeta,
    image_path: ?[]const u8,
) !BridgeResult {
    const s1 = pipe_meta.stage1;
    const s2 = pipe_meta.stage2;
    const fps = pipe_meta.frame_rate;
    const sigma_0: f32 = model.stage2_distilled_sigmas[0];

    const F = s1.f_lat;
    const H_s1 = s1.h_lat;
    const W_s1 = s1.w_lat;
    const H_s2 = s2.h_lat;
    const W_s2 = s2.w_lat;
    const T_a: i64 = s1.t_audio;
    const C: i64 = 128;
    const T_v1 = F * H_s1 * W_s1;
    const T_v2 = F * H_s2 * W_s2;

    // ---- Open checkpoint stores ----
    std.log.info("Opening bridge checkpoint stores...", .{});
    var up_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, upsampler_ckpt_path) catch |err| {
        std.log.err("Failed to open upsampler checkpoint: {s}", .{upsampler_ckpt_path});
        return err;
    };
    defer up_reg.deinit();
    var up_store: zml.io.TensorStore = .fromRegistry(allocator, &up_reg);
    defer up_store.deinit();

    var main_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, main_ckpt_path) catch |err| {
        std.log.err("Failed to open main checkpoint: {s}", .{main_ckpt_path});
        return err;
    };
    defer main_reg.deinit();
    var main_store: zml.io.TensorStore = .fromRegistry(allocator, &main_reg);
    defer main_store.deinit();

    var noise_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, noise_path) catch |err| {
        std.log.err("Failed to open stage2 noise: {s}", .{noise_path});
        return err;
    };
    defer noise_reg.deinit();
    var noise_store: zml.io.TensorStore = .fromRegistry(allocator, &noise_reg);
    defer noise_store.deinit();

    // ========================================================================
    // Step 1: Unpatchify video: [1, T_v1, 128] → [1, 128, F, H_s1, W_s1]
    // ========================================================================
    std.log.info("Compiling unpatchify...", .{});
    const patchified_shape = zml.Shape.init(.{ 1, T_v1, C }, .bf16);
    const video_5d_s1_shape = zml.Shape.init(.{ 1, C, F, H_s1, W_s1 }, .bf16);

    var unpatch_exe = try platform.compileFn(
        allocator, io,
        upsampler.forwardUnpatchifyVideo,
        .{
            zml.Tensor.fromShape(patchified_shape),
            video_5d_s1_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer unpatch_exe.deinit();

    std.log.info("Running unpatchify...", .{});
    var unpatch_args = try unpatch_exe.args(allocator);
    defer unpatch_args.deinit(allocator);
    var unpatch_results = try unpatch_exe.results(allocator);
    defer unpatch_results.deinit(allocator);
    unpatch_args.set(.{s1_video});
    unpatch_exe.call(unpatch_args, &unpatch_results);
    var video_5d_buf = unpatch_results.get(zml.Buffer);
    defer video_5d_buf.deinit();
    var s1_video_mut = s1_video;
    s1_video_mut.deinit();
    std.log.info("  Unpatchified: {any}", .{video_5d_buf.shape().dims()});

    // ========================================================================
    // Step 2: Upsample: [1, 128, F, H_s1, W_s1] → [1, 128, F, H_s2, W_s2]
    // ========================================================================
    std.log.info("Compiling upsampler...", .{});
    const upsampler_shape = upsampler.initUpsamplerParams(up_store.view());
    const stats_shape = conv_ops.initPerChannelStats(main_store.view());

    var upsample_exe = try platform.compileFn(
        allocator, io,
        upsampler.forwardUpsample,
        .{
            zml.Tensor.fromShape(video_5d_buf.shape()),
            upsampler_shape,
            stats_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer upsample_exe.deinit();

    std.log.info("Loading upsampler weights...", .{});
    const up_bufs = try zml.io.load(
        upsampler.UpsamplerParams, &upsampler_shape,
        allocator, io, platform,
        .{
            .store = &up_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );

    std.log.info("Loading per-channel statistics...", .{});
    const stats_bufs = try zml.io.load(
        conv_ops.PerChannelStats, &stats_shape,
        allocator, io, platform,
        .{
            .store = &main_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );

    std.log.info("Running upsampler...", .{});
    var up_args = try upsample_exe.args(allocator);
    defer up_args.deinit(allocator);
    var up_results = try upsample_exe.results(allocator);
    defer up_results.deinit(allocator);
    up_args.set(.{ video_5d_buf, up_bufs, stats_bufs });
    upsample_exe.call(up_args, &up_results);
    var upsampled_buf = up_results.get(zml.Buffer);
    defer upsampled_buf.deinit();
    std.log.info("  Upsampled: {any}", .{upsampled_buf.shape().dims()});

    // ========================================================================
    // Step 3: Re-patchify video: [1, 128, F, H_s2, W_s2] → [1, T_v2, 128]
    // ========================================================================
    std.log.info("Compiling patchify...", .{});
    var patchify_exe = try platform.compileFn(
        allocator, io,
        upsampler.forwardPatchifyVideo,
        .{zml.Tensor.fromShape(upsampled_buf.shape())},
        .{ .shardings = &.{sharding} },
    );
    defer patchify_exe.deinit();

    std.log.info("Running patchify...", .{});
    var patch_args = try patchify_exe.args(allocator);
    defer patch_args.deinit(allocator);
    var patch_results = try patchify_exe.results(allocator);
    defer patch_results.deinit(allocator);
    patch_args.set(.{upsampled_buf});
    patchify_exe.call(patch_args, &patch_results);
    var video_clean_buf = patch_results.get(zml.Buffer);
    std.log.info("  Re-patchified video: {any}", .{video_clean_buf.shape().dims()});

    // ========================================================================
    // Step 4: Audio passthrough (already patchified from Stage 1)
    // ========================================================================
    var audio_clean_buf = s1_audio; // ownership transferred

    // ========================================================================
    // Step 5: Compute positions and masks on host
    // ========================================================================
    std.log.info("Computing video positions...", .{});
    const video_pos_bytes = try computeVideoPositions(allocator, F, H_s2, W_s2, fps);
    defer allocator.free(video_pos_bytes);
    const video_pos_shape = zml.Shape.init(.{ 1, 3, T_v2, 2 }, .bf16);
    const video_pos_buf = try zml.Buffer.fromBytes(io, platform, video_pos_shape, sharding, video_pos_bytes);

    std.log.info("Computing audio positions...", .{});
    const audio_pos_bytes = try computeAudioPositions(allocator, T_a);
    defer allocator.free(audio_pos_bytes);
    const audio_pos_shape = zml.Shape.init(.{ 1, 1, T_a, 2 }, .f32);
    const audio_pos_buf = try zml.Buffer.fromBytes(io, platform, audio_pos_shape, sharding, audio_pos_bytes);

    // Denoise masks: all ones
    std.log.info("Creating denoise masks...", .{});
    const v_mask_shape = zml.Shape.init(.{ 1, T_v2, 1 }, .f32);
    const a_mask_shape = zml.Shape.init(.{ 1, T_a, 1 }, .f32);
    const v_mask_host = try allocator.alloc(u8, v_mask_shape.byteSize());
    defer allocator.free(v_mask_host);
    const a_mask_host = try allocator.alloc(u8, a_mask_shape.byteSize());
    defer allocator.free(a_mask_host);
    fillOnesF32(v_mask_host);
    fillOnesF32(a_mask_host);
    var v_mask_buf = try zml.Buffer.fromBytes(io, platform, v_mask_shape, sharding, v_mask_host);
    var a_mask_buf = try zml.Buffer.fromBytes(io, platform, a_mask_shape, sharding, a_mask_host);

    // ========================================================================
    // Step 6: Load noise and run noise init
    // ========================================================================
    std.log.info("Loading noise tensors...", .{});
    var v_noise_buf = try loadBuf(allocator, io, platform, &noise_store, "video_noise_s2", sharding);
    defer v_noise_buf.deinit();
    var a_noise_buf = try loadBuf(allocator, io, platform, &noise_store, "audio_noise_s2", sharding);
    defer a_noise_buf.deinit();
    std.log.info("  video_noise: {any}", .{v_noise_buf.shape().dims()});
    std.log.info("  audio_noise: {any}", .{a_noise_buf.shape().dims()});

    // Compile noise init
    std.log.info("Compiling noise init...", .{});
    const sigma_scalar_shape = zml.Shape.init(.{}, .f32);

    var noise_init_v_exe = try platform.compileFn(
        allocator, io,
        model.forwardNoiseInit,
        .{
            zml.Tensor.fromShape(video_clean_buf.shape()),
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
            zml.Tensor.fromShape(audio_clean_buf.shape()),
            zml.Tensor.fromShape(a_noise_buf.shape()),
            zml.Tensor.fromShape(a_mask_buf.shape()),
            zml.Tensor.fromShape(sigma_scalar_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer noise_init_a_exe.deinit();

    std.log.info("Running noise init (sigma_0={d:.6})...", .{sigma_0});
    var sigma0_buf = try zml.Buffer.scalar(io, platform, sigma_0, .f32, sharding);
    defer sigma0_buf.deinit();

    // Video noise init
    var ni_v_args = try noise_init_v_exe.args(allocator);
    defer ni_v_args.deinit(allocator);
    var ni_v_results = try noise_init_v_exe.results(allocator);
    defer ni_v_results.deinit(allocator);
    ni_v_args.set(.{ video_clean_buf, v_noise_buf, v_mask_buf, sigma0_buf });
    noise_init_v_exe.call(ni_v_args, &ni_v_results);
    var v_latent_buf = ni_v_results.get(zml.Buffer);

    // Audio noise init
    var ni_a_args = try noise_init_a_exe.args(allocator);
    defer ni_a_args.deinit(allocator);
    var ni_a_results = try noise_init_a_exe.results(allocator);
    defer ni_a_results.deinit(allocator);
    ni_a_args.set(.{ audio_clean_buf, a_noise_buf, a_mask_buf, sigma0_buf });
    noise_init_a_exe.call(ni_a_args, &ni_a_results);
    const a_latent_buf = ni_a_results.get(zml.Buffer);

    std.log.info("  video_latent (noised)", .{});
    std.log.info("  audio_latent (noised)", .{});

    // ---- Image conditioning for Stage 2 (optional) ----
    if (image_path) |img_path| {
        std.log.info("Applying image conditioning to Stage 2...", .{});

        // Load image from disk, resize to Stage 2 pixel resolution, normalize to bf16 [-1,1].
        // Stage 2 uses full resolution: pixel dims = h_lat * 32, w_lat * 32.
        const s2_pixel_h: u32 = @intCast(pipe_meta.stage2.h_lat * 32);
        const s2_pixel_w: u32 = @intCast(pipe_meta.stage2.w_lat * 32);
        var image_s2_buf = try image_loading.loadAndPreprocess(allocator, io, platform, sharding, img_path, s2_pixel_h, s2_pixel_w);
        defer image_s2_buf.deinit();
        std.log.info("  image_s2: {any}", .{image_s2_buf.shape().dims()});

        // Encode image → patchified tokens
        var img_tokens = try encodeImageToTokens(allocator, io, platform, sharding, main_ckpt_path, image_s2_buf);
        defer img_tokens.deinit();
        std.log.info("  image tokens: {any}", .{img_tokens.shape().dims()});

        // Apply conditioning: replace first-frame tokens in latent/clean/mask
        const cond = try applyConditioning(allocator, io, platform, sharding, v_latent_buf, video_clean_buf, v_mask_buf, img_tokens);

        // Replace old buffers with conditioned ones
        v_latent_buf.deinit();
        v_latent_buf = cond.latent;
        video_clean_buf.deinit();
        video_clean_buf = cond.clean;
        v_mask_buf.deinit();
        v_mask_buf = cond.mask;

        std.log.info("  Conditioning applied (n_img={d})", .{img_tokens.shape().dim(1)});
    }

    std.log.info("Bridge complete.", .{});

    return .{
        .v_latent = v_latent_buf,
        .a_latent = a_latent_buf,
        .v_positions = video_pos_buf,
        .a_positions = audio_pos_buf,
        .v_mask = v_mask_buf,
        .a_mask = a_mask_buf,
        .v_context = v_context,
        .a_context = a_context,
        .v_clean = video_clean_buf,
        .a_clean = audio_clean_buf,
    };
}

// ============================================================================
// Phase 3: Stage 2 — 3-step distilled denoising
// ============================================================================

fn runStage2(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    ckpt_path: []const u8,
    bridge: *BridgeResult,
    use_bf16_attn: bool,
    stage2_sigmas: []const f32,
) !Stage2Result {
    // ---- Open checkpoint store ----
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    // Use bridge buffers directly (already noised by bridge's noise init)
    var v_latent_buf = bridge.v_latent;
    var a_latent_buf = bridge.a_latent;

    const sigma_scalar_shape = zml.Shape.init(.{}, .f32);

    // ---- Compile preprocessing exe ----
    std.log.info("Compiling Stage 2 preprocessing exe...", .{});
    const preprocess_shape = model.initPreprocessParams(ckpt_store.view());
    var preprocess_exe = try platform.compileFn(
        allocator, io,
        model.forwardPreprocess,
        .{
            zml.Tensor.fromShape(v_latent_buf.shape()),
            zml.Tensor.fromShape(a_latent_buf.shape()),
            zml.Tensor.fromShape(bridge.v_mask.shape()),
            zml.Tensor.fromShape(bridge.a_mask.shape()),
            zml.Tensor.fromShape(zml.Shape.init(.{1}, .f32)),
            zml.Tensor.fromShape(zml.Shape.init(.{1}, .f32)),
            zml.Tensor.fromShape(bridge.v_positions.shape()),
            zml.Tensor.fromShape(bridge.a_positions.shape()),
            zml.Tensor.fromShape(bridge.v_context.shape()),
            zml.Tensor.fromShape(bridge.a_context.shape()),
            preprocess_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer preprocess_exe.deinit();
    std.log.info("Stage 2 preprocessing exe compiled.", .{});

    // Load preprocessing weights and run once for shape discovery
    std.log.info("Loading Stage 2 preprocessing weights...", .{});
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
    var sigma_1d_init = try sigma1dBuffer(io, platform, stage2_sigmas[0], sharding);
    defer sigma_1d_init.deinit();

    var pre_args_init = try preprocess_exe.args(allocator);
    defer pre_args_init.deinit(allocator);
    var pre_results_init = try preprocess_exe.results(allocator);
    defer pre_results_init.deinit(allocator);
    pre_args_init.set(.{
        v_latent_buf, a_latent_buf,
        bridge.v_mask, bridge.a_mask,
        sigma_1d_init, sigma_1d_init,
        bridge.v_positions, bridge.a_positions,
        bridge.v_context, bridge.a_context,
        preprocess_bufs,
    });
    preprocess_exe.call(pre_args_init, &pre_results_init);
    const init_pre_out = pre_results_init.get(zml.Bufferized(model.PreprocessOutput));

    // ---- Compile block exe ----
    std.log.info("Compiling Stage 2 block exe (bf16_attn={})...", .{use_bf16_attn});
    var block_params_shape = try allocator.create(model.FullStepParams);
    defer allocator.destroy(block_params_shape);
    block_params_shape.* = model.initFullStepParams(ckpt_store.view());

    const block_compile_args = .{
        zml.Tensor.fromShape(init_pre_out.vx.shape()),
        zml.Tensor.fromShape(init_pre_out.ax.shape()),
        zml.Tensor.fromShape(init_pre_out.video_timesteps.shape()),
        zml.Tensor.fromShape(init_pre_out.audio_timesteps.shape()),
        zml.Tensor.fromShape(init_pre_out.video_timesteps_zero.shape()),
        zml.Tensor.fromShape(init_pre_out.audio_timesteps_zero.shape()),
        zml.Tensor.fromShape(init_pre_out.v_denoise_mask.shape()),
        zml.Tensor.fromShape(init_pre_out.a_denoise_mask.shape()),
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

    // ---- Compile output projection exes ----
    std.log.info("Compiling Stage 2 output projection exes...", .{});
    const v_emb_shape = init_pre_out.v_embedded_timestep.shape();
    const a_emb_shape = init_pre_out.a_embedded_timestep.shape();

    var proj_v_exe = try platform.compileFn(
        allocator, io,
        model.forwardOutputProjection,
        .{
            zml.Tensor.fromShape(init_pre_out.vx.shape()).withPartialTags(.{ .b, .t, .d }),
            zml.Tensor.fromShape(v_emb_shape).withPartialTags(.{ .b, .t, .d_emb }),
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
            zml.Tensor.fromShape(a_emb_shape).withPartialTags(.{ .b, .t, .d_emb }),
            block_params_shape.audio_norm_proj_out,
        },
        .{ .shardings = &.{sharding} },
    );
    defer proj_a_exe.deinit();

    // ---- Compile denoising step exes ----
    // Stage 2 uses forwardDenoisingStep (velocity-based, not from x0)
    std.log.info("Compiling Stage 2 denoising step exes...", .{});

    var denoise_v_exe = try platform.compileFn(
        allocator, io,
        model.forwardDenoisingStep,
        .{
            zml.Tensor.fromShape(v_latent_buf.shape()),
            zml.Tensor.fromShape(v_latent_buf.shape()),
            zml.Tensor.fromShape(bridge.v_mask.shape()),
            zml.Tensor.fromShape(bridge.v_clean.shape()),
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
            zml.Tensor.fromShape(bridge.a_mask.shape()),
            zml.Tensor.fromShape(bridge.a_clean.shape()),
            zml.Tensor.fromShape(sigma_scalar_shape),
            zml.Tensor.fromShape(sigma_scalar_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer denoise_a_exe.deinit();
    std.log.info("All Stage 2 exes compiled.", .{});

    // ---- Load weights ----
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
    std.log.info("All Stage 2 weights loaded.", .{});

    // ---- Denoising loop: N-step Euler ----
    const num_steps = stage2_sigmas.len - 1;
    std.log.info("Starting {d}-step denoising loop...", .{num_steps});

    for (0..num_steps) |step_idx| {
        const sigma = stage2_sigmas[step_idx];
        const sigma_next = stage2_sigmas[step_idx + 1];

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
            bridge.v_mask, bridge.a_mask,
            sigma_1d, sigma_1d,
            bridge.v_positions, bridge.a_positions,
            bridge.v_context, bridge.a_context,
            preprocess_bufs,
        });
        preprocess_exe.call(pre_args, &pre_results);
        const pre_out = pre_results.get(zml.Bufferized(model.PreprocessOutput));

        // ---- 2. 48-block chain ----
        std.log.info("  Running 48-block chain...", .{});
        var h_v = pre_out.vx;
        var h_a = pre_out.ax;

        for (0..48) |i| {
            var blk_args = try block_exe.args(allocator);
            defer blk_args.deinit(allocator);
            var blk_results = try block_exe.results(allocator);
            defer blk_results.deinit(allocator);

            blk_args.set(.{
                h_v, h_a,
                pre_out.video_timesteps, pre_out.audio_timesteps,
                pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                pre_out.v_denoise_mask, pre_out.a_denoise_mask,
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
            block_exe.call(blk_args, &blk_results);

            const out = blk_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
            if (i > 0) { h_v.deinit(); h_a.deinit(); }
            h_v = out.vx_out;
            h_a = out.ax_out;
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
        dv_args.set(.{ v_latent_buf, video_vel, bridge.v_mask, bridge.v_clean, sigma_buf, sigma_next_buf });
        denoise_v_exe.call(dv_args, &dv_results);
        const dv_out = dv_results.get(zml.Bufferized(model.DenoisingStepResult));

        var da_args = try denoise_a_exe.args(allocator);
        defer da_args.deinit(allocator);
        var da_results = try denoise_a_exe.results(allocator);
        defer da_results.deinit(allocator);
        da_args.set(.{ a_latent_buf, audio_vel, bridge.a_mask, bridge.a_clean, sigma_buf, sigma_next_buf });
        denoise_a_exe.call(da_args, &da_results);
        const da_out = da_results.get(zml.Bufferized(model.DenoisingStepResult));

        video_vel.deinit();
        audio_vel.deinit();

        v_latent_buf.deinit();
        a_latent_buf.deinit();
        v_latent_buf = dv_out.next_latent;
        a_latent_buf = da_out.next_latent;

        std.log.info("  Step {d} complete.", .{step_idx});
    }

    std.log.info("Stage 2 denoising complete.", .{});

    // Mark bridge latent fields as consumed (they were freed in the loop)
    bridge.v_latent = undefined;
    bridge.a_latent = undefined;

    return .{
        .v_latent = v_latent_buf,
        .a_latent = a_latent_buf,
    };
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

// ============================================================================
// Phase 4: Video VAE Decode
// ============================================================================

fn runVideoVaeDecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    ckpt_path: []const u8,
    v_latent_patchified: zml.Buffer,
    pipe_meta: PipelineMeta,
) !VideoFrames {
    const s2 = pipe_meta.stage2;
    const F = s2.f_lat;
    const H = s2.h_lat;
    const W = s2.w_lat;
    const C: i64 = 128;
    const T_v = F * H * W;

    // ---- Open checkpoint store (reuse main checkpoint for VAE + per_channel_stats) ----
    std.log.info("Opening checkpoint for VAE decoder...", .{});
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    // ========================================================================
    // Step 1: Unpatchify latent — [1, T_v, 128] → [1, 128, F, H, W]
    // ========================================================================
    std.log.info("Compiling unpatchify for VAE input...", .{});
    const patchified_shape = zml.Shape.init(.{ 1, T_v, C }, .bf16);
    const video_5d_shape = zml.Shape.init(.{ 1, C, F, H, W }, .bf16);

    var unpatch_exe = try platform.compileFn(
        allocator, io,
        upsampler.forwardUnpatchifyVideo,
        .{
            zml.Tensor.fromShape(patchified_shape),
            video_5d_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer unpatch_exe.deinit();

    std.log.info("Running unpatchify...", .{});
    var unpatch_args = try unpatch_exe.args(allocator);
    defer unpatch_args.deinit(allocator);
    var unpatch_results = try unpatch_exe.results(allocator);
    defer unpatch_results.deinit(allocator);
    unpatch_args.set(.{v_latent_patchified});
    unpatch_exe.call(unpatch_args, &unpatch_results);
    const v_latent_5d = unpatch_results.get(zml.Buffer);
    std.log.info("  Unpatchified: {any}", .{v_latent_5d.shape().dims()});

    // ========================================================================
    // Step 2: Load VAE decoder weights + per-channel stats
    // ========================================================================
    std.log.info("Loading VAE decoder weights...", .{});
    var vae_params = video_vae.initVideoVaeDecoderParams(ckpt_store.view());
    const vae_bufs = try zml.io.load(
        video_vae.VideoVaeDecoderParams,
        &vae_params,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );

    std.log.info("Loading per-channel statistics...", .{});
    var stats_shape = conv_ops.initPerChannelStats(ckpt_store.view());
    const stats_bufs = try zml.io.load(
        conv_ops.PerChannelStats,
        &stats_shape,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );

    // ========================================================================
    // Step 3: Compile and run the VAE decoder
    // ========================================================================
    std.log.info("Compiling VAE decoder...", .{});
    var vae_exe = try platform.compileFn(
        allocator, io,
        video_vae.forwardVideoVaeDecode,
        .{
            zml.Tensor.fromShape(v_latent_5d.shape()),
            stats_shape,
            vae_params,
        },
        .{ .shardings = &.{sharding} },
    );
    defer vae_exe.deinit();

    std.log.info("Running VAE decode...", .{});
    var vae_args = try vae_exe.args(allocator);
    defer vae_args.deinit(allocator);
    var vae_results = try vae_exe.results(allocator);
    defer vae_results.deinit(allocator);
    vae_args.set(.{ v_latent_5d, stats_bufs, vae_bufs });
    vae_exe.call(vae_args, &vae_results);
    const decoded_video = vae_results.get(zml.Buffer); // [1, 3, F_out, H_out, W_out] bf16
    std.log.info("  Decoded video: {any}", .{decoded_video.shape().dims()});

    // ========================================================================
    // Step 4: Post-process to uint8 and write frames.bin
    // ========================================================================
    std.log.info("Post-processing decoded video to frames...", .{});

    // Download bf16 tensor to host
    const decoded_slice = try decoded_video.toSliceAlloc(allocator, io);
    defer decoded_slice.free(allocator);

    // Decode shape: [1, 3, F_out, H_out, W_out]
    const F_out: usize = @intCast(decoded_video.shape().dim(2));
    const H_out: usize = @intCast(decoded_video.shape().dim(3));
    const W_out: usize = @intCast(decoded_video.shape().dim(4));

    const num_pixels = F_out * H_out * W_out * 3;
    const frames_u8 = try allocator.alloc(u8, num_pixels);

    // Convert bf16 [1, 3, F, H, W] → u8 [F, H, W, 3] (NHWC)
    // (x + 1) / 2 * 255, clamped to [0, 255]
    const bf16_data = decoded_slice.constData();
    for (0..F_out) |f| {
        for (0..H_out) |h| {
            for (0..W_out) |w| {
                for (0..3) |c| {
                    // bf16 layout: [1, 3, F, H, W] → offset = c*F*H*W + f*H*W + h*W + w
                    const src_idx = (c * F_out * H_out * W_out + f * H_out * W_out + h * W_out + w) * 2;
                    // Read bf16 as u16, convert to f32
                    const bf16_bits = std.mem.readInt(u16, bf16_data[src_idx..][0..2], .little);
                    const f32_bits: u32 = @as(u32, bf16_bits) << 16;
                    const val: f32 = @bitCast(f32_bits);
                    // (val + 1) / 2 * 255, clamped
                    const normalized = @min(@max((val + 1.0) * 0.5, 0.0), 1.0);
                    const pixel: u8 = @intFromFloat(normalized * 255.0);
                    // NHWC layout: f*H*W*3 + h*W*3 + w*3 + c
                    frames_u8[f * H_out * W_out * 3 + h * W_out * 3 + w * 3 + c] = pixel;
                }
            }
        }
    }

    std.log.info("  Video frames: {d}x{d}, {d} frames", .{ W_out, H_out, F_out });

    return .{
        .data = frames_u8,
        .width = W_out,
        .height = H_out,
        .num_frames = F_out,
        .allocator = allocator,
    };
}

fn encodeOutputMp4(
    allocator: std.mem.Allocator,
    io: std.Io,
    video: VideoFrames,
    audio_interleaved: []const u8,
    audio_channels: usize,
    fps: f64,
    output_dir: []const u8,
) !void {
    const output_path = try std.fs.path.join(allocator, &.{ output_dir, "output.mp4" });
    defer allocator.free(output_path);

    // Write interleaved audio to a temp file (ffmpeg reads video from stdin,
    // audio from file — can't pipe both to stdin).
    const audio_path = try std.fs.path.join(allocator, &.{ output_dir, "_audio_tmp.raw" });
    defer allocator.free(audio_path);
    {
        const audio_file = try std.Io.Dir.createFile(.cwd(), io, audio_path, .{});
        defer audio_file.close(io);
        var wr_buf: [64 * 1024]u8 = undefined;
        var wr = audio_file.writer(io, &wr_buf);
        try wr.interface.writeAll(audio_interleaved);
        try wr.interface.flush();
    }

    var size_buf: [32]u8 = undefined;
    const size_str = std.fmt.bufPrint(&size_buf, "{d}x{d}", .{ video.width, video.height }) catch unreachable;

    var fps_buf: [16]u8 = undefined;
    const fps_str = std.fmt.bufPrint(&fps_buf, "{d:.0}", .{fps}) catch unreachable;

    var ac_buf: [8]u8 = undefined;
    const ac_str = std.fmt.bufPrint(&ac_buf, "{d}", .{audio_channels}) catch unreachable;

    std.log.info("Encoding video+audio with ffmpeg → {s}", .{output_path});

    var child = try std.process.spawn(io, .{
        .argv = &.{
            "ffmpeg",   "-y",
            // Video input: raw RGB24 from stdin
            "-f",       "rawvideo",
            "-pix_fmt", "rgb24",
            "-s",       size_str,
            "-r",       fps_str,
            "-i",       "pipe:0",
            // Audio input: interleaved f32le from file
            "-f",       "f32le",
            "-ar",      "48000",
            "-ac",      ac_str,
            "-i",       audio_path,
            // Output encoding
            "-c:v",     "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a",     "aac",
            "-b:a",     "192k",
            "-shortest",
            output_path,
        },
        .stdin = .pipe,
        .stdout = .inherit,
        .stderr = .inherit,
    });

    // Write all frame data to ffmpeg's stdin
    const stdin_file = child.stdin.?;
    var write_buf: [64 * 1024]u8 = undefined;
    var writer = stdin_file.writer(io, &write_buf);
    try writer.interface.writeAll(video.data);
    try writer.interface.flush();
    stdin_file.close(io);
    child.stdin = null;

    // Wait for ffmpeg to finish
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| {
            if (code != 0) {
                std.log.err("ffmpeg exited with code {d}", .{code});
                return error.FfmpegFailed;
            }
        },
        else => {
            std.log.err("ffmpeg terminated abnormally", .{});
            return error.FfmpegFailed;
        },
    }

    std.log.info("  Wrote {s}", .{output_path});

    // Clean up temp audio file
    try std.Io.Dir.deleteFile(.cwd(), io, audio_path);
}

// Phase 5: Audio VAE Decode
// ============================================================================

fn runAudioVaeDecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    ckpt_path: []const u8,
    a_latent_patchified: zml.Buffer,
    pipe_meta: PipelineMeta,
) !zml.Buffer {
    const T_aud = pipe_meta.stage2.t_audio;

    // ---- Open checkpoint store ----
    std.log.info("Opening checkpoint for audio VAE decoder...", .{});
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    // ========================================================================
    // Step 1: Unpatchify audio latent — [1, T_aud, 128] → [1, 8, T_aud, 16]
    // ========================================================================
    std.log.info("Compiling unpatchify for audio input...", .{});
    const patchified_shape = zml.Shape.init(.{ 1, T_aud, 128 }, .bf16);

    var unpatch_exe = try platform.compileFn(
        allocator, io,
        audio_vae.forwardUnpatchifyAudio,
        .{
            zml.Tensor.fromShape(patchified_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer unpatch_exe.deinit();

    std.log.info("Running audio unpatchify...", .{});
    var unpatch_args = try unpatch_exe.args(allocator);
    defer unpatch_args.deinit(allocator);
    var unpatch_results = try unpatch_exe.results(allocator);
    defer unpatch_results.deinit(allocator);
    unpatch_args.set(.{a_latent_patchified});
    unpatch_exe.call(unpatch_args, &unpatch_results);
    const a_latent_4d = unpatch_results.get(zml.Buffer);
    std.log.info("  Unpatchified audio: {any}", .{a_latent_4d.shape().dims()});

    // ========================================================================
    // Step 2: Load audio VAE decoder weights + per-channel stats
    // ========================================================================
    std.log.info("Loading audio VAE decoder weights...", .{});
    var audio_vae_params = audio_vae.initAudioVaeDecoderParams(ckpt_store.view());
    const audio_vae_bufs = try zml.io.load(
        audio_vae.AudioVaeDecoderParams,
        &audio_vae_params,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );

    std.log.info("Loading audio per-channel statistics...", .{});
    var audio_stats_shape = audio_vae.initAudioPerChannelStats(ckpt_store.view());
    const audio_stats_bufs = try zml.io.load(
        audio_vae.AudioPerChannelStats,
        &audio_stats_shape,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );

    // ========================================================================
    // Step 3: Compile and run the audio VAE decoder
    // ========================================================================
    std.log.info("Compiling audio VAE decoder...", .{});
    var audio_vae_exe = try platform.compileFn(
        allocator, io,
        audio_vae.forwardAudioVaeDecode,
        .{
            zml.Tensor.fromShape(a_latent_4d.shape()),
            audio_stats_shape,
            audio_vae_params,
        },
        .{ .shardings = &.{sharding} },
    );
    defer audio_vae_exe.deinit();

    std.log.info("Running audio VAE decode...", .{});
    var audio_vae_args = try audio_vae_exe.args(allocator);
    defer audio_vae_args.deinit(allocator);
    var audio_vae_results = try audio_vae_exe.results(allocator);
    defer audio_vae_results.deinit(allocator);
    audio_vae_args.set(.{ a_latent_4d, audio_stats_bufs, audio_vae_bufs });
    audio_vae_exe.call(audio_vae_args, &audio_vae_results);
    const decoded_audio = audio_vae_results.get(zml.Buffer); // [1, 2, T_out, 64] bf16
    std.log.info("  Decoded audio mel: {any}", .{decoded_audio.shape().dims()});

    return decoded_audio;
}

// ============================================================================
// Phase 6: Vocoder + BWE (mel spectrogram → 48kHz stereo waveform)
// ============================================================================

fn runVocoderWithBWE(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    ckpt_path: []const u8,
    audio_mel: zml.Buffer,
) !zml.Buffer {
    // ---- Open checkpoint store ----
    std.log.info("Opening checkpoint for vocoder...", .{});
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    // ---- Load vocoder weights (split: main 667, BWE pipeline 559) ----
    std.log.info("Loading main vocoder weights...", .{});
    var main_voc_params: vocoder.MainVocoderParams = undefined;
    vocoder.initMainVocoderParams(&main_voc_params, ckpt_store.view().withPrefix("vocoder").withPrefix("vocoder"));
    const main_voc_bufs = try zml.io.load(
        vocoder.MainVocoderParams,
        &main_voc_params,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );

    std.log.info("Loading BWE pipeline weights...", .{});
    var bwe_params: vocoder.BWEPipelineParams = undefined;
    vocoder.initBWEPipelineParams(&bwe_params, ckpt_store.view());
    const bwe_bufs = try zml.io.load(
        vocoder.BWEPipelineParams,
        &bwe_params,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );

    // ---- Stage 1: Main vocoder — mel → 16kHz waveform ----
    std.log.info("Compiling main vocoder (input: {any})...", .{audio_mel.shape().dims()});
    var main_voc_exe = try platform.compileFn(
        allocator, io,
        vocoder.forwardMainVocoder,
        .{ zml.Tensor.fromShape(audio_mel.shape()), &main_voc_params },
        .{ .shardings = &.{sharding} },
    );
    defer main_voc_exe.deinit();

    std.log.info("Running main vocoder...", .{});
    var main_voc_args = try main_voc_exe.args(allocator);
    defer main_voc_args.deinit(allocator);
    var main_voc_results = try main_voc_exe.results(allocator);
    defer main_voc_results.deinit(allocator);
    main_voc_args.set(.{ audio_mel, &main_voc_bufs });
    main_voc_exe.call(main_voc_args, &main_voc_results);
    const waveform_16k = main_voc_results.get(zml.Buffer);
    std.log.info("  16kHz waveform: {any}", .{waveform_16k.shape().dims()});

    // ---- Stage 2: BWE pipeline — 16kHz → 48kHz ----
    std.log.info("Compiling BWE pipeline (input: {any})...", .{waveform_16k.shape().dims()});
    var bwe_exe = try platform.compileFn(
        allocator, io,
        vocoder.forwardBWEPipeline,
        .{ zml.Tensor.fromShape(waveform_16k.shape()), &bwe_params },
        .{ .shardings = &.{sharding} },
    );
    defer bwe_exe.deinit();

    std.log.info("Running BWE pipeline...", .{});
    var bwe_args = try bwe_exe.args(allocator);
    defer bwe_args.deinit(allocator);
    var bwe_results = try bwe_exe.results(allocator);
    defer bwe_results.deinit(allocator);
    bwe_args.set(.{ waveform_16k, &bwe_bufs });
    bwe_exe.call(bwe_args, &bwe_results);
    const waveform = bwe_results.get(zml.Buffer);
    std.log.info("  48kHz waveform: {any}", .{waveform.shape().dims()});

    return waveform;
}

fn writeRawBytes(
    allocator: std.mem.Allocator,
    io: std.Io,
    data: []const u8,
    dir: []const u8,
    filename: []const u8,
) !void {
    const path = try std.fs.path.join(allocator, &.{ dir, filename });
    defer allocator.free(path);

    const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer file.close(io);

    var write_buf: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &write_buf);
    try writer.interface.writeAll(data);
    try writer.interface.flush();

    std.log.info("  Wrote {s} ({d} bytes)", .{ path, data.len });
}

fn writeBuffer (
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

    std.log.info("  Wrote {s} ({d} bytes, dims {any})", .{ path, slice.constData().len, buf.shape().dims() });
}

// ============================================================================
// Load pipeline_meta.json
// ============================================================================

fn loadPipelineMeta(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !PipelineMeta {
    const file = try std.Io.Dir.openFile(.cwd(), io, path, .{});
    defer file.close(io);

    var buf: [256]u8 = undefined;
    var file_reader = file.reader(io, &buf);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    const JsonStageMeta = struct {
        h_lat: i64,
        w_lat: i64,
        f_lat: i64,
        t_audio: i64,
    };

    const JsonMeta = struct {
        frame_rate: f64,
        stage1: JsonStageMeta,
        stage2: JsonStageMeta,
    };

    const parsed = try std.json.parseFromTokenSource(JsonMeta, allocator, &reader, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const v = parsed.value;

    return .{
        .frame_rate = v.frame_rate,
        .stage1 = .{
            .h_lat = v.stage1.h_lat,
            .w_lat = v.stage1.w_lat,
            .f_lat = v.stage1.f_lat,
            .t_audio = v.stage1.t_audio,
        },
        .stage2 = .{
            .h_lat = v.stage2.h_lat,
            .w_lat = v.stage2.w_lat,
            .f_lat = v.stage2.f_lat,
            .t_audio = v.stage2.t_audio,
        },
    };
}

// ============================================================================
// Video positions: [1, 3, T_v, 2] bf16
// ============================================================================

fn computeVideoPositions(allocator: std.mem.Allocator, F: i64, H: i64, W: i64, fps: f64) ![]u8 {
    const T_v: usize = @intCast(F * H * W);
    const num_vals = 3 * T_v * 2;
    const out = try allocator.alloc(u8, num_vals * 2);

    const scale_factors = [3]f32{ 8.0, 32.0, 32.0 };
    const fps_f32: f32 = @floatCast(fps);
    const Fi: usize = @intCast(F);
    const Hi: usize = @intCast(H);
    const Wi: usize = @intCast(W);

    for (0..Fi) |f| {
        for (0..Hi) |h| {
            for (0..Wi) |w| {
                const patch_idx = f * Hi * Wi + h * Wi + w;

                const coords = [3][2]f32{
                    .{ @floatFromInt(f), @floatFromInt(f + 1) },
                    .{ @floatFromInt(h), @floatFromInt(h + 1) },
                    .{ @floatFromInt(w), @floatFromInt(w + 1) },
                };

                for (0..3) |axis| {
                    var start = coords[axis][0] * scale_factors[axis];
                    var end = coords[axis][1] * scale_factors[axis];

                    if (axis == 0) {
                        start = @max(start + 1.0 - scale_factors[0], 0.0);
                        end = @max(end + 1.0 - scale_factors[0], 0.0);
                        start /= fps_f32;
                        end /= fps_f32;
                    }

                    const base = (axis * T_v * 2 + patch_idx * 2) * 2;
                    storeBf16(out[base..], start);
                    storeBf16(out[base + 2 ..], end);
                }
            }
        }
    }

    return out;
}

// ============================================================================
// Audio positions: [1, 1, T_a, 2] f32
// ============================================================================

fn computeAudioPositions(allocator: std.mem.Allocator, T_a: i64) ![]u8 {
    const T: usize = @intCast(T_a);
    const num_vals = T * 2;
    const out = try allocator.alloc(u8, num_vals * 4);

    for (0..T) |i| {
        const fi: f32 = @floatFromInt(i);
        const fi_end: f32 = @floatFromInt(i + 1);
        const mel_start: f32 = @max(fi * 4.0 + 1.0 - 4.0, 0.0);
        const mel_end: f32 = @max(fi_end * 4.0 + 1.0 - 4.0, 0.0);
        const start_sec: f32 = mel_start * 160.0 / 16000.0;
        const end_sec: f32 = mel_end * 160.0 / 16000.0;

        const base = i * 2 * 4;
        storeF32(out[base..], start_sec);
        storeF32(out[base + 4 ..], end_sec);
    }

    return out;
}

fn fillOnesF32(buf: []u8) void {
    const one_bits: u32 = @bitCast(@as(f32, 1.0));
    var i: usize = 0;
    while (i + 4 <= buf.len) : (i += 4) {
        std.mem.writeInt(u32, buf[i..][0..4], one_bits, .little);
    }
}

fn storeBf16(buf: []u8, val: f32) void {
    const bits: u32 = @bitCast(val);
    const lower: u32 = bits & 0xFFFF;
    const round_bit: u32 = if (lower > 0x8000 or (lower == 0x8000 and (bits & 0x10000) != 0)) 1 else 0;
    const bf16_bits: u16 = @truncate((bits >> 16) +% round_bit);
    std.mem.writeInt(u16, buf[0..2], bf16_bits, .little);
}

fn storeF32(buf: []u8, val: f32) void {
    std.mem.writeInt(u32, buf[0..4], @bitCast(val), .little);
}
