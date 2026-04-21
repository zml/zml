/// Unified end-to-end pipeline: Stage 1 → Bridge → Stage 2
///
/// Runs the full LTX-2.3 denoising pipeline in a single binary:
///   1. Stage 1: N-step guided denoising (4-pass per step × 48 blocks, default 30)
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
///       --gemma-hidden-states-pos /root/e2e_demo/pos_hidden_states.safetensors \
///       --gemma-hidden-states-neg /root/e2e_demo/neg_hidden_states.safetensors \
///       --seed 42 \
///       --output-dir /root/e2e_demo/unified_out/ \
///       --height 1024 --width 1536 --num-frames 121 --fps 24 \
///       --bf16-attn-stage2
const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const conv_ops = @import("conv_ops.zig");
const image_loading = @import("image_loading.zig");
const text_embeddings = @import("text_embeddings.zig");
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
// Phase timing
// ============================================================================

const PhaseTimer = struct {
    name: []const u8,
    io: std.Io,
    compile_ns: i96 = 0,
    load_ns: i96 = 0,
    exec_ns: i96 = 0,
    other_ns: i96 = 0,
    step_times_ns: [MAX_STEPS]i96 = [_]i96{0} ** MAX_STEPS,
    num_steps: usize = 0,
    lap_ts: std.Io.Timestamp = .zero,

    const MAX_STEPS = 64;

    fn init(name: []const u8, io: std.Io) PhaseTimer {
        return .{ .name = name, .io = io };
    }

    fn start(self: *PhaseTimer) void {
        self.lap_ts = std.Io.Timestamp.now(self.io, .awake);
    }

    fn readLap(self: *PhaseTimer) i96 {
        const now = std.Io.Timestamp.now(self.io, .awake);
        const dt = now.nanoseconds - self.lap_ts.nanoseconds;
        self.lap_ts = now;
        return dt;
    }

    fn addCompile(self: *PhaseTimer) void {
        self.compile_ns += self.readLap();
    }

    fn addLoad(self: *PhaseTimer) void {
        self.load_ns += self.readLap();
    }

    fn addExec(self: *PhaseTimer) void {
        self.exec_ns += self.readLap();
    }

    fn addOther(self: *PhaseTimer) void {
        self.other_ns += self.readLap();
    }

    fn recordStep(self: *PhaseTimer, dt_ns: i96) void {
        if (self.num_steps < MAX_STEPS) {
            self.step_times_ns[self.num_steps] = dt_ns;
            self.num_steps += 1;
        }
    }

    fn totalNs(self: *const PhaseTimer) i96 {
        return self.compile_ns + self.load_ns + self.exec_ns + self.other_ns;
    }

    fn fmtSecs(ns: i96) f64 {
        return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
    }

    fn printRow(self: *const PhaseTimer) void {
        std.log.info("  {s:<20} | {d:>8.1} | {d:>8.1} | {d:>8.1} | {d:>8.1} | {d:>8.1}", .{
            self.name,
            fmtSecs(self.compile_ns),
            fmtSecs(self.load_ns),
            fmtSecs(self.exec_ns),
            fmtSecs(self.other_ns),
            fmtSecs(self.totalNs()),
        });
        if (self.num_steps > 0) {
            var total_step: i96 = 0;
            for (self.step_times_ns[0..self.num_steps]) |s| total_step += s;
            const avg = fmtSecs(@divTrunc(total_step, @as(i96, @intCast(self.num_steps))));
            std.log.info("    (per step avg)     |          |          | {d:>8.1} |          |", .{avg});
        }
    }
};

fn printTimingSummary(timers: []const *const PhaseTimer) void {
    var total_compile: i96 = 0;
    var total_load: i96 = 0;
    var total_exec: i96 = 0;
    var total_other: i96 = 0;
    for (timers) |t| {
        total_compile += t.compile_ns;
        total_load += t.load_ns;
        total_exec += t.exec_ns;
        total_other += t.other_ns;
    }
    const total = total_compile + total_load + total_exec + total_other;

    std.log.info("", .{});
    std.log.info("====================== TIMING SUMMARY ======================", .{});
    std.log.info("  {s:<20} | {s:>8} | {s:>8} | {s:>8} | {s:>8} | {s:>8}", .{
        "Phase", "Compile", "Load", "Execute", "Other", "Total",
    });
    std.log.info("  {s:-<20}-+-{s:->8}-+-{s:->8}-+-{s:->8}-+-{s:->8}-+-{s:->8}", .{
        "", "", "", "", "", "",
    });
    for (timers) |t| t.printRow();
    std.log.info("  {s:-<20}-+-{s:->8}-+-{s:->8}-+-{s:->8}-+-{s:->8}-+-{s:->8}", .{
        "", "", "", "", "", "",
    });
    std.log.info("  {s:<20} | {d:>8.1} | {d:>8.1} | {d:>8.1} | {d:>8.1} | {d:>8.1}", .{
        "TOTAL",
        PhaseTimer.fmtSecs(total_compile),
        PhaseTimer.fmtSecs(total_load),
        PhaseTimer.fmtSecs(total_exec),
        PhaseTimer.fmtSecs(total_other),
        PhaseTimer.fmtSecs(total),
    });
    std.log.info("=============================================================", .{});
}

// ============================================================================
// Bufferized struct helpers
// ============================================================================

/// Recursively free all GPU buffers inside a Bufferized(T) struct.
/// Handles nested structs, arrays, and optionals.
fn deinitBufferizedFields(val: anytype) void {
    const T = @TypeOf(val.*);
    const fields = @typeInfo(T).@"struct".fields;
    inline for (fields) |field| {
        if (field.type == zml.Buffer) {
            @field(val, field.name).deinit();
        } else if (@typeInfo(field.type) == .@"struct") {
            deinitBufferizedFields(&@field(val, field.name));
        } else if (@typeInfo(field.type) == .array) {
            const Child = @typeInfo(field.type).array.child;
            if (Child == zml.Buffer) {
                for (&@field(val, field.name)) |*buf| buf.deinit();
            } else if (@typeInfo(Child) == .@"struct") {
                for (&@field(val, field.name)) |*elem| deinitBufferizedFields(elem);
            }
        } else if (@typeInfo(field.type) == .optional) {
            const Child = @typeInfo(field.type).optional.child;
            if (@field(val, field.name) != null) {
                if (Child == zml.Buffer) {
                    @field(val, field.name).?.deinit();
                } else if (@typeInfo(Child) == .@"struct") {
                    deinitBufferizedFields(&(@field(val, field.name).?));
                }
            }
        }
    }
}

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
    meta: ?[]const u8,
    output_dir: []const u8,
    seed: u64,
    bf16_attn_stage1: bool,
    bf16_attn_stage2: bool,
    num_inference_steps: usize,
    dump_intermediates: bool,
    image: ?[]const u8,
    gemma_hidden_states_pos: []const u8,
    gemma_hidden_states_neg: []const u8,
    // Generation params (used when --meta is not provided)
    height: u32,
    width: u32,
    num_frames: u32,
    fps: f64,
};

fn parseArgs(it: anytype) !CliArgs {
    var args: CliArgs = .{
        .stage1_ckpt = undefined,
        .stage2_ckpt = undefined,
        .upsampler_ckpt = undefined,
        .meta = null,
        .output_dir = undefined,
        .seed = 42,
        .bf16_attn_stage1 = false,
        .bf16_attn_stage2 = false,
        .num_inference_steps = model.stage1_default_schedule.num_steps,
        .dump_intermediates = false,
        .image = null,
        .gemma_hidden_states_pos = undefined,
        .gemma_hidden_states_neg = undefined,
        .height = 1024,
        .width = 1536,
        .num_frames = 121,
        .fps = 24.0,
    };
    var have = [_]bool{false} ** 6;

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
        } else if (std.mem.eql(u8, arg, "--meta")) {
            args.meta = it.next() orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--output-dir")) {
            args.output_dir = it.next() orelse return error.InvalidArgs;
            have[3] = true;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            const seed_str = it.next() orelse return error.InvalidArgs;
            args.seed = std.fmt.parseInt(u64, seed_str, 10) catch return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--bf16-attn-stage1")) {
            args.bf16_attn_stage1 = true;
        } else if (std.mem.eql(u8, arg, "--bf16-attn-stage2")) {
            args.bf16_attn_stage2 = true;
        } else if (std.mem.eql(u8, arg, "--num-inference-steps")) {
            const val = it.next() orelse return error.InvalidArgs;
            args.num_inference_steps = std.fmt.parseInt(usize, val, 10) catch return error.InvalidArgs;
            if (args.num_inference_steps == 0 or args.num_inference_steps > MAX_STAGE1_STEPS) {
                std.log.err("--num-inference-steps must be between 1 and {d}", .{MAX_STAGE1_STEPS});
                return error.InvalidArgs;
            }
        } else if (std.mem.eql(u8, arg, "--dump-intermediates")) {
            args.dump_intermediates = true;
        } else if (std.mem.eql(u8, arg, "--image")) {
            args.image = it.next() orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--gemma-hidden-states-pos")) {
            args.gemma_hidden_states_pos = it.next() orelse return error.InvalidArgs;
            have[4] = true;
        } else if (std.mem.eql(u8, arg, "--gemma-hidden-states-neg")) {
            args.gemma_hidden_states_neg = it.next() orelse return error.InvalidArgs;
            have[5] = true;
        } else if (std.mem.eql(u8, arg, "--height")) {
            const val = it.next() orelse return error.InvalidArgs;
            args.height = std.fmt.parseInt(u32, val, 10) catch return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--width")) {
            const val = it.next() orelse return error.InvalidArgs;
            args.width = std.fmt.parseInt(u32, val, 10) catch return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--num-frames")) {
            const val = it.next() orelse return error.InvalidArgs;
            args.num_frames = std.fmt.parseInt(u32, val, 10) catch return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--fps")) {
            const val = it.next() orelse return error.InvalidArgs;
            args.fps = std.fmt.parseFloat(f64, val) catch return error.InvalidArgs;
        }
    }

    for (have) |h| {
        if (!h) {
            std.log.err(
                "Usage: inference --stage1-ckpt <path> --stage2-ckpt <path> " ++
                    "--upsampler-ckpt <path> --output-dir <path> " ++
                    "--gemma-hidden-states-pos <path> --gemma-hidden-states-neg <path> " ++
                    "[--height <int>] [--width <int>] [--num-frames <int>] [--fps <float>] " ++
                    "[--num-inference-steps <int>] [--meta <path>] [--seed <int>] [--image <path>] " ++
                    "[--bf16-attn-stage1] [--bf16-attn-stage2] [--dump-intermediates]",
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
    rng_state: zml.Buffer, // [2] u64 — RNG state after Stage 1 noise draws

    fn deinit(self: *Stage1Result) void {
        self.v_latent.deinit();
        self.a_latent.deinit();
        self.v_context_pos.deinit();
        self.a_context_pos.deinit();
        self.rng_state.deinit();
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

/// Comptime upper bound for Stage 1 step count (sizes the sigma schedule array).
const MAX_STAGE1_STEPS: usize = 50;

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
        allocator,
        io,
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
    var encoder_bufs = try zml.io.load(
        video_vae_encoder.VideoVaeEncoderParams,
        &encoder_shape,
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer deinitBufferizedFields(&encoder_bufs);
    var stats_bufs = try zml.io.load(
        conv_ops.PerChannelStats,
        &stats_shape,
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer deinitBufferizedFields(&stats_bufs);

    std.log.info("  Running encoder...", .{});
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ image_buf, stats_bufs, encoder_bufs });
    exe.callOpts(io, args, &results, .{ .wait = true });
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
        allocator,
        io,
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
    exe.callOpts(io, args, &results, .{ .wait = true });
    const out = results.get(zml.Bufferized(ConditioningResult));
    return .{ .latent = out.latent, .clean = out.clean_latent, .mask = out.denoise_mask };
}

// ============================================================================
// Main
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("LTX-2.3 Pipeline (Stage 1 → Bridge → Stage 2)", .{});

    var it = init.minimal.args.iterate();
    const args = try parseArgs(&it);

    std.log.info("  stage1-ckpt:     {s}", .{args.stage1_ckpt});
    std.log.info("  stage2-ckpt:     {s}", .{args.stage2_ckpt});
    std.log.info("  upsampler-ckpt:  {s}", .{args.upsampler_ckpt});
    std.log.info("  seed:            {d}", .{args.seed});
    std.log.info("  meta:            {s}", .{args.meta orelse "(computed from --height/--width/--num-frames/--fps)"});
    std.log.info("  output-dir:      {s}", .{args.output_dir});
    std.log.info("  height:            {d}", .{args.height});
    std.log.info("  width:             {d}", .{args.width});
    std.log.info("  num-frames:        {d}", .{args.num_frames});
    std.log.info("  fps:               {d:.1}", .{args.fps});
    std.log.info("  bf16-attn-stage1:  {}", .{args.bf16_attn_stage1});
    std.log.info("  bf16-attn-stage2:  {}", .{args.bf16_attn_stage2});
    std.log.info("  num-inference-steps: {d}", .{args.num_inference_steps});
    std.log.info("  dump-intermediates: {}", .{args.dump_intermediates});
    std.log.info("  image:             {s}", .{args.image orelse "(none)"});
    std.log.info("  gemma-hs-pos:      {s}", .{args.gemma_hidden_states_pos});
    std.log.info("  gemma-hs-neg:      {s}", .{args.gemma_hidden_states_neg});

    // ---- Validate dimensions ----
    if (args.height % 32 != 0 or args.width % 32 != 0) {
        std.log.err("--height ({d}) and --width ({d}) must be divisible by 32 (VAE spatial factor).", .{ args.height, args.width });
        return error.InvalidArgs;
    }
    if (args.height % 64 != 0 or args.width % 64 != 0) {
        std.log.warn("--height ({d}) or --width ({d}) not divisible by 64; Stage 1 will generate at a slightly different " ++
            "aspect ratio and the bridge will crop to Stage 2 resolution. For best quality, use multiples of 64.", .{ args.height, args.width });
    }
    if ((args.num_frames - 1) % 8 != 0) {
        const valid = ((args.num_frames - 1) / 8) * 8 + 1;
        std.log.err("--num-frames ({d}) must be of the form 8k+1 (e.g. {d} or {d}).", .{ args.num_frames, valid, valid + 8 });
        return error.InvalidArgs;
    }

    // ---- Ensure output directory exists ----
    try std.Io.Dir.createDirPath(.cwd(), io, args.output_dir);

    // ---- Load pipeline metadata ----
    const pipe_meta = if (args.meta) |meta_path|
        try loadPipelineMeta(allocator, io, meta_path)
    else
        computePipelineMeta(args.height, args.width, args.num_frames, args.fps);
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
    defer platform.deinit(allocator, io);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ---- Phase timers ----
    var timer_s1 = PhaseTimer.init("Stage 1", io);
    var timer_bridge = PhaseTimer.init("Bridge", io);
    var timer_s2 = PhaseTimer.init("Stage 2", io);
    var timer_video_vae = PhaseTimer.init("Video VAE Decode", io);
    var timer_audio_vae = PhaseTimer.init("Audio VAE Decode", io);
    var timer_vocoder = PhaseTimer.init("Vocoder + BWE", io);
    var timer_mp4 = PhaseTimer.init("MP4 Encoding", io);

    // ========================================================================
    // Phase 1: Stage 1
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Phase 1: Stage 1 — {d}-step guided denoising ===", .{args.num_inference_steps});

    const s1 = try runStage1(
        allocator,
        io,
        platform,
        sharding,
        args.stage1_ckpt,
        args.bf16_attn_stage1,
        args.image,
        pipe_meta,
        args.dump_intermediates,
        args.output_dir,
        args.seed,
        args.gemma_hidden_states_pos,
        args.gemma_hidden_states_neg,
        args.num_inference_steps,
        &timer_s1,
    );

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
        pipe_meta,
        args.image,
        s1.rng_state,
        args.dump_intermediates,
        args.output_dir,
        &timer_bridge,
    );
    // Stage 1 latent/context buffers are consumed by bridge (ownership transferred)

    // ========================================================================
    // Phase 3: Stage 2
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Phase 3: Stage 2 — {d}-step distilled denoising ===", .{stage2_sigmas.len - 1});

    const s2 = try runStage2(allocator, io, platform, sharding, args.stage2_ckpt, &bridge, args.bf16_attn_stage2, stage2_sigmas, &timer_s2);

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

    var video_frames = try runVideoVaeDecode(allocator, io, platform, sharding, args.stage2_ckpt, s2.v_latent, pipe_meta, &timer_video_vae);
    defer video_frames.deinit();

    if (args.dump_intermediates) {
        try writeRawBytes(allocator, io, video_frames.data, args.output_dir, "frames.bin");
    }

    // ========================================================================
    // Phase 5: Audio VAE Decode
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Phase 5: Audio VAE Decode ===", .{});

    const audio_mel = try runAudioVaeDecode(allocator, io, platform, sharding, args.stage2_ckpt, s2.a_latent, pipe_meta, &timer_audio_vae);

    if (args.dump_intermediates) {
        try writeBuffer(allocator, io, audio_mel, args.output_dir, "audio_mel.bin");
    }

    // ========================================================================
    // Phase 6: Vocoder + BWE (mel → 48kHz waveform)
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Phase 6: Vocoder + BWE ===", .{});

    var waveform_buf = try runVocoderWithBWE(allocator, io, platform, sharding, args.stage2_ckpt, audio_mel, &timer_vocoder);
    defer waveform_buf.deinit();
    // audio_mel ownership transferred to runVocoderWithBWE (freed inside after main vocoder runs).

    // s2.v_latent was already freed inside runVideoVaeDecode.
    // s2.a_latent was already freed inside runAudioVaeDecode.

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
    const planar_f32: [*]const f32 = @ptrCast(@alignCast(waveform_bytes.ptr));
    const interleaved = try allocator.alloc(f32, num_channels * num_samples);
    defer allocator.free(interleaved);
    for (0..num_samples) |i| {
        for (0..num_channels) |ch| {
            interleaved[i * num_channels + ch] = planar_f32[ch * num_samples + i];
        }
    }
    const interleaved_bytes = std.mem.sliceAsBytes(interleaved);

    if (args.dump_intermediates) {
        try writeRawBytes(allocator, io, interleaved_bytes, args.output_dir, "waveform.bin");
    }

    timer_mp4.start();
    try encodeOutputMp4(allocator, io, video_frames, interleaved_bytes, num_channels, pipe_meta.frame_rate, args.output_dir);
    timer_mp4.addExec();

    // ========================================================================
    // Timing Summary
    // ========================================================================
    const all_timers = [_]*const PhaseTimer{
        &timer_s1,        &timer_bridge,    &timer_s2,
        &timer_video_vae, &timer_audio_vae, &timer_vocoder,
        &timer_mp4,
    };
    printTimingSummary(&all_timers);

    std.log.info("Done.", .{});
}

// ============================================================================
// Phase 1: Stage 1 — guided denoising
// ============================================================================

fn runStage1(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    ckpt_path: []const u8,
    use_bf16_attn: bool,
    image_path: ?[]const u8,
    pipe_meta: PipelineMeta,
    dump_intermediates: bool,
    output_dir: []const u8,
    seed: u64,
    gemma_hs_pos_path: []const u8,
    gemma_hs_neg_path: []const u8,
    num_stage1_steps: usize,
    timer: *PhaseTimer,
) !Stage1Result {
    timer.start();

    // ---- Sigma schedule ----
    // sigmas is [MAX_STAGE1_STEPS + 1]f32; only entries [0..num_stage1_steps] are valid.
    const sigmas = model.computeSigmaSchedule(
        MAX_STAGE1_STEPS,
        num_stage1_steps,
        model.MAX_SHIFT_ANCHOR, // In python reference implementation, the number of tokens used for computing the sigma shift when no latent tensor is available, is the same regardless of the resolution.
        model.stage1_default_schedule.max_shift,
        model.stage1_default_schedule.base_shift,
        model.stage1_default_schedule.terminal,
    );
    std.log.info("Sigma schedule ({d} steps): [{d:.6} ... {d:.6}]", .{
        num_stage1_steps, sigmas[0], sigmas[num_stage1_steps],
    });

    // ---- Open checkpoint store ----
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    // ---- Compute Stage 1 initial state on host ----
    const s1 = pipe_meta.stage1;
    const fps = pipe_meta.frame_rate;
    const F = s1.f_lat;
    const H_s1 = s1.h_lat;
    const W_s1 = s1.w_lat;
    const T_a: i64 = s1.t_audio;
    const C: i64 = 128;
    const T_v1 = F * H_s1 * W_s1;

    std.log.info("Computing Stage 1 initial state (T_v={d}, T_a={d})...", .{ T_v1, T_a });

    // Video positions: [1, 3, T_v1, 2] bf16
    var v_positions_buf = blk: {
        const video_pos_bytes = try computeVideoPositions(allocator, F, H_s1, W_s1, fps);
        defer allocator.free(video_pos_bytes);
        const video_pos_shape = zml.Shape.init(.{ 1, 3, T_v1, 2 }, .bf16);
        break :blk try zml.Buffer.fromBytes(io, platform, video_pos_shape, sharding, video_pos_bytes);
    };
    defer v_positions_buf.deinit();

    // Audio positions: [1, 1, T_a, 2] f32
    var a_positions_buf = blk: {
        const audio_pos_bytes = try computeAudioPositions(allocator, T_a);
        defer allocator.free(audio_pos_bytes);
        const audio_pos_shape = zml.Shape.init(.{ 1, 1, T_a, 2 }, .f32);
        break :blk try zml.Buffer.fromBytes(io, platform, audio_pos_shape, sharding, audio_pos_bytes);
    };
    defer a_positions_buf.deinit();

    // Denoise masks: all ones [1, T, 1] f32
    var v_mask_buf = blk: {
        const shape = zml.Shape.init(.{ 1, T_v1, 1 }, .f32);
        const host = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(host);
        fillOnesF32(host);
        break :blk try zml.Buffer.fromBytes(io, platform, shape, sharding, host);
    };
    defer v_mask_buf.deinit();
    var a_mask_buf = blk: {
        const shape = zml.Shape.init(.{ 1, T_a, 1 }, .f32);
        const host = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(host);
        fillOnesF32(host);
        break :blk try zml.Buffer.fromBytes(io, platform, shape, sharding, host);
    };
    defer a_mask_buf.deinit();

    // Clean latents: all zeros [1, T, 128] bf16
    var v_clean_buf = blk: {
        const shape = zml.Shape.init(.{ 1, T_v1, C }, .bf16);
        const host = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(host);
        @memset(host, 0);
        break :blk try zml.Buffer.fromBytes(io, platform, shape, sharding, host);
    };
    defer v_clean_buf.deinit();
    var a_clean_buf = blk: {
        const shape = zml.Shape.init(.{ 1, T_a, C }, .bf16);
        const host = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(host);
        @memset(host, 0);
        break :blk try zml.Buffer.fromBytes(io, platform, shape, sharding, host);
    };
    defer a_clean_buf.deinit();

    std.log.info("  video_positions:   {any}", .{v_positions_buf.shape().dims()});
    std.log.info("  audio_positions:   {any}", .{a_positions_buf.shape().dims()});
    std.log.info("  video_mask:        {any}", .{v_mask_buf.shape().dims()});
    std.log.info("  audio_mask:        {any}", .{a_mask_buf.shape().dims()});
    std.log.info("  video_clean:       {any}", .{v_clean_buf.shape().dims()});
    std.log.info("  audio_clean:       {any}", .{a_clean_buf.shape().dims()});

    if (dump_intermediates) {
        try writeBuffer(allocator, io, v_positions_buf, output_dir, "s1_video_positions.bin");
        try writeBuffer(allocator, io, a_positions_buf, output_dir, "s1_audio_positions.bin");
        try writeBuffer(allocator, io, v_mask_buf, output_dir, "s1_video_mask.bin");
        try writeBuffer(allocator, io, a_mask_buf, output_dir, "s1_audio_mask.bin");
        try writeBuffer(allocator, io, v_clean_buf, output_dir, "s1_video_clean.bin");
        try writeBuffer(allocator, io, a_clean_buf, output_dir, "s1_audio_clean.bin");
    }

    // Text contexts — positive AND negative for Stage 1 guidance.
    // Positive contexts are kept alive and returned for Stage 2.
    // Compute from Gemma hidden states using the Zig EmbeddingsProcessor.
    std.log.info("Computing text embeddings from Gemma hidden states...", .{});

    const emb_result = try computeTextEmbeddings(allocator, io, platform, sharding, &ckpt_store, gemma_hs_pos_path, gemma_hs_neg_path);
    var v_context_pos_buf = emb_result.v_context_pos;
    var a_context_pos_buf = emb_result.a_context_pos;
    var v_context_neg_buf = emb_result.v_context_neg;
    var a_context_neg_buf = emb_result.a_context_neg;
    defer v_context_neg_buf.deinit();
    defer a_context_neg_buf.deinit();

    // ---- Generate noise and run noise init ----
    std.log.info("Generating Stage 1 noise (seed={d})...", .{seed});

    var rng_buf = try zml.Tensor.Rng.initBuffer(platform, seed, io, sharding);

    // Generate video noise (draw #1)
    var v_noise_buf = blk: {
        var exe = try platform.compileFn(
            allocator,
            io,
            model.forwardGenerateNoise,
            .{
                zml.Tensor.Rng.init(),
                zml.Tensor.fromShape(v_clean_buf.shape()),
            },
            .{ .shardings = &.{sharding} },
        );
        defer exe.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ rng_buf, v_clean_buf });
        exe.callOpts(io, args, &results, .{ .wait = true });
        rng_buf._state.deinit();
        rng_buf, const noise = results.get(struct { zml.Bufferized(zml.Tensor.Rng), zml.Buffer });
        break :blk noise;
    };
    defer v_noise_buf.deinit();

    // Generate audio noise (draw #2)
    var a_noise_buf = blk: {
        var exe = try platform.compileFn(
            allocator,
            io,
            model.forwardGenerateNoise,
            .{
                zml.Tensor.Rng.init(),
                zml.Tensor.fromShape(a_clean_buf.shape()),
            },
            .{ .shardings = &.{sharding} },
        );
        defer exe.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ rng_buf, a_clean_buf });
        exe.callOpts(io, args, &results, .{ .wait = true });
        rng_buf._state.deinit();
        rng_buf, const noise = results.get(struct { zml.Bufferized(zml.Tensor.Rng), zml.Buffer });
        break :blk noise;
    };
    defer a_noise_buf.deinit();

    std.log.info("  video_noise: {any}", .{v_noise_buf.shape().dims()});
    std.log.info("  audio_noise: {any}", .{a_noise_buf.shape().dims()});

    if (dump_intermediates) {
        try writeBuffer(allocator, io, v_noise_buf, output_dir, "s1_video_noise.bin");
        try writeBuffer(allocator, io, a_noise_buf, output_dir, "s1_audio_noise.bin");
    }

    // Compile noise init
    const sigma_scalar_shape = zml.Shape.init(.{}, .f32);

    // Stage 1 noise_scale = 1.0 (pure noise for unconditioned, mask-weighted for image-conditioned)
    const noise_scale: f32 = 1.0;
    std.log.info("Running Stage 1 noise init (noise_scale={d:.6})...", .{noise_scale});
    var noise_scale_buf = try zml.Buffer.scalar(io, platform, noise_scale, .f32, sharding);
    defer noise_scale_buf.deinit();

    // Video noise init
    var v_latent_buf = blk: {
        var exe = try platform.compileFn(
            allocator,
            io,
            model.forwardNoiseInit,
            .{
                zml.Tensor.fromShape(v_clean_buf.shape()),
                zml.Tensor.fromShape(v_noise_buf.shape()),
                zml.Tensor.fromShape(v_mask_buf.shape()),
                zml.Tensor.fromShape(sigma_scalar_shape),
            },
            .{ .shardings = &.{sharding} },
        );
        defer exe.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ v_clean_buf, v_noise_buf, v_mask_buf, noise_scale_buf });
        exe.callOpts(io, args, &results, .{ .wait = true });
        break :blk results.get(zml.Buffer);
    };

    // Audio noise init
    var a_latent_buf = blk: {
        var exe = try platform.compileFn(
            allocator,
            io,
            model.forwardNoiseInit,
            .{
                zml.Tensor.fromShape(a_clean_buf.shape()),
                zml.Tensor.fromShape(a_noise_buf.shape()),
                zml.Tensor.fromShape(a_mask_buf.shape()),
                zml.Tensor.fromShape(sigma_scalar_shape),
            },
            .{ .shardings = &.{sharding} },
        );
        defer exe.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ a_clean_buf, a_noise_buf, a_mask_buf, noise_scale_buf });
        exe.callOpts(io, args, &results, .{ .wait = true });
        break :blk results.get(zml.Buffer);
    };

    std.log.info("  video_latent (noised): {any}", .{v_latent_buf.shape().dims()});
    std.log.info("  audio_latent (noised): {any}", .{a_latent_buf.shape().dims()});

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
    timer.addOther(); // host prep + text embeddings + noise gen/init + image cond

    std.log.info("Compiling preprocessing exe...", .{});
    const preprocess_shape = model.initPreprocessParams(ckpt_store.view());
    var preprocess_exe = try platform.compileFn(
        allocator,
        io,
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
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer model.PreprocessParams.unloadBuffers(&preprocess_bufs);

    // Run preprocessing once to discover output shapes for block compilation.
    // Shapes are config-deterministic, but we derive them from an actual run rather than
    // duplicating the shape arithmetic, so they never drift if forwardPreprocess changes.
    std.log.info("Running initial preprocessing for shape discovery...", .{});
    var sigma_1d_init = try sigma1dBuffer(io, platform, sigmas[0], sharding);
    defer sigma_1d_init.deinit();

    var pre_args_init = try preprocess_exe.args(allocator);
    defer pre_args_init.deinit(allocator);
    var pre_results_init = try preprocess_exe.results(allocator);
    defer pre_results_init.deinit(allocator);
    pre_args_init.set(.{
        v_latent_buf,      a_latent_buf,
        v_mask_buf,        a_mask_buf,
        sigma_1d_init,     sigma_1d_init,
        v_positions_buf,   a_positions_buf,
        v_context_pos_buf, a_context_pos_buf,
        preprocess_bufs,
    });
    preprocess_exe.callOpts(io, pre_args_init, &pre_results_init, .{ .wait = true });
    var init_pre_out = pre_results_init.get(zml.Bufferized(model.PreprocessOutput));

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
    const init_vx_shape = init_pre_out.vx.shape();
    const init_ax_shape = init_pre_out.ax.shape();

    // Free shape-discovery preprocessing outputs — all shapes extracted.
    deinitBufferizedFields(&init_pre_out);

    var proj_v_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardOutputProjection,
        .{
            zml.Tensor.fromShape(init_vx_shape).withPartialTags(.{ .b, .t, .d }),
            zml.Tensor.fromShape(v_emb_shape).withPartialTags(.{ .b, .t, .d_emb }),
            block_params_shape.norm_proj_out,
        },
        .{ .shardings = &.{sharding} },
    );
    defer proj_v_exe.deinit();

    var proj_a_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardOutputProjection,
        .{
            zml.Tensor.fromShape(init_ax_shape).withPartialTags(.{ .b, .t, .d }),
            zml.Tensor.fromShape(a_emb_shape).withPartialTags(.{ .b, .t, .d_emb }),
            block_params_shape.audio_norm_proj_out,
        },
        .{ .shardings = &.{sharding} },
    );
    defer proj_a_exe.deinit();

    // ---- Compile vel→x0 exes ----
    std.log.info("Compiling vel→x0 (toDenoised) exes...", .{});
    var to_denoised_v_exe = try platform.compileFn(
        allocator,
        io,
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
        allocator,
        io,
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
        allocator,
        io,
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
        allocator,
        io,
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
        allocator,
        io,
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
    timer.addCompile();

    // ---- Load weights ----
    std.log.info("Loading {d} block weights...", .{model.num_transformer_blocks});
    var block_params_bufs = try allocator.create([model.num_transformer_blocks]zml.Bufferized(model.Block0FullParams));
    defer allocator.destroy(block_params_bufs);
    for (0..model.num_transformer_blocks) |i| {
        block_params_bufs[i] = try zml.io.load(
            model.Block0FullParams,
            &block_params_shape.blocks[i],
            allocator,
            io,
            platform,
            &ckpt_store,
            .{
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
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
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
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer model.OutputProjection.Params.unloadBuffers(&proj_a_bufs);
    std.log.info("All Stage 1 weights loaded.", .{});
    timer.addLoad();

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

    // ---- Denoising loop ----
    std.log.info("Starting {d}-step denoising loop (4-pass guidance)...", .{num_stage1_steps});

    for (0..num_stage1_steps) |step_idx| {
        const step_start: std.Io.Timestamp = .now(io, .awake);
        const sigma = sigmas[step_idx];
        const sigma_next = sigmas[step_idx + 1];

        std.log.info("", .{});
        std.log.info("===== Step {d}/{d}: sigma={d:.6} -> {d:.6} =====", .{
            step_idx + 1, num_stage1_steps, sigma, sigma_next,
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
            v_latent_buf,      a_latent_buf,
            v_mask_buf,        a_mask_buf,
            sigma_1d,          sigma_1d,
            v_positions_buf,   a_positions_buf,
            v_context_pos_buf, a_context_pos_buf,
            preprocess_bufs,
        });
        preprocess_exe.callOpts(io, pre_args, &pre_results, .{ .wait = true });
        var pre_out = pre_results.get(zml.Bufferized(model.PreprocessOutput));

        // ---- Pass 1: Conditional (positive context, normal blocks) ----
        std.log.info("  Pass 1 (conditional): {d}-block chain...", .{model.num_transformer_blocks});
        var cond_h_v = pre_out.vx;
        var cond_h_a = pre_out.ax;

        {
            var blk_args = try block_normal_exe.args(allocator);
            defer blk_args.deinit(allocator);
            var blk_results = try block_normal_exe.results(allocator);
            defer blk_results.deinit(allocator);

            for (0..model.num_transformer_blocks) |i| {
                blk_args.set(.{
                    cond_h_v,                     cond_h_a,
                    pre_out.video_timesteps,      pre_out.audio_timesteps,
                    pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                    pre_out.v_denoise_mask,       pre_out.a_denoise_mask,
                    pre_out.v_prompt_timestep,    pre_out.a_prompt_timestep,
                    pre_out.v_pe_cos,             pre_out.v_pe_sin,
                    pre_out.a_pe_cos,             pre_out.a_pe_sin,
                    pre_out.v_text_ctx,           pre_out.a_text_ctx,
                    pre_out.v_cross_ss_ts,        pre_out.v_cross_gate_ts,
                    pre_out.a_cross_ss_ts,        pre_out.a_cross_gate_ts,
                    pre_out.a2v_pe_cos,           pre_out.a2v_pe_sin,
                    pre_out.a2v_k_pe_cos,         pre_out.a2v_k_pe_sin,
                    pre_out.v2a_pe_cos,           pre_out.v2a_pe_sin,
                    pre_out.v2a_k_pe_cos,         pre_out.v2a_k_pe_sin,
                    block_params_bufs[i],
                });
                block_normal_exe.callOpts(io, blk_args, &blk_results, .{ .wait = true });
                const out = blk_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
                if (i > 0) {
                    cond_h_v.deinit();
                    cond_h_a.deinit();
                }
                cond_h_v = out.vx_out;
                cond_h_a = out.ax_out;
            }
        }

        var cond_v_vel = try runOutputProjection(allocator, io, &proj_v_exe, cond_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var cond_a_vel = try runOutputProjection(allocator, io, &proj_a_exe, cond_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        cond_h_v.deinit(); // Free last block's hidden state after projection
        cond_h_a.deinit(); // Free last block's hidden state after projection

        var cond_v_x0 = try runToDenoised(allocator, io, &to_denoised_v_exe, v_latent_buf, cond_v_vel, v_mask_buf, sigma_buf);
        var cond_a_x0 = try runToDenoised(allocator, io, &to_denoised_a_exe, a_latent_buf, cond_a_vel, a_mask_buf, sigma_buf);
        cond_v_vel.deinit();
        cond_a_vel.deinit();

        // ---- Pass 2: Negative/CFG (negative context, normal blocks) ----
        std.log.info("  Pass 2 (negative/CFG): {d}-block chain...", .{model.num_transformer_blocks});
        var neg_h_v = pre_out.vx;
        var neg_h_a = pre_out.ax;

        {
            var blk_args = try block_normal_exe.args(allocator);
            defer blk_args.deinit(allocator);
            var blk_results = try block_normal_exe.results(allocator);
            defer blk_results.deinit(allocator);

            for (0..model.num_transformer_blocks) |i| {
                blk_args.set(.{
                    neg_h_v,                      neg_h_a,
                    pre_out.video_timesteps,      pre_out.audio_timesteps,
                    pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                    pre_out.v_denoise_mask,       pre_out.a_denoise_mask,
                    pre_out.v_prompt_timestep,    pre_out.a_prompt_timestep,
                    pre_out.v_pe_cos,             pre_out.v_pe_sin,
                    pre_out.a_pe_cos,             pre_out.a_pe_sin,
                    v_context_neg_buf,            a_context_neg_buf,
                    pre_out.v_cross_ss_ts,        pre_out.v_cross_gate_ts,
                    pre_out.a_cross_ss_ts,        pre_out.a_cross_gate_ts,
                    pre_out.a2v_pe_cos,           pre_out.a2v_pe_sin,
                    pre_out.a2v_k_pe_cos,         pre_out.a2v_k_pe_sin,
                    pre_out.v2a_pe_cos,           pre_out.v2a_pe_sin,
                    pre_out.v2a_k_pe_cos,         pre_out.v2a_k_pe_sin,
                    block_params_bufs[i],
                });
                block_normal_exe.callOpts(io, blk_args, &blk_results, .{ .wait = true });
                const out = blk_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
                if (i > 0) {
                    neg_h_v.deinit();
                    neg_h_a.deinit();
                }
                neg_h_v = out.vx_out;
                neg_h_a = out.ax_out;
            }
        }

        var neg_v_vel = try runOutputProjection(allocator, io, &proj_v_exe, neg_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var neg_a_vel = try runOutputProjection(allocator, io, &proj_a_exe, neg_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        neg_h_v.deinit();
        neg_h_a.deinit();

        var neg_v_x0 = try runToDenoised(allocator, io, &to_denoised_v_exe, v_latent_buf, neg_v_vel, v_mask_buf, sigma_buf);
        var neg_a_x0 = try runToDenoised(allocator, io, &to_denoised_a_exe, a_latent_buf, neg_a_vel, a_mask_buf, sigma_buf);
        neg_v_vel.deinit();
        neg_a_vel.deinit();

        // ---- Pass 3: STG (positive context, V-passthrough at block 28) ----
        std.log.info("  Pass 3 (STG): {d}-block chain (STG at block {d})...", .{ model.num_transformer_blocks, STG_BLOCK_IDX });
        var ptb_h_v = pre_out.vx;
        var ptb_h_a = pre_out.ax;

        {
            var blk_normal_args = try block_normal_exe.args(allocator);
            defer blk_normal_args.deinit(allocator);
            var blk_normal_results = try block_normal_exe.results(allocator);
            defer blk_normal_results.deinit(allocator);
            var blk_stg_args = try block_stg_exe.args(allocator);
            defer blk_stg_args.deinit(allocator);
            var blk_stg_results = try block_stg_exe.results(allocator);
            defer blk_stg_results.deinit(allocator);

            for (0..model.num_transformer_blocks) |i| {
                // Special handling for the STG block, which has additional inputs/outputs and a different forward fn.
                if (i == STG_BLOCK_IDX) {
                    blk_stg_args.set(.{
                        ptb_h_v,                      ptb_h_a,
                        pre_out.video_timesteps,      pre_out.audio_timesteps,
                        pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                        pre_out.v_denoise_mask,       pre_out.a_denoise_mask,
                        pre_out.v_prompt_timestep,    pre_out.a_prompt_timestep,
                        pre_out.v_pe_cos,             pre_out.v_pe_sin,
                        pre_out.a_pe_cos,             pre_out.a_pe_sin,
                        pre_out.v_text_ctx,           pre_out.a_text_ctx,
                        pre_out.v_cross_ss_ts,        pre_out.v_cross_gate_ts,
                        pre_out.a_cross_ss_ts,        pre_out.a_cross_gate_ts,
                        pre_out.a2v_pe_cos,           pre_out.a2v_pe_sin,
                        pre_out.a2v_k_pe_cos,         pre_out.a2v_k_pe_sin,
                        pre_out.v2a_pe_cos,           pre_out.v2a_pe_sin,
                        pre_out.v2a_k_pe_cos,         pre_out.v2a_k_pe_sin,
                        block_params_bufs[i],
                    });
                    block_stg_exe.callOpts(io, blk_stg_args, &blk_stg_results, .{ .wait = true });
                    const out = blk_stg_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
                    if (i > 0) {
                        ptb_h_v.deinit();
                        ptb_h_a.deinit();
                    }
                    ptb_h_v = out.vx_out;
                    ptb_h_a = out.ax_out;
                } else {
                    blk_normal_args.set(.{
                        ptb_h_v,                      ptb_h_a,
                        pre_out.video_timesteps,      pre_out.audio_timesteps,
                        pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                        pre_out.v_denoise_mask,       pre_out.a_denoise_mask,
                        pre_out.v_prompt_timestep,    pre_out.a_prompt_timestep,
                        pre_out.v_pe_cos,             pre_out.v_pe_sin,
                        pre_out.a_pe_cos,             pre_out.a_pe_sin,
                        pre_out.v_text_ctx,           pre_out.a_text_ctx,
                        pre_out.v_cross_ss_ts,        pre_out.v_cross_gate_ts,
                        pre_out.a_cross_ss_ts,        pre_out.a_cross_gate_ts,
                        pre_out.a2v_pe_cos,           pre_out.a2v_pe_sin,
                        pre_out.a2v_k_pe_cos,         pre_out.a2v_k_pe_sin,
                        pre_out.v2a_pe_cos,           pre_out.v2a_pe_sin,
                        pre_out.v2a_k_pe_cos,         pre_out.v2a_k_pe_sin,
                        block_params_bufs[i],
                    });
                    block_normal_exe.callOpts(io, blk_normal_args, &blk_normal_results, .{ .wait = true });
                    const out = blk_normal_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
                    if (i > 0) {
                        ptb_h_v.deinit();
                        ptb_h_a.deinit();
                    }
                    ptb_h_v = out.vx_out;
                    ptb_h_a = out.ax_out;
                }
            }
        }

        var ptb_v_vel = try runOutputProjection(allocator, io, &proj_v_exe, ptb_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var ptb_a_vel = try runOutputProjection(allocator, io, &proj_a_exe, ptb_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        ptb_h_v.deinit();
        ptb_h_a.deinit();

        var ptb_v_x0 = try runToDenoised(allocator, io, &to_denoised_v_exe, v_latent_buf, ptb_v_vel, v_mask_buf, sigma_buf);
        var ptb_a_x0 = try runToDenoised(allocator, io, &to_denoised_a_exe, a_latent_buf, ptb_a_vel, a_mask_buf, sigma_buf);
        ptb_v_vel.deinit();
        ptb_a_vel.deinit();

        // ---- Pass 4: Isolated (positive context, zero AV masks) ----
        std.log.info("  Pass 4 (isolated): {d}-block chain...", .{model.num_transformer_blocks});
        var iso_h_v = pre_out.vx;
        var iso_h_a = pre_out.ax;

        {
            var blk_args = try block_iso_exe.args(allocator);
            defer blk_args.deinit(allocator);
            var blk_results = try block_iso_exe.results(allocator);
            defer blk_results.deinit(allocator);

            for (0..model.num_transformer_blocks) |i| {
                blk_args.set(.{
                    iso_h_v,                      iso_h_a,
                    pre_out.video_timesteps,      pre_out.audio_timesteps,
                    pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                    pre_out.v_denoise_mask,       pre_out.a_denoise_mask,
                    pre_out.v_prompt_timestep,    pre_out.a_prompt_timestep,
                    pre_out.v_pe_cos,             pre_out.v_pe_sin,
                    pre_out.a_pe_cos,             pre_out.a_pe_sin,
                    pre_out.v_text_ctx,           pre_out.a_text_ctx,
                    pre_out.v_cross_ss_ts,        pre_out.v_cross_gate_ts,
                    pre_out.a_cross_ss_ts,        pre_out.a_cross_gate_ts,
                    pre_out.a2v_pe_cos,           pre_out.a2v_pe_sin,
                    pre_out.a2v_k_pe_cos,         pre_out.a2v_k_pe_sin,
                    zero_mask_buf,                pre_out.v2a_pe_cos,
                    pre_out.v2a_pe_sin,           pre_out.v2a_k_pe_cos,
                    pre_out.v2a_k_pe_sin,         zero_mask_buf,
                    block_params_bufs[i],
                });
                block_iso_exe.callOpts(io, blk_args, &blk_results, .{ .wait = true });
                const out = blk_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
                if (i > 0) {
                    iso_h_v.deinit();
                    iso_h_a.deinit();
                }
                iso_h_v = out.vx_out;
                iso_h_a = out.ax_out;
            }
        }

        // Free block-chain-only preprocessing outputs early — only v/a_embedded_timestep
        // are still needed for output projection. This reduces peak GPU memory.
        pre_out.vx.deinit();
        pre_out.ax.deinit();
        pre_out.video_timesteps.deinit();
        pre_out.audio_timesteps.deinit();
        pre_out.video_timesteps_zero.deinit();
        pre_out.audio_timesteps_zero.deinit();
        pre_out.v_denoise_mask.deinit();
        pre_out.a_denoise_mask.deinit();
        pre_out.v_prompt_timestep.deinit();
        pre_out.a_prompt_timestep.deinit();
        pre_out.v_pe_cos.deinit();
        pre_out.v_pe_sin.deinit();
        pre_out.a_pe_cos.deinit();
        pre_out.a_pe_sin.deinit();
        pre_out.v_text_ctx.deinit();
        pre_out.a_text_ctx.deinit();
        pre_out.v_cross_ss_ts.deinit();
        pre_out.v_cross_gate_ts.deinit();
        pre_out.a_cross_ss_ts.deinit();
        pre_out.a_cross_gate_ts.deinit();
        pre_out.a2v_pe_cos.deinit();
        pre_out.a2v_pe_sin.deinit();
        pre_out.a2v_k_pe_cos.deinit();
        pre_out.a2v_k_pe_sin.deinit();
        pre_out.v2a_pe_cos.deinit();
        pre_out.v2a_pe_sin.deinit();
        pre_out.v2a_k_pe_cos.deinit();
        pre_out.v2a_k_pe_sin.deinit();

        var iso_v_vel = try runOutputProjection(allocator, io, &proj_v_exe, iso_h_v, pre_out.v_embedded_timestep, proj_v_bufs);
        var iso_a_vel = try runOutputProjection(allocator, io, &proj_a_exe, iso_h_a, pre_out.a_embedded_timestep, proj_a_bufs);
        iso_h_v.deinit();
        iso_h_a.deinit();
        // Free remaining preprocessing outputs (embedded timesteps).
        pre_out.v_embedded_timestep.deinit();
        pre_out.a_embedded_timestep.deinit();

        var iso_v_x0 = try runToDenoised(allocator, io, &to_denoised_v_exe, v_latent_buf, iso_v_vel, v_mask_buf, sigma_buf);
        var iso_a_x0 = try runToDenoised(allocator, io, &to_denoised_a_exe, a_latent_buf, iso_a_vel, a_mask_buf, sigma_buf);
        iso_v_vel.deinit();
        iso_a_vel.deinit();

        // ---- Guider combine ----
        std.log.info("  Guider combine...", .{});
        var gc_args = try guider_combine_exe.args(allocator);
        defer gc_args.deinit(allocator);
        var gc_results = try guider_combine_exe.results(allocator);
        defer gc_results.deinit(allocator);

        gc_args.set(.{
            cond_v_x0, neg_v_x0,  ptb_v_x0,  iso_v_x0,
            cond_a_x0, neg_a_x0,  ptb_a_x0,  iso_a_x0,
            cfg_v_buf, stg_v_buf, mod_v_buf, rescale_v_buf,
            cfg_a_buf, stg_a_buf, mod_a_buf, rescale_a_buf,
        });
        guider_combine_exe.callOpts(io, gc_args, &gc_results, .{ .wait = true });
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
        denoise_v_exe.callOpts(io, dv_args, &dv_results, .{ .wait = true });
        const dv_out = dv_results.get(zml.Bufferized(model.DenoisingStepResult));

        var da_args = try denoise_a_exe.args(allocator);
        defer da_args.deinit(allocator);
        var da_results = try denoise_a_exe.results(allocator);
        defer da_results.deinit(allocator);
        da_args.set(.{ a_latent_buf, guided_a_x0, a_mask_buf, a_clean_buf, sigma_buf, sigma_next_buf });
        denoise_a_exe.callOpts(io, da_args, &da_results, .{ .wait = true });
        const da_out = da_results.get(zml.Bufferized(model.DenoisingStepResult));

        guided_v_x0.deinit();
        guided_a_x0.deinit();

        v_latent_buf.deinit();
        a_latent_buf.deinit();
        v_latent_buf = dv_out.next_latent;
        a_latent_buf = da_out.next_latent;

        const step_ns = step_start.untilNow(io, .awake).nanoseconds;
        timer.recordStep(step_ns);
        std.log.info("  Step {d} complete ({d:.1}s).", .{ step_idx + 1, PhaseTimer.fmtSecs(step_ns) });
    }
    timer.addExec();

    std.log.info("Stage 1 denoising complete.", .{});

    return .{
        .v_latent = v_latent_buf,
        .a_latent = a_latent_buf,
        .v_context_pos = v_context_pos_buf,
        .a_context_pos = a_context_pos_buf,
        .rng_state = rng_buf._state,
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
    pipe_meta: PipelineMeta,
    image_path: ?[]const u8,
    rng_state: zml.Buffer,
    dump_intermediates: bool,
    output_dir: []const u8,
    timer: *PhaseTimer,
) !BridgeResult {
    timer.start();
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
    const C: i64 = 128; // Both stage 1 and stage 2 use the same latent channel dimension.
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

    // ========================================================================
    // Step 1: Unpatchify video: [1, T_v1, 128] → [1, 128, F, H_s1, W_s1]
    // ========================================================================
    std.log.info("Compiling unpatchify...", .{});
    timer.addOther(); // checkpoint stores opening
    const patchified_shape = zml.Shape.init(.{ 1, T_v1, C }, .bf16);
    const video_5d_s1_shape = zml.Shape.init(.{ 1, C, F, H_s1, W_s1 }, .bf16);

    var unpatch_exe = try platform.compileFn(
        allocator,
        io,
        upsampler.forwardUnpatchifyVideo,
        .{
            zml.Tensor.fromShape(patchified_shape),
            video_5d_s1_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer unpatch_exe.deinit();
    timer.addCompile();

    std.log.info("Running unpatchify...", .{});
    var unpatch_args = try unpatch_exe.args(allocator);
    defer unpatch_args.deinit(allocator);
    var unpatch_results = try unpatch_exe.results(allocator);
    defer unpatch_results.deinit(allocator);
    unpatch_args.set(.{s1_video});
    unpatch_exe.callOpts(io, unpatch_args, &unpatch_results, .{ .wait = true });
    var video_5d_buf = unpatch_results.get(zml.Buffer);
    var s1_video_mut = s1_video;
    s1_video_mut.deinit();
    std.log.info("  Unpatchified: {any}", .{video_5d_buf.shape().dims()});
    timer.addExec();

    // ========================================================================
    // Step 2: Upsample: [1, 128, F, H_s1, W_s1] → [1, 128, F, H_s2, W_s2]
    // ========================================================================
    std.log.info("Compiling upsampler...", .{});
    const upsampler_shape = upsampler.initUpsamplerParams(up_store.view());
    const stats_shape = conv_ops.initPerChannelStats(main_store.view());

    var upsample_exe = try platform.compileFn(
        allocator,
        io,
        upsampler.forwardUpsample,
        .{
            zml.Tensor.fromShape(video_5d_buf.shape()),
            upsampler_shape,
            stats_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer upsample_exe.deinit();
    timer.addCompile();

    std.log.info("Loading upsampler weights...", .{});
    const up_bufs = try zml.io.load(
        upsampler.UpsamplerParams,
        &upsampler_shape,
        allocator,
        io,
        platform,
        &up_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );

    std.log.info("Loading per-channel statistics...", .{});
    const stats_bufs = try zml.io.load(
        conv_ops.PerChannelStats,
        &stats_shape,
        allocator,
        io,
        platform,
        &main_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    timer.addLoad(); // upsampler + stats weights

    std.log.info("Running upsampler...", .{});
    var up_args = try upsample_exe.args(allocator);
    defer up_args.deinit(allocator);
    var up_results = try upsample_exe.results(allocator);
    defer up_results.deinit(allocator);
    up_args.set(.{ video_5d_buf, up_bufs, stats_bufs });
    upsample_exe.callOpts(io, up_args, &up_results, .{ .wait = true });

    // Free upsampler weights — no longer needed after upsample.
    {
        var up_bufs_mut = up_bufs;
        deinitBufferizedFields(&up_bufs_mut);
        var stats_bufs_mut = stats_bufs;
        deinitBufferizedFields(&stats_bufs_mut);
    }
    var upsampled_buf = up_results.get(zml.Buffer);
    video_5d_buf.deinit(); // Free early — no longer needed after upsample
    std.log.info("  Upsampled: {any}", .{upsampled_buf.shape().dims()});
    timer.addExec();

    // ========================================================================
    // Step 3: Crop + re-patchify video: [1, 128, F, 2*H_s1, 2*W_s1] → [1, T_v2, 128]
    // When 2*H_s1 > H_s2 (height not divisible by 64), crop before patchify.
    // ========================================================================
    std.log.info("Compiling patchify...", .{});
    const video_5d_s2_shape = zml.Shape.init(.{ 1, C, F, H_s2, W_s2 }, .bf16);
    var patchify_exe = try platform.compileFn(
        allocator,
        io,
        upsampler.forwardCropAndPatchifyVideo,
        .{ zml.Tensor.fromShape(upsampled_buf.shape()), video_5d_s2_shape },
        .{ .shardings = &.{sharding} },
    );
    defer patchify_exe.deinit();
    timer.addCompile();

    std.log.info("Running patchify...", .{});
    var patch_args = try patchify_exe.args(allocator);
    defer patch_args.deinit(allocator);
    var patch_results = try patchify_exe.results(allocator);
    defer patch_results.deinit(allocator);
    patch_args.set(.{upsampled_buf});
    patchify_exe.callOpts(io, patch_args, &patch_results, .{ .wait = true });
    var video_clean_buf = patch_results.get(zml.Buffer);
    upsampled_buf.deinit(); // Free early — no longer needed after patchify
    std.log.info("  Re-patchified video: {any}", .{video_clean_buf.shape().dims()});
    timer.addExec();

    // ========================================================================
    // Step 4: Audio passthrough (already patchified from Stage 1)
    // ========================================================================
    var audio_clean_buf = s1_audio; // ownership transferred

    // ========================================================================
    // Step 5: Compute positions and masks on host
    // ========================================================================
    std.log.info("Computing video positions...", .{});
    const video_pos_buf = blk: {
        const bytes = try computeVideoPositions(allocator, F, H_s2, W_s2, fps);
        defer allocator.free(bytes);
        const shape = zml.Shape.init(.{ 1, 3, T_v2, 2 }, .bf16);
        break :blk try zml.Buffer.fromBytes(io, platform, shape, sharding, bytes);
    };

    std.log.info("Computing audio positions...", .{});
    const audio_pos_buf = blk: {
        const bytes = try computeAudioPositions(allocator, T_a);
        defer allocator.free(bytes);
        const shape = zml.Shape.init(.{ 1, 1, T_a, 2 }, .f32);
        break :blk try zml.Buffer.fromBytes(io, platform, shape, sharding, bytes);
    };

    // Denoise masks: all ones
    std.log.info("Creating denoise masks...", .{});
    var v_mask_buf = blk: {
        const shape = zml.Shape.init(.{ 1, T_v2, 1 }, .f32);
        const host = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(host);
        fillOnesF32(host);
        break :blk try zml.Buffer.fromBytes(io, platform, shape, sharding, host);
    };
    var a_mask_buf = blk: {
        const shape = zml.Shape.init(.{ 1, T_a, 1 }, .f32);
        const host = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(host);
        fillOnesF32(host);
        break :blk try zml.Buffer.fromBytes(io, platform, shape, sharding, host);
    };

    // ========================================================================
    // Step 6: Generate or load noise, then run noise init
    // ========================================================================
    var v_noise_buf: zml.Buffer = undefined;
    var a_noise_buf: zml.Buffer = undefined;
    {
        // Generate noise from RNG state
        std.log.info("Generating Stage 2 noise from RNG...", .{});
        var rng_buf: zml.Bufferized(zml.Tensor.Rng) = .{ ._state = rng_state };

        var noise_gen_v_exe = try platform.compileFn(
            allocator,
            io,
            model.forwardGenerateNoise,
            .{
                zml.Tensor.Rng.init(),
                zml.Tensor.fromShape(video_clean_buf.shape()),
            },
            .{ .shardings = &.{sharding} },
        );
        defer noise_gen_v_exe.deinit();

        var noise_gen_a_exe = try platform.compileFn(
            allocator,
            io,
            model.forwardGenerateNoise,
            .{
                zml.Tensor.Rng.init(),
                zml.Tensor.fromShape(audio_clean_buf.shape()),
            },
            .{ .shardings = &.{sharding} },
        );
        defer noise_gen_a_exe.deinit();

        // Generate video noise (draw #3)
        var gen_v_args = try noise_gen_v_exe.args(allocator);
        defer gen_v_args.deinit(allocator);
        var gen_v_results = try noise_gen_v_exe.results(allocator);
        defer gen_v_results.deinit(allocator);
        gen_v_args.set(.{ rng_buf, video_clean_buf });
        noise_gen_v_exe.callOpts(io, gen_v_args, &gen_v_results, .{ .wait = true });
        rng_buf._state.deinit();
        rng_buf, v_noise_buf = gen_v_results.get(struct { zml.Bufferized(zml.Tensor.Rng), zml.Buffer });

        // Generate audio noise (draw #4)
        var gen_a_args = try noise_gen_a_exe.args(allocator);
        defer gen_a_args.deinit(allocator);
        var gen_a_results = try noise_gen_a_exe.results(allocator);
        defer gen_a_results.deinit(allocator);
        gen_a_args.set(.{ rng_buf, audio_clean_buf });
        noise_gen_a_exe.callOpts(io, gen_a_args, &gen_a_results, .{ .wait = true });
        rng_buf._state.deinit();
        var final_rng: zml.Bufferized(zml.Tensor.Rng) = undefined;
        final_rng, a_noise_buf = gen_a_results.get(struct { zml.Bufferized(zml.Tensor.Rng), zml.Buffer });
        final_rng._state.deinit(); // Free final RNG state — not needed after bridge
    }
    // v_noise_buf and a_noise_buf freed explicitly after noise init below

    std.log.info("  video_noise: {any}", .{v_noise_buf.shape().dims()});
    std.log.info("  audio_noise: {any}", .{a_noise_buf.shape().dims()});

    if (dump_intermediates) {
        try writeBuffer(allocator, io, v_noise_buf, output_dir, "s2_video_noise.bin");
        try writeBuffer(allocator, io, a_noise_buf, output_dir, "s2_audio_noise.bin");
    }

    // Compile noise init
    std.log.info("Compiling noise init...", .{});
    const sigma_scalar_shape = zml.Shape.init(.{}, .f32);

    var noise_init_v_exe = try platform.compileFn(
        allocator,
        io,
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
        allocator,
        io,
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
    noise_init_v_exe.callOpts(io, ni_v_args, &ni_v_results, .{ .wait = true });
    var v_latent_buf = ni_v_results.get(zml.Buffer);

    // Audio noise init
    var ni_a_args = try noise_init_a_exe.args(allocator);
    defer ni_a_args.deinit(allocator);
    var ni_a_results = try noise_init_a_exe.results(allocator);
    defer ni_a_results.deinit(allocator);
    ni_a_args.set(.{ audio_clean_buf, a_noise_buf, a_mask_buf, sigma0_buf });
    noise_init_a_exe.callOpts(io, ni_a_args, &ni_a_results, .{ .wait = true });
    const a_latent_buf = ni_a_results.get(zml.Buffer);

    // Free noise buffers — no longer needed after noise init
    v_noise_buf.deinit();
    a_noise_buf.deinit();

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
    timer.addOther(); // positions, masks, noise gen/init, image cond

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
    timer: *PhaseTimer,
) !Stage2Result {
    timer.start();
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
        allocator,
        io,
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
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer model.PreprocessParams.unloadBuffers(&preprocess_bufs);

    // Run preprocessing once to discover output shapes for block compilation.
    // Shapes are config-deterministic, but we derive them from an actual run rather than
    // duplicating the shape arithmetic, so they never drift if forwardPreprocess changes.
    std.log.info("Running initial preprocessing for shape discovery...", .{});
    var sigma_1d_init = try sigma1dBuffer(io, platform, stage2_sigmas[0], sharding);
    defer sigma_1d_init.deinit();

    var pre_args_init = try preprocess_exe.args(allocator);
    defer pre_args_init.deinit(allocator);
    var pre_results_init = try preprocess_exe.results(allocator);
    defer pre_results_init.deinit(allocator);
    pre_args_init.set(.{
        v_latent_buf,       a_latent_buf,
        bridge.v_mask,      bridge.a_mask,
        sigma_1d_init,      sigma_1d_init,
        bridge.v_positions, bridge.a_positions,
        bridge.v_context,   bridge.a_context,
        preprocess_bufs,
    });
    preprocess_exe.callOpts(io, pre_args_init, &pre_results_init, .{ .wait = true });
    var init_pre_out = pre_results_init.get(zml.Bufferized(model.PreprocessOutput));

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
    const init_vx_shape = init_pre_out.vx.shape();
    const init_ax_shape = init_pre_out.ax.shape();

    // Free shape-discovery preprocessing outputs — all shapes extracted.
    deinitBufferizedFields(&init_pre_out);

    var proj_v_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardOutputProjection,
        .{
            zml.Tensor.fromShape(init_vx_shape).withPartialTags(.{ .b, .t, .d }),
            zml.Tensor.fromShape(v_emb_shape).withPartialTags(.{ .b, .t, .d_emb }),
            block_params_shape.norm_proj_out,
        },
        .{ .shardings = &.{sharding} },
    );
    defer proj_v_exe.deinit();

    var proj_a_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardOutputProjection,
        .{
            zml.Tensor.fromShape(init_ax_shape).withPartialTags(.{ .b, .t, .d }),
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
        allocator,
        io,
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
        allocator,
        io,
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
    timer.addCompile();

    // ---- Load weights ----
    std.log.info("Loading {d} block weights...", .{model.num_transformer_blocks});
    var block_params_bufs = try allocator.create([model.num_transformer_blocks]zml.Bufferized(model.Block0FullParams));
    defer allocator.destroy(block_params_bufs);
    for (0..model.num_transformer_blocks) |i| {
        block_params_bufs[i] = try zml.io.load(
            model.Block0FullParams,
            &block_params_shape.blocks[i],
            allocator,
            io,
            platform,
            &ckpt_store,
            .{
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
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
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
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer model.OutputProjection.Params.unloadBuffers(&proj_a_bufs);
    std.log.info("All Stage 2 weights loaded.", .{});
    timer.addLoad();

    // ---- Denoising loop: N-step Euler ----
    const num_steps = stage2_sigmas.len - 1;
    std.log.info("Starting {d}-step denoising loop...", .{num_steps});

    for (0..num_steps) |step_idx| {
        const step_start: std.Io.Timestamp = .now(io, .awake);
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
            v_latent_buf,       a_latent_buf,
            bridge.v_mask,      bridge.a_mask,
            sigma_1d,           sigma_1d,
            bridge.v_positions, bridge.a_positions,
            bridge.v_context,   bridge.a_context,
            preprocess_bufs,
        });
        preprocess_exe.callOpts(io, pre_args, &pre_results, .{ .wait = true });
        var pre_out = pre_results.get(zml.Bufferized(model.PreprocessOutput));

        // ---- 2. Block chain ----
        std.log.info("  Running {d}-block chain...", .{model.num_transformer_blocks});
        var h_v = pre_out.vx;
        var h_a = pre_out.ax;

        {
            var blk_args = try block_exe.args(allocator);
            defer blk_args.deinit(allocator);
            var blk_results = try block_exe.results(allocator);
            defer blk_results.deinit(allocator);

            for (0..model.num_transformer_blocks) |i| {
                blk_args.set(.{
                    h_v,                          h_a,
                    pre_out.video_timesteps,      pre_out.audio_timesteps,
                    pre_out.video_timesteps_zero, pre_out.audio_timesteps_zero,
                    pre_out.v_denoise_mask,       pre_out.a_denoise_mask,
                    pre_out.v_prompt_timestep,    pre_out.a_prompt_timestep,
                    pre_out.v_pe_cos,             pre_out.v_pe_sin,
                    pre_out.a_pe_cos,             pre_out.a_pe_sin,
                    pre_out.v_text_ctx,           pre_out.a_text_ctx,
                    pre_out.v_cross_ss_ts,        pre_out.v_cross_gate_ts,
                    pre_out.a_cross_ss_ts,        pre_out.a_cross_gate_ts,
                    pre_out.a2v_pe_cos,           pre_out.a2v_pe_sin,
                    pre_out.a2v_k_pe_cos,         pre_out.a2v_k_pe_sin,
                    pre_out.v2a_pe_cos,           pre_out.v2a_pe_sin,
                    pre_out.v2a_k_pe_cos,         pre_out.v2a_k_pe_sin,
                    block_params_bufs[i],
                });
                block_exe.callOpts(io, blk_args, &blk_results, .{ .wait = true });

                const out = blk_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
                if (i > 0) {
                    h_v.deinit();
                    h_a.deinit();
                }
                h_v = out.vx_out;
                h_a = out.ax_out;
            }
        }

        // Free block-chain-only preprocessing outputs early — only v/a_embedded_timestep
        // are still needed for output projection. This reduces peak GPU memory.
        pre_out.vx.deinit();
        pre_out.ax.deinit();
        pre_out.video_timesteps.deinit();
        pre_out.audio_timesteps.deinit();
        pre_out.video_timesteps_zero.deinit();
        pre_out.audio_timesteps_zero.deinit();
        pre_out.v_denoise_mask.deinit();
        pre_out.a_denoise_mask.deinit();
        pre_out.v_prompt_timestep.deinit();
        pre_out.a_prompt_timestep.deinit();
        pre_out.v_pe_cos.deinit();
        pre_out.v_pe_sin.deinit();
        pre_out.a_pe_cos.deinit();
        pre_out.a_pe_sin.deinit();
        pre_out.v_text_ctx.deinit();
        pre_out.a_text_ctx.deinit();
        pre_out.v_cross_ss_ts.deinit();
        pre_out.v_cross_gate_ts.deinit();
        pre_out.a_cross_ss_ts.deinit();
        pre_out.a_cross_gate_ts.deinit();
        pre_out.a2v_pe_cos.deinit();
        pre_out.a2v_pe_sin.deinit();
        pre_out.a2v_k_pe_cos.deinit();
        pre_out.a2v_k_pe_sin.deinit();
        pre_out.v2a_pe_cos.deinit();
        pre_out.v2a_pe_sin.deinit();
        pre_out.v2a_k_pe_cos.deinit();
        pre_out.v2a_k_pe_sin.deinit();

        // ---- 3. Output projection ----
        std.log.info("  Running output projection...", .{});

        var proj_v_args = try proj_v_exe.args(allocator);
        defer proj_v_args.deinit(allocator);
        var proj_v_results = try proj_v_exe.results(allocator);
        defer proj_v_results.deinit(allocator);
        proj_v_args.set(.{ h_v, pre_out.v_embedded_timestep, proj_v_bufs });
        proj_v_exe.callOpts(io, proj_v_args, &proj_v_results, .{ .wait = true });
        var video_vel = proj_v_results.get(zml.Buffer);

        var proj_a_args = try proj_a_exe.args(allocator);
        defer proj_a_args.deinit(allocator);
        var proj_a_results = try proj_a_exe.results(allocator);
        defer proj_a_results.deinit(allocator);
        proj_a_args.set(.{ h_a, pre_out.a_embedded_timestep, proj_a_bufs });
        proj_a_exe.callOpts(io, proj_a_args, &proj_a_results, .{ .wait = true });
        var audio_vel = proj_a_results.get(zml.Buffer);

        h_v.deinit();
        h_a.deinit();
        // Free remaining preprocessing outputs (embedded timesteps).
        pre_out.v_embedded_timestep.deinit();
        pre_out.a_embedded_timestep.deinit();

        // ---- 4. Denoising step ----
        std.log.info("  Running denoising step...", .{});

        var dv_args = try denoise_v_exe.args(allocator);
        defer dv_args.deinit(allocator);
        var dv_results = try denoise_v_exe.results(allocator);
        defer dv_results.deinit(allocator);
        dv_args.set(.{ v_latent_buf, video_vel, bridge.v_mask, bridge.v_clean, sigma_buf, sigma_next_buf });
        denoise_v_exe.callOpts(io, dv_args, &dv_results, .{ .wait = true });
        const dv_out = dv_results.get(zml.Bufferized(model.DenoisingStepResult));

        var da_args = try denoise_a_exe.args(allocator);
        defer da_args.deinit(allocator);
        var da_results = try denoise_a_exe.results(allocator);
        defer da_results.deinit(allocator);
        da_args.set(.{ a_latent_buf, audio_vel, bridge.a_mask, bridge.a_clean, sigma_buf, sigma_next_buf });
        denoise_a_exe.callOpts(io, da_args, &da_results, .{ .wait = true });
        const da_out = da_results.get(zml.Bufferized(model.DenoisingStepResult));

        video_vel.deinit();
        audio_vel.deinit();

        v_latent_buf.deinit();
        a_latent_buf.deinit();
        v_latent_buf = dv_out.next_latent;
        a_latent_buf = da_out.next_latent;

        const step_ns = step_start.untilNow(io, .awake).nanoseconds;
        timer.recordStep(step_ns);
        std.log.info("  Step {d} complete ({d:.1}s).", .{ step_idx, PhaseTimer.fmtSecs(step_ns) });
    }
    timer.addExec();

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
    io: std.Io,
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
    exe.callOpts(io, args, &results, .{ .wait = true });
    return results.get(zml.Buffer);
}

fn runToDenoised(
    allocator: std.mem.Allocator,
    io: std.Io,
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
    exe.callOpts(io, args, &results, .{ .wait = true });
    return results.get(zml.Buffer);
}

// ============================================================================
// Text Embeddings from Gemma Hidden States
// ============================================================================

const TextEmbeddingsResult = struct {
    v_context_pos: zml.Buffer,
    a_context_pos: zml.Buffer,
    v_context_neg: zml.Buffer,
    a_context_neg: zml.Buffer,
};

/// Compile and run the text embeddings processor on both pos and neg hidden states.
/// Uses weights from the LTX checkpoint (text_embedding_projection.* and
/// model.diffusion_model.{video,audio}_embeddings_connector.*).
fn computeTextEmbeddings(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    ckpt_store: *zml.io.TensorStore,
    pos_path: []const u8,
    neg_path: []const u8,
) !TextEmbeddingsResult {
    // ---- Initialize processor params from checkpoint ----
    const proc_init = text_embeddings.EmbeddingsProcessor.initParams(ckpt_store.view());
    const processor = proc_init.processor;
    const proc_params = proc_init.params;

    // ---- Open pos hidden states to get shapes for compilation ----
    var pos_hs_buf: zml.Buffer = undefined;
    var pos_mask_buf: zml.Buffer = undefined;
    {
        var pos_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, pos_path) catch |err| {
            std.log.err("Failed to open pos hidden states: {s}", .{pos_path});
            return err;
        };
        defer pos_reg.deinit();
        var pos_store: zml.io.TensorStore = .fromRegistry(allocator, &pos_reg);
        defer pos_store.deinit();

        pos_hs_buf = try loadBuf(allocator, io, platform, &pos_store, "stacked_hidden_states", sharding);
        errdefer pos_hs_buf.deinit();
        pos_mask_buf = try loadBuf(allocator, io, platform, &pos_store, "attention_mask", sharding);
    }
    defer pos_hs_buf.deinit();
    defer pos_mask_buf.deinit();

    std.log.info("  pos stacked_hidden_states: {s} {any}", .{ @tagName(pos_hs_buf.shape().dtype()), pos_hs_buf.shape().dims() });
    std.log.info("  pos attention_mask:        {s} {any}", .{ @tagName(pos_mask_buf.shape().dtype()), pos_mask_buf.shape().dims() });

    // ---- Compile graph ----
    std.log.info("Compiling text embeddings processor...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        text_embeddings.forwardEmbeddingsProcessor,
        .{
            &processor,
            zml.Tensor.fromShape(pos_hs_buf.shape()),
            zml.Tensor.fromShape(pos_mask_buf.shape()),
            proc_params,
        },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();

    // ---- Load model weights ----
    std.log.info("Loading text embeddings weights...", .{});
    var weight_bufs = try zml.io.load(
        text_embeddings.EmbeddingsProcessor.Params,
        &proc_params,
        allocator,
        io,
        platform,
        ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer text_embeddings.EmbeddingsProcessor.unloadBuffers(&weight_bufs);

    // ---- Run on positive hidden states ----
    std.log.info("Running text embeddings on positive prompt...", .{});
    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    exe_args.set(.{ pos_hs_buf, pos_mask_buf, weight_bufs });
    exe.callOpts(io, exe_args, &results, .{ .wait = true });

    var pos_out = results.get(zml.Bufferized(text_embeddings.EmbeddingsProcessor.Result));
    const v_context_pos = pos_out.v_context;
    const a_context_pos = pos_out.a_context;
    pos_out.binary_mask.deinit();

    std.log.info("  v_context_pos: {s} {any}", .{ @tagName(v_context_pos.shape().dtype()), v_context_pos.shape().dims() });
    std.log.info("  a_context_pos: {s} {any}", .{ @tagName(a_context_pos.shape().dtype()), a_context_pos.shape().dims() });

    // ---- Run on negative hidden states ----
    var neg_hs_buf: zml.Buffer = undefined;
    var neg_mask_buf: zml.Buffer = undefined;
    {
        std.log.info("Running text embeddings on negative prompt...", .{});
        var neg_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, neg_path) catch |err| {
            std.log.err("Failed to open neg hidden states: {s}", .{neg_path});
            return err;
        };
        defer neg_reg.deinit();
        var neg_store: zml.io.TensorStore = .fromRegistry(allocator, &neg_reg);
        defer neg_store.deinit();

        neg_hs_buf = try loadBuf(allocator, io, platform, &neg_store, "stacked_hidden_states", sharding);
        errdefer neg_hs_buf.deinit();
        neg_mask_buf = try loadBuf(allocator, io, platform, &neg_store, "attention_mask", sharding);
    }
    defer neg_hs_buf.deinit();
    defer neg_mask_buf.deinit();

    // Reuse the same compiled exe — validate that neg shapes match pos (both must be S=1024).
    if (!neg_hs_buf.shape().eql(pos_hs_buf.shape())) {
        std.log.err(
            "Negative hidden states shape {any} ({s}) does not match positive {any} ({s}). " ++
                "Both must have the same shape/dtype for the compiled graph.",
            .{
                neg_hs_buf.shape().dims(), @tagName(neg_hs_buf.shape().dtype()),
                pos_hs_buf.shape().dims(), @tagName(pos_hs_buf.shape().dtype()),
            },
        );
        return error.ShapeMismatch;
    }
    if (!neg_mask_buf.shape().eql(pos_mask_buf.shape())) {
        std.log.err(
            "Negative attention mask shape {any} ({s}) does not match positive {any} ({s}). " ++
                "Both must have the same shape/dtype for the compiled graph.",
            .{
                neg_mask_buf.shape().dims(), @tagName(neg_mask_buf.shape().dtype()),
                pos_mask_buf.shape().dims(), @tagName(pos_mask_buf.shape().dtype()),
            },
        );
        return error.ShapeMismatch;
    }
    exe_args.set(.{ neg_hs_buf, neg_mask_buf, weight_bufs });
    exe.callOpts(io, exe_args, &results, .{ .wait = true });

    var neg_out = results.get(zml.Bufferized(text_embeddings.EmbeddingsProcessor.Result));
    const v_context_neg = neg_out.v_context;
    const a_context_neg = neg_out.a_context;
    neg_out.binary_mask.deinit();

    std.log.info("  v_context_neg: {s} {any}", .{ @tagName(v_context_neg.shape().dtype()), v_context_neg.shape().dims() });
    std.log.info("  a_context_neg: {s} {any}", .{ @tagName(a_context_neg.shape().dtype()), a_context_neg.shape().dims() });

    return .{
        .v_context_pos = v_context_pos,
        .a_context_pos = a_context_pos,
        .v_context_neg = v_context_neg,
        .a_context_neg = a_context_neg,
    };
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
    timer: *PhaseTimer,
) !VideoFrames {
    timer.start();

    // Ensure the patchified input is freed on error paths.
    var patchified_input = v_latent_patchified;
    errdefer patchified_input.deinit();

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
    errdefer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    errdefer ckpt_store.deinit();

    // ========================================================================
    // Step 1: Unpatchify latent — [1, T_v, 128] → [1, 128, F, H, W]
    // ========================================================================
    std.log.info("Compiling unpatchify for VAE input...", .{});
    const patchified_shape = zml.Shape.init(.{ 1, T_v, C }, .bf16);
    const video_5d_shape = zml.Shape.init(.{ 1, C, F, H, W }, .bf16);

    var unpatch_exe = try platform.compileFn(
        allocator,
        io,
        upsampler.forwardUnpatchifyVideo,
        .{
            zml.Tensor.fromShape(patchified_shape),
            video_5d_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    errdefer unpatch_exe.deinit();
    timer.addCompile();

    std.log.info("Running unpatchify...", .{});
    var unpatch_args = try unpatch_exe.args(allocator);
    errdefer unpatch_args.deinit(allocator);
    var unpatch_results = try unpatch_exe.results(allocator);
    unpatch_args.set(.{v_latent_patchified});
    unpatch_exe.callOpts(io, unpatch_args, &unpatch_results, .{ .wait = true });
    var v_latent_5d = unpatch_results.get(zml.Buffer);
    std.log.info("  Unpatchified: {any}", .{v_latent_5d.shape().dims()});
    timer.addExec();

    // Free unpatchify resources — exe, args, results no longer needed.
    unpatch_args.deinit(allocator);
    unpatch_results.deinit(allocator);
    unpatch_exe.deinit();

    // Free the patchified input — no longer needed after unpatchify.
    patchified_input.deinit();

    // ========================================================================
    // Step 2: Tiling decision — for tiled path, download latent to host
    // and free GPU buffer BEFORE loading VAE weights to minimize peak memory.
    // ========================================================================
    const tiling_config: video_vae.TemporalTilingConfig = .{};
    const use_tiling = F > tiling_config.tile_latent_frames;

    // For tiled path: download to host and free GPU latent early
    var latent_host: ?zml.Slice = null;
    if (use_tiling) {
        std.log.info("Temporal tiling enabled: F'={d}, tile={d}, overlap={d}, stride={d}", .{
            F, tiling_config.tile_latent_frames, tiling_config.overlap_latent_frames, tiling_config.stride(),
        });
        std.log.info("Downloading latent to host for tiling...", .{});
        latent_host = try v_latent_5d.toSliceAlloc(allocator, io);
        // Free 5D GPU latent — we have the host copy, tiles will be uploaded individually.
        v_latent_5d.deinit();
    }
    defer if (latent_host) |lh| lh.free(allocator);

    // ========================================================================
    // Step 3: Load VAE decoder weights + per-channel stats
    // ========================================================================
    std.log.info("Loading VAE decoder weights...", .{});
    var vae_params = video_vae.initVideoVaeDecoderParams(ckpt_store.view());
    const vae_bufs = try zml.io.load(
        video_vae.VideoVaeDecoderParams,
        &vae_params,
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );
    defer {
        var vae_bufs_mut = vae_bufs;
        deinitBufferizedFields(&vae_bufs_mut);
    }

    std.log.info("Loading per-channel statistics...", .{});
    var stats_shape = conv_ops.initPerChannelStats(ckpt_store.view());
    const stats_bufs = try zml.io.load(
        conv_ops.PerChannelStats,
        &stats_shape,
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );
    defer {
        var stats_bufs_mut = stats_bufs;
        deinitBufferizedFields(&stats_bufs_mut);
    }

    timer.addLoad(); // VAE + stats weights

    // Free checkpoint resources — no longer needed after weight loading.
    ckpt_store.deinit();
    ckpt_reg.deinit();

    // ========================================================================
    // Step 4: Dispatch to tiled or single-shot decode
    // ========================================================================
    if (use_tiling) {
        return runVideoVaeDecodeTiled(
            allocator,
            io,
            platform,
            sharding,
            latent_host.?,
            vae_params,
            vae_bufs,
            stats_shape,
            stats_bufs,
            F,
            H,
            W,
            tiling_config,
            timer,
        );
    }

    // ========================================================================
    // Fast path: single decode, no tiling
    // ========================================================================
    std.log.info("Compiling VAE decoder (single, no tiling)...", .{});
    var vae_exe = try platform.compileFn(
        allocator,
        io,
        video_vae.forwardVideoVaeDecode,
        .{
            zml.Tensor.fromShape(v_latent_5d.shape()),
            stats_shape,
            vae_params,
        },
        .{ .shardings = &.{sharding} },
    );
    defer vae_exe.deinit();
    timer.addCompile();

    std.log.info("Running VAE decode...", .{});
    var vae_args = try vae_exe.args(allocator);
    defer vae_args.deinit(allocator);
    var vae_results = try vae_exe.results(allocator);
    defer vae_results.deinit(allocator);
    vae_args.set(.{ v_latent_5d, stats_bufs, vae_bufs });
    vae_exe.callOpts(io, vae_args, &vae_results, .{ .wait = true });
    v_latent_5d.deinit(); // Free GPU latent after VAE consumes it
    var decoded_video = vae_results.get(zml.Buffer); // [1, 3, F_out, H_out, W_out] bf16
    std.log.info("  Decoded video: {any}", .{decoded_video.shape().dims()});
    timer.addExec();

    defer decoded_video.deinit(); // Free GPU buffer after download to host
    return postProcessDecodedVideo(allocator, io, decoded_video);
}

/// Tiled VAE decode path: decode overlapping temporal chunks and blend on host.
/// The latent has already been downloaded to host (latent_host) and the GPU buffer freed
/// to minimize peak GPU memory during VAE weight loading and execution.
fn runVideoVaeDecodeTiled(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    latent_host: zml.Slice,
    vae_params: video_vae.VideoVaeDecoderParams,
    vae_bufs: zml.Bufferized(video_vae.VideoVaeDecoderParams),
    stats_shape: conv_ops.PerChannelStats,
    stats_bufs: zml.Bufferized(conv_ops.PerChannelStats),
    F: i64,
    H: i64,
    W: i64,
    config: video_vae.TemporalTilingConfig,
    timer: *PhaseTimer,
) !VideoFrames {
    const C: i64 = 128;

    // Compute tile plan
    const tile_plan = video_vae.computeTemporalTiles(F, config);
    std.log.info("  Tile plan: {d} tiles", .{tile_plan.count});

    // Total pixel output dimensions
    const F_px: usize = @intCast(8 * (F - 1) + 1);
    const H_px: usize = @intCast(32 * H);
    const W_px: usize = @intCast(32 * W);

    // Compile VAE decoder once for the tile shape [1, 128, tile_latent_frames, H, W]
    const tile_latent_shape = zml.Shape.init(.{ 1, C, config.tile_latent_frames, H, W }, .bf16);
    std.log.info("Compiling VAE decoder for tile shape: {any}...", .{tile_latent_shape.dims()});
    var vae_exe = try platform.compileFn(
        allocator,
        io,
        video_vae.forwardVideoVaeDecode,
        .{
            zml.Tensor.fromShape(tile_latent_shape),
            stats_shape,
            vae_params,
        },
        .{ .shardings = &.{sharding} },
    );
    defer vae_exe.deinit();
    timer.addCompile();

    const latent_bytes = latent_host.constData();

    // Host accumulation buffers (f32 for precision during blending)
    // Layout: [F_px, H_px, W_px, 3] in NHWC for easy frame-sequential access
    const accum_len = F_px * H_px * W_px * 3;
    const pixel_accum = try allocator.alloc(f32, accum_len);
    defer allocator.free(pixel_accum);
    @memset(pixel_accum, 0.0);

    const weight_accum = try allocator.alloc(f32, F_px);
    defer allocator.free(weight_accum);
    @memset(weight_accum, 0.0);

    // Process each tile
    const tile_lat_frames_usize: usize = @intCast(config.tile_latent_frames);
    const tile_byte_size: usize = @intCast(1 * C * config.tile_latent_frames * H * W * 2); // bf16=2 bytes

    // Pre-allocate tile buffer, args, and results outside the loop (same shape every tile).
    const tile_buf = try allocator.alloc(u8, tile_byte_size);
    defer allocator.free(tile_buf);

    var vae_args = try vae_exe.args(allocator);
    defer vae_args.deinit(allocator);
    var vae_results = try vae_exe.results(allocator);
    defer vae_results.deinit(allocator);

    for (0..tile_plan.count) |tile_idx| {
        const tile = tile_plan.tiles[tile_idx];
        const actual_lat_frames: usize = @intCast(tile.lat_actual_end - tile.lat_start);

        std.log.info("  Tile {d}/{d}: latent [{d}..{d}), actual={d} frames, pixel [{d}..{d})", .{
            tile_idx + 1,
            tile_plan.count,
            tile.lat_start,
            tile.lat_actual_end,
            actual_lat_frames,
            tile.px_start,
            tile.px_end,
        });

        // Create tile latent on host: [1, 128, tile_latent_frames, H, W] bf16
        // Copy actual frames, zero-pad if last tile is shorter.
        @memset(tile_buf, 0); // zero-fill (handles padding for short last tile)

        // Copy frame data. Layout is [B=1, C=128, F, H, W] contiguous.
        // Offset for frame f in the full latent: f * C * H * W * 2 bytes,
        // but C is dim 1 (outermost after B), so frames are NOT contiguous
        // in the standard [B,C,F,H,W] layout. We need to copy per-channel.
        //
        // Actually, in [B, C, F, H, W] layout with C=128:
        //   stride of dim 2 (F) = H * W * 2 bytes
        //   stride of dim 1 (C) = F * H * W * 2 bytes
        // To slice frames [start..end) we need to copy, for each channel c:
        //   source: c * F_full * HW2 + start * HW2 .. + end * HW2
        //   dest:   c * tile_F * HW2 + 0 .. actual * HW2
        const HW2: usize = @intCast(H * W * 2);
        const F_full: usize = @intCast(F);
        const start: usize = @intCast(tile.lat_start);

        for (0..128) |c| {
            const src_base = c * F_full * HW2 + start * HW2;
            const dst_base = c * tile_lat_frames_usize * HW2;
            const copy_len = actual_lat_frames * HW2;
            @memcpy(tile_buf[dst_base..][0..copy_len], latent_bytes[src_base..][0..copy_len]);
        }

        // Upload tile to GPU
        const tile_slice = zml.Slice.init(tile_latent_shape, tile_buf);
        var tile_gpu = try zml.Buffer.fromSlice(io, platform, tile_slice, sharding);

        // Run VAE decoder
        vae_args.set(.{ tile_gpu, stats_bufs, vae_bufs });
        vae_exe.callOpts(io, vae_args, &vae_results, .{ .wait = true });
        var decoded_tile = vae_results.get(zml.Buffer); // [1, 3, F_tile_px, H_px, W_px] bf16
        defer decoded_tile.deinit();

        // Free GPU tile input immediately
        tile_gpu.deinit();

        // Download decoded tile to host
        const tile_decoded_slice = try decoded_tile.toSliceAlloc(allocator, io);
        defer tile_decoded_slice.free(allocator);
        const tile_decoded_bytes = tile_decoded_slice.constData();

        // Determine actual pixel frame count for this tile (crop padding)
        const tile_F_px_full: usize = @intCast(decoded_tile.shape().dim(2)); // 8*(tile_lat-1)+1
        const actual_px_frames: usize = @intCast(tile.px_end - tile.px_start);
        const tile_F_px = @min(tile_F_px_full, actual_px_frames);

        // Compute blend weights for this tile
        const blend_weights = try video_vae.computeBlendWeights(allocator, tile, config);
        defer allocator.free(blend_weights);

        // Accumulate into pixel_accum with blending
        // Decoded tile layout: [1, 3, F_tile_px, H_px, W_px] bf16
        // Target layout: [F_px, H_px, W_px, 3] f32
        const px_offset: usize = @intCast(tile.px_start);
        for (0..tile_F_px) |f| {
            const w = blend_weights[f];
            const global_f = px_offset + f;
            weight_accum[global_f] += w;

            for (0..H_px) |h| {
                for (0..W_px) |ww| {
                    for (0..3) |c| {
                        // Source: [1, 3, F, H, W] bf16
                        const src_idx = (c * tile_F_px_full * H_px * W_px + f * H_px * W_px + h * W_px + ww) * 2;
                        const bf16_bits = std.mem.readInt(u16, tile_decoded_bytes[src_idx..][0..2], .little);
                        const f32_bits: u32 = @as(u32, bf16_bits) << 16;
                        const val: f32 = @bitCast(f32_bits);

                        // Accumulate weighted value
                        const dst_idx = global_f * H_px * W_px * 3 + h * W_px * 3 + ww * 3 + c;
                        pixel_accum[dst_idx] += val * w;
                    }
                }
            }
        }
    }
    timer.addExec();

    // ========================================================================
    // Normalize and convert to u8
    // ========================================================================
    std.log.info("Normalizing blended frames...", .{});
    const num_pixels = F_px * H_px * W_px * 3;
    const frames_u8 = try allocator.alloc(u8, num_pixels);

    for (0..F_px) |f| {
        const w = weight_accum[f];
        if (w == 0.0) continue; // shouldn't happen
        const inv_w = 1.0 / w;
        for (0..H_px) |h| {
            for (0..W_px) |ww| {
                for (0..3) |c| {
                    const idx = f * H_px * W_px * 3 + h * W_px * 3 + ww * 3 + c;
                    const val = pixel_accum[idx] * inv_w;
                    // (val + 1) / 2 * 255, clamped
                    const normalized = @min(@max((val + 1.0) * 0.5, 0.0), 1.0);
                    frames_u8[idx] = @intFromFloat(normalized * 255.0);
                }
            }
        }
    }

    std.log.info("  Video frames: {d}x{d}, {d} frames (tiled)", .{ W_px, H_px, F_px });

    return .{
        .data = frames_u8,
        .width = W_px,
        .height = H_px,
        .num_frames = F_px,
        .allocator = allocator,
    };
}

/// Post-process a decoded video buffer to u8 frames (non-tiled fast path).
fn postProcessDecodedVideo(
    allocator: std.mem.Allocator,
    io: std.Io,
    decoded_video: zml.Buffer,
) !VideoFrames {
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
    // Clean up temp audio file on all exit paths (success and error).
    defer std.Io.Dir.deleteFile(.cwd(), io, audio_path) catch {};

    var size_buf: [32]u8 = undefined;
    const size_str = std.fmt.bufPrint(&size_buf, "{d}x{d}", .{ video.width, video.height }) catch unreachable;

    var fps_buf: [16]u8 = undefined;
    const fps_str = std.fmt.bufPrint(&fps_buf, "{d:.0}", .{fps}) catch unreachable;

    var ac_buf: [8]u8 = undefined;
    const ac_str = std.fmt.bufPrint(&ac_buf, "{d}", .{audio_channels}) catch unreachable;

    std.log.info("Encoding video+audio with ffmpeg → {s}", .{output_path});

    var child = try std.process.spawn(io, .{
        .argv = &.{
            "ffmpeg",    "-y",
            // Video input: raw RGB24 from stdin
            "-f",        "rawvideo",
            "-pix_fmt",  "rgb24",
            "-s",        size_str,
            "-r",        fps_str,
            "-i",        "pipe:0",
            // Audio input: interleaved f32le from file
            "-f",        "f32le",
            "-ar",       "48000",
            "-ac",       ac_str,
            "-i",        audio_path,
            // Output encoding
            "-c:v",      "libx264",
            "-pix_fmt",  "yuv420p",
            "-c:a",      "aac",
            "-b:a",      "192k",
            "-shortest", output_path,
        },
        .stdin = .pipe,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    errdefer {
        // Ensure ffmpeg is not orphaned on error: close stdin so it gets EOF, then reap.
        if (child.stdin) |s| {
            s.close(io);
            child.stdin = null;
        }
        _ = child.wait(io) catch {};
    }

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
    timer: *PhaseTimer,
) !zml.Buffer {
    timer.start();

    // Ensure the patchified input is freed on error paths.
    var patchified_input = a_latent_patchified;
    errdefer patchified_input.deinit();

    const T_aud = pipe_meta.stage2.t_audio;

    // ---- Open checkpoint store ----
    std.log.info("Opening checkpoint for audio VAE decoder...", .{});
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    errdefer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    errdefer ckpt_store.deinit();

    // ========================================================================
    // Step 1: Unpatchify audio latent — [1, T_aud, 128] → [1, 8, T_aud, 16]
    // ========================================================================
    std.log.info("Compiling unpatchify for audio input...", .{});
    const patchified_shape = zml.Shape.init(.{ 1, T_aud, 128 }, .bf16);

    var unpatch_exe = try platform.compileFn(
        allocator,
        io,
        audio_vae.forwardUnpatchifyAudio,
        .{
            zml.Tensor.fromShape(patchified_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    errdefer unpatch_exe.deinit();
    timer.addCompile();

    std.log.info("Running audio unpatchify...", .{});
    var unpatch_args = try unpatch_exe.args(allocator);
    errdefer unpatch_args.deinit(allocator);
    var unpatch_results = try unpatch_exe.results(allocator);
    unpatch_args.set(.{patchified_input});
    unpatch_exe.callOpts(io, unpatch_args, &unpatch_results, .{ .wait = true });
    patchified_input.deinit(); // Free patchified input — consumed by unpatchify
    var a_latent_4d = unpatch_results.get(zml.Buffer);
    errdefer a_latent_4d.deinit();
    std.log.info("  Unpatchified audio: {any}", .{a_latent_4d.shape().dims()});
    timer.addExec();

    // Free unpatchify resources — exe, args, results no longer needed.
    unpatch_results.deinit(allocator);
    unpatch_args.deinit(allocator);
    unpatch_exe.deinit();

    // ========================================================================
    // Step 2: Load audio VAE decoder weights + per-channel stats
    // ========================================================================
    std.log.info("Loading audio VAE decoder weights...", .{});
    var audio_vae_params = audio_vae.initAudioVaeDecoderParams(ckpt_store.view());
    const audio_vae_bufs = try zml.io.load(
        audio_vae.AudioVaeDecoderParams,
        &audio_vae_params,
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );
    errdefer {
        var audio_vae_bufs_mut = audio_vae_bufs;
        deinitBufferizedFields(&audio_vae_bufs_mut);
    }

    std.log.info("Loading audio per-channel statistics...", .{});
    var audio_stats_shape = audio_vae.initAudioPerChannelStats(ckpt_store.view());
    const audio_stats_bufs = try zml.io.load(
        audio_vae.AudioPerChannelStats,
        &audio_stats_shape,
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );
    errdefer {
        var audio_stats_bufs_mut = audio_stats_bufs;
        deinitBufferizedFields(&audio_stats_bufs_mut);
    }

    // Free checkpoint resources — no longer needed after weight loading.
    ckpt_store.deinit();
    ckpt_reg.deinit();

    // ========================================================================
    // Step 3: Compile and run the audio VAE decoder
    // ========================================================================
    timer.addLoad(); // audio VAE + stats weights
    std.log.info("Compiling audio VAE decoder...", .{});
    var audio_vae_exe = try platform.compileFn(
        allocator,
        io,
        audio_vae.forwardAudioVaeDecode,
        .{
            zml.Tensor.fromShape(a_latent_4d.shape()),
            audio_stats_shape,
            audio_vae_params,
        },
        .{ .shardings = &.{sharding} },
    );
    defer audio_vae_exe.deinit();
    timer.addCompile();

    std.log.info("Running audio VAE decode...", .{});
    var audio_vae_args = try audio_vae_exe.args(allocator);
    defer audio_vae_args.deinit(allocator);
    var audio_vae_results = try audio_vae_exe.results(allocator);
    defer audio_vae_results.deinit(allocator);
    audio_vae_args.set(.{ a_latent_4d, audio_stats_bufs, audio_vae_bufs });
    audio_vae_exe.callOpts(io, audio_vae_args, &audio_vae_results, .{ .wait = true });
    a_latent_4d.deinit(); // Free intermediate unpatchified latent
    const decoded_audio = audio_vae_results.get(zml.Buffer); // [1, 2, T_out, 64] bf16
    std.log.info("  Decoded audio mel: {any}", .{decoded_audio.shape().dims()});
    timer.addExec();

    // Free audio VAE weights — no longer needed after decode.
    {
        var audio_vae_bufs_mut = audio_vae_bufs;
        deinitBufferizedFields(&audio_vae_bufs_mut);
        var audio_stats_bufs_mut = audio_stats_bufs;
        deinitBufferizedFields(&audio_stats_bufs_mut);
    }

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
    timer: *PhaseTimer,
) !zml.Buffer {
    timer.start();

    // Take ownership of the input mel buffer (consistent with runVideoVaeDecode/runAudioVaeDecode).
    var mel_input = audio_mel;
    errdefer mel_input.deinit();

    // ---- Open checkpoint store ----
    std.log.info("Opening checkpoint for vocoder...", .{});
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    errdefer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    errdefer ckpt_store.deinit();

    // ---- Load vocoder weights (split: main 667, BWE pipeline 559) ----
    // Heap-allocate param structs — too large for the stack
    std.log.info("Loading main vocoder weights...", .{});
    const main_voc_params = try allocator.create(vocoder.MainVocoderParams);
    defer allocator.destroy(main_voc_params);
    vocoder.initMainVocoderParams(main_voc_params, ckpt_store.view().withPrefix("vocoder").withPrefix("vocoder"));
    const main_voc_bufs = try allocator.create(zml.Bufferized(vocoder.MainVocoderParams));
    defer allocator.destroy(main_voc_bufs);
    main_voc_bufs.* = try zml.io.load(
        vocoder.MainVocoderParams,
        main_voc_params,
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );
    errdefer deinitBufferizedFields(main_voc_bufs);

    std.log.info("Loading BWE pipeline weights...", .{});
    const bwe_params = try allocator.create(vocoder.BWEPipelineParams);
    defer allocator.destroy(bwe_params);
    vocoder.initBWEPipelineParams(bwe_params, ckpt_store.view());
    const bwe_bufs = try allocator.create(zml.Bufferized(vocoder.BWEPipelineParams));
    defer allocator.destroy(bwe_bufs);
    bwe_bufs.* = try zml.io.load(
        vocoder.BWEPipelineParams,
        bwe_params,
        allocator,
        io,
        platform,
        &ckpt_store,
        .{
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * 1024 * 1024,
        },
    );
    errdefer deinitBufferizedFields(bwe_bufs);

    // Free checkpoint resources — no longer needed after weight loading.
    ckpt_store.deinit();
    ckpt_reg.deinit();

    // ---- Stage 1: Main vocoder — mel → 16kHz waveform ----
    timer.addLoad(); // main + BWE weights
    std.log.info("Compiling main vocoder (input: {any})...", .{mel_input.shape().dims()});
    var main_voc_exe = try platform.compileFn(
        allocator,
        io,
        vocoder.forwardMainVocoder,
        .{ zml.Tensor.fromShape(mel_input.shape()), main_voc_params },
        .{ .shardings = &.{sharding} },
    );
    errdefer main_voc_exe.deinit();
    timer.addCompile();

    std.log.info("Running main vocoder...", .{});
    var main_voc_args = try main_voc_exe.args(allocator);
    errdefer main_voc_args.deinit(allocator);
    var main_voc_results = try main_voc_exe.results(allocator);
    main_voc_args.set(.{ mel_input, main_voc_bufs });
    main_voc_exe.callOpts(io, main_voc_args, &main_voc_results, .{ .wait = true });
    mel_input.deinit(); // Free input mel — consumed by main vocoder
    var waveform_16k = main_voc_results.get(zml.Buffer);
    errdefer waveform_16k.deinit();
    std.log.info("  16kHz waveform: {any}", .{waveform_16k.shape().dims()});
    timer.addExec();

    // Free main vocoder exe, args, results — no longer needed.
    main_voc_results.deinit(allocator);
    main_voc_args.deinit(allocator);
    main_voc_exe.deinit();

    // Free main vocoder weights — no longer needed, BWE uses separate weights.
    deinitBufferizedFields(main_voc_bufs);

    // ---- Stage 2: BWE pipeline — 16kHz → 48kHz ----
    std.log.info("Compiling BWE pipeline (input: {any})...", .{waveform_16k.shape().dims()});
    var bwe_exe = try platform.compileFn(
        allocator,
        io,
        vocoder.forwardBWEPipeline,
        .{ zml.Tensor.fromShape(waveform_16k.shape()), bwe_params },
        .{ .shardings = &.{sharding} },
    );
    defer bwe_exe.deinit();
    timer.addCompile();

    std.log.info("Running BWE pipeline...", .{});
    var bwe_args = try bwe_exe.args(allocator);
    defer bwe_args.deinit(allocator);
    var bwe_results = try bwe_exe.results(allocator);
    defer bwe_results.deinit(allocator);
    bwe_args.set(.{ waveform_16k, bwe_bufs });
    bwe_exe.callOpts(io, bwe_args, &bwe_results, .{ .wait = true });
    waveform_16k.deinit(); // Free intermediate 16kHz waveform
    const waveform = bwe_results.get(zml.Buffer);
    std.log.info("  48kHz waveform: {any}", .{waveform.shape().dims()});
    timer.addExec();

    // Free BWE weights — no longer needed.
    deinitBufferizedFields(bwe_bufs);

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

    std.log.info("  Wrote {s} ({d} bytes, dims {any})", .{ path, slice.constData().len, buf.shape().dims() });
}

// ============================================================================
// Load pipeline_meta.json (legacy path, used when --meta is provided)
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
// Compute PipelineMeta from generation params (no JSON file needed)
// ============================================================================

fn computePipelineMeta(height: u32, width: u32, num_frames: u32, fps: f64) PipelineMeta {
    const f_lat: i64 = @as(i64, @intCast((num_frames - 1) / 8)) + 1;
    // Stage 2: full resolution
    const h_lat_s2: i64 = @intCast(height / 32);
    const w_lat_s2: i64 = @intCast(width / 32);
    // Stage 1: half of Stage 2 (ceiling division so that 2*s1 >= s2,
    // enabling the 2× spatial upsampler to produce at least s2 resolution)
    const h_lat_s1: i64 = @divTrunc(h_lat_s2 + 1, 2);
    const w_lat_s1: i64 = @divTrunc(w_lat_s2 + 1, 2);
    // Audio tokens: round(num_frames / fps * 25)
    // Main vocoder produces 16kHz audio (BWE then upsamples it at 48kHz), with hop_length=160 (step size for the sliding window) and downsample_factor=4
    // Explanation for 25 tokens/second:
    // Constants: sample_rate=16000, hop_length=160, downsample_factor=4
    // → latents_per_second = 16000 / (160 * 4) = 25
    const duration: f64 = @as(f64, @floatFromInt(num_frames)) / fps;
    const t_audio: i64 = @intFromFloat(@round(duration * 25.0));

    return .{
        .frame_rate = fps,
        .stage1 = .{
            .h_lat = h_lat_s1,
            .w_lat = w_lat_s1,
            .f_lat = f_lat,
            .t_audio = t_audio,
        },
        .stage2 = .{
            .h_lat = h_lat_s2,
            .w_lat = w_lat_s2,
            .f_lat = f_lat,
            .t_audio = t_audio,
        },
    };
}

// ============================================================================
// Video positions: [1, 3, T_v, 2] bf16
// ============================================================================

fn computeVideoPositions(allocator: std.mem.Allocator, F: i64, H: i64, W: i64, fps: f64) ![]u8 {
    const T_v: usize = @intCast(F * H * W);
    const num_vals = 3 * T_v * 2; // 3 axes (frame, height, width) × 2 values (start, end) per axis
    const out = try allocator.alloc(u8, num_vals * 2);

    const scale_factors = [3]f32{ 8.0, 32.0, 32.0 }; // 1 latent unit = 8 frames temporally, 32x32 pixels spatially
    const fps_f32: f32 = @floatCast(fps);
    const Fi: usize = @intCast(F);
    const Hi: usize = @intCast(H);
    const Wi: usize = @intCast(W);

    for (0..Fi) |f| {
        for (0..Hi) |h| {
            for (0..Wi) |w| {
                const patch_idx = f * Hi * Wi + h * Wi + w; // 3 dimensions tiled in order (frame, height, width)

                const coords = [3][2]f32{
                    .{ @floatFromInt(f), @floatFromInt(f + 1) },
                    .{ @floatFromInt(h), @floatFromInt(h + 1) },
                    .{ @floatFromInt(w), @floatFromInt(w + 1) },
                };

                for (0..3) |axis| {
                    // Compute start and end positions for this axis, scaled to the **latent space**
                    var start = coords[axis][0] * scale_factors[axis];
                    var end = coords[axis][1] * scale_factors[axis];

                    // Temporal axis: adjust for causal-conv offset (first latent covers 1 frame, not 8), then convert from frame indices to seconds.
                    if (axis == 0) {
                        start = @max(start + 1.0 - scale_factors[0], 0.0);
                        end = @max(end + 1.0 - scale_factors[0], 0.0);

                        start /= fps_f32;
                        end /= fps_f32;
                    }

                    const base = (axis * T_v * 2 + patch_idx * 2) * 2; // 2 values (start, end) per axis, 2 bytes per bf16
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
