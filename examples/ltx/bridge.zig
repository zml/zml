/// Stage 1 → Stage 2 Bridge — replaces bridge_s1_to_s2.py.
///
/// Takes Stage 1 denoised outputs (video_latent.bin, audio_latent.bin),
/// runs unpatchify → upsample → re-patchify, computes positions/masks,
/// noises latents, and writes all 12 tensors to stage2_inputs.safetensors.
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:bridge -- \
///       --stage1-video /root/mixed/stage1_out/video_latent.bin \
///       --stage1-audio /root/mixed/stage1_out/audio_latent.bin \
///       --stage2-noise /root/mixed/stage2_noise.safetensors \
///       --stage1-inputs /root/mixed/stage1_inputs.safetensors \
///       --meta /root/mixed/pipeline_meta.json \
///       --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
///       --main-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
///       --output /root/mixed/stage2_inputs.safetensors \
///       --ref /root/mixed/ref_stage2_inputs.safetensors

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const conv_ops = @import("conv_ops.zig");
const upsampler = @import("upsampler.zig");

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
    sigma_0: f64 = 0,
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
    stage1_video: []const u8,
    stage1_audio: []const u8,
    stage2_noise: []const u8,
    stage1_inputs: []const u8,
    meta: []const u8,
    upsampler_ckpt: []const u8,
    main_ckpt: []const u8,
    output: []const u8,
    ref: ?[]const u8,
};

fn parseArgs(it: anytype) !CliArgs {
    var args: CliArgs = .{
        .stage1_video = undefined,
        .stage1_audio = undefined,
        .stage2_noise = undefined,
        .stage1_inputs = undefined,
        .meta = undefined,
        .upsampler_ckpt = undefined,
        .main_ckpt = undefined,
        .output = undefined,
        .ref = null,
    };
    var have_s1v = false;
    var have_s1a = false;
    var have_noise = false;
    var have_s1i = false;
    var have_meta = false;
    var have_up = false;
    var have_main = false;
    var have_out = false;

    _ = it.next(); // exe name

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--stage1-video")) {
            args.stage1_video = it.next() orelse return error.InvalidArgs;
            have_s1v = true;
        } else if (std.mem.eql(u8, arg, "--stage1-audio")) {
            args.stage1_audio = it.next() orelse return error.InvalidArgs;
            have_s1a = true;
        } else if (std.mem.eql(u8, arg, "--stage2-noise")) {
            args.stage2_noise = it.next() orelse return error.InvalidArgs;
            have_noise = true;
        } else if (std.mem.eql(u8, arg, "--stage1-inputs")) {
            args.stage1_inputs = it.next() orelse return error.InvalidArgs;
            have_s1i = true;
        } else if (std.mem.eql(u8, arg, "--meta")) {
            args.meta = it.next() orelse return error.InvalidArgs;
            have_meta = true;
        } else if (std.mem.eql(u8, arg, "--upsampler-ckpt")) {
            args.upsampler_ckpt = it.next() orelse return error.InvalidArgs;
            have_up = true;
        } else if (std.mem.eql(u8, arg, "--main-ckpt")) {
            args.main_ckpt = it.next() orelse return error.InvalidArgs;
            have_main = true;
        } else if (std.mem.eql(u8, arg, "--output")) {
            args.output = it.next() orelse return error.InvalidArgs;
            have_out = true;
        } else if (std.mem.eql(u8, arg, "--ref")) {
            args.ref = it.next() orelse return error.InvalidArgs;
        }
    }

    if (!have_s1v or !have_s1a or !have_noise or !have_s1i or !have_meta or !have_up or !have_main or !have_out) {
        std.log.err(
            "Usage: bridge --stage1-video <path> --stage1-audio <path> " ++
                "--stage2-noise <path> --stage1-inputs <path> --meta <path> " ++
                "--upsampler-ckpt <path> --main-ckpt <path> --output <path> [--ref <path>]",
            .{},
        );
        return error.InvalidArgs;
    }

    return args;
}

// ============================================================================
// Main
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("LTX-2.3 Stage 1 → Stage 2 Bridge", .{});

    // ---- Parse CLI ----
    var it = init.minimal.args.iterate();
    const args = try parseArgs(&it);

    std.log.info("  stage1-video:    {s}", .{args.stage1_video});
    std.log.info("  stage1-audio:    {s}", .{args.stage1_audio});
    std.log.info("  stage2-noise:    {s}", .{args.stage2_noise});
    std.log.info("  stage1-inputs:   {s}", .{args.stage1_inputs});
    std.log.info("  meta:            {s}", .{args.meta});
    std.log.info("  upsampler-ckpt:  {s}", .{args.upsampler_ckpt});
    std.log.info("  main-ckpt:       {s}", .{args.main_ckpt});
    std.log.info("  output:          {s}", .{args.output});

    // ---- Load pipeline_meta.json ----
    std.log.info("Loading pipeline metadata...", .{});
    const pipe_meta = try loadPipelineMeta(allocator, io, args.meta);
    const s1 = pipe_meta.stage1;
    const s2 = pipe_meta.stage2;
    const fps = pipe_meta.frame_rate;
    const sigma_0: f32 = @floatCast(s2.sigma_0);

    std.log.info("  Stage 1: F={d} H={d} W={d} T_a={d}", .{ s1.f_lat, s1.h_lat, s1.w_lat, s1.t_audio });
    std.log.info("  Stage 2: F={d} H={d} W={d} T_a={d} sigma_0={d:.6}", .{ s2.f_lat, s2.h_lat, s2.w_lat, s2.t_audio, sigma_0 });
    std.log.info("  fps={d:.1}", .{fps});

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
    std.log.info("Opening checkpoint stores...", .{});
    var up_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, args.upsampler_ckpt) catch |err| {
        std.log.err("Failed to open upsampler checkpoint: {s}", .{args.upsampler_ckpt});
        return err;
    };
    defer up_reg.deinit();
    var up_store: zml.io.TensorStore = .fromRegistry(allocator, &up_reg);
    defer up_store.deinit();

    var main_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, args.main_ckpt) catch |err| {
        std.log.err("Failed to open main checkpoint: {s}", .{args.main_ckpt});
        return err;
    };
    defer main_reg.deinit();
    var main_store: zml.io.TensorStore = .fromRegistry(allocator, &main_reg);
    defer main_store.deinit();

    var noise_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, args.stage2_noise) catch |err| {
        std.log.err("Failed to open stage2 noise: {s}", .{args.stage2_noise});
        return err;
    };
    defer noise_reg.deinit();
    var noise_store: zml.io.TensorStore = .fromRegistry(allocator, &noise_reg);
    defer noise_store.deinit();

    var s1_inputs_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, args.stage1_inputs) catch |err| {
        std.log.err("Failed to open stage1 inputs: {s}", .{args.stage1_inputs});
        return err;
    };
    defer s1_inputs_reg.deinit();
    var s1_inputs_store: zml.io.TensorStore = .fromRegistry(allocator, &s1_inputs_reg);
    defer s1_inputs_store.deinit();

    // ---- Init platform ----
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ========================================================================
    // Step 1-2: Load video latent → unpatchify → upsample
    // ========================================================================
    std.log.info("Loading Stage 1 video latent...", .{});
    const patchified_shape = zml.Shape.init(.{ 1, T_v1, C }, .bf16);
    const video_host = try readRawFile(allocator, io, args.stage1_video, patchified_shape.byteSize());
    defer allocator.free(video_host);

    var patchified_buf = try zml.Buffer.fromBytes(io, platform, patchified_shape, sharding, video_host);
    defer patchified_buf.deinit();
    std.log.info("  Loaded video: {any}", .{patchified_shape});

    // Unpatchify: [1, T_v1, 128] → [1, 128, F, H_s1, W_s1]
    const video_5d_s1_shape = zml.Shape.init(.{ 1, C, F, H_s1, W_s1 }, .bf16);

    std.log.info("Compiling unpatchify...", .{});
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
    unpatch_args.set(.{patchified_buf});
    unpatch_exe.call(unpatch_args, &unpatch_results);
    var video_5d_buf = unpatch_results.get(zml.Buffer);
    defer video_5d_buf.deinit();
    std.log.info("  Unpatchified: {any}", .{video_5d_buf.shape()});

    // Upsample: [1, 128, F, H_s1, W_s1] → [1, 128, F, H_s2, W_s2]
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
    std.log.info("  Upsampled: {any}", .{upsampled_buf.shape()});

    // ========================================================================
    // Step 3: Re-patchify video for Stage 2
    // [1, 128, F, H_s2, W_s2] → [1, T_v2, 128]
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
    defer video_clean_buf.deinit();
    std.log.info("  Re-patchified video: {any}", .{video_clean_buf.shape()});

    // ========================================================================
    // Step 4: Audio passthrough — load as-is
    // ========================================================================
    std.log.info("Loading Stage 1 audio latent...", .{});
    const audio_shape = zml.Shape.init(.{ 1, T_a, C }, .bf16);
    const audio_host = try readRawFile(allocator, io, args.stage1_audio, audio_shape.byteSize());
    defer allocator.free(audio_host);
    var audio_clean_buf = try zml.Buffer.fromBytes(io, platform, audio_shape, sharding, audio_host);
    defer audio_clean_buf.deinit();
    std.log.info("  Audio clean: {any}", .{audio_clean_buf.shape()});

    // ========================================================================
    // Step 5: Compute positions and masks on host
    // ========================================================================
    std.log.info("Computing video positions...", .{});
    const video_pos_bytes = try computeVideoPositions(allocator, F, H_s2, W_s2, fps);
    defer allocator.free(video_pos_bytes);
    const video_pos_shape = zml.Shape.init(.{ 1, 3, T_v2, 2 }, .bf16);
    var video_pos_buf = try zml.Buffer.fromBytes(io, platform, video_pos_shape, sharding, video_pos_bytes);
    defer video_pos_buf.deinit();
    std.log.info("  Video positions: {any}", .{video_pos_buf.shape()});

    std.log.info("Computing audio positions...", .{});
    const audio_pos_bytes = try computeAudioPositions(allocator, T_a);
    defer allocator.free(audio_pos_bytes);
    const audio_pos_shape = zml.Shape.init(.{ 1, 1, T_a, 2 }, .f32);
    var audio_pos_buf = try zml.Buffer.fromBytes(io, platform, audio_pos_shape, sharding, audio_pos_bytes);
    defer audio_pos_buf.deinit();
    std.log.info("  Audio positions: {any}", .{audio_pos_buf.shape()});

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
    defer v_mask_buf.deinit();
    var a_mask_buf = try zml.Buffer.fromBytes(io, platform, a_mask_shape, sharding, a_mask_host);
    defer a_mask_buf.deinit();

    // ========================================================================
    // Step 6: Load noise and text contexts, then noise the latents
    // ========================================================================
    std.log.info("Loading noise tensors...", .{});
    var v_noise_buf = try loadBuf(allocator, io, platform, &noise_store, "video_noise_s2", sharding);
    defer v_noise_buf.deinit();
    var a_noise_buf = try loadBuf(allocator, io, platform, &noise_store, "audio_noise_s2", sharding);
    defer a_noise_buf.deinit();
    std.log.info("  video_noise: {any}", .{v_noise_buf.shape()});
    std.log.info("  audio_noise: {any}", .{a_noise_buf.shape()});

    std.log.info("Loading text contexts...", .{});
    var v_context_buf = try loadBuf(allocator, io, platform, &s1_inputs_store, "v_context_pos", sharding);
    defer v_context_buf.deinit();
    var a_context_buf = try loadBuf(allocator, io, platform, &s1_inputs_store, "a_context_pos", sharding);
    defer a_context_buf.deinit();
    std.log.info("  v_context: {any}", .{v_context_buf.shape()});
    std.log.info("  a_context: {any}", .{a_context_buf.shape()});

    // Compile and run noise init
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

    var ni_v_args = try noise_init_v_exe.args(allocator);
    defer ni_v_args.deinit(allocator);
    var ni_v_results = try noise_init_v_exe.results(allocator);
    defer ni_v_results.deinit(allocator);
    ni_v_args.set(.{ video_clean_buf, v_noise_buf, v_mask_buf, sigma0_buf });
    noise_init_v_exe.call(ni_v_args, &ni_v_results);
    var v_latent_buf = ni_v_results.get(zml.Buffer);
    defer v_latent_buf.deinit();

    var ni_a_args = try noise_init_a_exe.args(allocator);
    defer ni_a_args.deinit(allocator);
    var ni_a_results = try noise_init_a_exe.results(allocator);
    defer ni_a_results.deinit(allocator);
    ni_a_args.set(.{ audio_clean_buf, a_noise_buf, a_mask_buf, sigma0_buf });
    noise_init_a_exe.call(ni_a_args, &ni_a_results);
    var a_latent_buf = ni_a_results.get(zml.Buffer);
    defer a_latent_buf.deinit();

    std.log.info("  video_latent (noised): {any}", .{v_latent_buf.shape()});
    std.log.info("  audio_latent (noised): {any}", .{a_latent_buf.shape()});

    // ========================================================================
    // Write all 12 tensors to stage2_inputs.safetensors
    // ========================================================================
    std.log.info("Downloading tensors to host...", .{});

    const v_latent_slice = try v_latent_buf.toSliceAlloc(allocator, io);
    defer v_latent_slice.free(allocator);
    const a_latent_slice = try a_latent_buf.toSliceAlloc(allocator, io);
    defer a_latent_slice.free(allocator);
    const v_noise_slice = try v_noise_buf.toSliceAlloc(allocator, io);
    defer v_noise_slice.free(allocator);
    const a_noise_slice = try a_noise_buf.toSliceAlloc(allocator, io);
    defer a_noise_slice.free(allocator);
    const v_clean_slice = try video_clean_buf.toSliceAlloc(allocator, io);
    defer v_clean_slice.free(allocator);
    const a_clean_slice = try audio_clean_buf.toSliceAlloc(allocator, io);
    defer a_clean_slice.free(allocator);
    const v_mask_slice = try v_mask_buf.toSliceAlloc(allocator, io);
    defer v_mask_slice.free(allocator);
    const a_mask_slice = try a_mask_buf.toSliceAlloc(allocator, io);
    defer a_mask_slice.free(allocator);
    // Video/audio positions already in host bytes
    const v_ctx_slice = try v_context_buf.toSliceAlloc(allocator, io);
    defer v_ctx_slice.free(allocator);
    const a_ctx_slice = try a_context_buf.toSliceAlloc(allocator, io);
    defer a_ctx_slice.free(allocator);

    std.log.info("Writing stage2_inputs.safetensors...", .{});

    const entries = [_]SafetensorEntry{
        .{ .name = "video_latent", .shape = v_latent_buf.shape(), .data = v_latent_slice.constData() },
        .{ .name = "audio_latent", .shape = a_latent_buf.shape(), .data = a_latent_slice.constData() },
        .{ .name = "video_noise", .shape = v_noise_buf.shape(), .data = v_noise_slice.constData() },
        .{ .name = "audio_noise", .shape = a_noise_buf.shape(), .data = a_noise_slice.constData() },
        .{ .name = "video_clean_latent", .shape = video_clean_buf.shape(), .data = v_clean_slice.constData() },
        .{ .name = "audio_clean_latent", .shape = audio_clean_buf.shape(), .data = a_clean_slice.constData() },
        .{ .name = "video_denoise_mask", .shape = v_mask_buf.shape(), .data = v_mask_slice.constData() },
        .{ .name = "audio_denoise_mask", .shape = a_mask_buf.shape(), .data = a_mask_slice.constData() },
        .{ .name = "video_positions", .shape = video_pos_buf.shape(), .data = video_pos_bytes },
        .{ .name = "audio_positions", .shape = audio_pos_buf.shape(), .data = audio_pos_bytes },
        .{ .name = "v_context", .shape = v_context_buf.shape(), .data = v_ctx_slice.constData() },
        .{ .name = "a_context", .shape = a_context_buf.shape(), .data = a_ctx_slice.constData() },
    };

    try writeSafetensors(allocator, io, args.output, &entries);
    std.log.info("Done. Output: {s}", .{args.output});

    // ========================================================================
    // Optional: compare against reference
    // ========================================================================
    if (args.ref) |ref_path| {
        std.log.info("Comparing against reference: {s}", .{ref_path});
        try compareRef(allocator, io, platform, sharding, ref_path, &entries);
    }
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

    const JsonMeta = struct {
        frame_rate: f64,
        stage1: struct {
            h_lat: i64,
            w_lat: i64,
            f_lat: i64,
            t_audio: i64,
            sigma_0: f64 = 0,
        },
        stage2: struct {
            h_lat: i64,
            w_lat: i64,
            f_lat: i64,
            t_audio: i64,
            sigma_0: f64 = 0,
        },
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
            .sigma_0 = v.stage1.sigma_0,
        },
        .stage2 = .{
            .h_lat = v.stage2.h_lat,
            .w_lat = v.stage2.w_lat,
            .f_lat = v.stage2.f_lat,
            .t_audio = v.stage2.t_audio,
            .sigma_0 = v.stage2.sigma_0,
        },
    };
}

// ============================================================================
// Video positions: [1, 3, T_v, 2] bf16
//
// Mirrors PyTorch's _get_video_latent_pixel_coords (patchifiers.py):
//   latent_coords = meshgrid(arange(F), arange(H), arange(W), dtype=f32)
//   pixel_coords  = latent_coords * scale_factor        (f32 mul)
//   pixel_coords[time] = (pixel_coords[time] + 1 - 8).clamp(0)  (causal fix)
//   pixel_coords[time] /= fps                           (f32 div)
//   result = pixel_coords.to(torch.bfloat16)
//
// The reference is generated on GPU (CUDA). Division (e.g. / fps) on CUDA
// can differ by 1 ULP from CPU due to hardware FMA-based reciprocal.
// However, the bf16 conversion truncates the lower 16 mantissa bits where
// such 1-ULP f32 differences live, so GPU and CPU results are bitwise
// identical after bf16 quantization.
// ============================================================================

fn computeVideoPositions(allocator: std.mem.Allocator, F: i64, H: i64, W: i64, fps: f64) ![]u8 {
    const T_v: usize = @intCast(F * H * W);
    // 3 axes × T_v patches × 2 (start, end) = 6 * T_v bf16 values
    const num_vals = 3 * T_v * 2;
    const out = try allocator.alloc(u8, num_vals * 2); // bf16 = 2 bytes

    // Use f32 intermediates to match Python (PyTorch default float arithmetic).
    const scale_factors = [3]f32{ 8.0, 32.0, 32.0 };
    const fps_f32: f32 = @floatCast(fps);
    const Fi: usize = @intCast(F);
    const Hi: usize = @intCast(H);
    const Wi: usize = @intCast(W);

    // Layout: [3, T_v, 2] where axis 0 = time, 1 = height, 2 = width
    // T_v patches indexed as (f, h, w) in row-major order
    for (0..Fi) |f| {
        for (0..Hi) |h| {
            for (0..Wi) |w| {
                const patch_idx = f * Hi * Wi + h * Wi + w;

                // For each axis: compute start and end in latent coords,
                // scale to pixel coords, apply causal fix for time, divide time by fps
                const coords = [3][2]f32{
                    .{ @floatFromInt(f), @floatFromInt(f + 1) }, // time
                    .{ @floatFromInt(h), @floatFromInt(h + 1) }, // height
                    .{ @floatFromInt(w), @floatFromInt(w + 1) }, // width
                };

                for (0..3) |axis| {
                    var start = coords[axis][0] * scale_factors[axis];
                    var end = coords[axis][1] * scale_factors[axis];

                    // Causal fix: time axis only
                    if (axis == 0) {
                        start = @max(start + 1.0 - scale_factors[0], 0.0);
                        end = @max(end + 1.0 - scale_factors[0], 0.0);
                        // Divide by fps
                        start /= fps_f32;
                        end /= fps_f32;
                    }

                    // Convert to bf16 and store
                    // bf16 index: [axis * T_v * 2 + patch_idx * 2 + {0,1}]
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
//
// Mirrors PyTorch's _get_audio_latent_time_in_sec (patchifiers.py):
//   audio_latent_frame = arange(shift, T+shift, dtype=f32)
//   audio_mel_frame    = audio_latent_frame * 4            (f32 mul)
//   audio_mel_frame    = (audio_mel_frame + 1 - 4).clip(0) (causal fix)
//   result             = audio_mel_frame * 160 / 16000     (f32 mul then f32 div)
//
// IMPORTANT: The computation uses two separate f32 ops (* 160, then / 16000),
// NOT a single * 0.01. This matters because 0.01 is not exactly representable
// in IEEE 754 and the two-op chain produces different rounding.
//
// Known limitation: The reference is generated on CUDA GPU, which uses
// FMA-based Newton-Raphson for f32 division. This produces results that
// differ by 1 ULP from CPU division for some inputs (e.g. 160/16000 = 0.01).
// Since audio_positions stays in f32 (unlike video_positions which goes to
// bf16), the 1-ULP difference is preserved and causes a bitwise mismatch
// vs the GPU-generated reference, despite cosim=1.0 and max_abs_diff=0.0.
// ============================================================================

fn computeAudioPositions(allocator: std.mem.Allocator, T_a: i64) ![]u8 {
    const T: usize = @intCast(T_a);
    // 1 axis × T_a × 2 = 2 * T_a f32 values
    const num_vals = T * 2;
    const out = try allocator.alloc(u8, num_vals * 4); // f32 = 4 bytes

    for (0..T) |i| {
        // Match PyTorch's exact f32 operation chain:
        //   audio_latent_frame = arange(0..T, dtype=f32)   (and +1 for end)
        //   audio_mel_frame = audio_latent_frame * 4
        //   audio_mel_frame = (audio_mel_frame + 1 - 4).clip(min=0)
        //   result = audio_mel_frame * 160 / 16000
        // All operations are f32.
        const fi: f32 = @floatFromInt(i);
        const fi_end: f32 = @floatFromInt(i + 1);
        const mel_start: f32 = @max(fi * 4.0 + 1.0 - 4.0, 0.0);
        const mel_end: f32 = @max(fi_end * 4.0 + 1.0 - 4.0, 0.0);
        const start_sec: f32 = mel_start * 160.0 / 16000.0;
        const end_sec: f32 = mel_end * 160.0 / 16000.0;

        // Layout: [T_a, 2] row-major f32
        const base = i * 2 * 4;
        storeF32(out[base..], start_sec);
        storeF32(out[base + 4 ..], end_sec);
    }

    return out;
}

// ============================================================================
// Safetensors writer
// ============================================================================

const SafetensorEntry = struct {
    name: []const u8,
    shape: zml.Shape,
    data: []const u8,
};

fn dtypeToString(dtype: zml.DataType) []const u8 {
    return switch (dtype) {
        .f64 => "F64",
        .f32 => "F32",
        .f16 => "F16",
        .bf16 => "BF16",
        .f8e4m3fn => "F8_E4M3",
        .i64 => "I64",
        .i32 => "I32",
        .i16 => "I16",
        .i8 => "I8",
        .u64 => "U64",
        .u32 => "U32",
        .u16 => "U16",
        .u8 => "U8",
        .bool => "BOOL",
        else => "F32",
    };
}

fn writeSafetensors(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    entries: []const SafetensorEntry,
) !void {
    // Build JSON header
    var json_buf: std.ArrayList(u8) = .empty;
    defer json_buf.deinit(allocator);

    try json_buf.appendSlice(allocator, "{");

    // Compute data offsets
    var data_offset: usize = 0;
    for (entries, 0..) |entry, idx| {
        if (idx > 0) try json_buf.appendSlice(allocator, ",");

        // "name": {"dtype": "BF16", "shape": [1, 24576, 128], "data_offsets": [0, 6291456]}
        try json_buf.appendSlice(allocator, "\"");
        try json_buf.appendSlice(allocator, entry.name);
        try json_buf.appendSlice(allocator, "\":{\"dtype\":\"");
        try json_buf.appendSlice(allocator, dtypeToString(entry.shape.dtype()));
        try json_buf.appendSlice(allocator, "\",\"shape\":[");

        const dims = entry.shape.dims();
        for (dims, 0..) |d, di| {
            if (di > 0) try json_buf.appendSlice(allocator, ",");
            var num_buf: [32]u8 = undefined;
            const num_str = std.fmt.bufPrint(&num_buf, "{d}", .{d}) catch unreachable;
            try json_buf.appendSlice(allocator, num_str);
        }

        try json_buf.appendSlice(allocator, "],\"data_offsets\":[");
        {
            var num_buf: [32]u8 = undefined;
            const start_str = std.fmt.bufPrint(&num_buf, "{d}", .{data_offset}) catch unreachable;
            try json_buf.appendSlice(allocator, start_str);
        }
        data_offset += entry.data.len;
        {
            try json_buf.appendSlice(allocator, ",");
            var num_buf: [32]u8 = undefined;
            const end_str = std.fmt.bufPrint(&num_buf, "{d}", .{data_offset}) catch unreachable;
            try json_buf.appendSlice(allocator, end_str);
        }
        try json_buf.appendSlice(allocator, "]}");
    }

    try json_buf.appendSlice(allocator, "}");

    // Pad JSON header to 8-byte alignment (safetensors convention)
    const header_len = json_buf.items.len;
    const padded_len = (header_len + 7) & ~@as(usize, 7);
    while (json_buf.items.len < padded_len) {
        try json_buf.append(allocator, ' ');
    }

    // Write file: [8-byte LE header length][JSON header][tensor data...]
    const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer file.close(io);

    var write_buf: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &write_buf);

    // Header length (8 bytes LE)
    const len_val: u64 = @intCast(json_buf.items.len);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, len_val, .little);
    try writer.interface.writeAll(&len_bytes);

    // JSON header
    try writer.interface.writeAll(json_buf.items);

    // Tensor data (concatenated in order)
    for (entries) |entry| {
        try writer.interface.writeAll(entry.data);
    }

    try writer.interface.flush();

    var total_data: usize = 0;
    for (entries) |entry| total_data += entry.data.len;
    std.log.info("  Wrote {s} (header={d}, data={d} bytes, {d} tensors)", .{
        path, json_buf.items.len, total_data, entries.len,
    });
}

// ============================================================================
// Reference comparison
// ============================================================================

fn compareRef(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    ref_path: []const u8,
    entries: []const SafetensorEntry,
) !void {
    var ref_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ref_path) catch |err| {
        std.log.err("Failed to open reference: {s}", .{ref_path});
        return err;
    };
    defer ref_reg.deinit();
    var ref_store: zml.io.TensorStore = .fromRegistry(allocator, &ref_reg);
    defer ref_store.deinit();

    for (entries) |entry| {
        const ref_shape = ref_store.view().getShape(entry.name) orelse {
            std.log.warn("  {s}: not found in reference", .{entry.name});
            continue;
        };

        if (!entry.shape.eqlDims(ref_shape)) {
            std.log.warn("  {s}: shape mismatch zig={any} ref={any}", .{ entry.name, entry.shape, ref_shape });
            continue;
        }

        var ref_buf = try loadBuf(allocator, io, platform, &ref_store, entry.name, sharding);
        defer ref_buf.deinit();
        const ref_slice = try ref_buf.toSliceAlloc(allocator, io);
        defer ref_slice.free(allocator);

        const ref_data = ref_slice.constData();
        const our_data = entry.data;

        if (our_data.len != ref_data.len) {
            std.log.warn("  {s}: size mismatch zig={d} ref={d}", .{ entry.name, our_data.len, ref_data.len });
            continue;
        }

        // Check bitwise exact first
        if (std.mem.eql(u8, our_data, ref_data)) {
            std.log.info("  {s}: EXACT MATCH", .{entry.name});
            continue;
        }

        // Compute cosine similarity and stats based on dtype
        const dtype = entry.shape.dtype();
        if (dtype == .bf16) {
            const stats = computeBf16Stats(our_data, ref_data);
            std.log.info("  {s}: cosim={d:.6} mean_abs={d:.6} max_abs={d:.6} close={d:.4}", .{
                entry.name, stats.cosim, stats.mean_abs, stats.max_abs, stats.close_frac,
            });
        } else if (dtype == .f32) {
            const stats = computeF32Stats(our_data, ref_data);
            std.log.info("  {s}: cosim={d:.6} mean_abs={d:.6} max_abs={d:.6} close={d:.4}", .{
                entry.name, stats.cosim, stats.mean_abs, stats.max_abs, stats.close_frac,
            });
        } else {
            std.log.info("  {s}: NOT EXACT (unsupported dtype for comparison)", .{entry.name});
        }
    }
}

const CompareStats = struct {
    cosim: f64,
    mean_abs: f32,
    max_abs: f32,
    close_frac: f64,
};

fn computeBf16Stats(a_bytes: []const u8, b_bytes: []const u8) CompareStats {
    const n = a_bytes.len / 2;
    var max_abs: f32 = 0;
    var sum_abs: f64 = 0;
    var dot: f64 = 0;
    var sum_a2: f64 = 0;
    var sum_b2: f64 = 0;
    var num_close: u64 = 0;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const a16 = std.mem.readInt(u16, a_bytes[i * 2 ..][0..2], .little);
        const b16 = std.mem.readInt(u16, b_bytes[i * 2 ..][0..2], .little);
        const a: f32 = @bitCast(@as(u32, a16) << 16);
        const b: f32 = @bitCast(@as(u32, b16) << 16);

        const d = @abs(a - b);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
        dot += @as(f64, a) * @as(f64, b);
        sum_a2 += @as(f64, a) * @as(f64, a);
        sum_b2 += @as(f64, b) * @as(f64, b);

        if (d <= 5e-3 or d <= 1e-2 * @max(@abs(a), @abs(b))) num_close += 1;
    }

    const nf: f64 = @floatFromInt(n);
    return .{
        .cosim = dot / (@sqrt(sum_a2) * @sqrt(sum_b2)),
        .mean_abs = @floatCast(sum_abs / nf),
        .max_abs = max_abs,
        .close_frac = @as(f64, @floatFromInt(num_close)) / nf,
    };
}

fn computeF32Stats(a_bytes: []const u8, b_bytes: []const u8) CompareStats {
    const n = a_bytes.len / 4;
    var max_abs: f32 = 0;
    var sum_abs: f64 = 0;
    var dot: f64 = 0;
    var sum_a2: f64 = 0;
    var sum_b2: f64 = 0;
    var num_close: u64 = 0;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const a: f32 = @bitCast(std.mem.readInt(u32, a_bytes[i * 4 ..][0..4], .little));
        const b: f32 = @bitCast(std.mem.readInt(u32, b_bytes[i * 4 ..][0..4], .little));

        const d = @abs(a - b);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
        dot += @as(f64, a) * @as(f64, b);
        sum_a2 += @as(f64, a) * @as(f64, a);
        sum_b2 += @as(f64, b) * @as(f64, b);

        if (d <= 1e-6 or d <= 1e-5 * @max(@abs(a), @abs(b))) num_close += 1;
    }

    const nf: f64 = @floatFromInt(n);
    return .{
        .cosim = dot / (@sqrt(sum_a2) * @sqrt(sum_b2)),
        .mean_abs = @floatCast(sum_abs / nf),
        .max_abs = max_abs,
        .close_frac = @as(f64, @floatFromInt(num_close)) / nf,
    };
}

// ============================================================================
// Helpers
// ============================================================================

fn readRawFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expected_size: usize) ![]u8 {
    const file = try std.Io.Dir.openFile(.cwd(), io, path, .{});
    defer file.close(io);
    const host_data = try allocator.alloc(u8, expected_size);
    errdefer allocator.free(host_data);
    var read_buf: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    _ = try reader.interface.readSliceAll(host_data);
    return host_data;
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

fn fillOnesF32(buf: []u8) void {
    const one_bits: u32 = @bitCast(@as(f32, 1.0));
    var i: usize = 0;
    while (i + 4 <= buf.len) : (i += 4) {
        std.mem.writeInt(u32, buf[i..][0..4], one_bits, .little);
    }
}

fn storeBf16(buf: []u8, val: f32) void {
    // bf16 = f32 rounded to nearest-even, then take upper 16 bits.
    // This matches PyTorch's .to(torch.bfloat16) semantics.
    // NOTE: Simple truncation (drop lower 16 bits) does NOT match PyTorch —
    // it produces 1-ULP bf16 differences. Round-to-nearest-even is required.
    const bits: u32 = @bitCast(val);
    // Round-to-nearest-even: add rounding bias based on the lower 16 bits.
    // If lower bits > 0x8000: round up. If == 0x8000: round to even (tie-break).
    const lower: u32 = bits & 0xFFFF;
    const round_bit: u32 = if (lower > 0x8000 or (lower == 0x8000 and (bits & 0x10000) != 0)) 1 else 0;
    const bf16_bits: u16 = @truncate((bits >> 16) +% round_bit);
    std.mem.writeInt(u16, buf[0..2], bf16_bits, .little);
}

fn storeF32(buf: []u8, val: f32) void {
    std.mem.writeInt(u32, buf[0..4], @bitCast(val), .little);
}
