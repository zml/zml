/// Standalone Vocoder + BWE — for validation against Python reference.
///
/// Loads a 4D mel spectrogram from a safetensors file (exported by
/// export_vocoder_activations.py), runs the vocoder + BWE, writes the
/// decoded f32 waveform for comparison.
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:vocoder_decode -- \
///       /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
///       /root/e2e_demo/vocoder_ref/vocoder_activations.safetensors \
///       /root/e2e_demo/vocoder_zig_out/

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");

comptime {
    @setEvalBranchQuota(200000);
}

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("Vocoder + BWE (standalone validation)", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: vocoder_decode <checkpoint.safetensors> <vocoder_activations.safetensors> <output_dir/>", .{});
        return error.InvalidArgs;
    };
    const activations_path = it.next() orelse {
        std.log.err("Usage: vocoder_decode <checkpoint.safetensors> <vocoder_activations.safetensors> <output_dir/>", .{});
        return error.InvalidArgs;
    };
    const output_dir = it.next() orelse {
        std.log.err("Usage: vocoder_decode <checkpoint.safetensors> <vocoder_activations.safetensors> <output_dir/>", .{});
        return error.InvalidArgs;
    };

    std.log.info("  checkpoint:   {s}", .{ckpt_path});
    std.log.info("  activations:  {s}", .{activations_path});
    std.log.info("  output-dir:   {s}", .{output_dir});

    // ========================================================================
    // Open stores
    // ========================================================================
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    var act_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, activations_path) catch |err| {
        std.log.err("Failed to open activations: {s}", .{activations_path});
        return err;
    };
    defer act_reg.deinit();
    var act_store: zml.io.TensorStore = .fromRegistry(allocator, &act_reg);
    defer act_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ========================================================================
    // Load the mel spectrogram input [1, 2, T, 64]
    // ========================================================================
    std.log.info("Loading mel spectrogram from activations file...", .{});
    const mel_spec = try loadBuf(allocator, io, &act_store, "input_mel", platform, sharding);
    std.log.info("  input_mel: {any}", .{mel_spec.shape()});

    // ========================================================================
    // Load vocoder + BWE weights (split: main vocoder 667, BWE pipeline 559)
    // ========================================================================
    std.log.info("Loading main vocoder weights...", .{});
    var main_voc_params: model.MainVocoderParams = undefined;
    model.initMainVocoderParams(&main_voc_params, ckpt_store.view().withPrefix("vocoder").withPrefix("vocoder"));
    const main_voc_bufs = try zml.io.load(
        model.MainVocoderParams,
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
    var bwe_params: model.BWEPipelineParams = undefined;
    model.initBWEPipelineParams(&bwe_params, ckpt_store.view());
    const bwe_bufs = try zml.io.load(
        model.BWEPipelineParams,
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

    // ========================================================================
    // Debug: after ups[0] only
    // ========================================================================
    std.log.info("Debug: compiling after_ups0...", .{});
    var ups0_exe = try platform.compileFn(
        allocator, io,
        model.forwardAfterUps0,
        .{ zml.Tensor.fromShape(mel_spec.shape()), &main_voc_params },
        .{ .shardings = &.{sharding} },
    );
    defer ups0_exe.deinit();

    std.log.info("Debug: running after_ups0...", .{});
    var ups0_args = try ups0_exe.args(allocator);
    defer ups0_args.deinit(allocator);
    var ups0_results = try ups0_exe.results(allocator);
    defer ups0_results.deinit(allocator);
    ups0_args.set(.{ mel_spec, &main_voc_bufs });
    ups0_exe.call(ups0_args, &ups0_results);
    const ups0_buf = ups0_results.get(zml.Buffer);
    std.log.info("  after_ups0 output: {any}", .{ups0_buf.shape()});

    const ups0_slice = try ups0_buf.toSliceAlloc(allocator, io);
    defer ups0_slice.free(allocator);
    try writeRawBytes(allocator, io, ups0_slice.constData(), output_dir, "debug_after_ups0.bin");

    // ========================================================================
    // Debug: after stage 0 (ups[0] + resblocks mean)
    // ========================================================================
    std.log.info("Debug: compiling after_stage0...", .{});
    var stage0_exe = try platform.compileFn(
        allocator, io,
        model.forwardAfterStage0,
        .{ zml.Tensor.fromShape(mel_spec.shape()), &main_voc_params },
        .{ .shardings = &.{sharding} },
    );
    defer stage0_exe.deinit();

    std.log.info("Debug: running after_stage0...", .{});
    var stage0_args = try stage0_exe.args(allocator);
    defer stage0_args.deinit(allocator);
    var stage0_results = try stage0_exe.results(allocator);
    defer stage0_results.deinit(allocator);
    stage0_args.set(.{ mel_spec, &main_voc_bufs });
    stage0_exe.call(stage0_args, &stage0_results);
    const stage0_buf = stage0_results.get(zml.Buffer);
    std.log.info("  after_stage0 output: {any}", .{stage0_buf.shape()});

    const stage0_slice = try stage0_buf.toSliceAlloc(allocator, io);
    defer stage0_slice.free(allocator);
    try writeRawBytes(allocator, io, stage0_slice.constData(), output_dir, "debug_after_stage0.bin");

    // ========================================================================
    // Stage 1: Main vocoder — mel → 16kHz waveform
    // ========================================================================
    std.log.info("Compiling main vocoder (input: {any})...", .{mel_spec.shape()});
    var main_voc_exe = try platform.compileFn(
        allocator, io,
        model.forwardMainVocoder,
        .{ zml.Tensor.fromShape(mel_spec.shape()), &main_voc_params },
        .{ .shardings = &.{sharding} },
    );
    defer main_voc_exe.deinit();

    std.log.info("Running main vocoder...", .{});
    var main_voc_args = try main_voc_exe.args(allocator);
    defer main_voc_args.deinit(allocator);
    var main_voc_results = try main_voc_exe.results(allocator);
    defer main_voc_results.deinit(allocator);
    main_voc_args.set(.{ mel_spec, &main_voc_bufs });
    main_voc_exe.call(main_voc_args, &main_voc_results);
    const waveform_16k = main_voc_results.get(zml.Buffer);
    std.log.info("  16kHz waveform: {any}", .{waveform_16k.shape()});

    // ========================================================================
    // BWE Debug: compute mel spectrogram from 16kHz waveform
    // ========================================================================
    std.log.info("BWE Debug: compiling forwardBWEComputeMel (input: {any})...", .{waveform_16k.shape()});
    var bwe_mel_exe = try platform.compileFn(
        allocator, io,
        model.forwardBWEComputeMel,
        .{ zml.Tensor.fromShape(waveform_16k.shape()), &bwe_params },
        .{ .shardings = &.{sharding} },
    );
    defer bwe_mel_exe.deinit();

    std.log.info("BWE Debug: running forwardBWEComputeMel...", .{});
    var bwe_mel_args = try bwe_mel_exe.args(allocator);
    defer bwe_mel_args.deinit(allocator);
    var bwe_mel_results = try bwe_mel_exe.results(allocator);
    defer bwe_mel_results.deinit(allocator);
    bwe_mel_args.set(.{ waveform_16k, &bwe_bufs });
    bwe_mel_exe.call(bwe_mel_args, &bwe_mel_results);
    const bwe_mel_buf = bwe_mel_results.get(zml.Buffer);
    std.log.info("  bwe_mel: {any}", .{bwe_mel_buf.shape()});

    const bwe_mel_slice = try bwe_mel_buf.toSliceAlloc(allocator, io);
    defer bwe_mel_slice.free(allocator);
    try writeRawBytes(allocator, io, bwe_mel_slice.constData(), output_dir, "debug_bwe_mel.bin");

    // ========================================================================
    // BWE Debug: sinc resample skip connection
    // ========================================================================
    std.log.info("BWE Debug: compiling forwardBWESincSkip (input: {any})...", .{waveform_16k.shape()});
    var bwe_skip_exe = try platform.compileFn(
        allocator, io,
        model.forwardBWESincSkip,
        .{zml.Tensor.fromShape(waveform_16k.shape())},
        .{ .shardings = &.{sharding} },
    );
    defer bwe_skip_exe.deinit();

    std.log.info("BWE Debug: running forwardBWESincSkip...", .{});
    var bwe_skip_args = try bwe_skip_exe.args(allocator);
    defer bwe_skip_args.deinit(allocator);
    var bwe_skip_results = try bwe_skip_exe.results(allocator);
    defer bwe_skip_results.deinit(allocator);
    bwe_skip_args.set(.{waveform_16k});
    bwe_skip_exe.call(bwe_skip_args, &bwe_skip_results);
    const bwe_skip_buf = bwe_skip_results.get(zml.Buffer);
    std.log.info("  bwe_skip: {any}", .{bwe_skip_buf.shape()});

    const bwe_skip_slice = try bwe_skip_buf.toSliceAlloc(allocator, io);
    defer bwe_skip_slice.free(allocator);
    try writeRawBytes(allocator, io, bwe_skip_slice.constData(), output_dir, "debug_bwe_skip.bin");

    // ========================================================================
    // BWE Debug: residual (mel → BWE generator)
    // ========================================================================
    std.log.info("BWE Debug: compiling forwardBWEResidual (input: {any})...", .{waveform_16k.shape()});
    var bwe_res_exe = try platform.compileFn(
        allocator, io,
        model.forwardBWEResidual,
        .{ zml.Tensor.fromShape(waveform_16k.shape()), &bwe_params },
        .{ .shardings = &.{sharding} },
    );
    defer bwe_res_exe.deinit();

    std.log.info("BWE Debug: running forwardBWEResidual...", .{});
    var bwe_res_args = try bwe_res_exe.args(allocator);
    defer bwe_res_args.deinit(allocator);
    var bwe_res_results = try bwe_res_exe.results(allocator);
    defer bwe_res_results.deinit(allocator);
    bwe_res_args.set(.{ waveform_16k, &bwe_bufs });
    bwe_res_exe.call(bwe_res_args, &bwe_res_results);
    const bwe_res_buf = bwe_res_results.get(zml.Buffer);
    std.log.info("  bwe_residual: {any}", .{bwe_res_buf.shape()});

    const bwe_res_slice = try bwe_res_buf.toSliceAlloc(allocator, io);
    defer bwe_res_slice.free(allocator);
    try writeRawBytes(allocator, io, bwe_res_slice.constData(), output_dir, "debug_bwe_residual.bin");

    // ========================================================================
    // Stage 2: BWE pipeline — 16kHz → 48kHz waveform
    // ========================================================================
    std.log.info("Compiling BWE pipeline (input: {any})...", .{waveform_16k.shape()});
    var bwe_exe = try platform.compileFn(
        allocator, io,
        model.forwardBWEPipeline,
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
    std.log.info("  48kHz waveform: {any}", .{waveform.shape()});

    // ========================================================================
    // Write intermediate 16kHz and final 48kHz waveforms for comparison
    // ========================================================================
    std.log.info("Writing outputs...", .{});

    const waveform_16k_slice = try waveform_16k.toSliceAlloc(allocator, io);
    defer waveform_16k_slice.free(allocator);
    try writeRawBytes(allocator, io, waveform_16k_slice.constData(), output_dir, "waveform_16k.bin");

    const waveform_slice = try waveform.toSliceAlloc(allocator, io);
    defer waveform_slice.free(allocator);
    try writeRawBytes(allocator, io, waveform_slice.constData(), output_dir, "waveform.bin");

    std.log.info("  16kHz shape: {any}", .{waveform_16k.shape()});
    std.log.info("  48kHz shape: {any}", .{waveform.shape()});
    std.log.info("Done.", .{});
}

fn loadBuf(
    allocator: std.mem.Allocator,
    io: std.Io,
    store: *zml.io.TensorStore,
    name: []const u8,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const shape = store.view().getShape(name) orelse {
        std.log.err("Tensor '{s}' not found in store", .{name});
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
