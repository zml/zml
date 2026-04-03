/// Standalone Audio VAE Decoder — for validation against Python reference.
///
/// Loads a 4D audio latent from a safetensors file (exported by
/// export_audio_vae_activations.py), runs the audio VAE decoder, writes the
/// decoded bf16 mel spectrogram for comparison.
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:audio_vae_decode -- \
///       /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
///       /root/e2e_demo/audio_vae_ref/audio_vae_activations.safetensors \
///       /root/e2e_demo/audio_vae_zig_out/

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

    std.log.info("Audio VAE Decoder (standalone validation)", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: audio_vae_decode <checkpoint.safetensors> <audio_vae_activations.safetensors> <output_dir/>", .{});
        return error.InvalidArgs;
    };
    const activations_path = it.next() orelse {
        std.log.err("Usage: audio_vae_decode <checkpoint.safetensors> <audio_vae_activations.safetensors> <output_dir/>", .{});
        return error.InvalidArgs;
    };
    const output_dir = it.next() orelse {
        std.log.err("Usage: audio_vae_decode <checkpoint.safetensors> <audio_vae_activations.safetensors> <output_dir/>", .{});
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
    // Load the 4D input latent from activations file [1, 8, T, 16]
    // ========================================================================
    std.log.info("Loading input latent from activations file...", .{});
    const a_latent_4d = try loadBuf(allocator, io, &act_store, "input_latent", platform, sharding);
    std.log.info("  input_latent: {any}", .{a_latent_4d.shape()});

    // ========================================================================
    // Load audio VAE decoder weights + per-channel stats
    // ========================================================================
    std.log.info("Loading audio VAE decoder weights...", .{});
    var audio_vae_params = model.initAudioVaeDecoderParams(ckpt_store.view());
    const audio_vae_bufs = try zml.io.load(
        model.AudioVaeDecoderParams,
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
    var audio_stats_shape = model.initAudioPerChannelStats(ckpt_store.view());
    const audio_stats_bufs = try zml.io.load(
        model.AudioPerChannelStats,
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
    // Compile and run the audio VAE decoder
    // ========================================================================
    std.log.info("Compiling audio VAE decoder...", .{});
    var audio_vae_exe = try platform.compileFn(
        allocator, io,
        model.forwardAudioVaeDecode,
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
    std.log.info("  Decoded audio mel: {any}", .{decoded_audio.shape()});

    // ========================================================================
    // Write decoded tensor as raw bf16 binary (for PSNR comparison)
    // ========================================================================
    std.log.info("Writing outputs...", .{});
    const decoded_slice = try decoded_audio.toSliceAlloc(allocator, io);
    defer decoded_slice.free(allocator);

    try writeRawBytes(allocator, io, decoded_slice.constData(), output_dir, "decoded_audio.bin");

    std.log.info("  Mel shape: {any}", .{decoded_audio.shape()});
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
