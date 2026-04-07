/// Standalone Video VAE Decoder — for validation against Python reference.
///
/// Loads a 5D video latent from a safetensors file (exported by
/// export_vae_activations.py), runs the VAE decoder, writes the decoded
/// bf16 tensor and uint8 frames for comparison.
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:vae_decode -- \
///       /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
///       /root/e2e_demo/vae_ref/vae_activations.safetensors \
///       /root/e2e_demo/vae_zig_out/

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

    std.log.info("Video VAE Decoder (standalone validation)", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: vae_decode <checkpoint.safetensors> <vae_activations.safetensors> <output_dir/>", .{});
        return error.InvalidArgs;
    };
    const activations_path = it.next() orelse {
        std.log.err("Usage: vae_decode <checkpoint.safetensors> <vae_activations.safetensors> <output_dir/>", .{});
        return error.InvalidArgs;
    };
    const output_dir = it.next() orelse {
        std.log.err("Usage: vae_decode <checkpoint.safetensors> <vae_activations.safetensors> <output_dir/>", .{});
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
    // Load the 5D input latent from activations file
    // ========================================================================
    std.log.info("Loading input latent from activations file...", .{});
    const v_latent_5d = try loadBuf(allocator, io, &act_store, "input_latent", platform, sharding);
    std.log.info("  input_latent: {any}", .{v_latent_5d.shape()});

    // ========================================================================
    // Load VAE decoder weights + per-channel stats
    // ========================================================================
    std.log.info("Loading VAE decoder weights...", .{});
    var vae_params = model.initVideoVaeDecoderParams(ckpt_store.view());
    const vae_bufs = try zml.io.load(
        model.VideoVaeDecoderParams,
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
    var stats_shape = model.initPerChannelStats(ckpt_store.view());
    const stats_bufs = try zml.io.load(
        model.PerChannelStats,
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
    // Compile and run the VAE decoder
    // ========================================================================
    std.log.info("Compiling VAE decoder...", .{});
    var vae_exe = try platform.compileFn(
        allocator, io,
        model.forwardVideoVaeDecode,
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
    std.log.info("  Decoded video: {any}", .{decoded_video.shape()});

    // ========================================================================
    // Write decoded tensor as raw bf16 binary (for comparison)
    // ========================================================================
    std.log.info("Writing outputs...", .{});
    const decoded_slice = try decoded_video.toSliceAlloc(allocator, io);
    defer decoded_slice.free(allocator);

    // Write raw bf16 dump for Python comparison
    try writeRawBytes(allocator, io, decoded_slice.constData(), output_dir, "decoded_video.bin");

    // Also write uint8 frames for visual inspection
    const F_out: usize = @intCast(decoded_video.shape().dim(2));
    const H_out: usize = @intCast(decoded_video.shape().dim(3));
    const W_out: usize = @intCast(decoded_video.shape().dim(4));

    const num_pixels = F_out * H_out * W_out * 3;
    const frames_u8 = try allocator.alloc(u8, num_pixels);
    defer allocator.free(frames_u8);

    const bf16_data = decoded_slice.constData();
    for (0..F_out) |f| {
        for (0..H_out) |h| {
            for (0..W_out) |w| {
                for (0..3) |c| {
                    const src_idx = (c * F_out * H_out * W_out + f * H_out * W_out + h * W_out + w) * 2;
                    const bf16_bits = std.mem.readInt(u16, bf16_data[src_idx..][0..2], .little);
                    const f32_bits: u32 = @as(u32, bf16_bits) << 16;
                    const val: f32 = @bitCast(f32_bits);
                    const normalized = @min(@max((val + 1.0) * 0.5, 0.0), 1.0);
                    const pixel: u8 = @intFromFloat(normalized * 255.0);
                    frames_u8[f * H_out * W_out * 3 + h * W_out * 3 + w * 3 + c] = pixel;
                }
            }
        }
    }
    try writeRawBytes(allocator, io, frames_u8, output_dir, "frames.bin");

    std.log.info("  Video: {d}x{d}, {d} frames", .{ W_out, H_out, F_out });

    // Pipe frames to ffmpeg → output.mp4
    pipeToFfmpeg(allocator, io, frames_u8, W_out, H_out, 24.0, output_dir) catch |err| {
        std.log.warn("ffmpeg encoding failed ({s}), frames.bin still available", .{@errorName(err)});
    };

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

fn pipeToFfmpeg(
    allocator: std.mem.Allocator,
    io: std.Io,
    frames_u8: []const u8,
    width: usize,
    height: usize,
    fps: f64,
    output_dir: []const u8,
) !void {
    const output_path = try std.fs.path.join(allocator, &.{ output_dir, "output.mp4" });
    defer allocator.free(output_path);

    var size_buf: [32]u8 = undefined;
    const size_str = std.fmt.bufPrint(&size_buf, "{d}x{d}", .{ width, height }) catch unreachable;

    var fps_buf: [16]u8 = undefined;
    const fps_str = std.fmt.bufPrint(&fps_buf, "{d:.0}", .{fps}) catch unreachable;

    std.log.info("Encoding video with ffmpeg → {s}", .{output_path});

    var child = try std.process.spawn(io, .{
        .argv = &.{
            "ffmpeg", "-y",
            "-f",       "rawvideo",
            "-pix_fmt", "rgb24",
            "-s",       size_str,
            "-r",       fps_str,
            "-i",       "pipe:0",
            "-c:v",     "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        },
        .stdin = .pipe,
        .stdout = .inherit,
        .stderr = .inherit,
    });

    const stdin_file = child.stdin.?;
    var write_buf: [64 * 1024]u8 = undefined;
    var writer = stdin_file.writer(io, &write_buf);
    try writer.interface.writeAll(frames_u8);
    try writer.interface.flush();
    stdin_file.close(io);
    child.stdin = null;

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
