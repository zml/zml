/// Latent Upsampler — spatially upsamples Stage 1 video latent 2x.
///
/// Takes the raw bf16 Stage 1 video output, unpatchifies from
/// [1, T_v, 128] → [1, 128, F, H, W], runs the CNN upsampler to
/// [1, 128, F, H*2, W*2], and writes the result as raw bf16.
///
/// This replaces the Python step in bridge_s1_to_s2.py for the video
/// upsampling portion of the Stage 1 → Stage 2 bridge.
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:upsample -- \
///       --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
///       --main-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
///       --input /root/mixed/stage1_out/video_latent.bin \
///       --f-lat 7 --h-lat 16 --w-lat 28 \
///       --output-dir /root/mixed/upsampled/ \
///       --ref /root/mixed/ref_upsampled.bin
///
/// The optional --ref flag loads a reference raw bf16 .bin (or .safetensors)
/// and prints comparison metrics (cosim, mean_abs, close_fraction, histogram).

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

    std.log.info("LTX-2.3 Latent Upsampler", .{});

    // ========================================================================
    // Parse CLI arguments
    // ========================================================================
    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    var upsampler_ckpt_path: ?[]const u8 = null;
    var main_ckpt_path: ?[]const u8 = null;
    var input_path: ?[]const u8 = null;
    var output_dir: ?[]const u8 = null;
    var ref_path: ?[]const u8 = null;
    var f_lat: ?i64 = null;
    var h_lat: ?i64 = null;
    var w_lat: ?i64 = null;

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--upsampler-ckpt")) {
            upsampler_ckpt_path = it.next() orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--main-ckpt")) {
            main_ckpt_path = it.next() orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--input")) {
            input_path = it.next() orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--output-dir")) {
            output_dir = it.next() orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--ref")) {
            ref_path = it.next() orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--f-lat")) {
            f_lat = std.fmt.parseInt(i64, it.next() orelse return error.InvalidArgs, 10) catch return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--h-lat")) {
            h_lat = std.fmt.parseInt(i64, it.next() orelse return error.InvalidArgs, 10) catch return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--w-lat")) {
            w_lat = std.fmt.parseInt(i64, it.next() orelse return error.InvalidArgs, 10) catch return error.InvalidArgs;
        }
    }

    const ckpt_up = upsampler_ckpt_path orelse {
        std.log.err("Usage: upsample --upsampler-ckpt <path> --main-ckpt <path> --input <video_latent.bin> --f-lat F --h-lat H --w-lat W --output-dir <dir>", .{});
        return error.InvalidArgs;
    };
    const ckpt_main = main_ckpt_path orelse {
        std.log.err("Missing --main-ckpt (needed for per_channel_statistics)", .{});
        return error.InvalidArgs;
    };
    const in_path = input_path orelse {
        std.log.err("Missing --input (Stage 1 video_latent.bin)", .{});
        return error.InvalidArgs;
    };
    const out_dir = output_dir orelse {
        std.log.err("Missing --output-dir", .{});
        return error.InvalidArgs;
    };
    const F = f_lat orelse {
        std.log.err("Missing --f-lat (number of latent frames)", .{});
        return error.InvalidArgs;
    };
    const H = h_lat orelse {
        std.log.err("Missing --h-lat (latent height, Stage 1)", .{});
        return error.InvalidArgs;
    };
    const W = w_lat orelse {
        std.log.err("Missing --w-lat (latent width, Stage 1)", .{});
        return error.InvalidArgs;
    };

    std.log.info("  upsampler ckpt: {s}", .{ckpt_up});
    std.log.info("  main ckpt:      {s}", .{ckpt_main});
    std.log.info("  input:          {s}", .{in_path});
    std.log.info("  output dir:     {s}", .{out_dir});
    std.log.info("  latent dims:    F={d} H={d} W={d}", .{ F, H, W });

    // ========================================================================
    // Open checkpoint stores
    // ========================================================================
    var up_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_up) catch |err| {
        std.log.err("Failed to open upsampler checkpoint: {s}", .{ckpt_up});
        return err;
    };
    defer up_reg.deinit();
    var up_store: zml.io.TensorStore = .fromRegistry(allocator, &up_reg);
    defer up_store.deinit();

    var main_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_main) catch |err| {
        std.log.err("Failed to open main checkpoint: {s}", .{ckpt_main});
        return err;
    };
    defer main_reg.deinit();
    var main_store: zml.io.TensorStore = .fromRegistry(allocator, &main_reg);
    defer main_store.deinit();

    // ========================================================================
    // Platform init
    // ========================================================================
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ========================================================================
    // Load Stage 1 video latent from raw bf16 bin
    // ========================================================================
    std.log.info("Loading Stage 1 video latent...", .{});
    const C: i64 = 128; // latent channels
    const T_v: i64 = F * H * W; // VideoLatentPatchifier(patch_size=1): T = F*H*W
    const patchified_shape = zml.Shape.init(.{ 1, T_v, C }, .bf16);

    const file_bytes = @as(usize, @intCast(patchified_shape.count())) * 2; // bf16 = 2 bytes
    const host_data = try allocator.alloc(u8, file_bytes);
    defer allocator.free(host_data);

    {
        const file = try std.Io.Dir.openFile(.cwd(), io, in_path, .{});
        defer file.close(io);
        var read_buf: [64 * 1024]u8 = undefined;
        var reader = file.reader(io, &read_buf);
        _ = try reader.interface.readSliceAll(host_data);
    }
    std.log.info("  Loaded {d} bytes, patchified shape: {any}", .{ host_data.len, patchified_shape });

    // Unpatchify: [1, F*H*W, 128] → [1, 128, F, H, W]
    // VideoLatentPatchifier(patch_size=1): patchify = "b c f h w → b (f h w) c"
    // unpatchify = reshape [1, F, H, W, 128] then permute [0, 4, 1, 2, 3]
    // We load into patchified shape, upload to device, then reshape+transpose on device.
    //
    // Since ZML compiled functions can't capture runtime values, we do the
    // unpatchify on the host side before uploading: reinterpret the flat buffer
    // as [1, 128, F, H, W] directly.
    //
    // The patchified layout is [1, F*H*W, 128] stored as row-major bf16.
    // We need to transpose to [1, 128, F, H, W].
    // Load as [1, F, H, W, 128], transpose on host to [1, 128, F, H, W].
    const video_5d_shape = zml.Shape.init(.{ 1, C, F, H, W }, .bf16);

    // Upload patchified data, compile a reshape+transpose, and run it.
    var patchified_buf = try zml.Buffer.fromBytes(io, platform, patchified_shape, sharding, host_data);
    defer patchified_buf.deinit();

    // Compile unpatchify with the latent dimensions baked into the function via model.
    std.log.info("Compiling unpatchify...", .{});
    var unpatch_exe = try platform.compileFn(
        allocator, io,
        model.forwardUnpatchifyVideo,
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
    unpatch_args.set(.{patchified_buf});
    unpatch_exe.call(unpatch_args, &unpatch_results);
    var video_5d_buf = unpatch_results.get(zml.Buffer);
    defer video_5d_buf.deinit();
    std.log.info("  Unpatchified: {any}", .{video_5d_buf.shape()});

    // ========================================================================
    // Compile upsampler
    // ========================================================================
    std.log.info("Compiling upsampler...", .{});
    const upsampler_shape = model.initUpsamplerParams(up_store.view());
    const stats_shape = model.initPerChannelStats(main_store.view());

    var upsample_exe = try platform.compileFn(
        allocator, io,
        model.forwardUpsample,
        .{
            zml.Tensor.fromShape(video_5d_buf.shape()),
            upsampler_shape,
            stats_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer upsample_exe.deinit();
    std.log.info("Upsampler compiled.", .{});

    // ========================================================================
    // Load upsampler weights
    // ========================================================================
    std.log.info("Loading upsampler weights...", .{});
    const up_bufs = try zml.io.load(
        model.UpsamplerParams,
        &upsampler_shape,
        allocator, io, platform,
        .{
            .store = &up_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    // Note: we don't deinit up_bufs tensors individually — they'll be freed with the allocator.

    std.log.info("Loading per-channel statistics...", .{});
    const stats_bufs = try zml.io.load(
        model.PerChannelStats,
        &stats_shape,
        allocator, io, platform,
        .{
            .store = &main_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );

    // ========================================================================
    // Run upsampler
    // ========================================================================
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
    // Write output
    // ========================================================================
    std.log.info("Writing output...", .{});
    const out_slice = try upsampled_buf.toSliceAlloc(allocator, io);
    defer out_slice.free(allocator);
    try writeBytesToFile(allocator, io, out_slice.constData(), out_dir, "upsampled_video.bin");
    std.log.info("Done. Output written to {s}/upsampled_video.bin", .{out_dir});

    // ========================================================================
    // Optional numerical validation: compare against a reference .bin
    // ========================================================================
    if (ref_path) |ref| {
        std.log.info("Comparing against reference: {s}", .{ref});
        try compareOutputs(allocator, io, out_slice.constData(), ref);
    }
}

fn writeBytesToFile(
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

/// Load reference .bin, compare element-wise vs our output (both raw bf16).
/// Prints: max_abs_diff, mean_abs_diff, cosine_similarity, and a diff histogram.
fn compareOutputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_bytes: []const u8,
    ref_path: []const u8,
) !void {
    // Load reference file — supports raw .bin or .safetensors (skips JSON header)
    const ref_file = try std.Io.Dir.openFile(.cwd(), io, ref_path, .{});
    defer ref_file.close(io);
    const ref_bytes = try allocator.alloc(u8, out_bytes.len);
    defer allocator.free(ref_bytes);
    {
        var read_buf: [64 * 1024]u8 = undefined;
        var reader = ref_file.reader(io, &read_buf);
        if (std.mem.endsWith(u8, ref_path, ".safetensors")) {
            // Safetensors layout: [header_len: u64 LE][JSON header][raw tensor data]
            // data_offsets in the JSON are relative to the start of the data section.
            // For a single-tensor file the first tensor starts at offset 0 in data section.
            var hdr_len_buf: [8]u8 = undefined;
            _ = try reader.interface.readSliceAll(&hdr_len_buf);
            const hdr_len = std.mem.readInt(u64, &hdr_len_buf, .little);
            const hdr = try allocator.alloc(u8, hdr_len);
            defer allocator.free(hdr);
            _ = try reader.interface.readSliceAll(hdr); // discard JSON header
            std.log.info("  Safetensors header ({d} bytes): {s}", .{ hdr_len, hdr });
        } else {
            std.log.info("  Reading raw .bin reference ({d} bytes expected)", .{out_bytes.len});
        }
        _ = reader.interface.readSliceAll(ref_bytes) catch |err| {
            if (err == error.EndOfStream) {
                std.log.err("Reference file too small: expected {d} bytes but got EndOfStream. " ++
                    "Make sure --ref points to an upsampled reference output, not the Stage 1 input.", .{out_bytes.len});
                return error.SizeMismatch;
            }
            return err;
        };
    }

    if (out_bytes.len != ref_bytes.len) {
        std.log.err("Size mismatch: output={d} bytes, ref={d} bytes", .{ out_bytes.len, ref_bytes.len });
        return error.SizeMismatch;
    }

    const n = out_bytes.len / 2; // number of bf16 elements
    var max_abs: f32 = 0;
    var sum_abs: f64 = 0;
    var dot: f64 = 0;
    var sum_a2: f64 = 0;
    var sum_b2: f64 = 0;
    var num_close: u64 = 0;
    const atol: f32 = 5e-3; // absolute tolerance
    const rtol: f32 = 1e-2; // relative tolerance

    // Histogram buckets: <0.01, 0.01-0.1, 0.1-0.5, 0.5-1.0, 1.0-2.0, 2.0-5.0, >5.0
    const NUM_BUCKETS = 7;
    const bucket_bounds = [NUM_BUCKETS - 1]f32{ 0.01, 0.1, 0.5, 1.0, 2.0, 5.0 };
    const bucket_labels = [NUM_BUCKETS][]const u8{ "<0.01  ", "0.01-0.1", "0.1-0.5", "0.5-1.0", "1.0-2.0", "2.0-5.0", ">5.0   " };
    var buckets = [_]u64{0} ** NUM_BUCKETS;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        // Read bf16 bits and convert to f32: f32 = bf16 << 16
        const a16 = std.mem.readInt(u16, out_bytes[i * 2 ..][0..2], .little);
        const b16 = std.mem.readInt(u16, ref_bytes[i * 2 ..][0..2], .little);
        const a_f32: f32 = @bitCast(@as(u32, a16) << 16);
        const b_f32: f32 = @bitCast(@as(u32, b16) << 16);

        const d = @abs(a_f32 - b_f32);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
        dot += @as(f64, a_f32) * @as(f64, b_f32);
        sum_a2 += @as(f64, a_f32) * @as(f64, a_f32);
        sum_b2 += @as(f64, b_f32) * @as(f64, b_f32);

        // Bin the error
        var bucket: usize = NUM_BUCKETS - 1;
        for (bucket_bounds, 0..) |bound, bi| {
            if (d < bound) {
                bucket = bi;
                break;
            }
        }
        buckets[bucket] += 1;

        // approxEq: close if |a-b| <= atol or |a-b| <= rtol * max(|a|,|b|)
        const abs_a = @abs(a_f32);
        const abs_b = @abs(b_f32);
        const max_val = @max(abs_a, abs_b);
        if (d <= atol or d <= rtol * max_val) {
            num_close += 1;
        }
    }

    const mean_abs: f32 = @floatCast(sum_abs / @as(f64, @floatFromInt(n)));
    const cosim: f64 = dot / (@sqrt(sum_a2) * @sqrt(sum_b2));
    const close_frac: f64 = @as(f64, @floatFromInt(num_close)) / @as(f64, @floatFromInt(n));

    std.log.info("--- Numerical validation ({d} elements) ---", .{n});
    std.log.info("  max_abs_diff:      {d:.6}", .{max_abs});
    std.log.info("  mean_abs_diff:     {d:.6}", .{mean_abs});
    std.log.info("  cosine_similarity: {d:.8}", .{cosim});
    std.log.info("  close_fraction:    {d:.6} (atol={d:.4}, rtol={d:.4})", .{ close_frac, atol, rtol });

    // Print first 8 elements of both to detect layout/input mismatches
    std.log.info("  first 8 elements  zig:  ref:", .{});
    for (0..@min(8, n)) |k| {
        const a16 = std.mem.readInt(u16, out_bytes[k * 2 ..][0..2], .little);
        const b16 = std.mem.readInt(u16, ref_bytes[k * 2 ..][0..2], .little);
        const a_f32: f32 = @bitCast(@as(u32, a16) << 16);
        const b_f32: f32 = @bitCast(@as(u32, b16) << 16);
        std.log.info("    [{d}]  {d:.4}  {d:.4}", .{ k, a_f32, b_f32 });
    }

    std.log.info("  diff histogram:", .{});
    for (bucket_labels, buckets) |label, count| {
        const pct = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(n)) * 100.0;
        std.log.info("    {s} | {d:6.2}%  ({d})", .{ label, pct, count });
    }

    // Pass/fail thresholds for bf16 CNN with 8 ResBlocks:
    // cosim > 0.995 and mean_abs < 0.1 is expected when both use bf16 computation.
    const pass = cosim > 0.995 and mean_abs < 0.1;
    if (pass) {
        std.log.info("  PASS (cosim > 0.995, mean_abs < 0.1)", .{});
    } else {
        std.log.warn("  FAIL: cosim={d:.6}, mean_abs={d:.6}", .{ cosim, mean_abs });
    }
}
