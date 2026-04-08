/// Standalone VAE encoder validation binary.
///
/// Per-block isolated accuracy test: feeds Python reference input to each encoder
/// block individually and compares output against Python reference output.
/// This isolates whether divergence comes from a specific block (bug) or from
/// cumulative bf16 rounding across all blocks (expected).
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:validate_encoder -- \
///       /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
///       /root/imgcond_ref/image_preprocessed.safetensors \
///       /root/imgcond_ref/encoder_activations.safetensors

const std = @import("std");
const zml = @import("zml");
const conv_ops = @import("conv_ops.zig");
const video_vae = @import("video_vae.zig");
const enc = @import("video_vae_encoder.zig");

const Conv3dWeight = conv_ops.Conv3dWeight;
const PerChannelStats = conv_ops.PerChannelStats;
const VaeResBlock = video_vae.VaeResBlock;
const Tensor = zml.Tensor;

comptime {
    @setEvalBranchQuota(200000);
}

pub const std_options: std.Options = .{ .log_level = .info };

// ============================================================================
// Step functions — one per encoder block, for isolated per-block validation.
// Each takes the block's input + weights and produces the block's output.
// ============================================================================

fn stepDown0(x: Tensor, r0: VaeResBlock, r1: VaeResBlock, r2: VaeResBlock, r3: VaeResBlock) Tensor {
    var h = enc.forwardResBlock(x, r0);
    h = enc.forwardResBlock(h, r1);
    h = enc.forwardResBlock(h, r2);
    return enc.forwardResBlock(h, r3);
}

fn stepDown1(x: Tensor, conv: Conv3dWeight) Tensor {
    return enc.forwardSpaceToDepthDownsample(x, conv, .{ 1, 2, 2 }, 2);
}

fn stepDown2(x: Tensor, r0: VaeResBlock, r1: VaeResBlock, r2: VaeResBlock, r3: VaeResBlock, r4: VaeResBlock, r5: VaeResBlock) Tensor {
    var h = enc.forwardResBlock(x, r0);
    h = enc.forwardResBlock(h, r1);
    h = enc.forwardResBlock(h, r2);
    h = enc.forwardResBlock(h, r3);
    h = enc.forwardResBlock(h, r4);
    return enc.forwardResBlock(h, r5);
}

fn stepDown3(x: Tensor, conv: Conv3dWeight) Tensor {
    return enc.forwardSpaceToDepthDownsample(x, conv, .{ 2, 1, 1 }, 1);
}

fn stepDown4(x: Tensor, r0: VaeResBlock, r1: VaeResBlock, r2: VaeResBlock, r3: VaeResBlock) Tensor {
    var h = enc.forwardResBlock(x, r0);
    h = enc.forwardResBlock(h, r1);
    h = enc.forwardResBlock(h, r2);
    return enc.forwardResBlock(h, r3);
}

fn stepDown5(x: Tensor, conv: Conv3dWeight) Tensor {
    return enc.forwardSpaceToDepthDownsample(x, conv, .{ 2, 2, 2 }, 4);
}

fn stepDown6(x: Tensor, r0: VaeResBlock, r1: VaeResBlock) Tensor {
    const h = enc.forwardResBlock(x, r0);
    return enc.forwardResBlock(h, r1);
}

fn stepDown7(x: Tensor, conv: Conv3dWeight) Tensor {
    return enc.forwardSpaceToDepthDownsample(x, conv, .{ 2, 2, 2 }, 8);
}

fn stepDown8(x: Tensor, r0: VaeResBlock, r1: VaeResBlock) Tensor {
    const h = enc.forwardResBlock(x, r0);
    return enc.forwardResBlock(h, r1);
}

fn stepNormSilu(x: Tensor) Tensor {
    return enc.forwardPixelNorm(x).silu();
}

fn stepConvOutNormalize(x: Tensor, conv: Conv3dWeight, stats: PerChannelStats) Tensor {
    var h = enc.forwardCausalConv3d(x, conv);
    h = h.slice1d(1, .{ .end = 128 });
    const stats_shape = h.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1);
    const mean_broad = stats.mean_of_means.reshape(stats_shape).broad(h.shape());
    const std_broad = stats.std_of_means.reshape(stats_shape).broad(h.shape());
    return h.sub(mean_broad).div(std_broad);
}

// ============================================================================
// Main
// ============================================================================

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("VAE Encoder Validation — per-block isolated accuracy", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: validate_encoder <checkpoint.safetensors> <image.safetensors> <activations.safetensors>", .{});
        return error.InvalidArgs;
    };
    const image_path = it.next() orelse {
        std.log.err("Usage: validate_encoder <checkpoint.safetensors> <image.safetensors> <activations.safetensors>", .{});
        return error.InvalidArgs;
    };
    const ref_path = it.next() orelse {
        std.log.err("Usage: validate_encoder <checkpoint.safetensors> <image.safetensors> <activations.safetensors>", .{});
        return error.InvalidArgs;
    };

    // Open safetensors stores
    std.log.info("Opening stores...", .{});

    var ckpt_reg = try zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path);
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    var image_reg = try zml.safetensors.TensorRegistry.fromPath(allocator, io, image_path);
    defer image_reg.deinit();
    var image_store: zml.io.TensorStore = .fromRegistry(allocator, &image_reg);
    defer image_store.deinit();

    var ref_reg = try zml.safetensors.TensorRegistry.fromPath(allocator, io, ref_path);
    defer ref_reg.deinit();
    var ref_store: zml.io.TensorStore = .fromRegistry(allocator, &ref_reg);
    defer ref_store.deinit();

    // Platform init
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // Load input image
    std.log.info("Loading input image...", .{});
    var image_buf = try loadBuf(allocator, io, platform, &image_store, "image_s1", sharding);
    defer image_buf.deinit();
    std.log.info("  image_s1: {any}", .{image_buf.shape()});

    // Load encoder weights
    std.log.info("Loading encoder weights...", .{});
    const encoder_shape = enc.initVideoVaeEncoderParams(ckpt_store.view());
    const stats_shape = conv_ops.initPerChannelStats(ckpt_store.view());

    const encoder_bufs = try zml.io.load(
        enc.VideoVaeEncoderParams, &encoder_shape,
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
    std.log.info("  Encoder weights loaded.", .{});

    // ========================================================================
    // Section A: Per-block ISOLATED accuracy
    //   Feed Python reference input → run ONE block → compare to Python ref output.
    //   This tests each block independently, without accumulated error.
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Section A: Per-block ISOLATED accuracy ===", .{});
    std.log.info("(Each block fed Python reference input → compare output)", .{});
    std.log.info("", .{});

    // A0: patchify (pure reshape — exact)
    std.log.info("A0: patchify...", .{});
    {
        var exe = try platform.compileFn(allocator, io, enc.forwardPatchifyVae, .{Tensor.fromShape(image_buf.shape())}, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{image_buf});
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_patchify", buf, sharding);
    }

    // A1: conv_in (feed Python ref patchify output)
    std.log.info("A1: conv_in...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_patchify", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, enc.forwardCausalConv3d, .{ Tensor.fromShape(ref_in.shape()), encoder_shape.conv_in }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.conv_in });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_conv_in", buf, sharding);
    }

    // A2: down_blocks.0 — 4 ResBlocks @ 128ch
    std.log.info("A2: down_blocks.0 (4 ResBlocks)...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_conv_in", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepDown0, .{
            Tensor.fromShape(ref_in.shape()),
            encoder_shape.down0_res0, encoder_shape.down0_res1,
            encoder_shape.down0_res2, encoder_shape.down0_res3,
        }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.down0_res0, encoder_bufs.down0_res1, encoder_bufs.down0_res2, encoder_bufs.down0_res3 });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_down_0", buf, sharding);
    }

    // A3: down_blocks.1 — SpaceToDepth (1,2,2)
    std.log.info("A3: down_blocks.1 (S2D 1,2,2)...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_down_0", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepDown1, .{ Tensor.fromShape(ref_in.shape()), encoder_shape.down1_conv }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.down1_conv });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_down_1", buf, sharding);
    }

    // A4: down_blocks.2 — 6 ResBlocks @ 256ch
    std.log.info("A4: down_blocks.2 (6 ResBlocks)...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_down_1", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepDown2, .{
            Tensor.fromShape(ref_in.shape()),
            encoder_shape.down2_res0, encoder_shape.down2_res1,
            encoder_shape.down2_res2, encoder_shape.down2_res3,
            encoder_shape.down2_res4, encoder_shape.down2_res5,
        }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.down2_res0, encoder_bufs.down2_res1, encoder_bufs.down2_res2, encoder_bufs.down2_res3, encoder_bufs.down2_res4, encoder_bufs.down2_res5 });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_down_2", buf, sharding);
    }

    // A5: down_blocks.3 — SpaceToDepth (2,1,1)
    std.log.info("A5: down_blocks.3 (S2D 2,1,1)...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_down_2", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepDown3, .{ Tensor.fromShape(ref_in.shape()), encoder_shape.down3_conv }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.down3_conv });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_down_3", buf, sharding);
    }

    // A6: down_blocks.4 — 4 ResBlocks @ 512ch
    std.log.info("A6: down_blocks.4 (4 ResBlocks)...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_down_3", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepDown4, .{
            Tensor.fromShape(ref_in.shape()),
            encoder_shape.down4_res0, encoder_shape.down4_res1,
            encoder_shape.down4_res2, encoder_shape.down4_res3,
        }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.down4_res0, encoder_bufs.down4_res1, encoder_bufs.down4_res2, encoder_bufs.down4_res3 });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_down_4", buf, sharding);
    }

    // A7: down_blocks.5 — SpaceToDepth (2,2,2)
    std.log.info("A7: down_blocks.5 (S2D 2,2,2)...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_down_4", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepDown5, .{ Tensor.fromShape(ref_in.shape()), encoder_shape.down5_conv }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.down5_conv });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_down_5", buf, sharding);
    }

    // A8: down_blocks.6 — 2 ResBlocks @ 1024ch
    std.log.info("A8: down_blocks.6 (2 ResBlocks)...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_down_5", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepDown6, .{ Tensor.fromShape(ref_in.shape()), encoder_shape.down6_res0, encoder_shape.down6_res1 }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.down6_res0, encoder_bufs.down6_res1 });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_down_6", buf, sharding);
    }

    // A9: down_blocks.7 — SpaceToDepth (2,2,2)
    std.log.info("A9: down_blocks.7 (S2D 2,2,2)...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_down_6", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepDown7, .{ Tensor.fromShape(ref_in.shape()), encoder_shape.down7_conv }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.down7_conv });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_down_7", buf, sharding);
    }

    // A10: down_blocks.8 — 2 ResBlocks @ 1024ch
    std.log.info("A10: down_blocks.8 (2 ResBlocks)...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_down_7", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepDown8, .{ Tensor.fromShape(ref_in.shape()), encoder_shape.down8_res0, encoder_shape.down8_res1 }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.down8_res0, encoder_bufs.down8_res1 });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_down_8", buf, sharding);
    }

    // A11: PixelNorm + SiLU
    std.log.info("A11: norm_silu...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_down_8", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepNormSilu, .{Tensor.fromShape(ref_in.shape())}, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ref_in});
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/after_norm_silu", buf, sharding);
    }

    // A12: conv_out + extract means + normalize
    std.log.info("A12: conv_out + normalize...", .{});
    {
        var ref_in = try loadBuf(allocator, io, platform, &ref_store, "s1/after_norm_silu", sharding);
        defer ref_in.deinit();
        var exe = try platform.compileFn(allocator, io, stepConvOutNormalize, .{ Tensor.fromShape(ref_in.shape()), encoder_shape.conv_out, stats_shape }, .{ .shardings = &.{sharding} });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ref_in, encoder_bufs.conv_out, stats_bufs });
        exe.call(args, &results);
        var buf = results.get(zml.Buffer);
        defer buf.deinit();
        try compareReference(allocator, io, platform, &ref_store, "s1/encoded_normalized", buf, sharding);
    }

    // ========================================================================
    // Section B: Full end-to-end encoder (for cumulative error reference)
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Section B: Full end-to-end encoder ===", .{});
    std.log.info("", .{});

    var encode_exe = try platform.compileFn(
        allocator, io,
        enc.forwardVideoVaeEncode,
        .{ Tensor.fromShape(image_buf.shape()), stats_shape, encoder_shape },
        .{ .shardings = &.{sharding} },
    );
    defer encode_exe.deinit();

    var encode_args = try encode_exe.args(allocator);
    defer encode_args.deinit(allocator);
    var encode_results = try encode_exe.results(allocator);
    defer encode_results.deinit(allocator);
    encode_args.set(.{ image_buf, stats_bufs, encoder_bufs });
    encode_exe.call(encode_args, &encode_results);
    var encoded_buf = encode_results.get(zml.Buffer);
    defer encoded_buf.deinit();

    try compareReference(allocator, io, platform, &ref_store, "s1/encoded_normalized", encoded_buf, sharding);
}

// ============================================================================
// Comparison helpers (same as bridge.zig)
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
        std.log.err("Tensor not found: {s}", .{name});
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

fn compareReference(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    ref_store: *zml.io.TensorStore,
    ref_name: []const u8,
    our_buf: zml.Buffer,
    sharding: zml.sharding.Sharding,
) !void {
    const ref_shape = ref_store.view().getShape(ref_name) orelse {
        std.log.warn("  {s}: not found in reference", .{ref_name});
        return;
    };

    if (!our_buf.shape().eqlDims(ref_shape)) {
        std.log.warn("  {s}: shape mismatch zig={any} ref={any}", .{ ref_name, our_buf.shape(), ref_shape });
        return;
    }

    var ref_buf = try loadBuf(allocator, io, platform, ref_store, ref_name, sharding);
    defer ref_buf.deinit();
    const ref_slice = try ref_buf.toSliceAlloc(allocator, io);
    defer ref_slice.free(allocator);

    const our_slice = try our_buf.toSliceAlloc(allocator, io);
    defer our_slice.free(allocator);

    const ref_data = ref_slice.constData();
    const our_data = our_slice.constData();

    if (our_data.len != ref_data.len) {
        std.log.warn("  {s}: size mismatch zig={d} ref={d}", .{ ref_name, our_data.len, ref_data.len });
        return;
    }

    if (std.mem.eql(u8, our_data, ref_data)) {
        std.log.info("  {s}: EXACT MATCH", .{ref_name});
        return;
    }

    const dtype = our_buf.shape().dtype();
    if (dtype == .bf16) {
        const stats = computeBf16Stats(our_data, ref_data);
        std.log.info("  {s}: cosim={d:.6} mean_abs={d:.6} max_abs={d:.6} close={d:.4}", .{
            ref_name, stats.cosim, stats.mean_abs, stats.max_abs, stats.close_frac,
        });
    } else if (dtype == .f32) {
        const stats = computeF32Stats(our_data, ref_data);
        std.log.info("  {s}: cosim={d:.6} mean_abs={d:.6} max_abs={d:.6} close={d:.4}", .{
            ref_name, stats.cosim, stats.mean_abs, stats.max_abs, stats.close_frac,
        });
    } else {
        std.log.info("  {s}: NOT EXACT (unsupported dtype)", .{ref_name});
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
