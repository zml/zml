/// Noise init parity check.
///
/// Loads clean latent, noise, mask, and expected noised from a safetensors
/// fixture, applies forwardNoiseInit on device, and compares the result.
///
/// Usage:
///   bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:check_noise_init -- \
///       /root/fixtures/noise_init_fixture.safetensors

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

comptime {
    @setEvalBranchQuota(200000);
}

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("Noise init parity check", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const fixture_path = it.next() orelse {
        std.log.err("Usage: check_noise_init <noise_init_fixture.safetensors>", .{});
        return error.InvalidArgs;
    };

    // Open fixture
    var fixture_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
        std.log.err("Failed to open fixture: {s}", .{fixture_path});
        return err;
    };
    defer fixture_reg.deinit();
    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_reg);
    defer fixture_store.deinit();

    // Read sigma_0 from safetensors metadata
    const sigma_0_meta = fixture_reg.metadata.get("sigma_0") orelse {
        std.log.err("Missing 'sigma_0' in fixture metadata", .{});
        return error.MissingMetadata;
    };
    const sigma_0_str = switch (sigma_0_meta) {
        .string => |s| s,
        else => {
            std.log.err("'sigma_0' metadata is not a string", .{});
            return error.InvalidMetadata;
        },
    };
    const sigma_0 = try std.fmt.parseFloat(f32, sigma_0_str);
    std.log.info("sigma_0 = {d:.6}", .{sigma_0});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // Load tensors
    std.log.info("Loading fixture tensors...", .{});
    var v_clean = try loadBuf(allocator, io, platform, &fixture_store, "video_clean", sharding);
    defer v_clean.deinit();
    var a_clean = try loadBuf(allocator, io, platform, &fixture_store, "audio_clean", sharding);
    defer a_clean.deinit();
    var v_noise = try loadBuf(allocator, io, platform, &fixture_store, "video_noise", sharding);
    defer v_noise.deinit();
    var a_noise = try loadBuf(allocator, io, platform, &fixture_store, "audio_noise", sharding);
    defer a_noise.deinit();
    var v_mask = try loadBuf(allocator, io, platform, &fixture_store, "video_mask", sharding);
    defer v_mask.deinit();
    var a_mask = try loadBuf(allocator, io, platform, &fixture_store, "audio_mask", sharding);
    defer a_mask.deinit();
    var v_expected = try loadBuf(allocator, io, platform, &fixture_store, "video_expected", sharding);
    defer v_expected.deinit();
    var a_expected = try loadBuf(allocator, io, platform, &fixture_store, "audio_expected", sharding);
    defer a_expected.deinit();

    std.log.info("  video_clean:    {any}", .{v_clean.shape()});
    std.log.info("  video_noise:    {any}", .{v_noise.shape()});
    std.log.info("  video_mask:     {any}", .{v_mask.shape()});
    std.log.info("  video_expected: {any}", .{v_expected.shape()});
    std.log.info("  audio_clean:    {any}", .{a_clean.shape()});
    std.log.info("  audio_noise:    {any}", .{a_noise.shape()});
    std.log.info("  audio_mask:     {any}", .{a_mask.shape()});
    std.log.info("  audio_expected: {any}", .{a_expected.shape()});

    // Compile noise init exe
    std.log.info("Compiling noise init exe...", .{});
    const sigma_shape = zml.Shape.init(.{}, .f32);

    var noise_init_exe = try platform.compileFn(
        allocator, io,
        model.forwardNoiseInit,
        .{
            zml.Tensor.fromShape(v_clean.shape()),
            zml.Tensor.fromShape(v_noise.shape()),
            zml.Tensor.fromShape(v_mask.shape()),
            zml.Tensor.fromShape(sigma_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer noise_init_exe.deinit();

    // Recompile for audio (different shapes)
    var noise_init_a_exe = try platform.compileFn(
        allocator, io,
        model.forwardNoiseInit,
        .{
            zml.Tensor.fromShape(a_clean.shape()),
            zml.Tensor.fromShape(a_noise.shape()),
            zml.Tensor.fromShape(a_mask.shape()),
            zml.Tensor.fromShape(sigma_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer noise_init_a_exe.deinit();
    std.log.info("Noise init exes compiled.", .{});

    // Create sigma buffer
    var sigma_buf = try zml.Buffer.scalar(io, platform, sigma_0, .f32, sharding);
    defer sigma_buf.deinit();

    // Run video noise init
    std.log.info("Running video noise init...", .{});
    var v_args = try noise_init_exe.args(allocator);
    defer v_args.deinit(allocator);
    var v_results = try noise_init_exe.results(allocator);
    defer v_results.deinit(allocator);
    v_args.set(.{ v_clean, v_noise, v_mask, sigma_buf });
    noise_init_exe.call(v_args, &v_results);
    var v_computed = v_results.get(zml.Buffer);
    defer v_computed.deinit();

    // Run audio noise init
    std.log.info("Running audio noise init...", .{});
    var a_args = try noise_init_a_exe.args(allocator);
    defer a_args.deinit(allocator);
    var a_results = try noise_init_a_exe.results(allocator);
    defer a_results.deinit(allocator);
    a_args.set(.{ a_clean, a_noise, a_mask, sigma_buf });
    noise_init_a_exe.call(a_args, &a_results);
    var a_computed = a_results.get(zml.Buffer);
    defer a_computed.deinit();

    // Compare
    std.log.info("", .{});
    std.log.info("=== Video noise init ===", .{});
    const v_metrics = try check_utils.compareBuffersExtended(io, v_computed, v_expected, 1e-3, 1e-2);
    std.log.info("  cos_sim:     {d:.6}", .{v_metrics.cosine_similarity});
    std.log.info("  close:       {d:.6}", .{v_metrics.close_fraction});
    std.log.info("  max_abs_err: {e:.6}", .{v_metrics.max_abs_error});
    std.log.info("  mean_abs:    {e:.6}", .{v_metrics.mean_abs_error});

    std.log.info("", .{});
    std.log.info("=== Audio noise init ===", .{});
    const a_metrics = try check_utils.compareBuffersExtended(io, a_computed, a_expected, 1e-3, 1e-2);
    std.log.info("  cos_sim:     {d:.6}", .{a_metrics.cosine_similarity});
    std.log.info("  close:       {d:.6}", .{a_metrics.close_fraction});
    std.log.info("  max_abs_err: {e:.6}", .{a_metrics.max_abs_error});
    std.log.info("  mean_abs:    {e:.6}", .{a_metrics.mean_abs_error});

    // Verdict
    const v_pass = v_metrics.cosine_similarity >= 0.9999 and v_metrics.close_fraction >= 0.99;
    const a_pass = a_metrics.cosine_similarity >= 0.9999 and a_metrics.close_fraction >= 0.99;

    std.log.info("", .{});
    if (v_pass and a_pass) {
        std.log.info("PASS: Noise init matches Python reference.", .{});
    } else {
        std.log.err("FAIL: Noise init mismatch.", .{});
        if (!v_pass) std.log.err("  Video: cos_sim={d:.6}, close={d:.6}", .{ v_metrics.cosine_similarity, v_metrics.close_fraction });
        if (!a_pass) std.log.err("  Audio: cos_sim={d:.6}, close={d:.6}", .{ a_metrics.cosine_similarity, a_metrics.close_fraction });
    }
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
