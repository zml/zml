const std = @import("std");
const zml = @import("zml");
const flux_model = @import("flux2_transformer2d_model.zig");

pub const LatentPacker = struct {
    pub fn forward(input: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        var dummy: flux_model.Flux2Transformer2DModel = undefined;
        // Methods ignore self, so dummy is safe.
        const latents = dummy.pack_latents(input);
        const ids = dummy.prepare_latent_ids(input);
        return .{ latents, ids };
    }
};

pub fn pack_latents(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    latents_in: zml.Buffer,
) !struct { zml.Buffer, zml.Buffer } {
    const latents_tensor = zml.Tensor.fromShape(latents_in.shape());
    var exe = try zml.module.compile(allocator, io, LatentPacker.forward, .{latents_tensor}, platform);
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{latents_in});

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(args, &res);

    const out_buffers = res.get(struct { zml.Buffer, zml.Buffer });
    return .{ out_buffers[0], out_buffers[1] };
}

pub fn compute_empirical_mu(image_seq_len: f64, num_steps: f64) f32 {
    const a1 = 8.73809524e-05;
    const b1 = 1.89833333;
    const a2 = 0.00016927;
    const b2 = 0.45666666;

    if (image_seq_len > 4300.0) {
        return @floatCast(a2 * image_seq_len + b2);
    }

    const m_200 = a2 * image_seq_len + b2;
    const m_10 = a1 * image_seq_len + b1;
    const a = (m_200 - m_10) / 190.0;
    const b = m_200 - 200.0 * a;
    return @floatCast(a * num_steps + b);
}

pub fn get_latents(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    config: flux_model.Config,
    img_dim: usize,
) !struct { zml.Buffer, zml.Buffer } {
    const in_channels = config.in_channels;
    const vae_scale_factor: usize = 8;

    const batch_size = 1;
    const num_channels_latents = @as(usize, @intCast(in_channels)) / 4;
    const adjusted_height = 2 * (img_dim / (vae_scale_factor * 2));
    const adjusted_width = 2 * (img_dim / (vae_scale_factor * 2));

    // (1, 128, 8, 8) for 128x128
    const shape_latents = [_]usize{ batch_size, num_channels_latents * 4, adjusted_height / 2, adjusted_width / 2 };

    // Generate Raw Latents on Host
    var rng = BoxMullerGenerator.init(0);
    const latents_raw_data = try rng.randn(allocator, &shape_latents);
    defer allocator.free(latents_raw_data);

    // Print raw first 20 for user verification
    std.log.info("Latents Raw (first 20): {any}", .{latents_raw_data[0..20]});

    // Upload to Device
    const latents_raw_shape = zml.Shape.init(.{ .b = @as(i64, @intCast(shape_latents[0])), .c = @as(i64, @intCast(shape_latents[1])), .h = @as(i64, @intCast(shape_latents[2])), .w = @as(i64, @intCast(shape_latents[3])) }, .f32);

    var latents_raw_buffer = try zml.Buffer.fromBytes(io, platform, latents_raw_shape, std.mem.sliceAsBytes(latents_raw_data));
    defer latents_raw_buffer.deinit();

    // Compile and Execute Packing
    const latents_tensor = zml.Tensor.fromShape(latents_raw_shape);
    var exe = try zml.module.compile(allocator, io, LatentPacker.forward, .{latents_tensor}, platform);
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{latents_raw_buffer});

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(args, &res);

    const out_buffers = res.get(struct { zml.Buffer, zml.Buffer });
    return out_buffers;
}

pub const BoxMullerGenerator = struct {
    seed: i64,
    counter: i64,

    pub fn init(seed: i64) BoxMullerGenerator {
        return .{
            .seed = seed,
            .counter = 0,
        };
    }

    fn hash(self: BoxMullerGenerator, index: i64) f32 {
        var x = index +% self.seed;
        x = (x ^ (x >> 16)) *% 0x45d9f3b;
        x = (x ^ (x >> 16)) *% 0x45d9f3b;
        x = x ^ (x >> 16);

        // Normalize to [0, 1]
        // Python: (x % 1000000).float() / 1000000.0
        // Use @mod to emulate Python's modulo behavior (positive result).
        const m = @mod(x, 1000000);
        return @as(f32, @floatFromInt(m)) / 1000000.0;
    }

    pub fn randn(self: *BoxMullerGenerator, allocator: std.mem.Allocator, shape: []const usize) ![]f32 {
        var count: usize = 1;
        for (shape) |s| count *= s;

        const result = try allocator.alloc(f32, count);
        errdefer allocator.free(result);

        const constant_u2_offset = 0x9e3779b9;

        for (result, 0..) |*item, i| {
            // Python: indices = torch.arange(self.counter, self.counter + num_elements)
            const idx = self.counter + @as(i64, @intCast(i));

            // val1 = self._hash(indices)
            const val1 = self.hash(idx);
            // val2 = self._hash(indices + 0x9e3779b9)
            const val2 = self.hash(idx +% constant_u2_offset);

            // Box-Muller Transform
            // mag = sqrt(-2 ln(val1))
            const ln_val1 = @log(val1 + 1e-10);
            const mag = @sqrt(-2.0 * ln_val1);

            // dist = mag * cos(2pi * val2)
            const two_pi = 2.0 * 3.1415926535;
            const cos_val = @cos(two_pi * val2);

            item.* = mag * cos_val;
        }
        self.counter += @as(i64, @intCast(count));
        return result;
    }
};

test "BoxMullerGenerator matches Python reference" {
    const allocator = std.testing.allocator;
    // Python: generator = BoxMullerGenerator(seed=42, device="cpu")
    var gen = BoxMullerGenerator.init(42);

    // Python: samples = generator.randn((2, 3, 4))
    const shape = [_]usize{ 2, 3, 4 };
    const samples = try gen.randn(allocator, &shape);
    defer allocator.free(samples);

    // Reference output from Python script
    const expected = [_]f32{
        -0.0603, 0.9283,  0.1725,  1.3793,
        0.5367,  -0.1172, 0.3583,  1.0377,
        2.6232,  0.8390,  -1.3245, -0.3546,
        -0.0135, -1.3908, -2.3215, 1.5630,
        -0.4535, -1.4687, -0.3142, 0.5771,
        1.2876,  -1.0406, -0.5745, -1.1020,
    };

    var max_err: f32 = 0;
    for (samples, 0..) |val, i| {
        const diff = @abs(val - expected[i]);
        if (diff > max_err) max_err = diff;

        // Relaxed tolerance due to minor float implementation diffs (e.g. log, cos precision)
        try std.testing.expectApproxEqAbs(expected[i], val, 1e-3);
    }
    // std.debug.print("Max difference: {d}\n", .{max_err});
}
