const std = @import("std");
const zml = @import("zml");
const tools = @import("tools.zig");
const flux_model = @import("flux2_transformer2d_model.zig");
const autoencoder_kl = @import("autoencoder_kl_flux2.zig");
const flux_model_transformer2d = @import("flux2_transformer2d_model.zig");
const flow_match_euler_discrete_scheduler = @import("scheduling_flow_match_euler_discrete.zig");

const c_interface = @import("c");

const stdx = zml.stdx;
const log = std.log.scoped(.utils);

pub const LatentPacker = struct {
    pub fn forward(input: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        var dummy: flux_model.Flux2Transformer2DModel = undefined;
        const latents = dummy.pack_latents(input);
        const ids = dummy.prepare_latent_ids(input);
        return .{ latents, ids };
    }
};

pub const NormalMethod = enum {
    /// Classic Box-Muller: z = √(-2 ln u₁) · cos(2π u₂)
    box_muller,
    /// Marsaglia polar method with branchless fallback for rejected samples.
    marsaglia_polar,
};

/// Which random number generator to use for latent generation.
pub const GeneratorType = enum {
    /// Accelerator-based RNG with Box-Muller transform (zml.Tensor.Rng).
    accelerator_box_muller,
    /// Accelerator-based RNG with Marsaglia polar method (zml.Tensor.Rng).
    accelerator_marsaglia,
    /// CPU-based MT19937 + Box-Muller, bit-exact match with PyTorch's torch.Generator.
    torch,
};

/// Generates normally-distributed random latents on the accelerator,
/// then packs them in one compiled graph.
pub const LatentGenerator = struct {
    /// Generate standard-normal samples using the chosen method.
    fn randn(rng: zml.Tensor.Rng, shape: zml.Shape, method: NormalMethod) struct { zml.Tensor.Rng, zml.Tensor } {
        // Two independent uniform draws used by both methods.
        const rng1, const uniform1 = rng.uniform(shape, .{ .min = 1e-10, .max = 1.0 });
        const rng2, const uniform2 = rng1.uniform(shape, .{});

        const latents_raw = switch (method) {
            .box_muller => blk: {
                // z = √(-2 ln u₁) · cos(2π u₂)
                const mag = uniform1.log().scale(-2.0).sqrt();
                const phase = uniform2.scale(2.0 * std.math.pi);
                break :blk mag.mul(phase.cos());
            },
            .marsaglia_polar => blk: {
                // Map uniforms to [-1, 1]
                const v1 = uniform1.scale(2.0).addConstant(-1.0);
                const v2 = uniform2.scale(2.0).addConstant(-1.0);
                const s = v1.mul(v1).add(v2.mul(v2));

                // Marsaglia factor: √(-2 ln(s) / s)
                const factor = s.log().scale(-2.0).div(s).sqrt();
                const polar = v1.mul(factor);

                // Reject where s ≥ 1 (or s == 0): fall back to Box-Muller.
                const bm_mag = uniform1.log().scale(-2.0).sqrt();
                const bm_phase = uniform2.scale(2.0 * std.math.pi);
                const bm_fallback = bm_mag.mul(bm_phase.cos());

                const one = zml.Tensor.scalar(1.0, shape.dtype());
                const eps = zml.Tensor.scalar(1e-10, shape.dtype());
                const valid = s.cmp(.LT, one.broadcast(shape, &.{}))
                    .logical(.AND, s.cmp(.GT, eps.broadcast(shape, &.{})));

                break :blk valid.select(polar, bm_fallback);
            },
        };

        return .{ rng2, latents_raw };
    }

    pub fn forward(rng: zml.Tensor.Rng, latents_raw_shape: zml.Shape) struct { zml.Tensor.Rng, zml.Tensor, zml.Tensor } {
        const rng_out, const latents_raw = randn(rng, latents_raw_shape, .box_muller);

        var dummy: flux_model.Flux2Transformer2DModel = undefined;
        const latents = dummy.pack_latents(latents_raw);
        const ids = dummy.prepare_latent_ids(latents_raw);
        return .{ rng_out, latents.convert(.bf16), ids };
    }

    pub fn forwardPolar(rng: zml.Tensor.Rng, latents_raw_shape: zml.Shape) struct { zml.Tensor.Rng, zml.Tensor, zml.Tensor } {
        const rng_out, const latents_raw = randn(rng, latents_raw_shape, .marsaglia_polar);

        var dummy: flux_model.Flux2Transformer2DModel = undefined;
        const latents = dummy.pack_latents(latents_raw);
        const ids = dummy.prepare_latent_ids(latents_raw);
        return .{ rng_out, latents.convert(.bf16), ids };
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

pub const ResolutionInfo = struct {
    width: usize,
    height: usize,
};

pub fn get_latents(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    config: flux_model.Config,
    img_dim: ResolutionInfo,
    generator_type: GeneratorType,
    seed: u64,
) !struct { zml.Buffer, zml.Buffer } {
    const in_channels = config.in_channels;
    const vae_scale_factor: usize = 8;

    const batch_size = 1;
    const num_channels_latents = @as(usize, @intCast(in_channels));
    const adjusted_height = (img_dim.height / (vae_scale_factor * 2));
    const adjusted_width = (img_dim.width / (vae_scale_factor * 2));

    const shape_latents = [_]usize{ batch_size, num_channels_latents, adjusted_height, adjusted_width };

    switch (generator_type) {
        .torch => {
            const latents_raw_shape = zml.Shape.init(.{ .b = @as(i64, @intCast(shape_latents[0])), .c = @as(i64, @intCast(shape_latents[1])), .h = @as(i64, @intCast(shape_latents[2])), .w = @as(i64, @intCast(shape_latents[3])) }, .bf16);
            var gen = TorchGenerator.init(seed);

            const rand_data = try gen.randnBf16(allocator, &shape_latents);
            defer allocator.free(rand_data);

            var latents_buffer = try zml.Buffer.fromBytes(io, platform, latents_raw_shape, std.mem.sliceAsBytes(rand_data));
            defer latents_buffer.deinit();
            return pack_latents(allocator, io, platform, latents_buffer);
        },
        inline .accelerator_box_muller, .accelerator_marsaglia => |gt| {
            const latents_raw_shape = zml.Shape.init(.{ .b = @as(i64, @intCast(shape_latents[0])), .c = @as(i64, @intCast(shape_latents[1])), .h = @as(i64, @intCast(shape_latents[2])), .w = @as(i64, @intCast(shape_latents[3])) }, .f32);
            const forward_fn = switch (gt) {
                .accelerator_box_muller => LatentGenerator.forward,
                .accelerator_marsaglia => LatentGenerator.forwardPolar,
                else => unreachable,
            };

            var rng_buffer = try zml.Tensor.Rng.initBuffer(platform, seed, io);
            defer rng_buffer._state.deinit();

            var exe = try zml.module.compile(allocator, io, forward_fn, .{ zml.Tensor.Rng.init(), latents_raw_shape }, platform);
            defer exe.deinit();

            var args = try exe.args(allocator);
            defer args.deinit(allocator);
            args.set(.{rng_buffer});

            var res = try exe.results(allocator);
            defer res.deinit(allocator);

            exe.call(args, &res);

            const out = res.get(struct { zml.Bufferized(zml.Tensor.Rng), zml.Buffer, zml.Buffer });
            var rng_out_state = out[0]._state;
            rng_out_state.deinit(); // discard updated Rng state
            return .{ out[1], out[2] };
        },
    }
}

/// Bit-exact port of PyTorch's CPU `torch.Generator` (MT19937 + Box-Muller).
///
/// Produces the same sequence as:
///   gen = torch.Generator(device="cpu").manual_seed(seed)
///   torch.randn(shape, generator=gen, dtype=torch.float32)
pub const TorchGenerator = struct {
    const MERSENNE_STATE_N = 624;
    const MERSENNE_STATE_M = 397;
    const MATRIX_A: u32 = 0x9908b0df;
    const UMASK: u32 = 0x80000000;
    const LMASK: u32 = 0x7fffffff;

    state: [MERSENNE_STATE_N]u32,
    left: i32,
    next_idx: u32,
    // Box-Muller caching: PyTorch produces two normals per pair of uniforms
    // and caches the second one.
    has_cached: bool,
    cached_normal: f32,

    pub fn init(seed: u64) TorchGenerator {
        var self: TorchGenerator = undefined;
        self.has_cached = false;
        self.cached_normal = 0;

        // PyTorch: init_with_uint32(seed)
        self.state[0] = @as(u32, @truncate(seed));
        for (1..MERSENNE_STATE_N) |j| {
            self.state[j] = 1812433253 *% (self.state[j - 1] ^ (self.state[j - 1] >> 30)) +% @as(u32, @intCast(j));
        }
        self.left = 1;
        self.next_idx = 0;
        return self;
    }

    fn mixBits(u_val: u32, v_val: u32) u32 {
        return (u_val & UMASK) | (v_val & LMASK);
    }

    fn twist(u_val: u32, v_val: u32) u32 {
        return (mixBits(u_val, v_val) >> 1) ^ (if (v_val & 1 != 0) MATRIX_A else 0);
    }

    fn nextState(self: *TorchGenerator) void {
        self.left = MERSENNE_STATE_N;
        self.next_idx = 0;

        var j: usize = 0;
        // First loop: MERSENNE_STATE_N - MERSENNE_STATE_M iterations
        while (j < MERSENNE_STATE_N - MERSENNE_STATE_M) : (j += 1) {
            self.state[j] = self.state[j + MERSENNE_STATE_M] ^ twist(self.state[j], self.state[j + 1]);
        }
        // Second loop: MERSENNE_STATE_M - 1 iterations
        while (j < MERSENNE_STATE_N - 1) : (j += 1) {
            self.state[j] = self.state[j + MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(self.state[j], self.state[j + 1]);
        }
        // Final element wraps around
        self.state[MERSENNE_STATE_N - 1] = self.state[MERSENNE_STATE_M - 1] ^ twist(self.state[MERSENNE_STATE_N - 1], self.state[0]);
    }

    /// Generate one u32 (tempered) — equivalent to mt19937_engine::operator()
    fn random(self: *TorchGenerator) u32 {
        self.left -= 1;
        if (self.left == 0) {
            self.nextState();
        }
        var y = self.state[self.next_idx];
        self.next_idx += 1;

        // Tempering
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= (y >> 18);
        return y;
    }

    /// PyTorch's uniform_real<float>(val, 0.0, 1.0):
    ///   MASK = (1 << 24) - 1 = 0xFFFFFF
    ///   result = (val & MASK) * (1.0 / (1 << 24))
    fn uniformFloat(self: *TorchGenerator) f32 {
        const val = self.random();
        const mask: u32 = (1 << 24) - 1;
        const divisor: f32 = 1.0 / @as(f32, @floatFromInt(@as(u32, 1 << 24)));
        return @as(f32, @floatFromInt(val & mask)) * divisor;
    }

    /// Transform 16 uniform values in-place to normal values using Box-Muller.
    /// Matches PyTorch's `normal_fill_16` exactly:
    ///   data[0..8] = u1 → radius * cos(theta)
    ///   data[8..16] = u2 → radius * sin(theta)
    fn normalFill16(data: *[16]f32) void {
        for (0..8) |j| {
            const u1_val = 1.0 - data[j]; // (0, 1] for log safety
            const u2_val = data[j + 8];
            const radius = @sqrt(@as(f32, -2.0) * @log(u1_val));
            const theta = @as(f32, 2.0 * std.math.pi) * u2_val;
            data[j] = radius * @cos(theta);
            data[j + 8] = radius * @sin(theta);
        }
    }

    /// Generate one u64 (two tempered u32s combined) — equivalent to generator->random64()
    fn random64(self: *TorchGenerator) u64 {
        const hi = self.random();
        const lo = self.random();
        return (@as(u64, hi) << 32) | @as(u64, lo);
    }

    /// PyTorch's uniform_real<double>(val, 0.0, 1.0):
    ///   MASK = (1 << 53) - 1
    ///   result = (val & MASK) * (1.0 / (1 << 53))
    fn uniformDouble(self: *TorchGenerator) f64 {
        const val = self.random64();
        const mask: u64 = (@as(u64, 1) << 53) - 1;
        const divisor: f64 = 1.0 / @as(f64, @floatFromInt(@as(u64, 1) << 53));
        return @as(f64, @floatFromInt(val & mask)) * divisor;
    }

    /// Generate one standard-normal sample using the sequential cached approach.
    /// Matches PyTorch's `normal_distribution<double>` path used for tensors < 16 elements.
    /// Uses f64 accumulators and random64() as PyTorch does.
    fn normalFloat(self: *TorchGenerator) f32 {
        if (self.has_cached) {
            self.has_cached = false;
            return self.cached_normal;
        }

        const uf1 = self.uniformDouble();
        const uf2 = self.uniformDouble();

        // PyTorch: r = sqrt(-2.0 * log1p(-u2)),  theta = 2π * u1
        const r: f64 = @sqrt(-2.0 * @log(1.0 - uf2));
        const theta: f64 = 2.0 * std.math.pi * uf1;

        // Cache the sin term (cast to f32) for next call
        self.cached_normal = @floatCast(r * @sin(theta));
        self.has_cached = true;

        return @floatCast(r * @cos(theta));
    }

    /// Fill an f32 slice with standard-normal samples, matching torch.randn(shape, generator=gen).
    ///
    /// For size >= 16: uses PyTorch's `normal_fill` algorithm:
    ///   1. Fill entire buffer with uniform values
    ///   2. Transform in-place in blocks of 16 via Box-Muller
    ///   3. If size % 16 != 0, regenerate last 16 uniforms and re-transform
    /// For size < 16: uses sequential cached Box-Muller (PyTorch's scalar path).
    pub fn randn(self: *TorchGenerator, allocator: std.mem.Allocator, shape: []const usize) ![]f32 {
        var count: usize = 1;
        for (shape) |s| count *= s;

        const result = try allocator.alloc(f32, count);
        errdefer allocator.free(result);

        if (count >= 16) {
            // Step 1: Fill entire buffer with uniform values
            for (result) |*item| {
                item.* = self.uniformFloat();
            }

            // Step 2: Transform in blocks of 16
            var i: usize = 0;
            while (i + 16 <= count) : (i += 16) {
                normalFill16(result[i..][0..16]);
            }

            // Step 3: Handle tail if size % 16 != 0
            if (count % 16 != 0) {
                const tail_start = count - 16;
                // Regenerate the last 16 uniform values
                for (result[tail_start..][0..16]) |*item| {
                    item.* = self.uniformFloat();
                }
                normalFill16(result[tail_start..][0..16]);
            }
        } else {
            // Small tensor: use sequential cached approach
            for (result) |*item| {
                item.* = self.normalFloat();
            }
        }
        return result;
    }

    /// Generate bf16 samples by converting f32 to bf16 format.
    /// This matches PyTorch's behavior: generate in f32, then cast to bf16.
    /// BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits (upper 16 bits of f32).
    pub fn randnBf16(self: *TorchGenerator, allocator: std.mem.Allocator, shape: []const usize) ![]u16 {
        // Generate f32 samples first
        const f32_samples = try self.randn(allocator, shape);
        defer allocator.free(f32_samples);

        // Convert to bf16 by extracting upper 16 bits
        const result = try allocator.alloc(u16, f32_samples.len);
        errdefer allocator.free(result);

        for (f32_samples, 0..) |val, idx| {
            const bits: u32 = @bitCast(val);
            result[idx] = @truncate(bits >> 16);
        }

        return result;
    }
};

pub fn prepare_text_ids(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, seq_len: usize) !zml.Buffer {
    const batch_size = 1;
    const count = batch_size * seq_len * 4;
    const data = try allocator.alloc(i64, count);
    defer allocator.free(data);

    // For Python default t_coord=None -> t=0
    for (0..seq_len) |s| {
        data[s * 4 + 0] = 0; // t
        data[s * 4 + 1] = 0; // h
        data[s * 4 + 2] = 0; // w
        data[s * 4 + 3] = @as(i64, @intCast(s)); // l
    }

    const shape = zml.Shape.init(.{ @as(i64, @intCast(batch_size)), @as(i64, @intCast(seq_len)), 4 }, .i64);

    return try zml.Buffer.fromBytes(io, platform, shape, std.mem.sliceAsBytes(data));
}

test "TorchGenerator matches torch.randn reference (seed=0)" {
    const allocator = std.testing.allocator;
    var gen = TorchGenerator.init(0);

    const shape = [_]usize{24};
    const samples = try gen.randn(allocator, &shape);
    defer allocator.free(samples);

    // Reference: torch.randn(24, generator=torch.Generator('cpu').manual_seed(0), dtype=torch.float32)
    const expected = [_]f32{
        -1.1258398294, -1.1523602009, -0.2505785823, -0.4338788390,
        0.8487103581,  0.6920092106,  -0.3160127699, -2.1152195930,
        0.4680964053,  -0.1577124447, 1.4436601400,  0.2660494149,
        0.1664553285,  0.8743818402,  -0.1434742361, -0.1116093323,
        0.9318266511,  1.2590092421,  2.0049805641,  0.0537368990,
        0.6180566549,  -0.4128022194, -0.8410646915, -2.3160419464,
    };

    for (samples, 0..) |val, i| {
        try std.testing.expectApproxEqAbs(expected[i], val, 1e-4);
    }
}

/// SIMD-optimized i64 to f32 conversion
inline fn convertI64ToF32Simd(src: []const i64, dst: []f32) void {
    const VecWidth = 4; // i64 vectors are typically smaller
    const vec_limit = (src.len / VecWidth) * VecWidth;
    var i: usize = 0;

    // Vectorized conversion
    while (i < vec_limit) : (i += VecWidth) {
        inline for (0..VecWidth) |j| {
            dst[i + j] = @floatFromInt(src[i + j]);
        }
    }

    // Handle remaining elements
    while (i < src.len) : (i += 1) {
        dst[i] = @floatFromInt(src[i]);
    }
}

/// SIMD-optimized i32 to f32 conversion
inline fn convertI32ToF32Simd(src: []const i32, dst: []f32, offset: usize) void {
    const VecWidth = 8;
    const Vec = @Vector(VecWidth, f32);
    const vec_limit = (src.len / VecWidth) * VecWidth;
    var i: usize = 0;

    // Vectorized conversion
    while (i < vec_limit) : (i += VecWidth) {
        var vec_result: Vec = undefined;
        inline for (0..VecWidth) |j| {
            vec_result[j] = @floatFromInt(src[i + j]);
        }
        inline for (0..VecWidth) |j| {
            dst[offset + i + j] = vec_result[j];
        }
    }

    // Handle remaining elements
    while (i < src.len) : (i += 1) {
        dst[offset + i] = @floatFromInt(src[i]);
    }
}

/// SIMD-optimized f32 to bf16 conversion
inline fn convertF32ToBf16Simd(src: []const f32, dst: []u16) void {
    const VecWidth = 8;
    const vec_limit = (src.len / VecWidth) * VecWidth;
    var i: usize = 0;

    // Vectorized conversion
    while (i < vec_limit) : (i += VecWidth) {
        inline for (0..VecWidth) |j| {
            const bits: u32 = @bitCast(src[i + j]);
            dst[i + j] = @truncate(bits >> 16);
        }
    }

    // Handle remaining elements
    while (i < src.len) : (i += 1) {
        const bits: u32 = @bitCast(src[i]);
        dst[i] = @truncate(bits >> 16);
    }
}

/// Compute timestep embedding with cos/sin and convert to bf16
/// Optimized for batch processing
inline fn computeTimestepEmbedding(t_val: f32, freq_vals: []const f32, out_bf16: []u16) void {
    // First 128 elements: cos, next 128: sin
    var temp_f32: [256]f32 = undefined;

    // Compute cos and sin (unrolled for better performance)
    const VecWidth = 4;
    const vec_limit = (128 / VecWidth) * VecWidth;
    var j: usize = 0;

    while (j < vec_limit) : (j += VecWidth) {
        inline for (0..VecWidth) |i| {
            const arg = t_val * freq_vals[j + i];
            temp_f32[j + i] = @cos(arg); // First half: cos
            temp_f32[128 + j + i] = @sin(arg); // Second half: sin
        }
    }

    // Handle remaining elements
    while (j < 128) : (j += 1) {
        const arg = t_val * freq_vals[j];
        temp_f32[j] = @cos(arg);
        temp_f32[128 + j] = @sin(arg);
    }

    // Convert to bf16 using optimized function
    convertF32ToBf16Simd(&temp_f32, out_bf16);
}

pub fn loadInput(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, path: []const u8, fallback_shape: anytype) !zml.Buffer {
    const npy = tools.NumpyData.load(allocator, path) catch {
        log.warn("Input not found at {s}, using zeros", .{path});
        const shape = zml.Shape.init(fallback_shape, .f32);
        const data = try allocator.alloc(f32, shape.count());
        defer allocator.free(data);
        @memset(data, 0);
        return try zml.Buffer.fromBytes(io, platform, shape, std.mem.sliceAsBytes(data));
    };
    var npy_v = npy;
    defer npy_v.deinit();
    log.info("Loaded input from {s}", .{path});
    return try npy_v.toBuffer(io, platform);
}

pub fn schedule(
    model_ctx: *const flux_model_transformer2d.Flux2Transformer2D,
    scheduler: *flow_match_euler_discrete_scheduler.FlowMatchEulerDiscreteScheduler,
    latents: zml.Buffer,
    latent_ids: zml.Buffer,
    prompt_embeds: zml.Buffer,
    text_ids: zml.Buffer,
    num_inference_steps: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
) !zml.Buffer {
    @setEvalBranchQuota(10_000);
    const image_seq_len: f64 = @floatFromInt(latents.shape().dim(1));

    const empirical_mu = compute_empirical_mu(image_seq_len, @floatFromInt(num_inference_steps));

    var sigmas = try std.ArrayList(f32).initCapacity(allocator, num_inference_steps);
    defer sigmas.deinit(allocator);

    const start_sigma: f32 = 1.0;
    const end_sigma: f32 = 1.0 / @as(f32, @floatFromInt(num_inference_steps));
    const step_sigma = if (num_inference_steps > 1) (end_sigma - start_sigma) / @as(f32, @floatFromInt(num_inference_steps - 1)) else 0;

    for (0..num_inference_steps) |i| {
        sigmas.appendAssumeCapacity(start_sigma + step_sigma * @as(f32, @floatFromInt(i)));
    }

    try scheduler.set_timesteps(num_inference_steps, sigmas.items, empirical_mu, null);

    const freq_vals_arr = flux_model_transformer2d.computeTimestepFrequencies(model_ctx.config.timestep_guidance_channels);

    // --- RoPE Precompute ---
    const txt_shape = text_ids.shape();
    const img_shape = latent_ids.shape();
    const txt_len = txt_shape.dim(1);
    const img_len = img_shape.dim(1);
    const total_len = txt_len + img_len;

    var txt_ids_slice = try zml.Slice.alloc(allocator, txt_shape);
    defer txt_ids_slice.free(allocator);
    try text_ids.toSlice(io, txt_ids_slice);

    var img_ids_slice = try zml.Slice.alloc(allocator, img_shape);
    defer img_ids_slice.free(allocator);
    try latent_ids.toSlice(io, img_ids_slice);

    const comb_shape = zml.Shape.init(.{ .b = 1, .s = total_len, .coord = 4 }, .f32);
    const txt_bytes = txt_ids_slice.items(u8);
    const img_bytes = img_ids_slice.items(u8);

    var comb_slice = try zml.Slice.alloc(allocator, comb_shape);
    defer comb_slice.free(allocator);
    const comb_bytes_slice = comb_slice.items(u8);

    const comb_f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(comb_bytes_slice)));

    const txt_i64 = std.mem.bytesAsSlice(i64, @as([]align(8) u8, @alignCast(txt_bytes)));
    // SIMD-optimized int to float conversion
    convertI64ToF32Simd(txt_i64, comb_f32[0..txt_i64.len]);

    const img_i32 = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(img_bytes)));
    const offset = txt_i64.len;
    // SIMD-optimized int to float conversion with offset
    convertI32ToF32Simd(img_i32, comb_f32, offset);

    const ids_data_slice = comb_f32;
    // std.log.info("RoPE Theta: {d}", .{model_ctx.config.rope_theta});

    const rope_slices = try flux_model_transformer2d.computeRotaryEmbedding(allocator, ids_data_slice, comb_shape, model_ctx.config.axes_dims_rope, model_ctx.config.rope_theta);
    defer rope_slices[0].free(allocator);
    defer rope_slices[1].free(allocator);

    var rotary_cos_dev = try zml.Buffer.fromSlice(io, platform, rope_slices[0]);
    var rotary_sin_dev = try zml.Buffer.fromSlice(io, platform, rope_slices[1]);
    defer rotary_cos_dev.deinit();
    defer rotary_sin_dev.deinit();

    // -----------------------
    // Compilation of Step and Euler is now handled in Flux2Transformer2D.loadFromFile (stored in model_ctx)
    // -----------------------

    const t_proj_shape = zml.Shape.init(.{ .b = 1, .d = 256 }, .bf16);
    const sym_dt = zml.Tensor.fromShape(zml.Shape.init(.{}, .f32));

    var current_latents = latents;
    var is_first = true;
    scheduler.set_begin_index(0);

    // TODO: Make this configurable
    const guidance_scale: f32 = 3.5;

    // Guidance Projection (constant across all steps — compute once)
    const g_val: f32 = guidance_scale * 1000.0; // Python multiplies by 1000 internally
    var g_proj_bf16: [256]u16 = undefined;
    // SIMD-optimized timestep embedding computation
    computeTimestepEmbedding(g_val, &freq_vals_arr, &g_proj_bf16);
    var g_proj_buf = try zml.Buffer.fromBytes(io, platform, t_proj_shape, std.mem.sliceAsBytes(&g_proj_bf16));
    defer g_proj_buf.deinit();

    // Batch-precompute all timestep projection buffers (timesteps are known ahead of time)
    const t_proj_bufs = try allocator.alloc(zml.Buffer, scheduler.timesteps.len);
    defer {
        for (t_proj_bufs) |*buf| buf.deinit();
        allocator.free(t_proj_bufs);
    }
    for (scheduler.timesteps, 0..) |t_step, step_i| {
        const t_val: f32 = t_step;
        var t_proj_bf16: [256]u16 = undefined;
        // SIMD-optimized timestep embedding computation
        computeTimestepEmbedding(t_val, &freq_vals_arr, &t_proj_bf16);
        t_proj_bufs[step_i] = try zml.Buffer.fromBytes(io, platform, t_proj_shape, std.mem.sliceAsBytes(&t_proj_bf16));
    }

    var args_step = try model_ctx.step_exe.args(allocator);
    defer args_step.deinit(allocator);
    var res_step = try model_ctx.step_exe.results(allocator);
    defer res_step.deinit(allocator);
    var args_euler = try model_ctx.euler_exe.args(allocator);
    defer args_euler.deinit(allocator);
    var res_euler = try model_ctx.euler_exe.results(allocator);
    defer res_euler.deinit(allocator);

    for (0..scheduler.timesteps.len) |step_i| {
        // log.info("   Step {d}/{d} (t={d:.2})", .{ step_i + 1, num_inference_steps, scheduler.timesteps[step_i] });

        args_step.set(.{ model_ctx.weights, current_latents, prompt_embeds, t_proj_bufs[step_i], g_proj_buf, rotary_cos_dev, rotary_sin_dev });

        model_ctx.step_exe.call(args_step, &res_step);

        var noise_pred_buf = res_step.get(zml.Buffer);
        defer noise_pred_buf.deinit();

        // Calculate dt on CPU
        const step_idx = scheduler.step_index.?;
        const sigma = scheduler.sigmas[step_idx];
        const sigma_next = scheduler.sigmas[step_idx + 1];
        const dt = sigma_next - sigma;
        scheduler.step_index.? += 1;

        // Create dt buffer
        var dt_arr = [1]f32{dt};
        var dt_buf = try zml.Buffer.fromBytes(io, platform, sym_dt.shape(), std.mem.sliceAsBytes(&dt_arr));
        defer dt_buf.deinit();

        // Run Euler Step on Device
        args_euler.set(.{ current_latents, noise_pred_buf, dt_buf });

        model_ctx.euler_exe.call(args_euler, &res_euler);

        const next_latents_buf = res_euler.get(zml.Buffer);

        if (!is_first) current_latents.deinit();
        current_latents = next_latents_buf;
        is_first = false;
    }

    // try tools.printFlatten(allocator, io, current_latents, 20, "Latents after steps (first 20).", .{ .include_shape = true });
    // unpack_latents_with_ids - scatter tokens into correct spatial positions
    const unpacked = try unpackLatentsWithIds(allocator, io, platform, current_latents, latent_ids);
    // try tools.printFlatten(allocator, io, unpacked, 20, "Unpacked Latents (first 20).", .{ .include_shape = true });

    if (!is_first) current_latents.deinit();

    return unpacked;
}

/// Implements unpack_latents_with_ids from Python
/// Scatters latent tokens into correct spatial positions using position IDs
/// Optimized with SIMD vectorization and branchless operations
fn unpackLatentsWithIds(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    latents: zml.Buffer,
    latent_ids: zml.Buffer,
) !zml.Buffer {
    const lat_shape = latents.shape();
    const ids_shape = latent_ids.shape();

    const batch_size = lat_shape.dim(0);
    const seq_len = lat_shape.dim(1);
    const channels = lat_shape.dim(2);

    // Precompute dimension sizes as usize to avoid repeated @intCast in loops
    const bs: usize = @intCast(batch_size);
    const sl: usize = @intCast(seq_len);
    const ch: usize = @intCast(channels);

    // Read latents and IDs to CPU
    var lat_slice = try zml.Slice.alloc(allocator, lat_shape);
    defer lat_slice.free(allocator);
    try latents.toSlice(io, lat_slice);

    var ids_slice = try zml.Slice.alloc(allocator, ids_shape);
    defer ids_slice.free(allocator);
    try latent_ids.toSlice(io, ids_slice);

    const lat_data_bf16 = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(lat_slice.items(u8))));
    const ids_data = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(ids_slice.items(u8))));

    // Find max h and w from IDs (column 1 = h_ids, column 2 = w_ids)
    // Branchless max finding using SIMD for better performance
    var max_h: i32 = 0;
    var max_w: i32 = 0;

    // SIMD-accelerated max finding
    const VecWidth = 8;
    const Vec = @Vector(VecWidth, i32);
    var vec_max_h: Vec = @splat(0);
    var vec_max_w: Vec = @splat(0);

    const vec_limit = (sl / VecWidth) * VecWidth;
    var s: usize = 0;

    // Vectorized max finding
    while (s < vec_limit) : (s += VecWidth) {
        var vec_h: Vec = undefined;
        var vec_w: Vec = undefined;

        inline for (0..VecWidth) |i| {
            vec_h[i] = ids_data[(s + i) * 4 + 1];
            vec_w[i] = ids_data[(s + i) * 4 + 2];
        }

        vec_max_h = @max(vec_max_h, vec_h);
        vec_max_w = @max(vec_max_w, vec_w);
    }

    // Reduce vector to scalar using @reduce
    max_h = @reduce(.Max, vec_max_h);
    max_w = @reduce(.Max, vec_max_w);

    // Handle remaining elements
    while (s < sl) : (s += 1) {
        const h_id = ids_data[s * 4 + 1];
        const w_id = ids_data[s * 4 + 2];
        max_h = @max(max_h, h_id);
        max_w = @max(max_w, w_id);
    }

    const h: usize = @intCast(max_h + 1);
    const w: usize = @intCast(max_w + 1);

    // Allocate output: [B, C, H, W] — scatter bf16 values directly (no f32 intermediate)
    const out_shape = zml.Shape.init(.{ batch_size, channels, h, w }, .bf16);
    const out_count = out_shape.count();
    const out_bf16 = try allocator.alloc(u16, out_count);
    defer allocator.free(out_bf16);

    // Initialize to zero (optimized with @memset)
    @memset(out_bf16, 0);

    // Pre-compute stride for better performance
    const channel_hw_stride = h * w;

    // Scatter tokens into place using optimized channel copying
    for (0..bs) |b| {
        const b_out_base = b * ch * channel_hw_stride;
        const b_in_base = b * sl * ch;

        for (0..sl) |seq_idx| {
            const h_id: usize = @intCast(ids_data[b * sl * 4 + seq_idx * 4 + 1]);
            const w_id: usize = @intCast(ids_data[b * sl * 4 + seq_idx * 4 + 2]);
            const spatial_offset = h_id * w + w_id;
            const in_base = b_in_base + seq_idx * ch;

            // SIMD-optimized channel copying
            try scatterChannelsSimd(lat_data_bf16, out_bf16, in_base, b_out_base, spatial_offset, channel_hw_stride, ch);
        }
    }

    return try zml.Buffer.fromBytes(io, platform, out_shape, std.mem.sliceAsBytes(out_bf16));
}

/// SIMD-optimized scatter operation for channel data
/// Copies channel data from input to output at scattered spatial positions
inline fn scatterChannelsSimd(
    src: []const u16,
    dst: []u16,
    in_base: usize,
    out_base: usize,
    spatial_offset: usize,
    channel_hw_stride: usize,
    num_channels: usize,
) !void {
    // Use SIMD for bulk channel copying when possible
    const VecWidth = 8;
    const Vec = @Vector(VecWidth, u16);

    const vec_limit = (num_channels / VecWidth) * VecWidth;
    var c: usize = 0;

    // Vectorized channel copying
    while (c < vec_limit) : (c += VecWidth) {
        var vec_data: Vec = undefined;

        // Load from contiguous input
        inline for (0..VecWidth) |i| {
            vec_data[i] = src[in_base + c + i];
        }

        // Store to scattered output positions
        inline for (0..VecWidth) |i| {
            const out_idx = out_base + (c + i) * channel_hw_stride + spatial_offset;
            dst[out_idx] = vec_data[i];
        }
    }

    // Handle remaining channels (scalar tail)
    while (c < num_channels) : (c += 1) {
        const in_idx = in_base + c;
        const out_idx = out_base + c * channel_hw_stride + spatial_offset;
        dst[out_idx] = src[in_idx];
    }
}

const RgbImage = struct {
    w: c_int,
    h: c_int,
    comp: c_int,
    stride_in_bytes: c_int,
    data: []u8,

    pub fn free(self: @This(), allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

pub fn decodeImageToRgb(allocator: std.mem.Allocator, io: std.Io, image_decoded_buf: zml.Buffer) !RgbImage {
    const shape = image_decoded_buf.shape();
    const dim_c = shape.dim(1);
    _ = dim_c;
    const dim_h = shape.dim(2);
    const dim_w = shape.dim(3);

    const w: c_int = @intCast(dim_w);
    const h: c_int = @intCast(dim_h);
    const comp: c_int = 3;
    const stride_in_bytes: c_int = w * comp;

    // Fetch data from device
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    try image_decoded_buf.toSlice(io, slice);

    const rgb_data = try allocator.alloc(u8, @intCast(dim_h * dim_w * comp));

    const height: usize = @intCast(dim_h);
    const width: usize = @intCast(dim_w);
    const dtype = shape.dtype();

    // Pre-compute channel strides to avoid repeated calculations
    const channel_stride = height * width;

    // Fast path specialization for common data types
    switch (dtype) {
        .bf16 => try decodeImageToRgbTyped(u16, .bf16, slice, rgb_data, height, width, channel_stride),
        .f32 => try decodeImageToRgbTyped(f32, .f32, slice, rgb_data, height, width, channel_stride),
        .f16 => try decodeImageToRgbTyped(f16, .f16, slice, rgb_data, height, width, channel_stride),
        else => {
            // Fallback for other types (slower path)
            var idx: usize = 0;
            for (0..height) |y| {
                for (0..width) |x| {
                    for (0..3) |c| {
                        const in_idx = c * channel_stride + y * width + x;
                        const val_f64 = tools.getElementAsF64(slice, dtype, in_idx);
                        // Branchless clamping: clamp((val_f64 / 2.0 + 0.5), 0.0, 1.0)
                        const val_clamped = @max(0.0, @min(1.0, (val_f64 / 2.0) + 0.5));
                        rgb_data[idx] = @intFromFloat(val_clamped * 255.0);
                        idx += 1;
                    }
                }
            }
        },
    }

    return .{ .w = w, .h = h, .comp = comp, .stride_in_bytes = stride_in_bytes, .data = rgb_data };
}

/// Optimized type-specialized decoder using SIMD and branchless operations
inline fn decodeImageToRgbTyped(
    comptime T: type,
    comptime dtype_tag: zml.DataType,
    slice: zml.Slice,
    rgb_data: []u8,
    height: usize,
    width: usize,
    channel_stride: usize,
) !void {
    const src_data = slice.items(T);

    // SIMD vector width for processing multiple pixels at once
    const VecWidth = 8;
    const Vec = @Vector(VecWidth, f32);

    const vec_scale: Vec = @splat(0.5);
    const vec_offset: Vec = @splat(0.5);
    const vec_zero: Vec = @splat(0.0);
    const vec_one: Vec = @splat(1.0);
    const vec_255: Vec = @splat(255.0);

    const total_pixels = height * width;
    const vec_limit = (total_pixels / VecWidth) * VecWidth;

    // Process each channel
    for (0..3) |c| {
        const channel_offset = c * channel_stride;
        var pixel_idx: usize = 0;

        // SIMD vectorized loop - process 8 pixels at a time
        while (pixel_idx < vec_limit) : (pixel_idx += VecWidth) {
            var vec_vals: Vec = undefined;

            // Load values based on dtype
            inline for (0..VecWidth) |i| {
                const src_idx = channel_offset + pixel_idx + i;
                vec_vals[i] = switch (dtype_tag) {
                    .bf16 => blk: {
                        const raw = src_data[src_idx];
                        const u32_val = @as(u32, raw) << 16;
                        const f_val: f32 = @bitCast(u32_val);
                        break :blk f_val;
                    },
                    .f32 => src_data[src_idx],
                    .f16 => @floatCast(src_data[src_idx]),
                    else => unreachable,
                };
            }

            // Vectorized: (val / 2.0) + 0.5
            vec_vals = vec_vals * vec_scale + vec_offset;

            // Branchless clamping to [0, 1]
            vec_vals = @max(vec_zero, @min(vec_one, vec_vals));

            // Scale to [0, 255]
            vec_vals = vec_vals * vec_255;

            // Store results
            inline for (0..VecWidth) |i| {
                const out_idx = (pixel_idx + i) * 3 + c;
                rgb_data[out_idx] = @intFromFloat(vec_vals[i]);
            }
        }

        // Handle remaining pixels (scalar tail)
        while (pixel_idx < total_pixels) : (pixel_idx += 1) {
            const src_idx = channel_offset + pixel_idx;

            const val_f32: f32 = switch (dtype_tag) {
                .bf16 => blk: {
                    const raw = src_data[src_idx];
                    const u32_val = @as(u32, raw) << 16;
                    const f_val: f32 = @bitCast(u32_val);
                    break :blk f_val;
                },
                .f32 => src_data[src_idx],
                .f16 => @floatCast(src_data[src_idx]),
                else => unreachable,
            };

            // Branchless clamping
            const val_clamped = @max(0.0, @min(1.0, (val_f32 / 2.0) + 0.5));
            const out_idx = pixel_idx * 3 + c;
            rgb_data[out_idx] = @intFromFloat(val_clamped * 255.0);
        }
    }
}

pub fn saveFluxImageToPng(allocator: std.mem.Allocator, rgb: *const RgbImage, filename: []const u8) !void {
    const filename_z = try allocator.dupeZ(u8, filename);
    defer allocator.free(filename_z);

    if (c_interface.stbi_write_png(filename_z.ptr, rgb.w, rgb.h, rgb.comp, rgb.data.ptr, rgb.stride_in_bytes) == 0) {
        log.err("Failed to write PNG to {s}", .{filename});
    }
}

const StbiWriteContext = struct {
    list: std.ArrayList(u8),
    allocator: std.mem.Allocator,
    failed: bool = false,
};

fn stbiWriteCallback(context: ?*anyopaque, data: ?*anyopaque, size: c_int) callconv(.c) void {
    if (context == null or data == null or size <= 0) return;
    const ctx: *StbiWriteContext = @ptrCast(@alignCast(context.?));
    if (ctx.failed) return;

    const bytes = @as([*]const u8, @ptrCast(data.?))[0..@as(usize, @intCast(size))];
    ctx.list.appendSlice(ctx.allocator, bytes) catch {
        ctx.failed = true;
    };
}

fn encodePngToMemory(allocator: std.mem.Allocator, rgb_image_buffer: *const RgbImage) ![]u8 {
    var ctx = StbiWriteContext{ .list = .{}, .allocator = allocator };
    errdefer ctx.list.deinit(allocator);
    _ = c_interface.stbi_write_png_to_func(stbiWriteCallback, &ctx, rgb_image_buffer.w, rgb_image_buffer.h, rgb_image_buffer.comp, rgb_image_buffer.data.ptr, rgb_image_buffer.stride_in_bytes);

    if (ctx.failed) return error.OutOfMemory;
    return ctx.list.toOwnedSlice(allocator);
}

/// Print a PNG image to a Kitty-compatible terminal using the graphics protocol.
/// This avoids writing the image to disk and is suitable for Ghostty.
pub fn printFluxImageToTerminalKittyFromBuffer(allocator: std.mem.Allocator, rgb_image_buffer: *const RgbImage) !void {
    const png_bytes = try encodePngToMemory(allocator, rgb_image_buffer);
    defer allocator.free(png_bytes);

    const b64_len = std.base64.standard.Encoder.calcSize(png_bytes.len);
    const b64_buf = try allocator.alloc(u8, b64_len);
    defer allocator.free(b64_buf);
    const b64 = std.base64.standard.Encoder.encode(b64_buf, png_bytes);

    const esc = "\x1b_G";
    const st = "\x1b\\";
    const chunk_size: usize = 4096;

    var offset: usize = 0;
    while (offset < b64.len) {
        const remaining = b64.len - offset;
        const chunk_len = @min(chunk_size, remaining);
        const more: u8 = if (offset + chunk_len < b64.len) 1 else 0;

        var header_buf: [128]u8 = undefined;
        const header = try std.fmt.bufPrint(&header_buf, "{s}a=T,f=100,t=d,m={d};", .{ esc, more });

        std.debug.print("{s}{s}{s}", .{ header, b64[offset .. offset + chunk_len], st });
        offset += chunk_len;
    }

    std.debug.print("\n", .{});
}

pub fn unloadWeights(allocator: std.mem.Allocator, weights: anytype) void {
    const T = @TypeOf(weights.*);
    const type_info = @typeInfo(T);
    switch (type_info) {
        .@"struct" => |info| {
            if (T == zml.Buffer) {
                weights.deinit();
                return;
            }
            inline for (info.fields) |field| {
                unloadWeights(allocator, &@field(weights, field.name));
            }
        },
        .optional => {
            if (weights.*) |*w| {
                unloadWeights(allocator, w);
            }
        },
        .pointer => |info| {
            if (info.size == .slice) {
                for (weights.*) |*item| {
                    unloadWeights(allocator, item);
                }
                allocator.free(weights.*);
            }
        },
        else => {},
    }
}

pub const RoPE = struct {
    cos: zml.Tensor,
    sin: zml.Tensor,
};
