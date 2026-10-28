//! Port of PyTorch's CPU `torch.Generator` (MT19937 + Box-Muller), from the kevin/flux2 branch.
//!
//! Produces the same sequence as:
//!   gen = torch.Generator(device="cpu").manual_seed(seed)
//!   torch.randn(shape, generator=gen, dtype=torch.float32)
//!
//! `torch.randn(..., dtype=torch.bfloat16)` is exactly these f32 samples rounded to nearest-even, so convert
//! them with `Tensor.convert(.bf16)` rather than truncating the bits.
//! Note: `torch.Generator("cuda")` uses Philox, a different algorithm, see `gpu.zig`.

const std = @import("std");

const CpuGenerator = @This();

const MERSENNE_STATE_N = 624;
const MERSENNE_STATE_M = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UMASK: u32 = 0x80000000;
const LMASK: u32 = 0x7fffffff;

state: [MERSENNE_STATE_N]u32,
left: i32,
next_idx: u32,
// Box-Muller caching: PyTorch produces two normals per pair of uniforms and caches the second one.
has_cached: bool,
cached_normal: f32,

pub fn init(seed: u64) CpuGenerator {
    var self: CpuGenerator = undefined;
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

fn nextState(self: *CpuGenerator) void {
    self.left = MERSENNE_STATE_N;
    self.next_idx = 0;

    var j: usize = 0;
    while (j < MERSENNE_STATE_N - MERSENNE_STATE_M) : (j += 1) {
        self.state[j] = self.state[j + MERSENNE_STATE_M] ^ twist(self.state[j], self.state[j + 1]);
    }
    while (j < MERSENNE_STATE_N - 1) : (j += 1) {
        self.state[j] = self.state[j + MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(self.state[j], self.state[j + 1]);
    }
    self.state[MERSENNE_STATE_N - 1] = self.state[MERSENNE_STATE_M - 1] ^ twist(self.state[MERSENNE_STATE_N - 1], self.state[0]);
}

/// One tempered u32, like mt19937_engine::operator()
fn random(self: *CpuGenerator) u32 {
    self.left -= 1;
    if (self.left == 0) {
        self.nextState();
    }
    var y = self.state[self.next_idx];
    self.next_idx += 1;

    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);
    return y;
}

/// PyTorch's uniform_real<float>: the low 24 bits scaled to [0, 1).
fn uniformFloat(self: *CpuGenerator) f32 {
    const val = self.random();
    const mask: u32 = (1 << 24) - 1;
    const divisor: f32 = 1.0 / @as(f32, @floatFromInt(@as(u32, 1 << 24)));
    return @as(f32, @floatFromInt(val & mask)) * divisor;
}

/// PyTorch's `normal_fill_16`: data[0..8] become radius * cos(theta), data[8..16] radius * sin(theta).
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

/// Two tempered u32s combined, like generator->random64()
fn random64(self: *CpuGenerator) u64 {
    const hi = self.random();
    const lo = self.random();
    return (@as(u64, hi) << 32) | @as(u64, lo);
}

/// PyTorch's uniform_real<double>: the low 53 bits scaled to [0, 1).
fn uniformDouble(self: *CpuGenerator) f64 {
    const val = self.random64();
    const mask: u64 = (@as(u64, 1) << 53) - 1;
    const divisor: f64 = 1.0 / @as(f64, @floatFromInt(@as(u64, 1) << 53));
    return @as(f64, @floatFromInt(val & mask)) * divisor;
}

/// PyTorch's sequential `normal_distribution<double>` path, used for tensors of less than 16 elements.
fn normalFloat(self: *CpuGenerator) f32 {
    if (self.has_cached) {
        self.has_cached = false;
        return self.cached_normal;
    }

    const uf1 = self.uniformDouble();
    const uf2 = self.uniformDouble();

    // PyTorch: r = sqrt(-2.0 * log1p(-u2)), theta = 2π * u1
    const r: f64 = @sqrt(-2.0 * @log(1.0 - uf2));
    const theta: f64 = 2.0 * std.math.pi * uf1;

    self.cached_normal = @floatCast(r * @sin(theta));
    self.has_cached = true;

    return @floatCast(r * @cos(theta));
}

/// Standard-normal samples, like torch.randn(shape, generator=gen). Caller owns the result.
///
/// For 16 elements or more, PyTorch's `normal_fill`: fill the whole buffer with uniforms, transform it in
/// blocks of 16 with Box-Muller, and if the size isn't a multiple of 16 regenerate and transform the last 16.
pub fn randn(self: *CpuGenerator, allocator: std.mem.Allocator, count: usize) ![]f32 {
    const result = try allocator.alloc(f32, count);
    errdefer allocator.free(result);

    if (count >= 16) {
        for (result) |*item| {
            item.* = self.uniformFloat();
        }

        var i: usize = 0;
        while (i + 16 <= count) : (i += 16) {
            normalFill16(result[i..][0..16]);
        }

        if (count % 16 != 0) {
            const tail_start = count - 16;
            for (result[tail_start..][0..16]) |*item| {
                item.* = self.uniformFloat();
            }
            normalFill16(result[tail_start..][0..16]);
        }
    } else {
        for (result) |*item| {
            item.* = self.normalFloat();
        }
    }
    return result;
}

// Expected values from torch.randn(shape, generator=torch.Generator("cpu").manual_seed(42)).
// The tolerance covers PyTorch's vectorized log/cos/sin, which can differ from Zig's by an ulp.
fn expectSamples(expected: []const f32, actual: []const f32) !void {
    for (expected, actual) |e, a| try std.testing.expectApproxEqRel(e, a, 1e-6);
}

test "randn matches torch for the Qwen-Image 2.1 2048x2048 latents" {
    var gen: CpuGenerator = .init(42);
    // (1, 1, 64, 128, 128), what diffusers' prepare_latents draws
    const samples = try gen.randn(std.testing.allocator, 64 * 128 * 128);
    defer std.testing.allocator.free(samples);

    try expectSamples(&.{ 1.9269152879714966, 1.4872840642929077, 0.9007171988487244, -2.1055209636688232 }, samples[0..4]);
    try expectSamples(&.{-0.7395542860031128}, samples[524287..][0..1]);
    try expectSamples(&.{ -0.9463756084442139, -1.024017333984375, -0.40064486861228943, 0.285871297121048 }, samples[samples.len - 4 ..]);
}

test "randn matches torch when the size isn't a multiple of 16" {
    var gen: CpuGenerator = .init(42);
    const samples = try gen.randn(std.testing.allocator, 20);
    defer std.testing.allocator.free(samples);

    try expectSamples(&.{ -0.3097267746925354, -0.3957104980945587, 0.8034093379974365, -0.6215953826904297 }, samples[16..]);
}

test "randn matches torch for less than 16 elements" {
    var gen: CpuGenerator = .init(42);
    const samples = try gen.randn(std.testing.allocator, 5);
    defer std.testing.allocator.free(samples);

    try expectSamples(&.{ 0.33669036626815796, 0.12880940735340118, 0.23446236550807953, 0.23033303022384644, -1.1228563785552979 }, samples);
}
