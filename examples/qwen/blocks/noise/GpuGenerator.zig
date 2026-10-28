//! Port of PyTorch's CUDA `torch.Generator` (Philox4x32-10 + curand_normal4).
//!
//! Produces the same sequence as:
//!   gen = torch.Generator(device="cuda").manual_seed(seed)
//!   torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)
//!
//! `torch.randn(..., dtype=torch.bfloat16)` is exactly these f32 samples rounded to nearest-even, so convert
//! them with `Tensor.convert(.bf16)` rather than truncating the bits.
//!
//! Unlike the CPU generator, the sequence depends on the GPU: PyTorch's grid-stride kernel sizes its grid from
//! the SM count and the max threads per SM, and every thread draws from its own Philox subsequence. Use the
//! `Device` of the GPU the reference ran on.
//!
//! The device computes sin/cos with the `__sincosf` fast intrinsic, which isn't reproducible on the CPU, so the
//! samples are within ~1e-6 of PyTorch's rather than bit-exact. Rounded to bf16 they almost always match exactly.

const std = @import("std");

const GpuGenerator = @This();

/// `cudaDeviceProp` fields PyTorch's `calc_execution_policy` reads.
pub const Device = struct {
    multi_processor_count: u32,
    max_threads_per_multi_processor: u32,

    pub const rtx_5090: Device = .{ .multi_processor_count = 170, .max_threads_per_multi_processor = 1536 };
};

// aten/src/ATen/native/cuda/DistributionTemplates.h
const BLOCK_SIZE = 256;
const MAX_GENERATOR_OFFSETS_PER_CURAND_CALL = 4;
// curand_normal4 returns a float4
const UNROLL_FACTOR = 4;

const PHILOX_W32_0: u32 = 0x9E3779B9;
const PHILOX_W32_1: u32 = 0xBB67AE85;
const PHILOX_M4x32_0: u32 = 0xD2511F53;
const PHILOX_M4x32_1: u32 = 0xCD9E8D57;

// curand_globals.h, both products are evaluated in f32
const CURAND_2POW32_INV: f32 = 2.3283064e-10;
const CURAND_2POW32_INV_2PI: f32 = @as(f32, 2.3283064e-10) * @as(f32, 6.2831855);

seed: u64,
/// CUDAGeneratorImpl's philox_offset_per_thread_, in 32-bit draws.
offset: u64,
device: Device,

pub fn init(seed: u64, device: Device) GpuGenerator {
    return .{ .seed = seed, .offset = 0, .device = device };
}

/// One Philox4x32-10 block, like curand_Philox4x32_10.
fn philox(counter: u128, seed: u64) [4]u32 {
    var c: [4]u32 = .{ @truncate(counter), @truncate(counter >> 32), @truncate(counter >> 64), @truncate(counter >> 96) };
    var k: [2]u32 = .{ @truncate(seed), @truncate(seed >> 32) };
    for (0..10) |round| {
        if (round > 0) {
            k[0] +%= PHILOX_W32_0;
            k[1] +%= PHILOX_W32_1;
        }
        const p0 = @as(u64, PHILOX_M4x32_0) * c[0];
        const p1 = @as(u64, PHILOX_M4x32_1) * c[2];
        c = .{
            @as(u32, @truncate(p1 >> 32)) ^ c[1] ^ k[0],
            @truncate(p1),
            @as(u32, @truncate(p0 >> 32)) ^ c[3] ^ k[1],
            @truncate(p0),
        };
    }
    return c;
}

/// _curand_box_muller: two uniforms in (0, 1] to { radius * sin(theta), radius * cos(theta) }.
/// nvcc contracts the uniform scalings into FMAs.
fn boxMuller(x: u32, y: u32) [2]f32 {
    const u = @mulAdd(f32, @floatFromInt(x), CURAND_2POW32_INV, CURAND_2POW32_INV / 2);
    const v = @mulAdd(f32, @floatFromInt(y), CURAND_2POW32_INV_2PI, CURAND_2POW32_INV_2PI / 2);
    const s = @sqrt(-2.0 * @log(u));
    return .{ @sin(v) * s, @cos(v) * s };
}

/// Standard-normal samples, like torch.randn(shape, generator=gen) on a CUDA generator. Caller owns the result.
///
/// PyTorch's distribution_elementwise_grid_stride_kernel: thread `idx` of `threads` draws curand_normal4 from
/// subsequence `idx`, and its k-th float4 fills elements k * 4 * threads + j * threads + idx for j in 0..4.
pub fn randn(self: *GpuGenerator, allocator: std.mem.Allocator, count: usize) ![]f32 {
    const result = try allocator.alloc(f32, count);
    errdefer allocator.free(result);
    if (count == 0) return result;
    // Larger tensors are split in 32-bit indexable chunks, each with its own launch.
    std.debug.assert(count <= std.math.maxInt(i32));

    // calc_execution_policy
    const blocks_per_sm = self.device.max_threads_per_multi_processor / BLOCK_SIZE;
    const grid: usize = @min(self.device.multi_processor_count * blocks_per_sm, std.math.divCeil(usize, count, BLOCK_SIZE) catch unreachable);
    const threads = grid * BLOCK_SIZE;
    const calls_per_thread = (count - 1) / (threads * UNROLL_FACTOR) + 1;

    // philox_cuda_state: hands out the current offset and advances it, rounded up to a multiple of 4.
    const offset = self.offset;
    const counter_offset: u64 = calls_per_thread * MAX_GENERATOR_OFFSETS_PER_CURAND_CALL;
    self.offset += std.mem.alignForward(u64, counter_offset, 4);
    // The offset is always a multiple of 4, so curand4 returns whole Philox blocks (curandState STATE == 0).
    std.debug.assert(offset % 4 == 0);

    for (0..calls_per_thread) |k| {
        for (0..threads) |idx| {
            // curand_init(seed, idx, offset): the subsequence goes in the high 64 bits of the counter, the
            // offset in the low ones, then each curand4 call increments the whole 128-bit counter.
            const counter = (@as(u128, idx) << 64) + offset / 4 + k;
            const bits = philox(counter, self.seed);
            const n01 = boxMuller(bits[0], bits[1]);
            const n23 = boxMuller(bits[2], bits[3]);
            const normals: [4]f32 = .{ n01[0], n01[1], n23[0], n23[1] };
            for (normals, 0..) |n, j| {
                const i = k * threads * UNROLL_FACTOR + j * threads + idx;
                if (i < count) result[i] = n;
            }
        }
    }
    return result;
}

// Expected values from torch.randn(shape, generator=torch.Generator("cuda").manual_seed(42)) on an RTX 5090.
// The tolerance covers the device's `__sincosf` and `logf`, which differ from Zig's by a few ulps.
fn expectSample(expected: f32, actual: f32) !void {
    try std.testing.expectApproxEqAbs(expected, actual, 1e-5);
}

test "randn matches torch for the Qwen-Image 2.1 2048x2048 latents" {
    var gen: GpuGenerator = .init(42, .rtx_5090);
    // (1, 1, 64, 128, 128), what diffusers' prepare_latents draws
    const samples = try gen.randn(std.testing.allocator, 64 * 128 * 128);
    defer std.testing.allocator.free(samples);

    // 1020 blocks of 256 threads, so 261120 threads each drawing two float4.
    const threads = 1020 * 256;
    const expected = [_]struct { usize, f32 }{
        .{ 0, 0.19401879608631134 },
        .{ 1, 2.1613736152648926 },
        .{ 2, -0.17205022275447845 },
        .{ 3, 0.8490601181983948 },
        .{ threads - 1, 1.3229272365570068 },
        .{ threads, -0.9701897501945496 }, // 2nd value of thread 0's first float4
        .{ 2 * threads, 1.9239264726638794 },
        .{ 3 * threads, -1.2356209754943848 },
        .{ 4 * threads, 0.13913755118846893 }, // thread 0's second float4
        .{ 4 * threads + 1, -0.10821663588285446 },
        .{ samples.len - 1, 1.1164684295654297 },
    };
    for (expected) |e| try expectSample(e[1], samples[e[0]]);
}

test "randn advances the Philox offset between calls" {
    var gen: GpuGenerator = .init(42, .rtx_5090);
    const first = try gen.randn(std.testing.allocator, 64 * 128 * 128);
    defer std.testing.allocator.free(first);
    const samples = try gen.randn(std.testing.allocator, 20);
    defer std.testing.allocator.free(samples);

    const expected = [_]f32{
        -0.5187486410140991, 1.2267580032348633,   0.6254805326461792,   -0.9116847515106201,
        0.568683385848999,   -0.33366551995277405, 0.05579303577542305,  -0.7529569864273071,
        -1.3717703819274902, -0.0663466677069664,  0.9949638843536377,   1.569246530532837,
        0.0513431653380394,  -0.6866719126701355,  -0.21783246099948883, 0.10478829592466354,
        -1.5612664222717285, -0.8418194651603699,  0.11348822712898254,  -0.5082326531410217,
    };
    for (expected, samples) |e, a| try expectSample(e, a);
}
