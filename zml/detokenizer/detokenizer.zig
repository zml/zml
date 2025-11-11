const std = @import("std");

// AVX-512 32-bit float vector type.  We may want to
// generalize this later to other types (e.g. u16 for bfloat16).
pub const VECTOR_SIZE = 16;
// vector of floats
pub const VF = @Vector(VECTOR_SIZE, f32);

/// Options controlling generation. The default values correspond to greedy decoding.
const SamplingStrategy = struct { topk: u32 = 1, topp: ?f32 = null, temperature: f32 = 1.0 };

// for now, rng is anytype.  We'll do better later.

/// Selects a token given activations and sampling options.  The output is the
/// index of the selected token.
pub fn sample(activations: []const VF, opts: SamplingStrategy, rng: anytype) usize {
    _ = rng;
    if (opts.topk == 1) return greedy_sample(activations);

    // presort activations into descending order here.
    if (opts.topp) |top_p| {
        return nucleus_sample(activations, opts, rng);
    } else {
        return topk_sample(activations, opts, rng);
    }
}

const minfloat = -std.math.inf(f32);

// GREEDY SAMPLING

fn greedy_sample(activations: []const VF) usize {
    var max_index: usize = 0;
    var max_value = minfloat;
    for (activations, 0..) |chunk, chunk_index| {
        const this_max = @reduce(.Max, chunk);
        if (this_max > max_value) {
            max_value = this_max;
            const max_chunk: VF = @splat(max_value);
            const map = max_chunk == chunk;
            // take the first index that matches
            const sub_index = @ctz(@as(u16, @bitCast(map)));
            max_index = chunk_index * VECTOR_SIZE + sub_index;
        }
    }
    return max_index;
}

fn check_greedy(_: void, bytes: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(seedFromHash(bytes));
    var r = prng.random();

    const activations = try std.testing.allocator.alloc(VF, @intCast(r.int(u8)));
    defer std.testing.allocator.free(activations);

    // Fill with random values
    for (activations) |*chunk| {
        chunk.* = randvec(&r);
    }

    var biggestvalue = minfloat;
    var biggestindex: usize = 0;

    for (activations, 0..) |chunk, i| {
        for (0..VECTOR_SIZE) |j| {
            if (chunk[j] > biggestvalue) {
                biggestvalue = chunk[j];
                biggestindex = i * VECTOR_SIZE + j;
            }
        }
    }

    try std.testing.expectEqual(biggestindex, greedy_sample(activations));
}

test "greedy sample basic test" {
    const activations: [2]VF = .{
        .{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 },
        .{ -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, 2.0 },
    };
    const result = greedy_sample(&activations);
    try std.testing.expectEqual(result, 31);
}

test "greedy sample fuzz test" {
    try std.testing.fuzz({}, check_greedy, .{});
}

// NUCLEUS SAMPLING

fn nucleus_sample(activations: []const VF, opts: SamplingStrategy, rng: anytype) usize {
    _ = activations;
    _ = opts;
    _ = rng;
    @panic("not implemented");
}

// TOP-K SAMPLING

fn topk_sample(activations: []const VF, opts: SamplingStrategy, rng: anytype) usize {
    _ = activations;
    _ = opts;
    _ = rng;
    @panic("not implemented");
}

// RANDOMIZER UTILITIES

fn seedFromHash(bytes: []const u8) u64 {
    var hasher = std.crypto.hash.Blake3.init(.{});
    hasher.update(bytes);
    var hash: [32]u8 = undefined;
    hasher.final(&hash);
    return std.mem.readInt(u64, hash[0..8], .little);
}

fn randUniform(r: *std.Random) f32 {
    const mantissa = r.int(u32) >> 8; // top 24 bits
    return @as(f32, @floatFromInt(mantissa)) * (1.0 / 16777216.0);
}

fn randvec(r: *std.Random) VF {
    var result: VF = undefined;
    for (0..VECTOR_SIZE) |index| {
        result[index] = randUniform(r);
    }
    return result;
}
