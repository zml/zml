const std = @import("std");
const detokenizer = @import("detokenizer.zig");

const VECTOR_SIZE = detokenizer.VECTOR_SIZE;
const VF = detokenizer.VF;

/// vector of indices
const VI = @Vector(VECTOR_SIZE, u32);
/// select vector, comptime value only, used for simd select operations
const VS = @Vector(VECTOR_SIZE, i32);
/// mask vector, comptime value only.
const VM = @Vector(VECTOR_SIZE, bool);

// SORTING UTILITIES
const Direction = enum { asc, desc };
const SimdParams = struct {VS, VM};

pub fn bitonic_1(logits: *VF) VI {
    var indices: VI = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    // Block size 2: crossover desc
    simd_sort_block(logits, &indices, crossover(2), .desc);

    // Block size 4: crossover desc, merge 1
    simd_sort_block(logits, &indices, crossover(4), .desc);
    simd_sort_block(logits, &indices, merge(1), .asc);

    // Block size 8: crossover desc, merge 2, merge 1
    simd_sort_block(logits, &indices, crossover(8), .desc);
    simd_sort_block(logits, &indices, merge(2), .asc);
    simd_sort_block(logits, &indices, merge(1), .asc);

    // Block size 16: crossover desc, merge 4, merge 2, merge 1
    simd_sort_block(logits, &indices, crossover(16), .desc);
    simd_sort_block(logits, &indices, merge(4), .desc);
    simd_sort_block(logits, &indices, merge(2), .desc);
    simd_sort_block(logits, &indices, merge(1), .desc);

    return indices;
}

const REVERSE: VS = .{15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

pub fn bitonic_2(logits: *[2]VF) [2]VI {
    var indices: [2]VI = .{
        .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
        .{ 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
    };

    // TODO: reorder these steps to improve instruction-level parallelism
    inline for (0..2) |i| {
        // Block size 2: crossover desc
        simd_sort_block(&logits[i], &indices[i], crossover(2), .desc);

        // Block size 4: crossover desc, merge 1
        simd_sort_block(&logits[i], &indices[i], crossover(4), .desc);
        simd_sort_block(&logits[i], &indices[i], merge(1), .asc);

        // Block size 8: crossover desc, merge 2, merge 1
        simd_sort_block(&logits[i], &indices[i], crossover(8), .desc);
        simd_sort_block(&logits[i], &indices[i], merge(2), .asc);
        simd_sort_block(&logits[i], &indices[i], merge(1), .asc);

        // Block size 16: crossover desc, merge 4, merge 2, merge 1
        simd_sort_block(&logits[i], &indices[i], crossover(16), .desc);
        simd_sort_block(&logits[i], &indices[i], merge(4), .desc);
        simd_sort_block(&logits[i], &indices[i], merge(2), .desc);
        simd_sort_block(&logits[i], &indices[i], merge(1), .desc);
    }

    // block size 32: double crossover special direct comparison.
    const reverse_second_half_v = @shuffle(f32, logits[1], undefined, REVERSE);
    const reverse_second_half_i = @shuffle(u32, indices[1], undefined, REVERSE);

    const compare = reverse_second_half_v > logits[0];

    logits[1] = @select(f32, compare, logits[0], reverse_second_half_v);
    indices[1] = @select(u32, compare, indices[0], reverse_second_half_i);

    logits[0] = @select(f32, compare, reverse_second_half_v, logits[0]);
    indices[0] = @select(u32, compare, reverse_second_half_i, indices[0]);

    inline for (0..2) |i| {
        // Block size 32: crossover desc
        simd_sort_block(&logits[i], &indices[i], merge(8), .desc);
        simd_sort_block(&logits[i], &indices[i], merge(4), .desc);
        simd_sort_block(&logits[i], &indices[i], merge(2), .desc);
        simd_sort_block(&logits[i], &indices[i], merge(1), .desc);
    }

    return indices;
}

// SIMD OPERATIONS

fn crossover(comptime blocksize: usize) SimdParams {
    return switch (blocksize) {
      2 => .{.{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14 }, @bitCast(@as(u16, 0b1010_1010_1010_1010)) },
      4 => .{.{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12 }, @bitCast(@as(u16, 0b1100_1100_1100_1100)) },
      8 => .{.{7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8 }, @bitCast(@as(u16, 0b1111_0000_1111_0000)) },
      16 => .{.{15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }, @bitCast(@as(u16, 0b1111_1111_0000_0000)) },
      else => @compileError("unsupported blocksize"),
    };
}

fn merge(comptime stride: usize) SimdParams {
    return switch(stride) {
        1 => .{.{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14 }, @bitCast(@as(u16, 0b1010_1010_1010_1010)) },
        2 => .{.{2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13 }, @bitCast(@as(u16, 0b1100_1100_1100_1100)) },
        4 => .{.{4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11 }, @bitCast(@as(u16, 0b1111_0000_1111_0000)) },
        8 => .{.{8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7 }, @bitCast(@as(u16, 0b1111_1111_0000_0000)) },
        else => @compileError("unsupported stride"),
    };
}

fn simd_sort_block(values: *VF, indices: *VI, comptime simd_params: SimdParams, comptime direction: Direction) void {
    const permutation, const mask = simd_params;

    // create the comparison vectors
    const permuted_v = @shuffle(f32, values.*, undefined, permutation);
    const permuted_i = @shuffle(u32, indices.*, undefined, permutation);

    const compare = switch (direction) {
        .asc => (permuted_v < values.*) != mask,
        .desc => (permuted_v < values.*) == mask,
    };

    values.* = @select(f32, compare, permuted_v, values.*);
    indices.* = @select(u32, compare, permuted_i, indices.*);
}

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

fn check_bitonic_1(_: void, bytes: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(seedFromHash(bytes));
    var r = prng.random();

    // Generate random vector
    var vec = randvec(&r);
    const original = vec;

    // Sort it
    const indices = bitonic_1(&vec);

    // Check that vec is sorted in descending order
    for (1..VECTOR_SIZE) |i| {
        try std.testing.expect(vec[i - 1] >= vec[i]);
    }

    // Check that indices correctly map sorted values back to original positions
    for (0..VECTOR_SIZE) |i| {
        try std.testing.expectEqual(original[indices[i]], vec[i]);
    }
}

test "bitonic sort basic test" {
    var vec: VF = .{ 0.5, 0.2, 0.9, 0.1, 0.4, 0.8, 0.3, 0.7, 0.6, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75 };
    const original = vec;
    const indices = bitonic_1(&vec);

    // Check that vec is sorted in descending order
    for (1..VECTOR_SIZE) |i| {
        try std.testing.expect(vec[i - 1] >= vec[i]);
    }

    // Check that indices correctly map sorted values back to original positions
    for (0..VECTOR_SIZE) |i| {
        try std.testing.expectEqual(original[indices[i]], vec[i]);
    }
}

test "bitonic sort fuzz test" {
    try std.testing.fuzz({}, check_bitonic_1, .{});
}

fn check_bitonic_2(_: void, bytes: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(seedFromHash(bytes));
    var r = prng.random();

    // Generate random vectors
    var vecs: [2]VF = .{ randvec(&r), randvec(&r) };
    const originals: [32]f32 = @bitCast(vecs);

    // Sort them
    const result_indices = bitonic_2(&vecs);
    const indices: [32]u32 = @bitCast(result_indices);
    const sorted: [32]f32 = @bitCast(vecs);

    // Check that the entire 32-element sequence is sorted in descending order
    for (1..32) |i| {
        try std.testing.expect(sorted[i - 1] >= sorted[i]);
    }

    // Check that all 32 indices correctly map sorted values back to original positions
    for (0..32) |i| {
        try std.testing.expectEqual(originals[indices[i]], sorted[i]);
    }
}

test "bitonic sort 2 basic test" {
    var vecs: [2]VF = .{
        .{ 0.5, 0.2, 0.9, 0.1, 0.4, 0.8, 0.3, 0.7, 0.6, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75 },
        .{ 0.95, 0.85, 0.42, 0.12, 0.32, 0.62, 0.22, 0.52, 0.72, 0.18, 0.28, 0.38, 0.48, 0.58, 0.68, 0.78 },
    };

    const originals: [32]f32 = @bitCast(vecs);
    const result_indices = bitonic_2(&vecs);
    const indices: [32]u32 = @bitCast(result_indices);
    const sorted: [32]f32 = @bitCast(vecs);

    // Check that the entire 32-element sequence is sorted in descending order
    for (1..32) |i| {
        try std.testing.expect(sorted[i - 1] >= sorted[i]);
    }

    // Check that all 32 indices correctly map sorted values back to original positions
    for (0..32) |i| {
        try std.testing.expectEqual(originals[indices[i]], sorted[i]);
    }
}

test "bitonic sort 2 fuzz test" {
    try std.testing.fuzz({}, check_bitonic_2, .{});
}
