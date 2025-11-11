const std = @import("std");
const detokenizer = @import("detokenizer.zig");

const VECTOR_SIZE = detokenizer.VECTOR_SIZE;
const VF = detokenizer.VF;

/// vector of indices
const VI = @Vector(VECTOR_SIZE, u32);

// SORTING UTILITIES

const Direction = enum { asc, desc };

fn bitonic_sort_1(logits: *VF) VI {
    var indices: VI = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    // Block size 2: crossover desc
    simd_crossover(logits, &indices, 2, .desc);

    // Block size 4: crossover desc, merge 1
    simd_crossover(logits, &indices, 4, .desc);
    simd_merge(logits, &indices, 1);

    // Block size 8: crossover desc, merge 2, merge 1
    simd_crossover(logits, &indices, 8, .desc);
    simd_merge(logits, &indices, 2);
    simd_merge(logits, &indices, 1);

    // Block size 16: crossover desc, merge 4, merge 2, merge 1
    simd_crossover(logits, &indices, 16, .asc);
    simd_merge(logits, &indices, 4);
    simd_merge(logits, &indices, 2);
    simd_merge(logits, &indices, 1);

    return indices;
}

/// generates the permutation of the vector that corresponds to which element 
/// the crossover operation should be compared against.
fn crossover_permute(comptime blocksize: usize) @Vector(VECTOR_SIZE, i32) {
    switch (blocksize) {
        2 => return .{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14 },
        4 => return .{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12 },
        8 => return .{7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8 },
        16 => return .{15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 },
        else => @compileError("unsupported blocksize"),
    }
}

fn crossover_mask(comptime blocksize: usize) @Vector(VECTOR_SIZE, bool) {
    switch (blocksize) {
        2 => return .{ true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false },
        4 => return .{ true, true, false, false, true, true, false, false, true, true, false, false, true, true, false, false },
        8 => return .{ true, true, true, true, false, false, false, false, true, true, true, true, false, false, false, false },
        16 => return .{ true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false },
        else => @compileError("unsupported blocksize"),
    }
}

fn merge_permute(comptime stride: usize) @Vector(VECTOR_SIZE, i32) {
    switch (stride) {
        1 => return .{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14 },
        2 => return .{2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13 },
        4 => return .{4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11 },
        8 => return .{8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7 },
        else => @compileError("unsupported stride"),
    }
}

fn merge_mask(comptime stride: usize) @Vector(VECTOR_SIZE, bool) {
    switch (stride) {
        1 => return .{ true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false },
        2 => return .{ true, true, false, false, true, true, false, false, true, true, false, false, true, true, false, false },
        4 => return .{ true, true, true, true, false, false, false, false, true, true, true, true, false, false, false, false },
        8 => return .{ true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false },
        else => @compileError("unsupported stride"),
    }
}

inline fn simd_crossover(values: *VF, indices: *VI, comptime blocksize: usize, comptime direction: Direction) void {
    // create the comparison vectors
    const permuted_v = @shuffle(f32, values.*, undefined, crossover_permute(blocksize));
    const permuted_i = @shuffle(u32, indices.*, undefined, crossover_permute(blocksize));

    const compare = switch (direction) {
        .asc => (permuted_v < values.*) != crossover_mask(blocksize),
        .desc => (permuted_v < values.*) == crossover_mask(blocksize),
    };

    values.* = @select(f32, compare, permuted_v, values.*);
    indices.* = @select(u32, compare, permuted_i, indices.*);
}

inline fn simd_merge(values: *VF, indices: *VI, comptime stride: usize) void {
    const permuted_v = @shuffle(f32, values.*, undefined, merge_permute(stride));
    const permuted_i = @shuffle(u32, indices.*, undefined, merge_permute(stride));

    const compare = (permuted_v < values.*) != merge_mask(stride);

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

fn check_bitonic(_: void, bytes: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(seedFromHash(bytes));
    var r = prng.random();

    // Generate random vector
    var vec = randvec(&r);
    const original = vec;

    // Sort it
    const indices = bitonic_sort_1(&vec);

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
    const indices = bitonic_sort_1(&vec);

    std.debug.print("\nOriginal: {any}\n", .{original});
    std.debug.print("Sorted:   {any}\n", .{vec});
    std.debug.print("Indices:  {any}\n", .{indices});

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
    try std.testing.fuzz({}, check_bitonic, .{});
}
