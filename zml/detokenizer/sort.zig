const std = @import("std");
const detokenizer = @import("detokenizer.zig");

const VECTOR_SIZE = detokenizer.VECTOR_SIZE;
const VF = detokenizer.VF;

/// vector of indices
const VI = @Vector(VECTOR_SIZE, u32);

// SORTING UTILITIES

fn bitonic_sort_1(logits: *VF) VI {
    // Bitonic sort for 16 elements in descending order
    // Following PyRTL structure: crossover creates bitonic sequence, then merge with cleaner
    var indices: VI = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    // Block size 2: crossover desc
    compSwap(logits, &indices, 0, 1, true);
    compSwap(logits, &indices, 2, 3, true);
    compSwap(logits, &indices, 4, 5, true);
    compSwap(logits, &indices, 6, 7, true);
    compSwap(logits, &indices, 8, 9, true);
    compSwap(logits, &indices, 10, 11, true);
    compSwap(logits, &indices, 12, 13, true);
    compSwap(logits, &indices, 14, 15, true);

    // Block size 4: crossover desc
    compSwap(logits, &indices, 0, 3, true);
    compSwap(logits, &indices, 1, 2, true);
    compSwap(logits, &indices, 4, 7, true);
    compSwap(logits, &indices, 5, 6, true);
    compSwap(logits, &indices, 8, 11, true);
    compSwap(logits, &indices, 9, 10, true);
    compSwap(logits, &indices, 12, 15, true);
    compSwap(logits, &indices, 13, 14, true);

    // merge stride 1
    compSwap(logits, &indices, 0, 1, false);
    compSwap(logits, &indices, 2, 3, false);
    compSwap(logits, &indices, 4, 5, false);
    compSwap(logits, &indices, 6, 7, false);
    compSwap(logits, &indices, 8, 9, false);
    compSwap(logits, &indices, 10, 11, false);
    compSwap(logits, &indices, 12, 13, false);
    compSwap(logits, &indices, 14, 15, false);

    // Block size 8: crossover desc
    compSwap(logits, &indices, 0, 7, true);
    compSwap(logits, &indices, 1, 6, true);
    compSwap(logits, &indices, 2, 5, true);
    compSwap(logits, &indices, 3, 4, true);
    compSwap(logits, &indices, 8, 15, true);
    compSwap(logits, &indices, 9, 14, true);
    compSwap(logits, &indices, 10, 13, true);
    compSwap(logits, &indices, 11, 12, true);

    // merge stride 2
    compSwap(logits, &indices, 0, 2, false);
    compSwap(logits, &indices, 1, 3, false);
    compSwap(logits, &indices, 4, 6, false);
    compSwap(logits, &indices, 5, 7, false);
    compSwap(logits, &indices, 8, 10, false);
    compSwap(logits, &indices, 9, 11, false);
    compSwap(logits, &indices, 12, 14, false);
    compSwap(logits, &indices, 13, 15, false);

    // merge stride 1
    compSwap(logits, &indices, 0, 1, false);
    compSwap(logits, &indices, 2, 3, false);
    compSwap(logits, &indices, 4, 5, false);
    compSwap(logits, &indices, 6, 7, false);
    compSwap(logits, &indices, 8, 9, false);
    compSwap(logits, &indices, 10, 11, false);
    compSwap(logits, &indices, 12, 13, false);
    compSwap(logits, &indices, 14, 15, false);

    // Block size 16: crossover desc
    compSwap(logits, &indices, 0, 15, false);
    compSwap(logits, &indices, 1, 14, false);
    compSwap(logits, &indices, 2, 13, false);
    compSwap(logits, &indices, 3, 12, false);
    compSwap(logits, &indices, 4, 11, false);
    compSwap(logits, &indices, 5, 10, false);
    compSwap(logits, &indices, 6, 9, false);
    compSwap(logits, &indices, 7, 8, false);

    // merge stride 4
    compSwap(logits, &indices, 0, 4, false);
    compSwap(logits, &indices, 1, 5, false);
    compSwap(logits, &indices, 2, 6, false);
    compSwap(logits, &indices, 3, 7, false);
    compSwap(logits, &indices, 8, 12, false);
    compSwap(logits, &indices, 9, 13, false);
    compSwap(logits, &indices, 10, 14, false);
    compSwap(logits, &indices, 11, 15, false);  

    // merge stride 2
    compSwap(logits, &indices, 0, 2, false);
    compSwap(logits, &indices, 1, 3, false);
    compSwap(logits, &indices, 4, 6, false);
    compSwap(logits, &indices, 5, 7, false);
    compSwap(logits, &indices, 8, 10, false);
    compSwap(logits, &indices, 9, 11, false);
    compSwap(logits, &indices, 12, 14, false);
    compSwap(logits, &indices, 13, 15, false);

    // merge stride 1
    compSwap(logits, &indices, 0, 1, false);
    compSwap(logits, &indices, 2, 3, false);
    compSwap(logits, &indices, 4, 5, false);
    compSwap(logits, &indices, 6, 7, false);
    compSwap(logits, &indices, 8, 9, false);
    compSwap(logits, &indices, 10, 11, false);
    compSwap(logits, &indices, 12, 13, false);
    compSwap(logits, &indices, 14, 15, false);  

    return indices;
}

inline fn compSwap(values: *VF, indices: *VI, i: u32, j: u32, asc: bool) void {
    const should_swap = if (asc) values[i] > values[j] else values[i] < values[j];
    if (should_swap) {
        const tmp_val = values[i];
        values[i] = values[j];
        values[j] = tmp_val;

        const tmp_idx = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp_idx;
    }
}

inline fn compareExchange(values: *VF, indices: *VI, i: u32, j: u32) void {
    if (values[i] > values[j]) {
        const tmp_val = values[i];
        values[i] = values[j];
        values[j] = tmp_val;

        const tmp_idx = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp_idx;
    }
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
