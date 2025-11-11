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
    //simd_merge(logits, &indices, 1);

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
    simd_crossover(logits, &indices, 8, .desc);

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
    simd_crossover(logits, &indices, 16, .asc);

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

fn crossover_fwd(comptime blocksize: usize) @Vector(VECTOR_SIZE, i32) {
    switch (blocksize) {
        2 => return .{ 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 },
        4 => return .{ 0, 1, 4, 5, 8, 9, 12, 13, 3, 2, 7, 6, 11, 10, 15, 14 },
        8 => return .{ 0, 1, 2, 3, 8, 9, 10, 11, 7, 6, 5, 4, 15, 14, 13, 12 },
        16 => return .{ 0, 1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8 },
        else => @compileError("unsupported blocksize"),
    }
}

fn crossover_rev(comptime blocksize: usize) @Vector(VECTOR_SIZE, i32) {
    switch (blocksize) {
        2 => return .{ 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 },
        4 => return .{ 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15 },
        8 => return .{ 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15 },
        else => @compileError("unsupported blocksize"),
    }
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

fn permute_mask(comptime blocksize: usize) @Vector(VECTOR_SIZE, bool) {
    switch (blocksize) {
        2 => return .{ true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false },
        4 => return .{ true, true, false, false, true, true, false, false, true, true, false, false, true, true, false, false },
        8 => return .{ true, true, true, true, false, false, false, false, true, true, true, true, false, false, false, false },
        16 => return .{ true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false },
        else => @compileError("unsupported blocksize"),
    }
}

inline fn simd_crossover(values: *VF, indices: *VI, comptime blocksize: usize, comptime direction: Direction) void {
    // create the comparison vectors
    const permuted_v = @shuffle(f32, values.*, undefined, crossover_permute(blocksize));
    const permuted_i = @shuffle(u32, indices.*, undefined, crossover_permute(blocksize));

    const compare = switch (direction) {
        .asc => (permuted_v < values.*) != permute_mask(blocksize),
        .desc => (permuted_v < values.*) == permute_mask(blocksize),
    };

    values.* = @select(f32, compare, permuted_v, values.*);
    indices.* = @select(u32, compare, permuted_i, indices.*);
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
