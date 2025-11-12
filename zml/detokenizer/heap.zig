const detokenizer = @import("detokenizer.zig");
const VECTOR_SIZE = detokenizer.VECTOR_SIZE;
const VF = detokenizer.VF;
const VI = @Vector(VECTOR_SIZE, u32);
const sort = @import("sort.zig");
const std = @import("std");

// TODO: turn preconditions into switchable asserts.

/// Turns a slice of logits into a min-heap in-place.
/// Also, fills out an "indices" slice with the corresponding indices that
/// the chunks came from.
/// returns the heap's minimum value.
///
/// Contents of the indices slice may be undefined.
pub fn heapify(chunks: []VF, indices: []VI) f32 {
    // preconditions:
    // - at least three chunks.
    // - indices length matches chunks length.
    if (chunks.len <= 2) @panic("use sort functions for chunk counts of 1 or 2");
    if (indices.len != chunks.len) @panic("indices length must match chunks length");

    // postconditions:
    //   the return value is the minimum value in the heap.

    // first sort the bottom-most chunks, and save in the indices.
    // note that sorted list is by definition a min-heap.
    const first_chunks = sort.bitonic_2(chunks[0..2]);
    indices[0] = first_chunks[0];
    indices[1] = first_chunks[1];

    // switch out of SIMD mode for the rest of the chunks.
    const logits_ptr: [*]f32 = @ptrCast(chunks.ptr);
    const logits_len = chunks.len * VECTOR_SIZE;

    for (32..logits_len) |index| {
        heap_insert(logits_ptr, @ptrCast(indices.ptr), @intCast(index));
    }

    return logits_ptr[0];
}

fn heap_insert(logits: [*]f32, indices: [*]u32, start_index: u32) void {
    var index = start_index;
    indices[index] = start_index;
    while (index > 0) {
        const parent = parent_index(index);
        const value = logits[index];
        const parent_value = logits[parent];

        if (parent_value < value) return;
        logits[parent] = value;
        logits[index] = parent_value;
        // swap indices
        indices[index] = indices[parent];
        indices[parent] = start_index;
        index = parent;
    }
}

inline fn parent_index(index: u32) u32 {
    return (index - 1) / 2;
}

fn is_min_heap(data: []f32) bool {
    for (1..data.len) |i| {
        if (data[parent_index(@intCast(i))] > data[i]) {
            return false;
        }
    }
    return true;
}

test "heapify" {
    const original = [_]f32{ 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5 };

    var data: [3]VF = @bitCast(original);
    var indices: [3]VI = undefined;
    const min = heapify(&data, &indices);

    const indices_flat: []u32 = @ptrCast(&indices);
    const data_flat: []f32 = @ptrCast(&data);

    // check min-heap property
    try std.testing.expect(is_min_heap(@ptrCast(&data)));
    // check indices match.

    for (0..data_flat.len) |i| {
        try std.testing.expectEqual(data_flat[i], original[indices_flat[i]]);
    }
    // check that minimum value is at root
    try std.testing.expectEqual(data_flat[0], 0.5);
    try std.testing.expectEqual(min, 0.5);
}

fn check_heapify(_: void, bytes: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(seedFromHash(bytes));
    var r = prng.random();

    // Generate random chunk count between 3 and 128
    const chunk_count = @max(3, @as(usize, @intCast(r.int(u8))) % 128 + 3);

    const chunks = try std.testing.allocator.alloc(VF, chunk_count);
    defer std.testing.allocator.free(chunks);

    const indices = try std.testing.allocator.alloc(VI, chunk_count);
    defer std.testing.allocator.free(indices);

    // Create a copy for verification
    const original = try std.testing.allocator.alloc(f32, chunk_count * VECTOR_SIZE);
    defer std.testing.allocator.free(original);

    // Fill with random values
    for (chunks, 0..) |*chunk, i| {
        for (0..VECTOR_SIZE) |j| {
            const value = randUniform(&r);
            chunk.*[j] = value;
            original[i * VECTOR_SIZE + j] = value;
        }
    }

    const min = heapify(chunks, indices);

    const data_flat: [*]f32 = @ptrCast(chunks.ptr);
    const indices_flat: [*]u32 = @ptrCast(indices.ptr);
    const data_len = chunk_count * VECTOR_SIZE;

    // Verify min-heap property
    try std.testing.expect(is_min_heap(data_flat[0..data_len]));

    // Verify indices correctly map back to original values
    for (0..data_len) |i| {
        try std.testing.expectEqual(data_flat[i], original[indices_flat[i]]);
    }

    // Verify all values are >= the minimum value
    for (0..data_len) |i| {
        try std.testing.expect(data_flat[i] >= min);
    }
}

test "heapify fuzz test" {
    try std.testing.fuzz({}, check_heapify, .{});
}

// Utility functions for fuzz testing
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
