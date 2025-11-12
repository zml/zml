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

/// Discovers the top-K logits from source_chunks, and records them in dest_chunks
/// and their source indices (from source_chunks) in dest_indices.
///
/// This should be considered the main entrypoint for heap-based top-K selection.
///
/// Contents of the dest_chunks and dest_indices slices may be undefined beforehand.
pub fn partial_heapify(dest_chunks: []VF, dest_indices: []VI, source_chunks: []VF) void {
    // preconditions:
    // - at least three chunks in dest_chunks.
    // - dest_indices length matches dest_chunks length.
    // - source_chunks length is at least as long as dest_chunks length.
    if (dest_chunks.len <= 2) @panic("use sort functions for chunk counts of 1 or 2");
    if (dest_indices.len != dest_chunks.len) @panic("indices length must match chunks length");
    if (source_chunks.len < dest_chunks.len) @panic("source_chunks must be at least as long as dest_chunks");

    // first, copy source chunks into dest_chunks.
    for (dest_chunks, source_chunks[0..dest_chunks.len]) |*dest_chunk, source_chunk| {
        dest_chunk.* = source_chunk;
    }

    // next, heapify the dest_chunks.
    const min_value = heapify(dest_chunks, dest_indices);

    // finally, seek through the rest of the source_chunks and insert into the heap as needed.
    for (source_chunks[dest_chunks.len..], dest_chunks.len..) |source_chunk, index| {
        // check if any value in the source_chunk is larger than the min_value
        var chunk = source_chunk;
        var min: VF = @splat(min_value);
        var cmp = chunk > min;
        while (@as(u16, @bitCast(cmp)) != 0) {
            // find the maximum value and its index
            const max_val = @reduce(.Max, chunk);
            const max_mask = chunk == @as(VF, @splat(max_val));
            const sub_index: u4 = @intCast(@ctz(@as(u16, @bitCast(max_mask))));
            const new_min = heap_insert_one(@ptrCast(dest_chunks), @ptrCast(dest_indices), max_val, @intCast(index * VECTOR_SIZE + sub_index));
            // update chunk by ablating the found max_value.
            // these are preserved as SIMD instructions to prevent inefficent scalar moves.
            const ablation: @Vector(16, bool) = @bitCast(@as(u16, 1) << sub_index);
            chunk = @select(f32, ablation, @as(VF, @splat(0.0)), chunk);
            min = @splat(new_min);
            cmp = chunk > min;
        }
    }
}

fn heap_insert_one(logits: []f32, indices: []u32, new_value: f32, new_index: u32) f32 {
    // where we're looking at in the heap.
    var heap_index: usize = 0;

    const max_parent_index = (logits.len - 2) / 2;

    while (heap_index <= max_parent_index) {
        const left_index = 2 * heap_index + 1;
        const right_index = 2 * heap_index + 2;

        const left_child_value = logits[left_index];

        // Find the smaller child
        var smaller_child_index = left_index;
        var smaller_child_value = left_child_value;

        if (right_index < logits.len) {
            const right_child_value = logits[right_index];
            if (right_child_value < left_child_value) {
                smaller_child_index = right_index;
                smaller_child_value = right_child_value;
            }
        }

        // If new_value is smaller than or equal to the smaller child, we're done
        if (new_value <= smaller_child_value) {
            break;
        }

        // Move the smaller child up
        logits[heap_index] = smaller_child_value;
        indices[heap_index] = indices[smaller_child_index];
        heap_index = smaller_child_index;
    }

    // we are right where we belong.
    logits[heap_index] = new_value;
    indices[heap_index] = new_index;
    return logits[0];
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

fn check_partial_heapify(_: void, bytes: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(seedFromHash(bytes));
    var r = prng.random();

    // Generate random source chunk count between 6 and 128
    const source_chunk_count = @max(6, @as(usize, @intCast(r.int(u8))) % 123 + 6);

    // Dest chunk count is between 3 and half of source
    const dest_chunk_count = @max(3, @as(usize, @intCast(r.int(u8))) % (source_chunk_count / 2) + 3);

    const source_chunks = try std.testing.allocator.alloc(VF, source_chunk_count);
    defer std.testing.allocator.free(source_chunks);

    const dest_chunks = try std.testing.allocator.alloc(VF, dest_chunk_count);
    defer std.testing.allocator.free(dest_chunks);

    const dest_indices = try std.testing.allocator.alloc(VI, dest_chunk_count);
    defer std.testing.allocator.free(dest_indices);

    // Create a copy for verification
    const original = try std.testing.allocator.alloc(f32, source_chunk_count * VECTOR_SIZE);
    defer std.testing.allocator.free(original);

    // Fill with random values
    for (source_chunks, 0..) |*chunk, i| {
        for (0..VECTOR_SIZE) |j| {
            const value = randUniform(&r);
            chunk.*[j] = value;
            original[i * VECTOR_SIZE + j] = value;
        }
    }

    partial_heapify(dest_chunks, dest_indices, source_chunks);

    const dest_flat: [*]f32 = @ptrCast(dest_chunks.ptr);
    const dest_indices_flat: [*]u32 = @ptrCast(dest_indices.ptr);
    const dest_len = dest_chunk_count * VECTOR_SIZE;

    // Verify min-heap property
    try std.testing.expect(is_min_heap(dest_flat[0..dest_len]));

    // Verify indices correctly map back to original values
    for (0..dest_len) |i| {
        try std.testing.expectEqual(dest_flat[i], original[dest_indices_flat[i]]);
    }

    // Verify the minimum value in dest is the (dest_len)th largest value overall
    const min_in_dest = dest_flat[0];

    // Count how many values in original are >= min_in_dest
    var count_larger: usize = 0;
    for (original) |val| {
        if (val >= min_in_dest) {
            count_larger += 1;
        }
    }

    // Should be exactly dest_len values >= min_in_dest
    try std.testing.expectEqual(dest_len, count_larger);
}

test "partial_heapify" {
    // Create source data: 6 chunks (96 values)
    // We'll select top 3 chunks (48 values)
    const source_data = [_]f32{
        // Chunk 0: values 0-15
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        // Chunk 1: values 16-31
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        // Chunk 2: values 32-47
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
        // Chunk 3: values 48-63 (these should all be in top-48)
        80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0,
        // Chunk 4: values 64-79 (these should all be in top-48)
        64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
        // Chunk 5: values 80-95 (these should all be in top-48)
        96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
    };

    var source_chunks: [6]VF = @bitCast(source_data);
    var dest_chunks: [3]VF = undefined;
    var dest_indices: [3]VI = undefined;

    partial_heapify(&dest_chunks, &dest_indices, &source_chunks);

    const dest_flat: []f32 = @ptrCast(&dest_chunks);
    const indices_flat: []u32 = @ptrCast(&dest_indices);

    // Verify min-heap property
    try std.testing.expect(is_min_heap(dest_flat));

    // Verify indices correctly map back to original values
    for (0..dest_flat.len) |i| {
        try std.testing.expectEqual(dest_flat[i], source_data[indices_flat[i]]);
    }

    // Verify all values are >= 64 (the 48th largest value)
    // Since we're selecting top 48 from 0-111, the minimum should be around 64
    for (dest_flat) |val| {
        try std.testing.expect(val >= 64.0);
    }

    // Verify the minimum (root) is actually the 48th largest value
    const min_val = dest_flat[0];
    try std.testing.expectEqual(min_val, 64.0);
}

test "partial_heapify fuzz test" {
    try std.testing.fuzz({}, check_partial_heapify, .{});
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
