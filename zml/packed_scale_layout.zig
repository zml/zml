//! DeepGEMM's TMA-aligned, column-major packed UE8M0 scale storage.
//!
//! Four adjacent K scales form a little-endian u32. These words are column
//! major: their row stride is one and their column stride is paddedRows(rows).
//! This describes one device's local plane. Pad it before joining rank slices.
//! Reference: DeepGEMM 1fc9c669, csrc/jit_kernels/impls/smxx_layout.hpp,
//! get_mn_major_tma_aligned_packed_ue8m0_tensor_torch (MIT license).
const std = @import("std");

pub const row_alignment = 4;
pub const column_alignment = 4;

pub fn paddedRows(rows: usize) usize {
    return std.mem.alignForward(usize, rows, row_alignment);
}

pub fn paddedColumns(columns: usize) usize {
    return std.mem.alignForward(usize, columns, column_alignment);
}

pub fn byteSize(rows: usize, columns: usize) usize {
    return paddedRows(rows) * paddedColumns(columns);
}

pub fn offset(row: usize, column: usize, padded_rows: usize, padded_columns: usize) usize {
    std.debug.assert(padded_rows % row_alignment == 0 and row < padded_rows);
    std.debug.assert(padded_columns % column_alignment == 0 and column < padded_columns);
    return (column / column_alignment * padded_rows + row) * column_alignment + column % column_alignment;
}

test "packed column scales match four-byte column-major words including padding" {
    for ([_]usize{ 1, 3, 4, 5, 31, 32, 127, 128, 129, 136, 224, 256, 512, 768, 4096, 5120 }) |rows| {
        for ([_]usize{ 1, 3, 4, 5, 17, 192, 256 }) |columns| {
            const rp = paddedRows(rows);
            const cp = paddedColumns(columns);
            const seen = try std.testing.allocator.alloc(bool, byteSize(rows, columns));
            defer std.testing.allocator.free(seen);
            @memset(seen, false);
            for (0..rp) |row| {
                for (0..cp) |column| {
                    const address = offset(row, column, rp, cp);
                    try std.testing.expect(address < seen.len and !seen[address]);
                    seen[address] = true;
                    // Independent indexing of an int32 matrix with strides
                    // (1, rp), followed by selecting its little-endian byte.
                    const word = row + rp * (column >> 2);
                    try std.testing.expectEqual(word * @sizeOf(u32) + (column & 3), address);
                }
            }
            for (seen) |visited| try std.testing.expect(visited);
        }
    }
}

test "packed column scales preserve logical bytes and separately padded rank planes" {
    const rows = 5;
    const columns = 7;
    const rp = paddedRows(rows);
    const cp = paddedColumns(columns);
    const rank_bytes = comptime byteSize(rows, columns);
    var storage: [2 * rank_bytes]u8 = @splat(0);
    for (0..2) |rank| {
        for (0..rows) |row| {
            for (0..columns) |column| {
                const exponent: u8 = @intCast(105 + rank * 11 + row + 2 * column);
                storage[rank * rank_bytes + offset(row, column, rp, cp)] = exponent;
            }
        }
    }
    for (0..2) |rank| {
        for (0..rp) |row| {
            for (0..cp) |column| {
                const actual = storage[rank * rank_bytes + offset(row, column, rp, cp)];
                const expected: u8 = if (row < rows and column < columns) @intCast(105 + rank * 11 + row + 2 * column) else 0;
                try std.testing.expectEqual(expected, actual);
            }
        }
    }
    // A global row padding calculation would silently lose four rows here.
    try std.testing.expect(2 * paddedRows(rows) != paddedRows(2 * rows));
}
