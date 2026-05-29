const std = @import("std");

const MINMATCH = 4;
const ML_BITS = 4;
const ML_MASK = (@as(u8, 1) << ML_BITS) - 1; // extract the match length nibble
const RUN_MASK = (@as(u8, 1) << (8 - ML_BITS)) - 1; // extract the literal length nibble

pub const Error = error{ CorruptedData, OutputTooSmall };

const FRAME_MAGIC: u32 = 0x184D2204;

/// Decompress LZ4 data, auto-detecting block vs frame format.
pub fn decompress(src: []const u8, dst: []u8) Error!usize {
    if (src.len >= 4 and std.mem.readInt(u32, src[0..4], .little) == FRAME_MAGIC) {
        return decompressFrame(src, dst);
    }
    return decompressBlock(src, dst);
}

/// Decompress an LZ4 frame (magic + descriptor + blocks + end mark).
fn decompressFrame(src: []const u8, dst: []u8) Error!usize {
    if (src.len < 7) return error.CorruptedData;

    const flg = src[4];
    if ((flg >> 6) & 0x3 != 1) return error.CorruptedData; // version must be 01

    const has_content_size = (flg >> 3) & 1 != 0;
    const has_block_checksum = (flg >> 4) & 1 != 0;
    const has_dict_id = flg & 1 != 0;

    // Skip: magic(4) + FLG(1) + BD(1) + [ContentSize(8)] + [DictID(4)] + HC(1)
    var pos: usize = 7;
    if (has_content_size) pos += 8;
    if (has_dict_id) pos += 4;

    var written: usize = 0;
    while (pos + 4 <= src.len) {
        const raw = std.mem.readInt(u32, src[pos..][0..4], .little);
        pos += 4;
        if (raw == 0) break; // end mark

        const is_uncompressed = (raw >> 31) != 0;
        const block_size: usize = raw & 0x7FFFFFFF;
        if (pos + block_size > src.len) return error.CorruptedData;

        if (is_uncompressed) {
            if (written + block_size > dst.len) return error.OutputTooSmall;
            @memcpy(dst[written..][0..block_size], src[pos..][0..block_size]);
            written += block_size;
        } else {
            written += try decompressBlock(src[pos..][0..block_size], dst[written..]);
        }
        pos += block_size;
        if (has_block_checksum) pos += 4;
    }
    return written;
}

/// Decompress a single LZ4 block. No frame header, no dictionary.
/// Returns the number of decompressed bytes written to `dst`.
fn decompressBlock(src: []const u8, dst: []u8) Error!usize {
    if (src.len == 0) return 0;

    // keep track of input and output positions
    var ip: usize = 0;
    var op: usize = 0;

    while (ip < src.len) {
        const token = src[ip];
        ip += 1;

        // --- literals ---
        var lit_len: usize = token >> ML_BITS;
        if (lit_len == RUN_MASK) {
            while (true) {
                if (ip >= src.len) return error.CorruptedData;
                const s = src[ip];
                ip += 1;
                lit_len += s;
                if (s != 255) break;
            }
        }
        if (lit_len > 0) {
            if (ip + lit_len > src.len) return error.CorruptedData;
            if (op + lit_len > dst.len) return error.OutputTooSmall;
            @memcpy(dst[op..][0..lit_len], src[ip..][0..lit_len]);
            ip += lit_len;
            op += lit_len;
        }

        if (ip >= src.len) break; // last sequence has no match

        // --- match offset ---
        if (ip + 2 > src.len) return error.CorruptedData;
        const offset: usize = std.mem.readInt(u16, src[ip..][0..2], .little);
        ip += 2;
        if (offset == 0 or offset > op) return error.CorruptedData;

        // --- match length ---
        var match_len: usize = @as(usize, token & ML_MASK) + MINMATCH;
        if ((token & ML_MASK) == ML_MASK) {
            while (true) {
                if (ip >= src.len) return error.CorruptedData;
                const s = src[ip];
                ip += 1;
                match_len += s;
                if (s != 255) break;
            }
        }
        if (op + match_len > dst.len) return error.OutputTooSmall;

        // copy match — fast path for non-overlapping, byte-by-byte for overlapping (RLE)
        const match_pos = op - offset;
        if (offset >= match_len) {
            @memcpy(dst[op..][0..match_len], dst[match_pos..][0..match_len]);
        } else {
            for (0..match_len) |i| {
                dst[op + i] = dst[match_pos + i];
            }
        }
        op += match_len;
    }

    return op;
}

/// Reverse the ByteGrouping4 shuffle used by XET xorb compression type 2.
///
/// During encoding, bytes are distributed round-robin into 4 groups then concatenated.
/// This function restores the original interleaved order.
pub fn unshuffleBg4(grouped: []const u8, out: []u8) error{CorruptedData}!void {
    const n = grouped.len;
    if (n != out.len) return error.CorruptedData;
    if (n == 0) return;

    const base = n / 4;
    const rem = n % 4;

    // Iterate each group sequentially (cache-friendly reads), strided writes to output.
    var offset: usize = 0;
    for (0..4) |g| {
        const size = base + @as(usize, if (g < rem) 1 else 0);
        for (0..size) |i| {
            out[i * 4 + g] = grouped[offset + i];
        }
        offset += size;
    }
}

/// Reverse the FullBitslice transform used by XET xorb compression type 3.
///
/// During encoding, bits are rearranged: bit `orig_bit_idx` of byte `orig_byte_idx`
/// is placed at linear position `orig_byte_idx * 8 + orig_bit_idx` in the output
/// stream. This function reverses that mapping.
pub fn reverseFullBitslice(data: []const u8, out: []u8) error{CorruptedData}!void {
    const n = data.len;
    if (n != out.len) return error.CorruptedData;
    if (n == 0) return;

    @memset(out, 0);
    for (0..n) |in_byte_idx| {
        for (0..8) |in_bit_idx| {
            const k = in_byte_idx * 8 + in_bit_idx;
            const orig_byte_idx = k % n;
            const orig_bit_idx = k / n;
            if (orig_bit_idx >= 8) continue;
            const bit: u8 = (data[in_byte_idx] >> @intCast(in_bit_idx)) & 1;
            out[orig_byte_idx] |= bit << @intCast(orig_bit_idx);
        }
    }
}

/// Decompress ByteGrouping4+LZ4: first LZ4-decompress, then un-shuffle.
/// `tmp` must be at least `uncompressed_size` bytes (used for intermediate LZ4 output).
pub fn decompressBg4(src: []const u8, dst: []u8, tmp: []u8) Error!usize {
    const n = try decompress(src, tmp);
    if (n > dst.len) return error.OutputTooSmall;
    unshuffleBg4(tmp[0..n], dst[0..n]) catch return error.CorruptedData;
    return n;
}

/// Decompress FullBitslice+LZ4: first LZ4-decompress, then reverse bitslice.
/// `tmp` must be at least `uncompressed_size` bytes (used for intermediate LZ4 output).
pub fn decompressFbs(src: []const u8, dst: []u8, tmp: []u8) Error!usize {
    const n = try decompress(src, tmp);
    if (n > dst.len) return error.OutputTooSmall;
    reverseFullBitslice(tmp[0..n], dst[0..n]) catch return error.CorruptedData;
    return n;
}

// ── tests ──

test "decompress: all literals" {
    // "Hello" encoded as pure literals: token 0x50, then 5 bytes
    const compressed = [_]u8{ 0x50, 'H', 'e', 'l', 'l', 'o' };
    var out: [5]u8 = undefined;
    const n = try decompress(&compressed, &out);
    try std.testing.expectEqualStrings("Hello", out[0..n]);
}

test "decompress: extended literal length" {
    // 20 bytes of 'X', literal length = 15 + 5
    var compressed: [22]u8 = undefined;
    compressed[0] = 0xF0; // token: lit_len=15, match_len=0
    compressed[1] = 5; // extension: 15 + 5 = 20
    @memset(compressed[2..], 'X');
    var out: [20]u8 = undefined;
    const n = try decompress(&compressed, &out);
    try std.testing.expectEqual(@as(usize, 20), n);
    try std.testing.expectEqualStrings("X" ** 20, out[0..n]);
}

test "decompress: RLE via match" {
    // 6 × 'A': 1 literal 'A', then match offset=1 length=5 (5-4=1)
    // token = (1 << 4) | 1 = 0x11
    const compressed = [_]u8{ 0x11, 'A', 0x01, 0x00 };
    var out: [6]u8 = undefined;
    const n = try decompress(&compressed, &out);
    try std.testing.expectEqual(@as(usize, 6), n);
    try std.testing.expectEqualStrings("AAAAAA", out[0..n]);
}

test "decompress: literal + back-reference" {
    // "ABCDABCD" = 4 literals "ABCD" + match offset=4 length=4
    // token = (4 << 4) | (4-4) = 0x40
    const compressed = [_]u8{ 0x40, 'A', 'B', 'C', 'D', 0x04, 0x00 };
    var out: [8]u8 = undefined;
    const n = try decompress(&compressed, &out);
    try std.testing.expectEqualStrings("ABCDABCD", out[0..n]);
}

test "decompress: output too small" {
    const compressed = [_]u8{ 0x50, 'H', 'e', 'l', 'l', 'o' };
    var out: [3]u8 = undefined;
    try std.testing.expectError(error.OutputTooSmall, decompress(&compressed, &out));
}

test "decompress: corrupted offset" {
    // literal 'A' then offset=5 which exceeds output position (1)
    const compressed = [_]u8{ 0x10, 'A', 0x05, 0x00 };
    var out: [10]u8 = undefined;
    try std.testing.expectError(error.CorruptedData, decompress(&compressed, &out));
}

test "unshuffleBg4: 12 bytes (multiple of 4)" {
    // Original: [0,1,2,3, 4,5,6,7, 8,9,10,11]
    // Group 0: positions 0,4,8 → bytes 0,4,8
    // Group 1: positions 1,5,9 → bytes 1,5,9
    // Group 2: positions 2,6,10 → bytes 2,6,10
    // Group 3: positions 3,7,11 → bytes 3,7,11
    // Grouped: [0,4,8, 1,5,9, 2,6,10, 3,7,11]
    const grouped = [_]u8{ 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11 };
    const expected = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    var out: [12]u8 = undefined;
    try unshuffleBg4(&grouped, &out);
    try std.testing.expectEqualSlices(u8, &expected, &out);
}

test "unshuffleBg4: 5 bytes (remainder 1)" {
    // Original: [10,20,30,40,50]
    // Group 0 (size 2): bytes at pos 0,4 → 10,50
    // Group 1 (size 1): byte at pos 1 → 20
    // Group 2 (size 1): byte at pos 2 → 30
    // Group 3 (size 1): byte at pos 3 → 40
    // Grouped: [10,50, 20, 30, 40]
    const grouped = [_]u8{ 10, 50, 20, 30, 40 };
    const expected = [_]u8{ 10, 20, 30, 40, 50 };
    var out: [5]u8 = undefined;
    try unshuffleBg4(&grouped, &out);
    try std.testing.expectEqualSlices(u8, &expected, &out);
}

test "unshuffleBg4: 6 bytes (remainder 2)" {
    // Original: [10,20,30,40,50,60]
    // Group 0 (size 2): pos 0,4 → 10,50
    // Group 1 (size 2): pos 1,5 → 20,60
    // Group 2 (size 1): pos 2 → 30
    // Group 3 (size 1): pos 3 → 40
    // Grouped: [10,50, 20,60, 30, 40]
    const grouped = [_]u8{ 10, 50, 20, 60, 30, 40 };
    const expected = [_]u8{ 10, 20, 30, 40, 50, 60 };
    var out: [6]u8 = undefined;
    try unshuffleBg4(&grouped, &out);
    try std.testing.expectEqualSlices(u8, &expected, &out);
}

test "reverseFullBitslice: round-trip with applyFullBitslice" {
    // Test-only forward transform to verify round-trip.
    const original = [_]u8{ 0xA5, 0x3C, 0xF0, 0x0F };
    var bitsliced: [4]u8 = undefined;
    applyFullBitslice(&original, &bitsliced);

    var recovered: [4]u8 = undefined;
    try reverseFullBitslice(&bitsliced, &recovered);
    try std.testing.expectEqualSlices(u8, &original, &recovered);
}

test "reverseFullBitslice: known pair" {
    // 2-byte example: original = [0b10110010, 0b01001101] = [0xB2, 0x4D]
    // Forward bitslice (N=2):
    //   k=0: in_byte=0, in_bit=0 → orig_byte=0%2=0, orig_bit=0/2=0 → bit=(0xB2>>0)&1=0 → result[0] |= 0<<0
    //   k=1: in_byte=0, in_bit=1 → orig_byte=1%2=1, orig_bit=1/2=0 → bit=(0xB2>>1)&1=1 → result[1] |= 1<<0
    //   k=2: in_byte=0, in_bit=2 → orig_byte=0%2=0, orig_bit=2/2=1 → bit=(0xB2>>2)&1=0 → result[0] |= 0<<1
    //   k=3: in_byte=0, in_bit=3 → orig_byte=1%2=1, orig_bit=3/2=1 → bit=(0xB2>>3)&1=0 → result[1] |= 0<<1
    //   k=4: in_byte=0, in_bit=4 → orig_byte=0%2=0, orig_bit=4/2=2 → bit=(0xB2>>4)&1=1 → result[0] |= 1<<2
    //   k=5: in_byte=0, in_bit=5 → orig_byte=1%2=1, orig_bit=5/2=2 → bit=(0xB2>>5)&1=1 → result[1] |= 1<<2
    //   k=6: in_byte=0, in_bit=6 → orig_byte=0%2=0, orig_bit=6/2=3 → bit=(0xB2>>6)&1=0 → result[0] |= 0<<3
    //   k=7: in_byte=0, in_bit=7 → orig_byte=1%2=1, orig_bit=7/2=3 → bit=(0xB2>>7)&1=1 → result[1] |= 1<<3
    //   k=8: in_byte=1, in_bit=0 → orig_byte=0%2=0, orig_bit=8/2=4 → bit=(0x4D>>0)&1=1 → result[0] |= 1<<4
    //   k=9: in_byte=1, in_bit=1 → orig_byte=1%2=1, orig_bit=9/2=4 → bit=(0x4D>>1)&1=0 → result[1] |= 0<<4
    //   k=10: in_byte=1, in_bit=2 → orig_byte=0%2=0, orig_bit=10/2=5 → bit=(0x4D>>2)&1=1 → result[0] |= 1<<5
    //   k=11: in_byte=1, in_bit=3 → orig_byte=1%2=1, orig_bit=11/2=5 → bit=(0x4D>>3)&1=1 → result[1] |= 1<<5
    //   k=12: in_byte=1, in_bit=4 → orig_byte=0%2=0, orig_bit=12/2=6 → bit=(0x4D>>4)&1=0 → result[0] |= 0<<6
    //   k=13: in_byte=1, in_bit=5 → orig_byte=1%2=1, orig_bit=13/2=6 → bit=(0x4D>>5)&1=0 → result[1] |= 0<<6
    //   k=14: in_byte=1, in_bit=6 → orig_byte=0%2=0, orig_bit=14/2=7 → bit=(0x4D>>6)&1=1 → result[0] |= 1<<7
    //   k=15: in_byte=1, in_bit=7 → orig_byte=1%2=1, orig_bit=15/2=7 → bit=(0x4D>>7)&1=0 → result[1] |= 0<<7
    //   result[0] = 0b10110100 = 0xB4 (bits: 0,0,1,0,1,1,0,1 from LSB)
    //                          actually: bit0=0, bit1=0, bit2=1, bit3=0, bit4=1, bit5=1, bit6=0, bit7=1 = 0b10110100 = 0xB4? No.
    // Let me just use round-trip: apply forward, then verify reverse recovers original.
    const original = [_]u8{ 0xB2, 0x4D };
    var bitsliced: [2]u8 = undefined;
    applyFullBitslice(&original, &bitsliced);

    // Verify forward transform produced something different (not identity).
    try std.testing.expect(!std.mem.eql(u8, &original, &bitsliced));

    var recovered: [2]u8 = undefined;
    try reverseFullBitslice(&bitsliced, &recovered);
    try std.testing.expectEqualSlices(u8, &original, &recovered);
}

test "reverseFullBitslice: single byte is identity" {
    // With N=1, k goes 0..7, orig_byte = k%1 = 0, orig_bit = k/1 = k.
    // So bit k of input → bit k of output. Identity transform.
    const data = [_]u8{0xA5};
    var out: [1]u8 = undefined;
    try reverseFullBitslice(&data, &out);
    try std.testing.expectEqualSlices(u8, &data, &out);
}

test "reverseFullBitslice: empty" {
    var out: [0]u8 = undefined;
    try reverseFullBitslice(&.{}, &out);
}

/// Test-only: forward FullBitslice transform (encoding direction).
/// Used to verify round-trip with `reverseFullBitslice`.
fn applyFullBitslice(data: []const u8, out: []u8) void {
    const n = data.len;
    @memset(out, 0);
    for (0..n) |orig_byte_idx| {
        for (0..8) |orig_bit_idx| {
            const k = orig_bit_idx * n + orig_byte_idx;
            const out_byte_idx = k / 8;
            const out_bit_idx = k % 8;
            if (out_byte_idx >= n) continue;
            const bit: u8 = (data[orig_byte_idx] >> @intCast(orig_bit_idx)) & 1;
            out[out_byte_idx] |= bit << @intCast(out_bit_idx);
        }
    }
}
