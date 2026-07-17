const std = @import("std");

/// Decode an LZ4 block or frame payload directly into `out`.
pub fn decodeBlockInto(compressed: []const u8, out: []u8) !void {
    try BlockReader.readPayloadInto(compressed, out);
}

pub const BlockReader = struct {
    const frame_magic: u32 = 0x184d2204;
    const frame_magic_bytes = [_]u8{ 0x04, 0x22, 0x4d, 0x18 };
    const version_mask: u8 = 0xc0;
    const supported_version: u8 = 0x40;
    const block_independence_flag: u8 = 0x20;
    const block_checksum_flag: u8 = 0x10;
    const content_size_flag: u8 = 0x08;
    const content_checksum_flag: u8 = 0x04;
    const dict_id_flag: u8 = 0x01;
    const block_uncompressed_flag: u32 = @as(u32, 1) << 31;
    const block_size_mask: u32 = block_uncompressed_flag - 1;
    const min_match_len: usize = 4;
    const extended_len_marker: u8 = 15;
    const extended_len_step: u8 = 255;

    source: *std.Io.Reader,
    compressed_remaining: usize,
    decoded_size: usize,
    decoded_remaining: usize,
    interface: std.Io.Reader,

    pub fn init(source: *std.Io.Reader, compressed_size: usize, decoded_size: usize) BlockReader {
        return .{
            .source = source,
            .compressed_remaining = compressed_size,
            .decoded_size = decoded_size,
            .decoded_remaining = decoded_size,
            .interface = .{
                .vtable = &.{
                    .stream = stream,
                    .discard = discard,
                },
                .buffer = &.{},
                .seek = 0,
                .end = 0,
            },
        };
    }

    fn stream(reader: *std.Io.Reader, writer: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *BlockReader = @alignCast(@fieldParentPtr("interface", reader));
        if (self.decoded_remaining == 0) return error.EndOfStream;
        if (@intFromEnum(limit) < self.decoded_size) return error.ReadFailed;

        var decoded = writer.writableSliceGreedy(self.decoded_size) catch return error.WriteFailed;
        if (decoded.len < self.decoded_size) return error.WriteFailed;
        decoded = decoded[0..self.decoded_size];

        const payload_size = self.compressed_remaining;
        const payload = self.source.take(payload_size) catch return error.ReadFailed;
        self.compressed_remaining = 0;
        readPayloadInto(payload, decoded) catch return error.ReadFailed;

        writer.advance(self.decoded_size);
        self.decoded_remaining = 0;
        return self.decoded_size;
    }

    fn discard(reader: *std.Io.Reader, limit: std.Io.Limit) std.Io.Reader.Error!usize {
        const self: *BlockReader = @alignCast(@fieldParentPtr("interface", reader));
        if (self.decoded_remaining == 0) return error.EndOfStream;
        if (@intFromEnum(limit) < self.decoded_size) return error.ReadFailed;
        try self.source.discardAll(self.compressed_remaining);
        self.compressed_remaining = 0;
        self.decoded_remaining = 0;
        return self.decoded_size;
    }

    fn readPayloadInto(input: []const u8, out: []u8) !void {
        if (isFrame(input)) {
            try readFrameInto(input, out);
        } else {
            const written = try readTokenBlock(input, out);
            if (written != out.len) return error.InvalidLz4;
        }
    }

    fn isFrame(input: []const u8) bool {
        return input.len >= frame_magic_bytes.len and std.mem.eql(u8, input[0..frame_magic_bytes.len], &frame_magic_bytes);
    }

    fn readFrameInto(input: []const u8, out: []u8) !void {
        var ip: usize = 0;
        if (try takeInt(input, &ip, u32) != frame_magic) return error.InvalidLz4;

        const flag = try takeByte(input, &ip);
        _ = try takeByte(input, &ip);
        if ((flag & version_mask) != supported_version) return error.InvalidLz4;
        if ((flag & block_independence_flag) == 0) return error.UnsupportedLz4Frame;

        const has_block_checksum = (flag & block_checksum_flag) != 0;
        const has_content_size = (flag & content_size_flag) != 0;
        const has_content_checksum = (flag & content_checksum_flag) != 0;
        const has_dict_id = (flag & dict_id_flag) != 0;

        if (has_content_size) {
            const content_size = try takeInt(input, &ip, u64);
            if (content_size != out.len) return error.InvalidLz4;
        }
        if (has_dict_id) try skip(input, &ip, 4);
        _ = try takeByte(input, &ip);

        var op: usize = 0;
        while (true) {
            const raw_block_size = try takeInt(input, &ip, u32);
            if (raw_block_size == 0) break;

            const block_size: usize = @intCast(raw_block_size & block_size_mask);
            const block = try take(input, &ip, block_size);
            if ((raw_block_size & block_uncompressed_flag) != 0) {
                if (op + block.len > out.len) return error.InvalidLz4;
                @memcpy(out[op..][0..block.len], block);
                op += block.len;
            } else {
                op += try readTokenBlock(block, out[op..]);
            }

            if (has_block_checksum) try skip(input, &ip, 4);
        }

        if (has_content_checksum) try skip(input, &ip, 4);
        if (op != out.len or ip != input.len) return error.InvalidLz4;
    }

    fn take(input: []const u8, ip: *usize, n: usize) ![]const u8 {
        if (n > input.len - ip.*) return error.InvalidLz4;
        const start = ip.*;
        ip.* += n;
        return input[start..][0..n];
    }

    fn takeByte(input: []const u8, ip: *usize) !u8 {
        if (ip.* >= input.len) return error.InvalidLz4;
        defer ip.* += 1;
        return input[ip.*];
    }

    fn takeInt(input: []const u8, ip: *usize, comptime T: type) !T {
        return std.mem.readInt(T, (try take(input, ip, @sizeOf(T)))[0..@sizeOf(T)], .little);
    }

    fn skip(input: []const u8, ip: *usize, n: usize) !void {
        _ = try take(input, ip, n);
    }

    fn readTokenBlock(input: []const u8, out: []u8) !usize {
        var ip: usize = 0;
        var op: usize = 0;

        while (ip < input.len) {
            const token = input[ip];
            ip += 1;

            const literal_len = try readLength(input, &ip, token >> 4);
            if (literal_len > input.len - ip or literal_len > out.len - op) return error.InvalidLz4;
            @memcpy(out[op..][0..literal_len], input[ip..][0..literal_len]);
            ip += literal_len;
            op += literal_len;

            if (ip == input.len) break;
            if (input.len - ip < 2) return error.InvalidLz4;
            const match_offset: usize = @intCast(std.mem.readInt(u16, input[ip..][0..2], .little));
            ip += 2;
            if (match_offset == 0 or match_offset > op) return error.InvalidLz4;

            const match_len = try readLength(input, &ip, token & 0x0f) + min_match_len;
            if (match_len > out.len - op) return error.InvalidLz4;
            copyMatch(out, op, match_offset, match_len);
            op += match_len;
        }

        return op;
    }

    fn readLength(input: []const u8, ip: *usize, base: u8) !usize {
        var result: usize = base;
        if (base != extended_len_marker) return result;
        while (true) {
            if (ip.* >= input.len) return error.InvalidLz4;
            const extra = input[ip.*];
            ip.* += 1;
            result += extra;
            if (extra != extended_len_step) return result;
        }
    }

    fn copyMatch(out: []u8, write_pos: usize, match_offset: usize, match_len: usize) void {
        if (match_offset == 1) {
            @memset(out[write_pos..][0..match_len], out[write_pos - 1]);
            return;
        }

        // Overlap-focused path: for offsets >= 8, we can stream fixed-size copies.
        // Every 8-byte chunk read is fully behind the write cursor, so @memcpy is valid.
        if (match_offset >= 8) {
            var copied8: usize = 0;
            while (copied8 + 8 <= match_len) : (copied8 += 8) {
                @memcpy(
                    out[write_pos + copied8 ..][0..8],
                    out[write_pos + copied8 - match_offset ..][0..8],
                );
            }
            if (copied8 < match_len) {
                @memcpy(
                    out[write_pos + copied8 ..][0 .. match_len - copied8],
                    out[write_pos + copied8 - match_offset ..][0 .. match_len - copied8],
                );
            }
            return;
        }

        const prefix_len = @min(match_offset, match_len);
        @memcpy(out[write_pos..][0..prefix_len], out[write_pos - match_offset ..][0..prefix_len]);

        var copied = prefix_len;
        while (copied < match_len) {
            const n = @min(copied, match_len - copied);
            @memcpy(out[write_pos + copied ..][0..n], out[write_pos..][0..n]);
            copied += n;
        }
    }
};
