const std = @import("std");

pub const BlockReader = struct {
    const FRAME_MAGIC: u32 = 0x184d2204;
    const FRAME_MAGIC_BYTES = [_]u8{ 0x04, 0x22, 0x4d, 0x18 };
    const VERSION_MASK: u8 = 0xc0;
    const SUPPORTED_VERSION: u8 = 0x40;
    const BLOCK_INDEPENDENCE_FLAG: u8 = 0x20;
    const BLOCK_CHECKSUM_FLAG: u8 = 0x10;
    const CONTENT_SIZE_FLAG: u8 = 0x08;
    const CONTENT_CHECKSUM_FLAG: u8 = 0x04;
    const DICT_ID_FLAG: u8 = 0x01;
    const BLOCK_UNCOMPRESSED_FLAG: u32 = @as(u32, 1) << 31;
    const BLOCK_SIZE_MASK: u32 = BLOCK_UNCOMPRESSED_FLAG - 1;
    const MIN_MATCH_LEN: usize = 4;
    const EXTENDED_LEN_MARKER: u8 = 15;
    const EXTENDED_LEN_STEP: u8 = 255;

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
        return input.len >= FRAME_MAGIC_BYTES.len and std.mem.eql(u8, input[0..FRAME_MAGIC_BYTES.len], &FRAME_MAGIC_BYTES);
    }

    fn readFrameInto(input: []const u8, out: []u8) !void {
        var ip: usize = 0;
        if (try takeInt(input, &ip, u32) != FRAME_MAGIC) return error.InvalidLz4;

        const flag = try takeByte(input, &ip);
        _ = try takeByte(input, &ip);
        if ((flag & VERSION_MASK) != SUPPORTED_VERSION) return error.InvalidLz4;
        if ((flag & BLOCK_INDEPENDENCE_FLAG) == 0) return error.UnsupportedLz4Frame;

        const has_block_checksum = (flag & BLOCK_CHECKSUM_FLAG) != 0;
        const has_content_size = (flag & CONTENT_SIZE_FLAG) != 0;
        const has_content_checksum = (flag & CONTENT_CHECKSUM_FLAG) != 0;
        const has_dict_id = (flag & DICT_ID_FLAG) != 0;

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

            const block_size: usize = @intCast(raw_block_size & BLOCK_SIZE_MASK);
            const block = try take(input, &ip, block_size);
            if ((raw_block_size & BLOCK_UNCOMPRESSED_FLAG) != 0) {
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

            const match_len = try readLength(input, &ip, token & 0x0f) + MIN_MATCH_LEN;
            if (match_len > out.len - op) return error.InvalidLz4;
            copyMatch(out, op, match_offset, match_len);
            op += match_len;
        }

        return op;
    }

    fn readLength(input: []const u8, ip: *usize, base: u8) !usize {
        var result: usize = base;
        if (base != EXTENDED_LEN_MARKER) return result;
        while (true) {
            if (ip.* >= input.len) return error.InvalidLz4;
            const extra = input[ip.*];
            ip.* += 1;
            result += extra;
            if (extra != EXTENDED_LEN_STEP) return result;
        }
    }

    fn copyMatch(out: []u8, write_pos: usize, match_offset: usize, match_len: usize) void {
        if (match_offset == 1) {
            @memset(out[write_pos..][0..match_len], out[write_pos - 1]);
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
