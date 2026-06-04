const std = @import("std");
const bg4 = @import("bg4.zig");
const lz4 = @import("lz4.zig");
const stats = @import("stats.zig");

pub const MAX_DECODED_FRAME_BYTES = 128 * 1024;

const AtomicStats = stats.AtomicStats;

pub const Compression = enum(u8) {
    none = 0,
    lz4 = 1,
    byte_grouping_4_lz4 = 2,
};

pub const Frame = struct {
    compression: Compression,
    compressed_size: usize,
    uncompressed_size: usize,
};

pub const FrameReader = struct {
    reader: *std.Io.Reader,

    pub fn init(reader: *std.Io.Reader) FrameReader {
        return .{ .reader = reader };
    }

    pub fn next(self: *FrameReader) !Frame {
        var header_buf: [8]u8 = undefined;
        try self.reader.readSliceAll(&header_buf);

        if (header_buf[0] != 0) return error.UnsupportedXorbVersion;

        return .{
            .compressed_size = std.mem.readInt(u24, header_buf[1..4], .little),
            .compression = switch (header_buf[4]) {
                0 => .none,
                1 => .lz4,
                2 => .byte_grouping_4_lz4,
                else => return error.UnsupportedCompression,
            },
            .uncompressed_size = std.mem.readInt(u24, header_buf[5..8], .little),
        };
    }
};

pub const FrameDecoder = struct {
    stats: *AtomicStats,
    io: std.Io,

    pub fn decode(self: FrameDecoder, input: *std.Io.Reader, frame: Frame, destination: []u8, grouped_scratch: []u8) !void {
        const started = std.Io.Timestamp.now(self.io, .awake);

        switch (frame.compression) {
            .none => {
                try input.readSliceAll(destination);
            },
            .lz4 => {
                var out: std.Io.Writer = .fixed(destination);
                var lz4_reader: lz4.BlockReader = .init(input, frame.compressed_size, frame.uncompressed_size);
                _ = try lz4_reader.interface.streamRemaining(&out);
            },
            .byte_grouping_4_lz4 => {
                if (frame.uncompressed_size > grouped_scratch.len) return error.ChunkTooLarge;
                var grouped_writer: bg4.DegroupWriter = .init(grouped_scratch[0..frame.uncompressed_size], destination);
                var lz4_reader: lz4.BlockReader = .init(input, frame.compressed_size, frame.uncompressed_size);
                _ = try lz4_reader.interface.streamRemaining(&grouped_writer.interface);
                try grouped_writer.interface.flush();
            },
        }

        self.stats.addDecodeNs(@intCast(started.untilNow(self.io, .awake).nanoseconds));
    }
};
