const std = @import("std");

pub const Chunk = struct {
    pub const Id = enum(u32) {
        riff = @bitCast(@as([4]u8, "RIFF".*)),
        fmt = @bitCast(@as([4]u8, "fmt ".*)),
        data = @bitCast(@as([4]u8, "data".*)),
        _,
    };

    pub const Header = extern struct {
        id: Id align(1),
        size: u32 align(1),
    };

    pub const Riff = extern struct {
        pub const Format = enum(u32) {
            wave = @bitCast(@as([4]u8, "WAVE".*)),
            _,
        };

        format: Format align(1),
    };

    pub const Fmt = extern struct {
        pub const AudioFormat = enum(u16) {
            pcm = 1,
            _,
        };

        audio_format: AudioFormat align(1),
        num_channels: u16 align(1),
        sample_rate: u32 align(1),
        byte_rate: u32 align(1),
        block_align: u16 align(1),
        bits_per_sample: u16 align(1),
    };
};

pub fn readPcmWav(allocator: std.mem.Allocator, reader: *std.Io.Reader, out: *std.ArrayList(u8)) !Chunk.Fmt {
    std.debug.assert((try reader.takeStruct(Chunk.Header, .little)).id == .riff);
    std.debug.assert((try reader.takeStruct(Chunk.Riff, .little)).format == .wave);
    var fmt: ?Chunk.Fmt = null;
    var data_read: bool = false;
    while (fmt == null or !data_read) {
        const hdr = try reader.takeStruct(Chunk.Header, .little);
        switch (hdr.id) {
            .fmt => {
                fmt = try reader.takeStruct(Chunk.Fmt, .little);
                if (hdr.size > @sizeOf(Chunk.Fmt)) {
                    try reader.discardAll(hdr.size - @sizeOf(Chunk.Fmt));
                }
            },
            .data => {
                try out.resize(allocator, hdr.size);
                try reader.readSliceAll(out.items);
                data_read = true;
            },
            else => try reader.discardAll(hdr.size),
        }
    }
    return fmt.?;
}

// pub fn writePcmWav(writer: anytype, fmt: Chunk.Fmt, data: []const u8) !void {
//     try writer.writeStruct(Chunk.Header{ .id = .riff, .size = @sizeOf(Chunk.Riff) });
//     try writer.writeStruct(Chunk.Riff{ .format = .wave });
//     try writer.writeStruct(Chunk.Header{ .id = .fmt, .size = @sizeOf(Chunk.Fmt) });
//     try writer.writeStruct(fmt);
//     try writer.writeStruct(Chunk.Header{ .id = .data, .size = @intCast(data.len) });
//     try writer.writeAll(data);
// }
