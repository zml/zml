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

fn readPcmWav(allocator: std.mem.Allocator, reader: *std.Io.Reader, out: *std.ArrayList(u8)) !Chunk.Fmt {
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

pub fn loadWav(allocator: std.mem.Allocator, reader: *std.Io.Reader) ![]const f32 {
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    var sample_list: std.ArrayList(u8) = .empty;

    const wav_fmt = try readPcmWav(arena, reader, &sample_list);
    const byte_per_sample = wav_fmt.bits_per_sample / 8;
    const sample_count = sample_list.items.len / (byte_per_sample * wav_fmt.num_channels);

    const samples = try allocator.alloc(f32, sample_count);
    for (0..sample_count) |i| {
	const offset = i * byte_per_sample * wav_fmt.num_channels;

	const sample = switch(byte_per_sample) {
	    1 => (@as(f32, @floatFromInt(std.mem.bytesToValue(u8, sample_list.items[offset .. offset + 1]))) - 128.0) / 128.0,
	    2 => @as(f32, @floatFromInt(std.mem.bytesToValue(i16, sample_list.items[offset .. offset + 2]))) / 32768.0,
	    3 => @as(f32, @floatFromInt(std.mem.bytesToValue(i24, sample_list.items[offset .. offset + 3]))) / 8388608.0,
	    4 => @as(f32, @floatFromInt(std.mem.bytesToValue(i32, sample_list.items[offset .. offset + 4]))) / 2147483648.0,
	    else => unreachable,
	};

	samples[i] = sample;
    }

    return samples;
}
