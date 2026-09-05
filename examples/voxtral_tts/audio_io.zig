const std = @import("std");

pub const Voice = struct {
    storage: []u8,
    bytes: []const u8,
    frames: usize,

    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !Voice {
        if (name.len == 0) return error.InvalidVoice;
        for (name) |c| if (!std.ascii.isAlphabetic(c) and c != '_') return error.InvalidVoice;
        const path = try std.fmt.allocPrint(allocator, "voice_embedding/{s}.pt", .{name});
        defer allocator.free(path);
        const storage = try dir.readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
        errdefer allocator.free(storage);
        const meta = try storedZipMember(storage, "/data.pkl");
        const bytes = try storedZipMember(storage, "/data/0");
        const byteorder = try storedZipMember(storage, "/byteorder");
        if (!std.mem.eql(u8, byteorder, "little")) return error.UnsupportedVoice;
        // These public .pt files hold one contiguous BF16 tensor, not a state
        // dict. Inspect only its metadata; never execute/unpickle Python code.
        if (!std.mem.startsWith(u8, meta, "\x80\x02ctorch._utils\n_rebuild_tensor_v2\n") or std.mem.indexOf(u8, meta, "ctorch\nBFloat16Storage\n") == null) return error.UnsupportedVoice;
        var cursor = (std.mem.indexOfScalar(u8, meta, 'Q') orelse return error.UnsupportedVoice) + 1;
        if (try pickleInt(meta, &cursor) != 0) return error.UnsupportedVoice;
        const frames = try pickleInt(meta, &cursor);
        if (try pickleInt(meta, &cursor) != 3072) return error.UnsupportedVoice;
        if (cursor + 3 > meta.len or meta[cursor] != 0x86 or meta[cursor + 1] != 'q') return error.UnsupportedVoice;
        cursor += 3;
        if (try pickleInt(meta, &cursor) != 3072 or try pickleInt(meta, &cursor) != 1) return error.UnsupportedVoice;
        if (frames == 0 or frames > 2048 or bytes.len != frames * 3072 * 2) return error.UnsupportedVoice;
        return .{ .storage = storage, .bytes = bytes, .frames = frames };
    }

    pub fn deinit(self: Voice, allocator: std.mem.Allocator) void {
        allocator.free(self.storage);
    }
};

fn pickleInt(data: []const u8, cursor: *usize) !usize {
    if (cursor.* >= data.len) return error.UnsupportedVoice;
    const count: usize = switch (data[cursor.*]) {
        'K' => 1,
        'M' => 2,
        'J' => 4,
        else => return error.UnsupportedVoice,
    };
    cursor.* += 1;
    if (cursor.* + count > data.len) return error.UnsupportedVoice;
    var value: usize = 0;
    for (data[cursor.*..][0..count], 0..) |b, i| value |= @as(usize, b) << @intCast(8 * i);
    cursor.* += count;
    return value;
}

fn read(comptime T: type, bytes: []const u8, offset: usize) !T {
    if (offset > bytes.len or @sizeOf(T) > bytes.len - offset) return error.InvalidZip;
    return std.mem.readInt(T, bytes[offset..][0..@sizeOf(T)], .little);
}

fn storedZipMember(bytes: []const u8, suffix: []const u8) ![]const u8 {
    const end = std.mem.lastIndexOf(u8, bytes, "PK\x05\x06") orelse return error.InvalidZip;
    var cursor: usize = try read(u32, bytes, end + 16);
    const entries = try read(u16, bytes, end + 10);
    for (0..entries) |_| {
        if (try read(u32, bytes, cursor) != 0x02014b50) return error.InvalidZip;
        const name_len = try read(u16, bytes, cursor + 28);
        const extra_len = try read(u16, bytes, cursor + 30);
        const comment_len = try read(u16, bytes, cursor + 32);
        const next = cursor + 46 + name_len + extra_len + comment_len;
        if (next > bytes.len) return error.InvalidZip;
        const name = bytes[cursor + 46 ..][0..name_len];
        if (std.mem.endsWith(u8, name, suffix)) {
            if (try read(u16, bytes, cursor + 10) != 0 or (try read(u16, bytes, cursor + 8) & 1) != 0) return error.UnsupportedVoice;
            const len = try read(u32, bytes, cursor + 24);
            if (try read(u32, bytes, cursor + 20) != len) return error.InvalidZip;
            const local: usize = try read(u32, bytes, cursor + 42);
            if (try read(u32, bytes, local) != 0x04034b50) return error.InvalidZip;
            const start: usize = @as(usize, local) + 30 + try read(u16, bytes, local + 26) + try read(u16, bytes, local + 28);
            if (start > bytes.len or len > bytes.len - start) return error.InvalidZip;
            const data = bytes[start..][0..len];
            if (std.hash.Crc32.hash(data) != try read(u32, bytes, cursor + 16)) return error.InvalidZip;
            return data;
        }
        cursor = next;
    }
    return error.MissingVoiceTensor;
}

pub fn writeWav(io: std.Io, path: []const u8, samples: []const f32) !void {
    for (samples) |sample| if (!std.math.isFinite(sample)) return error.NonFiniteAudio;
    const header = try wavHeader(samples.len);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true });
    defer file.close(io);
    var buffer: [16384]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writer.interface.writeAll(&header);
    for (samples) |sample| {
        const pcm: i16 = @intFromFloat(@round(std.math.clamp(sample, -1, 1) * 32767));
        try writer.interface.writeInt(i16, pcm, .little);
    }
    try writer.interface.flush();
}

pub fn wavHeader(samples: usize) ![44]u8 {
    if (samples > (std.math.maxInt(u32) - 36) / 2) return error.AudioTooLong;
    var result: [44]u8 = @splat(0);
    @memcpy(result[0..4], "RIFF");
    std.mem.writeInt(u32, result[4..8], @intCast(36 + samples * 2), .little);
    @memcpy(result[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, result[16..20], 16, .little);
    std.mem.writeInt(u16, result[20..22], 1, .little);
    std.mem.writeInt(u16, result[22..24], 1, .little);
    std.mem.writeInt(u32, result[24..28], 24000, .little);
    std.mem.writeInt(u32, result[28..32], 48000, .little);
    std.mem.writeInt(u16, result[32..34], 2, .little);
    std.mem.writeInt(u16, result[34..36], 16, .little);
    @memcpy(result[36..40], "data");
    std.mem.writeInt(u32, result[40..44], @intCast(samples * 2), .little);
    return result;
}

test "WAV header describes mono 24kHz signed PCM16" {
    const header = try wavHeader(24000);
    try std.testing.expectEqualStrings("RIFF", header[0..4]);
    try std.testing.expectEqual(@as(u32, 48036), std.mem.readInt(u32, header[4..8], .little));
    try std.testing.expectEqual(@as(u32, 24000), std.mem.readInt(u32, header[24..28], .little));
    try std.testing.expectEqual(@as(u32, 48000), std.mem.readInt(u32, header[40..44], .little));
    try std.testing.expectError(error.InvalidZip, storedZipMember("not a zip", "/data/0"));
}
