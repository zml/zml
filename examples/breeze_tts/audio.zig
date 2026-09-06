const std = @import("std");
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

pub fn readWav(a: std.mem.Allocator, data: []const u8) ![]f32 {
    if (data.len < 12 or !std.mem.eql(u8, data[0..4], "RIFF") or !std.mem.eql(u8, data[8..12], "WAVE")) return error.InvalidWav;
    const end = @as(usize, std.mem.readInt(u32, data[4..8], .little)) + 8;
    if (end > data.len) return error.TruncatedWav;
    var format: ?[]const u8 = null;
    var pcm: ?[]const u8 = null;
    var pos: usize = 12;
    while (pos + 8 <= end) {
        const size: usize = std.mem.readInt(u32, data[pos + 4 ..][0..4], .little);
        const start = pos + 8;
        if (size > end - start) return error.TruncatedWav;
        if (std.mem.eql(u8, data[pos..][0..4], "fmt ")) format = data[start..][0..size];
        if (std.mem.eql(u8, data[pos..][0..4], "data")) pcm = data[start..][0..size];
        pos = start + size + (size % 2);
    }
    const fmt = format orelse return error.InvalidWav;
    const bytes = pcm orelse return error.InvalidWav;
    if (fmt.len < 16) return error.InvalidWav;
    var kind: u32 = std.mem.readInt(u16, fmt[0..2], .little);
    const channels = std.mem.readInt(u16, fmt[2..4], .little);
    const rate = std.mem.readInt(u32, fmt[4..8], .little);
    const bits = std.mem.readInt(u16, fmt[14..16], .little);
    if (kind == 0xfffe) {
        if (fmt.len < 40) return error.InvalidWav;
        const extension_size = std.mem.readInt(u16, fmt[16..18], .little);
        if (extension_size < 22 or extension_size > fmt.len - 18) return error.InvalidWav;
        // PCM and IEEE float use the standard wave subtype GUID, with the
        // original format tag in its first four bytes (little endian).
        const subtype_tail = [_]u8{ 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71 };
        if (!std.mem.eql(u8, fmt[28..40], &subtype_tail)) return error.UnsupportedWav;
        kind = std.mem.readInt(u32, fmt[24..28], .little);
        const valid_bits = std.mem.readInt(u16, fmt[18..20], .little);
        if (valid_bits == 0 or valid_bits > bits or (kind == 3 and valid_bits != bits)) return error.InvalidWav;
        // PCM valid bits are left aligned in the container, so scaling by
        // the container width below also handles e.g. 24 valid bits in i32.
    }
    if (rate != 24000) return error.ReferenceMustBe24kHz;
    if (channels == 0 or channels > 8 or !((kind == 1 and (bits == 16 or bits == 24 or bits == 32)) or (kind == 3 and bits == 32))) return error.UnsupportedWav;
    const width: usize = bits / 8;
    const stride = width * channels;
    if (std.mem.readInt(u16, fmt[12..14], .little) != stride or bytes.len % stride != 0 or bytes.len == 0) return error.InvalidWav;
    if (bytes.len / stride > 24000 * 30) return error.ReferenceTooLong;
    const samples = try a.alloc(f32, bytes.len / stride);
    errdefer a.free(samples);
    for (samples, 0..) |*sample, i| {
        sample.* = 0;
        for (0..channels) |c| {
            const b = bytes[i * stride + c * width ..][0..width];
            const v: f32 = if (kind == 3) @bitCast(std.mem.readInt(u32, b[0..4], .little)) else switch (bits) {
                16 => @as(f32, @floatFromInt(std.mem.readInt(i16, b[0..2], .little))) / 32768,
                24 => @as(f32, @floatFromInt(std.mem.readInt(i24, b[0..3], .little))) / 8388608,
                32 => @as(f32, @floatFromInt(std.mem.readInt(i32, b[0..4], .little))) / 2147483648,
                else => unreachable,
            };
            if (!std.math.isFinite(v)) return error.NonFiniteAudio;
            sample.* += v / @as(f32, @floatFromInt(channels));
        }
    }
    return samples;
}
test "PCM WAV round trip and truncated data rejection" {
    var bytes: [48]u8 = undefined;
    @memcpy(bytes[0..44], &(try wavHeader(2)));
    std.mem.writeInt(i16, bytes[44..46], -32768, .little);
    std.mem.writeInt(i16, bytes[46..48], 16384, .little);
    const samples = try readWav(std.testing.allocator, &bytes);
    defer std.testing.allocator.free(samples);
    try std.testing.expectEqualSlices(f32, &.{ -1, 0.5 }, samples);
    try std.testing.expectError(error.TruncatedWav, readWav(std.testing.allocator, bytes[0..47]));
}

fn extensibleFixture(subtype: u32, bits: u16, valid_bits: u16, pcm: []const u8) ![]u8 {
    const result = try std.testing.allocator.alloc(u8, 68 + pcm.len + pcm.len % 2);
    @memset(result, 0);
    const header = try wavHeader(0);
    @memcpy(result[0..36], header[0..36]);
    std.mem.writeInt(u32, result[4..8], @intCast(result.len - 8), .little);
    std.mem.writeInt(u32, result[16..20], 40, .little);
    std.mem.writeInt(u16, result[20..22], 0xfffe, .little);
    std.mem.writeInt(u32, result[28..32], @as(u32, 24000) * (bits / 8), .little);
    std.mem.writeInt(u16, result[32..34], bits / 8, .little);
    std.mem.writeInt(u16, result[34..36], bits, .little);
    std.mem.writeInt(u16, result[36..38], 22, .little);
    std.mem.writeInt(u16, result[38..40], valid_bits, .little);
    std.mem.writeInt(u32, result[40..44], 4, .little);
    std.mem.writeInt(u32, result[44..48], subtype, .little);
    @memcpy(result[48..60], &[_]u8{ 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71 });
    @memcpy(result[60..64], "data");
    std.mem.writeInt(u32, result[64..68], @intCast(pcm.len), .little);
    @memcpy(result[68..][0..pcm.len], pcm);
    return result;
}

test "extensible mono PCM16 recording header decodes without conversion" {
    const bytes = try extensibleFixture(1, 16, 16, &.{ 0, 0x80, 0, 0x40 });
    defer std.testing.allocator.free(bytes);
    const samples = try readWav(std.testing.allocator, bytes);
    defer std.testing.allocator.free(samples);
    try std.testing.expectEqualSlices(f32, &.{ -1, 0.5 }, samples);
}

test "extensible float32 and left aligned PCM24 in a 32 bit container" {
    const floats = try extensibleFixture(3, 32, 32, &.{ 0, 0, 0, 0xbf, 0, 0, 0x80, 0x3e });
    defer std.testing.allocator.free(floats);
    const f = try readWav(std.testing.allocator, floats);
    defer std.testing.allocator.free(f);
    try std.testing.expectEqualSlices(f32, &.{ -0.5, 0.25 }, f);
    const pcm = try extensibleFixture(1, 32, 24, &.{ 0, 0, 0, 0x80, 0, 0, 0, 0x40 });
    defer std.testing.allocator.free(pcm);
    const p = try readWav(std.testing.allocator, pcm);
    defer std.testing.allocator.free(p);
    try std.testing.expectEqualSlices(f32, &.{ -1, 0.5 }, p);
}

test "extensible WAV rejects malformed extensions and unsupported subtypes" {
    const bytes = try extensibleFixture(1, 16, 16, &.{ 0, 0 });
    defer std.testing.allocator.free(bytes);
    std.mem.writeInt(u16, bytes[36..38], 21, .little);
    try std.testing.expectError(error.InvalidWav, readWav(std.testing.allocator, bytes));
    std.mem.writeInt(u16, bytes[36..38], 23, .little);
    try std.testing.expectError(error.InvalidWav, readWav(std.testing.allocator, bytes));
    std.mem.writeInt(u16, bytes[36..38], 22, .little);
    std.mem.writeInt(u16, bytes[38..40], 17, .little);
    try std.testing.expectError(error.InvalidWav, readWav(std.testing.allocator, bytes));
    std.mem.writeInt(u16, bytes[38..40], 16, .little);
    bytes[59] ^= 1;
    try std.testing.expectError(error.UnsupportedWav, readWav(std.testing.allocator, bytes));
    bytes[59] ^= 1;
    std.mem.writeInt(u32, bytes[44..48], 6, .little);
    try std.testing.expectError(error.UnsupportedWav, readWav(std.testing.allocator, bytes));
}
