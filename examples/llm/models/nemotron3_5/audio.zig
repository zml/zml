const std = @import("std");

pub const DecodedAudio = struct {
    samples: []f32,
    sample_rate: u32,
    channels: u16,

    pub fn deinit(self: DecodedAudio, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
    }

    pub fn frameCount(self: DecodedAudio) usize {
        return self.samples.len / self.channels;
    }
};

pub const DecodeError = std.mem.Allocator.Error || error{
    UnsupportedAudioFormat,
    UnsupportedWavContainer,
    UnsupportedWavEncoding,
    TruncatedWav,
    InvalidWavFormat,
    InvalidWavData,
    DuplicateWavChunk,
    MissingWavFormat,
    MissingWavData,
    EmptyAudio,
    NonFiniteAudioSample,
};

pub const targetSampleRate: u32 = 16_000;

pub const PrepareError = std.mem.Allocator.Error || error{
    InvalidAudioFormat,
    InvalidAudioData,
    EmptyAudio,
    NonFiniteAudioSample,
};

const Format = struct {
    encoding: enum { pcm, ieee_float },
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    block_align: u16,
};

pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) DecodeError!DecodedAudio {
    if (bytes.len < 12) return error.TruncatedWav;
    if (std.mem.eql(u8, bytes[0..4], "RIFX") or std.mem.eql(u8, bytes[0..4], "RF64")) return error.UnsupportedWavContainer;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.UnsupportedAudioFormat;

    const riff_size: usize = readInt(u32, bytes, 4);
    if (riff_size < 4) return error.InvalidWavFormat;
    if (riff_size > bytes.len - 8) return error.TruncatedWav;
    const riff_end = 8 + riff_size;

    var format: ?Format = null;
    var data: ?[]const u8 = null;
    var offset: usize = 12;
    while (offset < riff_end) {
        if (riff_end - offset < 8) return error.TruncatedWav;
        const chunk_id = bytes[offset..][0..4];
        const chunk_size: usize = readInt(u32, bytes, offset + 4);
        const payload_start = offset + 8;
        if (chunk_size > riff_end - payload_start) return error.TruncatedWav;
        const payload_end = payload_start + chunk_size;
        const payload = bytes[payload_start..payload_end];

        if (std.mem.eql(u8, chunk_id, "fmt ")) {
            if (format != null) return error.DuplicateWavChunk;
            format = try parseFormat(payload);
        } else if (std.mem.eql(u8, chunk_id, "data")) {
            if (data != null) return error.DuplicateWavChunk;
            data = payload;
        }

        offset = payload_end;
        if (chunk_size % 2 != 0) {
            if (offset == riff_end) return error.TruncatedWav;
            offset += 1;
        }
    }

    const fmt = format orelse return error.MissingWavFormat;
    const encoded = data orelse return error.MissingWavData;
    if (encoded.len % fmt.block_align != 0) return error.InvalidWavData;
    if (encoded.len == 0) return error.EmptyAudio;

    const bytes_per_sample = fmt.bits_per_sample / 8;
    const samples = try allocator.alloc(f32, encoded.len / bytes_per_sample);
    errdefer allocator.free(samples);

    for (samples, 0..) |*sample, i| {
        const sample_offset = i * bytes_per_sample;
        sample.* = switch (fmt.encoding) {
            .pcm => switch (fmt.bits_per_sample) {
                8 => (@as(f32, @floatFromInt(encoded[sample_offset])) - 128.0) / 128.0,
                16 => @as(f32, @floatFromInt(readInt(i16, encoded, sample_offset))) / 32768.0,
                24 => @as(f32, @floatFromInt(readInt(i24, encoded, sample_offset))) / 8388608.0,
                32 => @as(f32, @floatFromInt(readInt(i32, encoded, sample_offset))) / 2147483648.0,
                else => unreachable, // Validated by parseFormat
            },
            .ieee_float => @bitCast(readInt(u32, encoded, sample_offset)),
        };
        if (!std.math.isFinite(sample.*)) return error.NonFiniteAudioSample;
    }

    return .{
        .samples = samples,
        .sample_rate = fmt.sample_rate,
        .channels = fmt.channels,
    };
}

/// Yoinks decoded audio and returns allocated mono, 16 kHz samples
pub fn prepare(allocator: std.mem.Allocator, audio: DecodedAudio) PrepareError!DecodedAudio {
    if (audio.channels == 0 or audio.sample_rate == 0) return error.InvalidAudioFormat;
    if (audio.samples.len % audio.channels != 0) return error.InvalidAudioData;
    if (audio.samples.len == 0) return error.EmptyAudio;

    const mono = try downmix(allocator, audio);
    if (audio.sample_rate == targetSampleRate) {
        return .{ .samples = mono, .sample_rate = targetSampleRate, .channels = 1 };
    }
    defer allocator.free(mono);

    return .{
        .samples = try resample(allocator, mono, audio.sample_rate),
        .sample_rate = targetSampleRate,
        .channels = 1,
    };
}

fn downmix(allocator: std.mem.Allocator, audio: DecodedAudio) PrepareError![]f32 {
    const mono = try allocator.alloc(f32, audio.frameCount());
    errdefer allocator.free(mono);
    for (mono, 0..) |*sample, frame| {
        const start = frame * audio.channels;
        var sum: f64 = 0;
        for (audio.samples[start..][0..audio.channels]) |channel| {
            if (!std.math.isFinite(channel)) return error.NonFiniteAudioSample;
            sum += channel;
        }
        sample.* = @floatCast(sum / @as(f64, @floatFromInt(audio.channels)));
    }
    return mono;
}

fn resample(allocator: std.mem.Allocator, mono: []const f32, sourceRate: u32) PrepareError![]f32 {
    const count = (@as(u128, mono.len) * targetSampleRate + sourceRate - 1) / sourceRate;
    if (count > std.math.maxInt(usize)) return error.OutOfMemory;
    const output = try allocator.alloc(f32, @intCast(count));
    errdefer allocator.free(output);

    const ratio = @as(f64, @floatFromInt(sourceRate)) / @as(f64, targetSampleRate);
    const cutoff = 0.94 * @min(1.0, 1.0 / ratio);
    const radius = 32.0 / cutoff;
    const halfWidth: usize = @intFromFloat(@ceil(radius));

    for (output, 0..) |*sample, i| {
        const position = @as(f64, @floatFromInt(i)) * ratio;
        const center: usize = @min(mono.len - 1, @as(usize, @intFromFloat(position)));
        const start = center -| halfWidth;
        const end = @min(mono.len, center +| halfWidth +| 1);
        var value: f64 = 0;
        var weightSum: f64 = 0;
        for (mono[start..end], start..) |input, index| {
            const distance = @as(f64, @floatFromInt(index)) - position;
            if (@abs(distance) >= radius) continue;
            const phase = std.math.pi * distance / radius;
            const window = 0.42 + 0.5 * @cos(phase) + 0.08 * @cos(2.0 * phase);
            const weight = cutoff * sinc(cutoff * distance) * window;
            value += @as(f64, input) * weight;
            weightSum += weight;
        }
        sample.* = @floatCast(value / weightSum);
        if (!std.math.isFinite(sample.*)) return error.NonFiniteAudioSample;
    }
    return output;
}

fn sinc(x: f64) f64 {
    if (@abs(x) < 1e-12) return 1.0;
    const phase = std.math.pi * x;
    return @sin(phase) / phase;
}

fn parseFormat(bytes: []const u8) DecodeError!Format {
    if (bytes.len < 16) return error.InvalidWavFormat;
    var format_tag = readInt(u16, bytes, 0);
    const channels = readInt(u16, bytes, 2);
    const sample_rate = readInt(u32, bytes, 4);
    const byte_rate = readInt(u32, bytes, 8);
    const block_align = readInt(u16, bytes, 12);
    const bits_per_sample = readInt(u16, bytes, 14);

    var extra_size: usize = 0;
    if (bytes.len > 16) {
        if (bytes.len < 18) return error.InvalidWavFormat;
        extra_size = readInt(u16, bytes, 16);
        if (extra_size > bytes.len - 18) return error.InvalidWavFormat;
    }

    if (format_tag == 0xfffe) {
        if (extra_size < 22 or bytes.len < 40) return error.InvalidWavFormat;
        const valid_bits = readInt(u16, bytes, 18);
        const channel_mask = readInt(u32, bytes, 20);
        if (valid_bits == 0 or valid_bits > bits_per_sample) return error.InvalidWavFormat;
        if (channel_mask != 0 and @popCount(channel_mask) != channels) return error.InvalidWavFormat;

        // KSDATAFORMAT_SUBTYPE_PCM and IEEE_FLOAT differ only in the first GUID word.
        const subtype_tail = "\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71";
        if (!std.mem.eql(u8, bytes[28..40], subtype_tail)) return error.UnsupportedWavEncoding;
        const subtype = readInt(u32, bytes, 24);
        format_tag = switch (subtype) {
            1, 3 => @intCast(subtype),
            else => return error.UnsupportedWavEncoding,
        };
        if (format_tag == 3 and valid_bits != bits_per_sample) return error.InvalidWavFormat;
        // Extensible PCM stores valid bits left-aligned in the container, so the
        // container-width scaling used by decode also handles e.g. 24-in-32 PCM.
    }

    const encoding: @FieldType(Format, "encoding") = switch (format_tag) {
        1 => switch (bits_per_sample) {
            8, 16, 24, 32 => .pcm,
            else => return error.UnsupportedWavEncoding,
        },
        3 => if (bits_per_sample == 32) .ieee_float else return error.UnsupportedWavEncoding,
        else => return error.UnsupportedWavEncoding,
    };
    if (channels == 0 or sample_rate == 0) return error.InvalidWavFormat;
    const expected_block_align = @as(u32, channels) * (bits_per_sample / 8);
    if (block_align != expected_block_align) return error.InvalidWavFormat;
    if (@as(u64, sample_rate) * block_align != byte_rate) return error.InvalidWavFormat;

    return .{
        .encoding = encoding,
        .channels = channels,
        .sample_rate = sample_rate,
        .bits_per_sample = bits_per_sample,
        .block_align = block_align,
    };
}

fn readInt(comptime T: type, bytes: []const u8, offset: usize) T {
    const size = @divExact(@bitSizeOf(T), 8);
    return std.mem.readInt(T, bytes[offset..][0..size], .little);
}
