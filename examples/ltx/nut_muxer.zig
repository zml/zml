//! Minimal NUT muxer (PIPE mode) for piping raw video+audio to ffmpeg via stdout.
//!
//! Stream 0: raw RGB24 video (all keyframes)
//! Stream 1: PCM f32le interleaved audio (all keyframes)
//!
//! Uses only the fully-coded escape (frame code 1) for data frames,
//! which keeps the implementation simple at the cost of a few extra bytes
//! per frame header.

const std = @import("std");

// NUT startcodes (big-endian u64).
const MAIN_STARTCODE: u64 = 0x4E4D7A561F5F04AD;
const STREAM_STARTCODE: u64 = 0x4E5311405BF2F9DB;
const SYNCPOINT_STARTCODE: u64 = 0x4E4BE4ADEECA4569;

const ID_STRING = "nut/multimedia container\x00";
const NUT_VERSION: u64 = 4;
const NUT_PIPE: u64 = 2;
const MAX_DISTANCE: u64 = 1024 * 32 - 1;

// Frame flags.
const FLAG_KEY: u64 = 1;
const FLAG_CODED_PTS: u64 = 8;
const FLAG_STREAM_ID: u64 = 16;
const FLAG_SIZE_MSB: u64 = 32;
const FLAG_CHECKSUM: u64 = 64;
const FLAG_CODED: u64 = 4096;
const FLAG_INVALID: u64 = 8192;

// Codec tags (little-endian fourcc).
fn mkTag(a: u8, b: u8, c: u8, d: u8) u32 {
    return @as(u32, a) | @as(u32, b) << 8 | @as(u32, c) << 16 | @as(u32, d) << 24;
}
const VIDEO_TAG: u32 = mkTag('R', 'G', 'B', 24);
const AUDIO_TAG: u32 = mkTag('P', 'F', 'D', 32);

/// CRC-32 with polynomial 0x04C11DB7 (non-reflected, matches ffmpeg).
const Crc32 = std.hash.crc.Crc(u32, .{
    .polynomial = 0x04C11DB7,
    .initial = 0,
    .reflect_input = false,
    .reflect_output = false,
    .xor_output = 0,
});

/// Fixed-capacity byte buffer for building packet payloads on the stack.
const PayloadBuf = struct {
    data: [4096]u8 = undefined,
    len: usize = 0,

    fn appendSlice(self: *PayloadBuf, bytes: []const u8) void {
        @memcpy(self.data[self.len..][0..bytes.len], bytes);
        self.len += bytes.len;
    }

    fn slice(self: *const PayloadBuf) []const u8 {
        return self.data[0..self.len];
    }

    fn putV(self: *PayloadBuf, val: u64) void {
        const n = encodeV(val, self.data[self.len..]);
        self.len += n;
    }

    fn putS(self: *PayloadBuf, val: i64) void {
        const enc: u64 = if (val > 0)
            @intCast(2 * val - 1)
        else if (val < 0)
            @intCast(-2 * val)
        else
            0;
        self.putV(enc);
    }
};

pub const NutMuxer = struct {
    writer: *std.Io.Writer,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    audio_rate: u32,
    audio_channels: u32,
    last_pts: [2]i64,

    pub fn init(
        writer: *std.Io.Writer,
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
        audio_channels: u32,
        audio_rate: u32,
    ) NutMuxer {
        const g = gcd(fps_num, fps_den);
        return .{
            .writer = writer,
            .width = width,
            .height = height,
            .fps_num = fps_num / g,
            .fps_den = fps_den / g,
            .audio_rate = audio_rate,
            .audio_channels = audio_channels,
            .last_pts = .{ 0, 0 },
        };
    }

    pub fn writeHeaders(self: *NutMuxer) !void {
        try self.writer.writeAll(ID_STRING);

        var main_payload: PayloadBuf = .{};
        self.buildMainPayload(&main_payload);
        try self.emitPacket(MAIN_STARTCODE, main_payload.slice());

        var video_payload: PayloadBuf = .{};
        self.buildVideoStreamPayload(&video_payload);
        try self.emitPacket(STREAM_STARTCODE, video_payload.slice());

        var audio_payload: PayloadBuf = .{};
        self.buildAudioStreamPayload(&audio_payload);
        try self.emitPacket(STREAM_STARTCODE, audio_payload.slice());

        // Initial syncpoint (required even in PIPE mode for the decoder
        // to transition from header parsing to frame reading).
        var sp_payload: PayloadBuf = .{};
        sp_payload.putV(0); // global_key_pts: pts=0, time_base_id=0
        sp_payload.putV(0); // back_ptr_div16: no previous syncpoint
        try self.emitPacket(SYNCPOINT_STARTCODE, sp_payload.slice());
    }

    /// Mux all video frames interleaved with proportional audio chunks.
    pub fn mux(
        self: *NutMuxer,
        video_data: []const u8,
        num_frames: usize,
        frame_w: usize,
        frame_h: usize,
        audio_data: []const u8,
        num_channels: usize,
    ) !void {
        const frame_bytes = frame_w * frame_h * 3;
        const sample_bytes = num_channels * 4;
        const total_samples = audio_data.len / sample_bytes;
        var audio_pos: usize = 0;

        for (0..num_frames) |f| {
            try self.writeFrame(0, @intCast(f), video_data[f * frame_bytes ..][0..frame_bytes]);

            const target = (f + 1) * total_samples / num_frames;
            if (target > audio_pos) {
                const start = audio_pos * sample_bytes;
                const end = target * sample_bytes;
                try self.writeFrame(1, @intCast(audio_pos), audio_data[start..end]);
                audio_pos = target;
            }
        }
        if (audio_pos * sample_bytes < audio_data.len) {
            try self.writeFrame(1, @intCast(audio_pos), audio_data[audio_pos * sample_bytes ..]);
        }
    }

    // ── Frame writing (fully-coded escape) ──────────────────────────────

    fn writeFrame(self: *NutMuxer, stream_id: u8, pts: i64, data: []const u8) !void {
        var flags: u64 = FLAG_KEY | FLAG_SIZE_MSB | FLAG_STREAM_ID | FLAG_CODED_PTS;
        if (data.len > 2 * MAX_DISTANCE) flags |= FLAG_CHECKSUM;

        var hdr: [64]u8 = undefined;
        var n: usize = 0;

        hdr[n] = 1; // frame code 1 = fully-coded escape
        n += 1;
        n += encodeV(flags, hdr[n..]);
        n += encodeV(stream_id, hdr[n..]);
        n += encodeV(codePts(self.last_pts[stream_id], pts, if (stream_id == 0) 7 else 14), hdr[n..]);
        n += encodeV(@intCast(data.len), hdr[n..]);

        if (flags & FLAG_CHECKSUM != 0) {
            const c = Crc32.hash(hdr[0..n]);
            @memcpy(hdr[n..][0..4], &toBE32(c));
            n += 4;
        }

        try self.writer.writeAll(hdr[0..n]);
        try self.writer.writeAll(data);
        self.last_pts[stream_id] = pts;
    }

    // ── Header packet emission ──────────────────────────────────────────

    fn emitPacket(self: *NutMuxer, startcode: u64, payload: []const u8) !void {
        const forw_ptr: u64 = @intCast(payload.len + 4);

        try self.writer.writeAll(&toBE64(startcode));

        var vbuf: [10]u8 = undefined;
        const vn = encodeV(forw_ptr, &vbuf);
        try self.writer.writeAll(vbuf[0..vn]);

        if (forw_ptr > 4096) {
            var tmp: [18]u8 = undefined;
            @memcpy(tmp[0..8], &toBE64(startcode));
            @memcpy(tmp[8..][0..vn], vbuf[0..vn]);
            try self.writer.writeAll(&toBE32(Crc32.hash(tmp[0 .. 8 + vn])));
        }

        try self.writer.writeAll(payload);
        try self.writer.writeAll(&toBE32(Crc32.hash(payload)));
    }

    // ── Payload builders ────────────────────────────────────────────────

    fn buildMainPayload(self: *NutMuxer, buf: *PayloadBuf) void {
        buf.putV(NUT_VERSION);
        buf.putV(1); // minor_version (required for version > 3)
        buf.putV(2); // stream_count
        buf.putV(MAX_DISTANCE);
        buf.putV(2); // time_base_count
        buf.putV(self.fps_den); // TB 0: video
        buf.putV(self.fps_num);
        buf.putV(1); // TB 1: audio
        buf.putV(self.audio_rate);
        buildFrameCodeTable(buf);
        buf.putV(0); // elision header count
        buf.putV(NUT_PIPE);
    }

    fn buildVideoStreamPayload(self: *NutMuxer, buf: *PayloadBuf) void {
        buf.putV(0); // stream_id
        buf.putV(0); // class: video
        buf.putV(4); // fourcc len
        buf.appendSlice(&std.mem.toBytes(VIDEO_TAG));
        buf.putV(0); // time_base_id
        buf.putV(7); // msb_pts_shift
        buf.putV(self.fps_num / self.fps_den); // max_pts_distance
        buf.putV(0); // decode_delay
        buf.putV(0); // stream_flags
        buf.putV(0); // codec_specific_data len
        buf.putV(self.width);
        buf.putV(self.height);
        buf.putV(1); // SAR num
        buf.putV(1); // SAR den
        buf.putV(0); // csp_type
        // Color info (version > 3):
        buf.putV(1); // color_range - 1 (JPEG/full = 2, write 2-1=1)
        buf.putV(0); // color_primaries
        buf.putV(0); // color_trc
        buf.putV(0); // color_space (0 = RGB)
    }

    fn buildAudioStreamPayload(self: *NutMuxer, buf: *PayloadBuf) void {
        buf.putV(1); // stream_id
        buf.putV(1); // class: audio
        buf.putV(4); // fourcc len
        buf.appendSlice(&std.mem.toBytes(AUDIO_TAG));
        buf.putV(1); // time_base_id
        buf.putV(14); // msb_pts_shift
        buf.putV(self.audio_rate); // max_pts_distance
        buf.putV(0); // decode_delay
        buf.putV(0); // stream_flags
        buf.putV(0); // codec_specific_data len
        buf.putV(self.audio_rate); // samplerate num
        buf.putV(1); // samplerate den
        buf.putV(self.audio_channels);
    }
};

// ── Frame code table ────────────────────────────────────────────────────

fn buildFrameCodeTable(buf: *PayloadBuf) void {
    // Code 0: invalid, 1 entry.
    buf.putV(FLAG_INVALID);
    buf.putV(0);

    // Code 1: fully-coded escape, 1 entry.
    buf.putV(FLAG_CODED);
    buf.putV(1); // fields=1
    buf.putS(0); // pts_delta=0

    // Codes 2..255: invalid, 253 entries (excludes 'N'=78 which the decoder
    // auto-marks invalid; ffmpeg validates count <= 256 - (i <= 'N') - i).
    buf.putV(FLAG_INVALID);
    buf.putV(6); // fields=6 (need count)
    buf.putS(0); // pts_delta
    buf.putV(1); // size_mul
    buf.putV(0); // stream_id
    buf.putV(0); // size_lsb
    buf.putV(0); // reserved
    buf.putV(253); // count
}

// ── Encoding primitives ─────────────────────────────────────────────────

fn codePts(last_pts: i64, pts: i64, msb_shift: u6) u64 {
    const mask: i64 = (@as(i64, 1) << msb_shift) - 1;
    const lsb = pts & mask;
    const delta = last_pts - @divTrunc(mask, 2);
    const full = @mod(lsb - delta, mask + 1) + delta;
    if (full == pts) return @intCast(lsb);
    return @intCast(pts + (@as(i64, 1) << msb_shift));
}

fn encodeV(val: u64, out: []u8) usize {
    const n = vLen(val);
    var v = val;
    var i = n;
    while (i > 0) {
        i -= 1;
        out[i] = @intCast(v & 0x7F | if (i < n - 1) @as(u64, 0x80) else 0);
        v >>= 7;
    }
    return n;
}

fn vLen(val: u64) usize {
    if (val == 0) return 1;
    return (@as(usize, 64) - @as(usize, @clz(val)) + 6) / 7;
}

fn gcd(a: u32, b: u32) u32 {
    var x = a;
    var y = b;
    while (y != 0) {
        const t = y;
        y = x % y;
        x = t;
    }
    return x;
}

fn toBE64(val: u64) [8]u8 {
    return std.mem.toBytes(std.mem.nativeToBig(u64, val));
}

fn toBE32(val: u32) [4]u8 {
    return std.mem.toBytes(std.mem.nativeToBig(u32, val));
}
