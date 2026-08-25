const std = @import("std");

const packing = @import("packing.zig");
const vae = @import("vae.zig");

const log = std.log.scoped(.minimax_h3_media);

pub const RgbImage = struct { w: u32, h: u32, rgb: []u8 };
pub const Size = struct { w: u32, h: u32 };

pub fn writePpm(
    io: std.Io,
    dir: std.Io.Dir,
    name: []const u8,
    width: u32,
    height: u32,
    rgb: []const u8,
) !void {
    std.debug.assert(rgb.len == @as(usize, width) * height * 3);
    const file = try dir.createFile(io, name, .{});
    defer file.close(io);
    var writer = file.writer(io, &.{});
    try writer.interface.print("P6\n{d} {d}\n255\n", .{ width, height });
    try writer.interface.writeAll(rgb);
}

pub fn writeWavS16(
    io: std.Io,
    dir: std.Io.Dir,
    name: []const u8,
    sample_rate: u32,
    channels: u16,
    pcm: []const i16,
) !void {
    const file = try dir.createFile(io, name, .{});
    defer file.close(io);
    var writer = file.writer(io, &.{});
    const data_bytes: u32 = @intCast(pcm.len * 2);
    const byte_rate = sample_rate * channels * 2;
    try writer.interface.writeAll("RIFF");
    try writer.interface.writeInt(u32, 36 + data_bytes, .little);
    try writer.interface.writeAll("WAVEfmt ");
    try writer.interface.writeInt(u32, 16, .little);
    try writer.interface.writeInt(u16, 1, .little);
    try writer.interface.writeInt(u16, channels, .little);
    try writer.interface.writeInt(u32, sample_rate, .little);
    try writer.interface.writeInt(u32, byte_rate, .little);
    try writer.interface.writeInt(u16, channels * 2, .little);
    try writer.interface.writeInt(u16, 16, .little);
    try writer.interface.writeAll("data");
    try writer.interface.writeInt(u32, data_bytes, .little);
    try writer.interface.writeAll(std.mem.sliceAsBytes(pcm));
}

pub fn rgbU8FromNchw(allocator: std.mem.Allocator, nchw: []const f32, frames: u32, height: u32, width: u32) ![]u8 {
    const plane = @as(usize, frames) * height * width;
    std.debug.assert(nchw.len >= plane * 3);
    const out = try allocator.alloc(u8, plane * 3);
    var i: usize = 0;
    while (i < plane) : (i += 1) {
        const r = std.math.clamp(nchw[i], 0, 1);
        const g = std.math.clamp(nchw[plane + i], 0, 1);
        const b = std.math.clamp(nchw[2 * plane + i], 0, 1);
        out[i * 3 + 0] = @intFromFloat(@round(r * 255.0));
        out[i * 3 + 1] = @intFromFloat(@round(g * 255.0));
        out[i * 3 + 2] = @intFromFloat(@round(b * 255.0));
    }
    return out;
}

pub fn writeFrameSequence(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    nchw: []const f32,
    frames: u32,
    height: u32,
    width: u32,
) !void {
    const rgb = try rgbU8FromNchw(allocator, nchw, frames, height, width);
    defer allocator.free(rgb);
    const stride = @as(usize, width) * height * 3;
    var f: u32 = 0;
    while (f < frames) : (f += 1) {
        var name_buf: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "frame_{d:0>4}.ppm", .{f});
        try writePpm(io, dir, name, width, height, rgb[f * stride ..][0..stride]);
    }
}

pub fn f32ToS16(allocator: std.mem.Allocator, samples: []const f32) ![]i16 {
    const out = try allocator.alloc(i16, samples.len);
    for (samples, out) |s, *d| {
        const v = std.math.clamp(s, -1.0, 1.0);
        d.* = @intFromFloat(@round(v * 32767.0));
    }
    return out;
}

pub fn interleaveStereo(allocator: std.mem.Allocator, left: []const f32, right: []const f32) ![]f32 {
    std.debug.assert(left.len == right.len);
    const out = try allocator.alloc(f32, left.len * 2);
    for (left, right, 0..) |l, r, i| {
        out[i * 2] = l;
        out[i * 2 + 1] = r;
    }
    return out;
}

const ffmpeg_bin = "ffmpeg";

var tmp_seq: u32 = 0;

fn tmpId() u64 {
    tmp_seq += 1;
    return (@as(u64, @intFromPtr(&tmp_seq)) << 16) ^ tmp_seq;
}

fn envDir(name: [:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    const path = std.mem.span(raw);
    return if (path.len == 0) null else path;
}

fn tempRoot() []const u8 {
    return envDir("TMPDIR") orelse envDir("TEMP") orelse envDir("TMP") orelse ".";
}

const Scratch = struct {
    path: []u8,

    fn init(allocator: std.mem.Allocator) !Scratch {
        // Host IO: VFS Dir.deleteTree panics on path-only scratch dirs.
        var threaded: std.Io.Threaded = .init_single_threaded;
        const io = threaded.io();
        const name = try std.fmt.allocPrint(allocator, "h3_{x}", .{tmpId()});
        defer allocator.free(name);
        const path = try std.fs.path.join(allocator, &.{ tempRoot(), name });
        errdefer allocator.free(path);
        try std.Io.Dir.cwd().createDirPath(io, path);
        return .{ .path = path };
    }

    fn join(self: Scratch, allocator: std.mem.Allocator, file: []const u8) ![]u8 {
        return std.fs.path.join(allocator, &.{ self.path, file });
    }

    fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
        var threaded: std.Io.Threaded = .init_single_threaded;
        const io = threaded.io();
        if (std.fs.path.isAbsolute(self.path)) {
            if (std.fs.path.dirname(self.path)) |parent_path| {
                var parent = std.Io.Dir.openDirAbsolute(io, parent_path, .{}) catch {
                    allocator.free(self.path);
                    self.* = undefined;
                    return;
                };
                defer parent.close(io);
                parent.deleteTree(io, std.fs.path.basename(self.path)) catch {};
            }
        } else {
            std.Io.Dir.cwd().deleteTree(io, self.path) catch {};
        }
        allocator.free(self.path);
        self.* = undefined;
    }
};

fn runFfmpeg(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !std.process.RunResult {
    return std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(16 * 1024),
    });
}

pub fn muxMp4(allocator: std.mem.Allocator, io: std.Io, out_path: []const u8, frames: u32) !bool {
    _ = frames;
    const frame_in = try std.fs.path.join(allocator, &.{ out_path, "frame_%04d.ppm" });
    defer allocator.free(frame_in);
    const audio_in = try std.fs.path.join(allocator, &.{ out_path, "audio.wav" });
    defer allocator.free(audio_in);
    const mp4 = try std.fs.path.join(allocator, &.{ out_path, "output.mp4" });
    defer allocator.free(mp4);
    const result = runFfmpeg(allocator, io, &.{
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        "24",
        "-i",
        frame_in,
        "-i",
        audio_in,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        mp4,
    }) catch |err| {
        log.warn("ffmpeg {s}: {s}", .{ ffmpeg_bin, @errorName(err) });
        return false;
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code == 0) {
            log.info("muxed output.mp4 with {s}", .{ffmpeg_bin});
            return true;
        } else {
            log.warn("{s} exited {d}: {s}", .{ ffmpeg_bin, code, result.stderr });
        },
        else => log.warn("{s} did not exit cleanly", .{ffmpeg_bin}),
    }
    return false;
}

pub fn readPpmRgb(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !RgbImage {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    const header = try parsePpmHeader(bytes);
    const need = @as(usize, header.w) * header.h * 3;
    if (bytes.len < header.data_off + need) return error.BadPpm;
    return .{ .w = header.w, .h = header.h, .rgb = try allocator.dupe(u8, bytes[header.data_off..][0..need]) };
}

pub fn resizeRgb(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
    const out = try allocator.alloc(u8, @as(usize, dst_w) * dst_h * 3);
    var y: u32 = 0;
    while (y < dst_h) : (y += 1) {
        const sy = @min(src_h - 1, @as(u32, @intCast((@as(u64, y) * src_h) / dst_h)));
        var x: u32 = 0;
        while (x < dst_w) : (x += 1) {
            const sx = @min(src_w - 1, @as(u32, @intCast((@as(u64, x) * src_w) / dst_w)));
            const si = (@as(usize, sy) * src_w + sx) * 3;
            const di = (@as(usize, y) * dst_w + x) * 3;
            @memcpy(out[di..][0..3], src[si..][0..3]);
        }
    }
    return out;
}

pub fn ppmSize(io: std.Io, path: []const u8) !Size {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var buf: [256]u8 = undefined;
    var reader = file.reader(io, &.{});
    const n = try reader.interface.readSliceShort(&buf);
    const img = try parsePpmHeader(buf[0..n]);
    return .{ .w = img.w, .h = img.h };
}

fn parsePpmHeader(bytes: []const u8) !struct { w: u32, h: u32, data_off: usize } {
    var rest = bytes;
    const magic_end = std.mem.indexOfScalar(u8, rest, '\n') orelse return error.BadPpm;
    if (!std.mem.eql(u8, std.mem.trim(u8, rest[0..magic_end], " \r"), "P6")) return error.UnsupportedImage;
    rest = rest[magic_end + 1 ..];
    var w: usize = 0;
    var h: usize = 0;
    var maxv: usize = 0;
    while (maxv == 0) {
        const line_end = std.mem.indexOfScalar(u8, rest, '\n') orelse return error.BadPpm;
        const line = std.mem.trim(u8, rest[0..line_end], " \r");
        rest = rest[line_end + 1 ..];
        if (line.len == 0 or line[0] == '#') continue;
        var it = std.mem.tokenizeScalar(u8, line, ' ');
        if (w == 0) w = try std.fmt.parseInt(usize, it.next() orelse return error.BadPpm, 10);
        if (h == 0) h = try std.fmt.parseInt(usize, it.next() orelse return error.BadPpm, 10);
        if (it.next()) |mv| maxv = try std.fmt.parseInt(usize, mv, 10);
    }
    if (maxv != 255) return error.UnsupportedPpmDepth;
    return .{ .w = @intCast(w), .h = @intCast(h), .data_off = bytes.len - rest.len };
}

pub fn imageSize(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Size {
    if (ppmSize(io, path)) |s| return s else |_| {}
    const img = try loadRgbRaw(allocator, io, path);
    defer allocator.free(img.rgb);
    return .{ .w = img.w, .h = img.h };
}

pub fn wavSampleCount(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !u32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024));
    defer allocator.free(bytes);
    const info = try parseWavHeader(bytes);
    return info.samples;
}

pub fn loadRgbRaw(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !RgbImage {
    if (readPpmRgb(allocator, io, path)) |img| return img else |_| {}
    var scratch = try Scratch.init(allocator);
    defer scratch.deinit(allocator);
    const tmp_name = try scratch.join(allocator, "in.ppm");
    defer allocator.free(tmp_name);
    const result = runFfmpeg(allocator, io, &.{
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-frames:v", "1", tmp_name,
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.ImageLoadFailed,
        else => return error.ImageLoadFailed,
    }
    return readPpmRgb(allocator, io, tmp_name);
}

pub fn loadRgb(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    dst_w: u32,
    dst_h: u32,
) ![]u8 {
    if (readPpmRgb(allocator, io, path)) |img| {
        defer allocator.free(img.rgb);
        if (img.w == dst_w and img.h == dst_h) return allocator.dupe(u8, img.rgb);
        return resizeRgb(allocator, img.rgb, img.w, img.h, dst_w, dst_h);
    } else |_| {}

    var scratch = try Scratch.init(allocator);
    defer scratch.deinit(allocator);
    const tmp_name = try scratch.join(allocator, "in.ppm");
    defer allocator.free(tmp_name);
    const scale = try std.fmt.allocPrint(allocator, "scale={d}:{d}", .{ dst_w, dst_h });
    defer allocator.free(scale);
    const result = runFfmpeg(allocator, io, &.{
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-vf", scale, "-frames:v", "1", tmp_name,
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.ImageLoadFailed,
        else => return error.ImageLoadFailed,
    }
    const img = try readPpmRgb(allocator, io, tmp_name);
    defer allocator.free(img.rgb);
    if (img.w == dst_w and img.h == dst_h) return allocator.dupe(u8, img.rgb);
    return resizeRgb(allocator, img.rgb, img.w, img.h, dst_w, dst_h);
}

pub fn loadVideoRgb(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    dst_w: u32,
    dst_h: u32,
    frames: u32,
) !struct { rgb: []u8, frames: u32, w: u32, h: u32 } {
    var scratch = try Scratch.init(allocator);
    defer scratch.deinit(allocator);
    const tmp_pat = try scratch.join(allocator, "f_%04d.ppm");
    defer allocator.free(tmp_pat);
    const fps = try std.fmt.allocPrint(allocator, "fps=24,scale={d}:{d}", .{ dst_w, dst_h });
    defer allocator.free(fps);
    const nbuf = try std.fmt.allocPrint(allocator, "{d}", .{frames});
    defer allocator.free(nbuf);
    const result = runFfmpeg(allocator, io, &.{
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-vf", fps, "-frames:v", nbuf, tmp_pat,
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.VideoLoadFailed,
        else => return error.VideoLoadFailed,
    }

    const plane = @as(usize, dst_w) * dst_h * 3;
    const out = try allocator.alloc(u8, frames * plane);
    errdefer allocator.free(out);
    var loaded: u32 = 0;
    while (loaded < frames) : (loaded += 1) {
        var name_buf: [16]u8 = undefined;
        const frame_name = try std.fmt.bufPrint(&name_buf, "f_{d:0>4}.ppm", .{loaded + 1});
        const name = try scratch.join(allocator, frame_name);
        defer allocator.free(name);
        const img = readPpmRgb(allocator, io, name) catch break;
        defer allocator.free(img.rgb);
        const rgb = if (img.w == dst_w and img.h == dst_h)
            img.rgb
        else
            try resizeRgb(allocator, img.rgb, img.w, img.h, dst_w, dst_h);
        defer if (rgb.ptr != img.rgb.ptr) allocator.free(rgb);
        @memcpy(out[loaded * plane ..][0..plane], rgb);
    }
    if (loaded == 0) return error.VideoLoadFailed;
    return .{ .rgb = out, .frames = loaded, .w = dst_w, .h = dst_h };
}

pub fn rgbVideoToNchwImagenet(allocator: std.mem.Allocator, rgb: []const u8, frames: u32, height: u32, width: u32) ![]f32 {
    const plane = @as(usize, height) * width;
    const out = try allocator.alloc(f32, 3 * frames * plane);
    var f: u32 = 0;
    while (f < frames) : (f += 1) {
        var i: usize = 0;
        while (i < plane) : (i += 1) {
            inline for (0..3) |c| {
                const v = @as(f32, @floatFromInt(rgb[(f * plane + i) * 3 + c])) / 255.0;
                out[(c * frames + f) * plane + i] = (v - vae.imagenet_mean[c]) / vae.imagenet_std[c];
            }
        }
    }
    return out;
}

pub fn loadWavStereo(allocator: std.mem.Allocator, io: std.Io, path: []const u8, sample_rate: u32) ![]f32 {
    if (readWavStereo(allocator, io, path, sample_rate)) |pcm| return pcm else |_| {}
    var scratch = try Scratch.init(allocator);
    defer scratch.deinit(allocator);
    const tmp = try scratch.join(allocator, "in.wav");
    defer allocator.free(tmp);
    const rate = try std.fmt.allocPrint(allocator, "{d}", .{sample_rate});
    defer allocator.free(rate);
    const result = runFfmpeg(allocator, io, &.{
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "2", "-ar", rate, tmp,
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.AudioLoadFailed,
        else => return error.AudioLoadFailed,
    }
    return readWavStereo(allocator, io, tmp, sample_rate);
}

const WavInfo = struct { samples: u32, ch: u16, rate: u32, bits: u16, data_off: usize };

fn parseWavHeader(bytes: []const u8) !WavInfo {
    if (bytes.len < 44) return error.BadWav;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.BadWav;
    var off: usize = 12;
    var data_off: usize = 0;
    var data_len: usize = 0;
    var ch: u16 = 0;
    var rate: u32 = 0;
    var bits: u16 = 0;
    while (off + 8 <= bytes.len) {
        const id = bytes[off..][0..4];
        const n = std.mem.readInt(u32, bytes[off + 4 ..][0..4], .little);
        off += 8;
        if (std.mem.eql(u8, id, "fmt ")) {
            if (n < 16 or off + 16 > bytes.len) return error.BadWav;
            ch = std.mem.readInt(u16, bytes[off + 2 ..][0..2], .little);
            rate = std.mem.readInt(u32, bytes[off + 4 ..][0..4], .little);
            bits = std.mem.readInt(u16, bytes[off + 14 ..][0..2], .little);
        } else if (std.mem.eql(u8, id, "data")) {
            data_off = off;
            data_len = n;
            break;
        }
        off += n;
    }
    if (data_off == 0 or ch == 0 or bits == 0) return error.BadWav;
    return .{
        .samples = @intCast(data_len / (ch * (bits / 8))),
        .ch = ch,
        .rate = rate,
        .bits = bits,
        .data_off = data_off,
    };
}

fn readWavStereo(allocator: std.mem.Allocator, io: std.Io, path: []const u8, sample_rate: u32) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    const info = try parseWavHeader(bytes);
    if (info.rate != sample_rate) return error.WavRateMismatch;
    const samples = info.samples;
    const ch = info.ch;
    const bits = info.bits;
    const data_off = info.data_off;
    const out = try allocator.alloc(f32, samples * 2);
    var i: usize = 0;
    while (i < samples) : (i += 1) {
        var c: u16 = 0;
        while (c < 2) : (c += 1) {
            const src_c = if (c < ch) c else 0;
            const idx = data_off + (i * ch + src_c) * (bits / 8);
            const s: f32 = switch (bits) {
                16 => @as(f32, @floatFromInt(std.mem.readInt(i16, bytes[idx..][0..2], .little))) / 32768.0,
                32 => std.mem.bytesAsValue(f32, bytes[idx..][0..4]).*,
                else => return error.UnsupportedWav,
            };
            out[i * 2 + c] = s;
        }
    }
    return out;
}

pub fn refsContainAudio(refs: []const u8) bool {
    var it = std.mem.splitScalar(u8, refs, ',');
    while (it.next()) |part| {
        const path = std.mem.trim(u8, part, " \t");
        if (path.len != 0 and guessKind(path) == .audio) return true;
    }
    return false;
}

pub fn guessKind(path: []const u8) packing.ReferenceKind {
    const ext = std.fs.path.extension(path);
    if (std.ascii.eqlIgnoreCase(ext, ".wav") or std.ascii.eqlIgnoreCase(ext, ".mp3") or std.ascii.eqlIgnoreCase(ext, ".flac") or std.ascii.eqlIgnoreCase(ext, ".m4a"))
        return .audio;
    if (std.ascii.eqlIgnoreCase(ext, ".mp4") or std.ascii.eqlIgnoreCase(ext, ".mov") or std.ascii.eqlIgnoreCase(ext, ".mkv") or std.ascii.eqlIgnoreCase(ext, ".webm"))
        return .video;
    return .image;
}

pub fn rgbToNchwImagenet(allocator: std.mem.Allocator, rgb: []const u8, height: u32, width: u32) ![]f32 {
    const plane = @as(usize, height) * width;
    const out = try allocator.alloc(f32, plane * 3);
    var i: usize = 0;
    while (i < plane) : (i += 1) {
        inline for (0..3) |c| {
            const v = @as(f32, @floatFromInt(rgb[i * 3 + c])) / 255.0;
            out[c * plane + i] = (v - vae.imagenet_mean[c]) / vae.imagenet_std[c];
        }
    }
    return out;
}
