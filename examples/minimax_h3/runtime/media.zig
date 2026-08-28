const builtin = @import("builtin");
const std = @import("std");

const geom = @import("../conditioning/geometry.zig");
const vae = @import("../vae/geometry.zig");

const log = std.log.scoped(.minimax_h3_media);

pub const RgbImage = struct { w: u32, h: u32, rgb: []u8 };

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

pub const Output = struct {
    dir: []const u8,
    mp4_name: []const u8,

    pub fn parse(path: []const u8) Output {
        if (path.len == 0) return .{ .dir = "output", .mp4_name = "output.mp4" };
        if (path.len >= 4 and std.ascii.eqlIgnoreCase(path[path.len - 4 ..], ".mp4")) {
            return .{
                .dir = std.fs.path.dirname(path) orelse ".",
                .mp4_name = std.fs.path.basename(path),
            };
        }
        return .{ .dir = path, .mp4_name = "output.mp4" };
    }

    pub fn isCwd(self: Output) bool {
        return self.dir.len == 0 or std.mem.eql(u8, self.dir, ".");
    }
};

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
    if (envDir("TMPDIR")) |p| return p;
    if (envDir("TEMP")) |p| return p;
    if (envDir("TMP")) |p| return p;
    return switch (builtin.os.tag) {
        .windows => "C:\\Windows\\Temp",
        else => "/tmp",
    };
}

pub const Scratch = struct {
    path: []u8,

    pub fn init(allocator: std.mem.Allocator) !Scratch {
        var threaded: std.Io.Threaded = .init_single_threaded;
        const io = threaded.io();
        const name = try std.fmt.allocPrint(allocator, "h3_{x}", .{tmpId()});
        defer allocator.free(name);
        const path = try std.fs.path.join(allocator, &.{ tempRoot(), name });
        errdefer allocator.free(path);
        try std.Io.Dir.cwd().createDirPath(io, path);
        return .{ .path = path };
    }

    pub fn join(self: Scratch, allocator: std.mem.Allocator, file: []const u8) ![]u8 {
        return std.fs.path.join(allocator, &.{ self.path, file });
    }

    pub fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
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

pub fn muxMp4(
    allocator: std.mem.Allocator,
    io: std.Io,
    frames_dir: []const u8,
    audio_path: []const u8,
    mp4_path: []const u8,
) !bool {
    const frame_in = try std.fs.path.join(allocator, &.{ frames_dir, "frame_%04d.ppm" });
    defer allocator.free(frame_in);
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
        audio_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        mp4_path,
    }) catch |err| {
        log.warn("ffmpeg {s}: {s}", .{ ffmpeg_bin, @errorName(err) });
        return false;
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code == 0) {
            log.info("muxed {s} with {s}", .{ mp4_path, ffmpeg_bin });
            return true;
        } else {
            log.warn("{s} exited {d}: {s}", .{ ffmpeg_bin, code, result.stderr });
        },
        else => log.warn("{s} did not exit cleanly", .{ffmpeg_bin}),
    }
    return false;
}

pub fn openPath(io: std.Io, path: []const u8) !std.Io.Dir {
    if (std.fs.path.isAbsolute(path)) return std.Io.Dir.openDirAbsolute(io, path, .{});
    return std.Io.Dir.cwd().openDir(io, path, .{});
}

pub fn writeGeneratedVideo(
    allocator: std.mem.Allocator,
    io: std.Io,
    dest_dir: std.Io.Dir,
    dest_path: []const u8,
    mp4_name: []const u8,
    nchw: []const f32,
    frames: u32,
    height: u32,
    width: u32,
    pcm: []const i16,
    sample_rate: u32,
) !bool {
    var scratch = try Scratch.init(allocator);
    defer scratch.deinit(allocator);
    var scratch_dir = try openPath(io, scratch.path);
    defer scratch_dir.close(io);
    try writeFrameSequence(allocator, io, scratch_dir, nchw, frames, height, width);
    try writeWavS16(io, scratch_dir, "audio.wav", sample_rate, 2, pcm);
    const audio_in = try scratch.join(allocator, "audio.wav");
    defer allocator.free(audio_in);
    const mp4 = try std.fs.path.join(allocator, &.{ dest_path, mp4_name });
    defer allocator.free(mp4);
    if (try muxMp4(allocator, io, scratch.path, audio_in, mp4)) return true;

    const frames_path = try std.fs.path.join(allocator, &.{ dest_path, "frames" });
    defer allocator.free(frames_path);
    try std.Io.Dir.cwd().createDirPath(io, frames_path);
    var fallback = try openPath(io, frames_path);
    defer fallback.close(io);
    try writeFrameSequence(allocator, io, fallback, nchw, frames, height, width);
    try writeWavS16(io, dest_dir, "audio.wav", sample_rate, 2, pcm);
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
    const raw = try loadRgbRaw(allocator, io, path);
    defer allocator.free(raw.rgb);
    if (raw.w == dst_w and raw.h == dst_h) return allocator.dupe(u8, raw.rgb);
    return geom.resizeLanczos(allocator, raw.rgb, raw.w, raw.h, dst_w, dst_h);
}

pub fn loadRgbCover(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    dst_w: u32,
    dst_h: u32,
) ![]u8 {
    const raw = try loadRgbRaw(allocator, io, path);
    defer allocator.free(raw.rgb);
    return geom.coverCropLanczos(allocator, raw.rgb, raw.w, raw.h, dst_w, dst_h);
}

pub const VideoClip = struct {
    rgb: []u8,
    frames: u32,
    w: u32,
    h: u32,
    fps: f32,
    has_audio: bool = false,
};

pub const VideoMeta = struct { w: u32, h: u32, fps: f32, has_audio: bool };

pub fn parseFfmpegProbe(text: []const u8) !VideoMeta {
    const video = std.mem.indexOf(u8, text, "Video:") orelse return error.VideoLoadFailed;
    const rest = text[video..];
    const size = findWxH(rest) orelse return error.VideoLoadFailed;
    const fps = findFps(rest) orelse return error.VideoLoadFailed;
    return .{
        .w = size.w,
        .h = size.h,
        .fps = fps,
        .has_audio = std.mem.indexOf(u8, text, "Audio:") != null,
    };
}

fn findWxH(text: []const u8) ?struct { w: u32, h: u32 } {
    var i: usize = 0;
    while (i + 3 < text.len) : (i += 1) {
        if (!std.ascii.isDigit(text[i])) continue;
        const x = std.mem.indexOfScalarPos(u8, text, i + 1, 'x') orelse return null;
        if (x == i) continue;
        var end = x + 1;
        while (end < text.len and std.ascii.isDigit(text[end])) end += 1;
        if (end == x + 1) continue;
        const w = std.fmt.parseInt(u32, text[i..x], 10) catch continue;
        const h = std.fmt.parseInt(u32, text[x + 1 .. end], 10) catch continue;
        if (w > 0 and h > 0) return .{ .w = w, .h = h };
    }
    return null;
}

fn findFps(text: []const u8) ?f32 {
    const needle = " fps";
    if (std.mem.indexOf(u8, text, needle)) |at| {
        var start = at;
        while (start > 0 and (std.ascii.isDigit(text[start - 1]) or text[start - 1] == '.')) start -= 1;
        if (start < at) {
            if (std.fmt.parseFloat(f32, text[start..at])) |v| {
                if (v > 0) return v;
            } else |_| {}
        }
    }
    if (std.mem.indexOf(u8, text, " tbr")) |at| {
        var start = at;
        while (start > 0 and (std.ascii.isDigit(text[start - 1]) or text[start - 1] == '.')) start -= 1;
        if (start < at) {
            if (std.fmt.parseFloat(f32, text[start..at])) |v| {
                if (v > 0) return v;
            } else |_| {}
        }
    }
    return null;
}

pub fn probeSize(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !struct { w: u32, h: u32 } {
    const result = runFfmpeg(allocator, io, &.{
        ffmpeg_bin, "-hide_banner", "-i", path,
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    const size = findWxH(result.stderr) orelse return error.ImageLoadFailed;
    return .{ .w = size.w, .h = size.h };
}

pub fn probeVideo(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !VideoMeta {
    const result = runFfmpeg(allocator, io, &.{
        ffmpeg_bin, "-hide_banner", "-i", path,
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    return parseFfmpegProbe(result.stderr);
}

pub fn loadVideoNative(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !VideoClip {
    const meta = try probeVideo(allocator, io, path);
    var scratch = try Scratch.init(allocator);
    defer scratch.deinit(allocator);
    const tmp_pat = try scratch.join(allocator, "f_%04d.ppm");
    defer allocator.free(tmp_pat);
    const result = runFfmpeg(allocator, io, &.{
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-vsync", "0", tmp_pat,
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.VideoLoadFailed,
        else => return error.VideoLoadFailed,
    }

    var frames_list: std.ArrayList([]u8) = .empty;
    defer {
        for (frames_list.items) |f| allocator.free(f);
        frames_list.deinit(allocator);
    }
    var loaded: u32 = 0;
    var fw: u32 = meta.w;
    var fh: u32 = meta.h;
    while (loaded < 4096) : (loaded += 1) {
        var name_buf: [16]u8 = undefined;
        const frame_name = try std.fmt.bufPrint(&name_buf, "f_{d:0>4}.ppm", .{loaded + 1});
        const name = try scratch.join(allocator, frame_name);
        defer allocator.free(name);
        const img = readPpmRgb(allocator, io, name) catch break;
        fw = img.w;
        fh = img.h;
        try frames_list.append(allocator, img.rgb);
    }
    if (frames_list.items.len == 0) return error.VideoLoadFailed;
    const plane = @as(usize, fw) * fh * 3;
    const out = try allocator.alloc(u8, frames_list.items.len * plane);
    for (frames_list.items, 0..) |frame, i| {
        if (frame.len != plane) {
            const resized = try geom.resizeLanczos(allocator, frame, fw, fh, fw, fh);
            defer allocator.free(resized);
            @memcpy(out[i * plane ..][0..plane], resized[0..plane]);
        } else {
            @memcpy(out[i * plane ..][0..plane], frame);
        }
    }
    return .{
        .rgb = out,
        .frames = @intCast(frames_list.items.len),
        .w = fw,
        .h = fh,
        .fps = meta.fps,
        .has_audio = meta.has_audio,
    };
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

pub const Pcm = struct { stereo: []f32, rate: u32 };

pub fn loadWavNative(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Pcm {
    if (readWavAny(allocator, io, path)) |pcm| return pcm else |_| {}
    var scratch = try Scratch.init(allocator);
    defer scratch.deinit(allocator);
    const tmp = try scratch.join(allocator, "native.wav");
    defer allocator.free(tmp);
    const result = runFfmpeg(allocator, io, &.{
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "2", tmp,
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.AudioLoadFailed,
        else => return error.AudioLoadFailed,
    }
    return readWavAny(allocator, io, tmp);
}

fn readWavAny(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Pcm {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    const info = try parseWavHeader(bytes);
    const stereo = try decodeWavStereo(allocator, bytes, info);
    return .{ .stereo = stereo, .rate = info.rate };
}

pub fn loadAudioOfficial(allocator: std.mem.Allocator, io: std.Io, path: []const u8, duration_s: f32, dst_rate: u32) ![]f32 {
    const native = try loadWavNative(allocator, io, path);
    defer allocator.free(native.stereo);
    const max_pcm: u32 = @intFromFloat(@round(duration_s * @as(f32, @floatFromInt(native.rate))));
    const truncated = try geom.truncateStereo(allocator, native.stereo, max_pcm);
    defer allocator.free(truncated);
    return geom.resampleLinear(allocator, truncated, native.rate, dst_rate);
}

const WavInfo = struct { samples: u32, ch: u16, rate: u32, bits: u16, data_off: usize };

pub fn parseWavHeader(bytes: []const u8) !WavInfo {
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

fn decodeWavStereo(allocator: std.mem.Allocator, bytes: []const u8, info: WavInfo) ![]f32 {
    const out = try allocator.alloc(f32, @as(usize, info.samples) * 2);
    var i: usize = 0;
    while (i < info.samples) : (i += 1) {
        var c: u16 = 0;
        while (c < 2) : (c += 1) {
            const src_c = if (c < info.ch) c else 0;
            const idx = info.data_off + (i * info.ch + src_c) * (info.bits / 8);
            const s: f32 = switch (info.bits) {
                16 => @as(f32, @floatFromInt(std.mem.readInt(i16, bytes[idx..][0..2], .little))) / 32768.0,
                32 => std.mem.bytesAsValue(f32, bytes[idx..][0..4]).*,
                else => return error.UnsupportedWav,
            };
            out[i * 2 + c] = s;
        }
    }
    return out;
}

pub fn rgbToNchwImagenet(allocator: std.mem.Allocator, rgb: []const u8, height: u32, width: u32) ![]f32 {
    return rgbVideoToNchwImagenet(allocator, rgb, 1, height, width);
}
