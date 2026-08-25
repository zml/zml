const builtin = @import("builtin");
const std = @import("std");

const config = @import("../core/config.zig");
const geom = @import("../conditioning/geom.zig");
const packing = @import("../model/packing.zig");
const vae = @import("../vae/geom.zig");

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

pub fn resizeRgb(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
    return geom.resizeLanczos(allocator, src, src_w, src_h, dst_w, dst_h);
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

pub fn loadJpegThumb(allocator: std.mem.Allocator, io: std.Io, path: []const u8, max_side: u32) ![]u8 {
    var scratch = try Scratch.init(allocator);
    defer scratch.deinit(allocator);
    const out = try scratch.join(allocator, "thumb.jpg");
    defer allocator.free(out);
    const vf = try std.fmt.allocPrint(
        allocator,
        "scale={d}:{d}:force_original_aspect_ratio=decrease",
        .{ max_side, max_side },
    );
    defer allocator.free(vf);
    const result = runFfmpeg(allocator, io, &.{
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-q:v",
        "5",
        out,
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.ImageLoadFailed,
        else => return error.ImageLoadFailed,
    }
    return std.Io.Dir.cwd().readFileAlloc(io, out, allocator, .limited(2 * 1024 * 1024));
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
    const raw = try loadRgbRaw(allocator, io, path);
    defer allocator.free(raw.rgb);
    if (raw.w == dst_w and raw.h == dst_h) return allocator.dupe(u8, raw.rgb);
    return resizeRgb(allocator, raw.rgb, raw.w, raw.h, dst_w, dst_h);
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

pub fn probeVideo(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !struct { w: u32, h: u32, fps: f32, has_audio: bool } {
    const result = runFfmpeg(allocator, io, &.{
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "csv=p=0",
        path,
    }) catch return error.FfmpegMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.VideoLoadFailed,
        else => return error.VideoLoadFailed,
    }
    var it = std.mem.splitScalar(u8, std.mem.trim(u8, result.stdout, " \r\n"), ',');
    const w = try std.fmt.parseInt(u32, it.next() orelse return error.VideoLoadFailed, 10);
    const h = try std.fmt.parseInt(u32, it.next() orelse return error.VideoLoadFailed, 10);
    const rate = it.next() orelse return error.VideoLoadFailed;
    const fps = parseRate(rate) orelse return error.VideoLoadFailed;

    const audio = runFfmpeg(allocator, io, &.{
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path,
    }) catch return .{ .w = w, .h = h, .fps = fps, .has_audio = false };
    defer allocator.free(audio.stdout);
    defer allocator.free(audio.stderr);
    const has_audio = std.mem.indexOf(u8, audio.stdout, "audio") != null;
    return .{ .w = w, .h = h, .fps = fps, .has_audio = has_audio };
}

fn parseRate(text: []const u8) ?f32 {
    const trimmed = std.mem.trim(u8, text, " \r\n");
    if (std.mem.indexOfScalar(u8, trimmed, '/')) |slash| {
        const num = std.fmt.parseFloat(f32, trimmed[0..slash]) catch return null;
        const den = std.fmt.parseFloat(f32, trimmed[slash + 1 ..]) catch return null;
        if (den == 0) return null;
        return num / den;
    }
    return std.fmt.parseFloat(f32, trimmed) catch null;
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
            const resized = try resizeRgb(allocator, frame, fw, fh, fw, fh);
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

pub fn loadVideoRgb(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    dst_w: u32,
    dst_h: u32,
    frames: u32,
) !struct { rgb: []u8, frames: u32, w: u32, h: u32 } {
    const clip = try loadVideoNative(allocator, io, path);
    defer allocator.free(clip.rgb);
    const indices = try geom.resampleFrameIndices(clip.frames, clip.fps, config.video_fps, allocator);
    defer allocator.free(indices);
    const keep = @min(frames, @as(u32, @intCast(indices.len)));
    const plane = @as(usize, dst_w) * dst_h * 3;
    const src_plane = @as(usize, clip.w) * clip.h * 3;
    const out = try allocator.alloc(u8, keep * plane);
    errdefer allocator.free(out);
    var i: u32 = 0;
    while (i < keep) : (i += 1) {
        const src_i = indices[i];
        const src = clip.rgb[src_i * src_plane ..][0..src_plane];
        const rgb = if (clip.w == dst_w and clip.h == dst_h)
            try allocator.dupe(u8, src)
        else
            try resizeRgb(allocator, src, clip.w, clip.h, dst_w, dst_h);
        defer allocator.free(rgb);
        @memcpy(out[i * plane ..][0..plane], rgb);
    }
    return .{ .rgb = out, .frames = keep, .w = dst_w, .h = dst_h };
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

fn readWavStereo(allocator: std.mem.Allocator, io: std.Io, path: []const u8, sample_rate: u32) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    const info = try parseWavHeader(bytes);
    if (info.rate != sample_rate) return error.WavRateMismatch;
    return decodeWavStereo(allocator, bytes, info);
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
    return rgbVideoToNchwImagenet(allocator, rgb, 1, height, width);
}
