const builtin = @import("builtin");
const std = @import("std");

const log = std.log.scoped(.minimax_h3_media);

// =============================================================================
// serve/media.zig — WAV / PPM / ffmpeg mux
// =============================================================================

fn writePpm(
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

fn rgbU8FromNchw(allocator: std.mem.Allocator, nchw: []const f32, frames: u32, height: u32, width: u32) ![]u8 {
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

fn writeFrameSequence(
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

var ffmpeg_resolved: ?[]const u8 = null;
var ffmpeg_home_buf: [512]u8 = undefined;

fn ffmpegExists(io: std.Io, path: []const u8) bool {
    var f = std.Io.Dir.openFileAbsolute(io, path, .{ .mode = .read_only }) catch return false;
    f.close(io);
    return true;
}

fn ffmpegBin(io: std.Io) []const u8 {
    if (ffmpeg_resolved) |p| return p;
    if (std.c.getenv("FFMPEG")) |raw| {
        ffmpeg_resolved = std.mem.span(raw);
        return ffmpeg_resolved.?;
    }
    if (std.c.getenv("HOME")) |home| {
        const p = std.fmt.bufPrint(&ffmpeg_home_buf, "{s}/.local/bin/ffmpeg", .{std.mem.span(home)}) catch "ffmpeg";
        if (ffmpegExists(io, p)) {
            ffmpeg_resolved = p;
            return p;
        }
    }
    if (ffmpegExists(io, "/usr/bin/ffmpeg")) {
        ffmpeg_resolved = "/usr/bin/ffmpeg";
        return "/usr/bin/ffmpeg";
    }
    ffmpeg_resolved = "ffmpeg";
    return "ffmpeg";
}

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

fn muxMp4(
    allocator: std.mem.Allocator,
    io: std.Io,
    frames_dir: []const u8,
    audio_path: []const u8,
    mp4_path: []const u8,
) !bool {
    const frame_in = try std.fs.path.join(allocator, &.{ frames_dir, "frame_%04d.ppm" });
    defer allocator.free(frame_in);
    const result = runFfmpeg(allocator, io, &.{
        ffmpegBin(io),
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
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        mp4_path,
    }) catch |err| {
        log.warn("ffmpeg {s}: {s}", .{ ffmpegBin(io), @errorName(err) });
        return false;
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code == 0) {
            log.info("muxed {s} with {s}", .{ mp4_path, ffmpegBin(io) });
            return true;
        } else {
            log.warn("{s} exited {d}: {s}", .{ ffmpegBin(io), code, result.stderr });
        },
        else => log.warn("{s} did not exit cleanly", .{ffmpegBin(io)}),
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
    const need = @as(usize, frames) * @as(usize, sample_rate) / 24 * 2;
    var owned_pad: ?[]i16 = null;
    defer if (owned_pad) |p| allocator.free(p);
    const audio = if (pcm.len >= need) pcm else blk: {
        const pad = try allocator.alloc(i16, need);
        owned_pad = pad;
        @memset(pad, 0);
        if (pcm.len != 0) @memcpy(pad[0..pcm.len], pcm);
        break :blk pad;
    };
    try writeWavS16(io, scratch_dir, "audio.wav", sample_rate, 2, audio);
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
