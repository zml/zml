const std = @import("std");

const config = @import("../core/config.zig");

pub const min_shot_ms: u32 = 1500;
pub const min_shot_floor_s: f32 = 1.2;
pub const seconds_per_shot: f32 = 3.4;
pub const max_shots: u32 = 4;
pub const max_ref_files = config.max_ref_files;
pub const max_ref_images = config.max_ref_images;
pub const max_ref_videos = config.max_ref_videos;
pub const max_ref_audios = config.max_ref_audios;

pub const instruction_openings = [_][]const u8{
    "For the target video, at 0.00 seconds",
    "How the reference pictures align with the target video",
};

pub fn framesFor(duration_s: f32) u32 {
    return config.alignFrameCount(config.frameCount(duration_s));
}

pub fn effectiveSeconds(duration_s: f32) f32 {
    return @as(f32, @floatFromInt(framesFor(duration_s))) / config.video_fps;
}

pub fn isOnGrid(duration_s: f32) bool {
    const n = config.frameCount(duration_s);
    const snapped = config.alignFrameCount(n);
    return snapped == n and @abs((@as(f32, @floatFromInt(n)) / config.video_fps) - duration_s) <= 0.0006;
}

pub fn sSs(duration_s: f32) f32 {
    const x = effectiveSeconds(duration_s) * 100.0;
    const n = @floor(x);
    return (if (x - n >= 0.5) n + 1.0 else n) / 100.0;
}

pub fn msToTimestamp(ms: u32) [12]u8 {
    var buf: [12]u8 = undefined;
    const total_s = ms / 1000;
    const milli = ms % 1000;
    const minutes = total_s / 60;
    const seconds = total_s % 60;
    _ = std.fmt.bufPrint(&buf, "{d:0>2}:{d:0>2}.{d:0>3}", .{ minutes, seconds, milli }) catch unreachable;
    return buf;
}

pub fn parseTimestamp(text: []const u8) ?f32 {
    if (text.len < 9) return null;
    const mm = std.fmt.parseInt(u32, text[0..2], 10) catch return null;
    if (text[2] != ':') return null;
    const ss = std.fmt.parseInt(u32, text[3..5], 10) catch return null;
    if (text[5] != '.') return null;
    const mmm = std.fmt.parseInt(u32, text[6..9], 10) catch return null;
    return @as(f32, @floatFromInt(mm * 60 + ss)) + @as(f32, @floatFromInt(mmm)) / 1000.0;
}

pub fn shotCount(duration_s: f32, variant: config.Variant, pinned: ?u32) u32 {
    if (pinned) |n| return @max(1, n);
    if (variant == .fl2va) return 1;
    const seconds = effectiveSeconds(duration_s);
    var n: u32 = @intFromFloat(@round(seconds / seconds_per_shot));
    n = @max(1, @min(max_shots, n));
    while (n > 1 and (seconds * 1000.0) / @as(f32, @floatFromInt(n)) < @as(f32, @floatFromInt(min_shot_ms))) {
        n -= 1;
    }
    return n;
}

pub fn cutBounds(duration_s: f32, n_shots: u32) [5]u32 {
    var bounds: [5]u32 = .{0} ** 5;
    const total_ms: u32 = @intFromFloat(@round(effectiveSeconds(duration_s) * 1000.0));
    bounds[0] = 0;
    var i: u32 = 1;
    while (i < n_shots) : (i += 1) {
        const raw = @as(f32, @floatFromInt(total_ms)) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_shots));
        const rounded: u32 = @intFromFloat(@round(raw / 100.0) * 100.0);
        bounds[i] = @max(bounds[i - 1] + min_shot_ms, rounded);
    }
    bounds[n_shots] = total_ms;
    var j: u32 = n_shots - 1;
    while (j > 0) : (j -= 1) {
        if (bounds[j] + min_shot_ms > bounds[j + 1]) {
            bounds[j] = bounds[j + 1] - min_shot_ms;
        }
        if (j == 1) break;
    }
    return bounds;
}

pub fn instructionLine(allocator: std.mem.Allocator, variant: config.Variant, last_shot: u32, duration_s: f32, n_pictures: u32, last_only: bool) ![]u8 {
    const mark = sSs(duration_s);
    return switch (variant) {
        .t2va => allocator.dupe(u8, ""),
        .ref2va => allocator.dupe(u8, ""),
        .fl2va => if (n_pictures == 1 and last_only)
            std.fmt.allocPrint(allocator, "How the reference pictures align with the target video — Picture 1 (from Shot {d}) aligns with the {d:.2}-second mark of the target video.", .{ last_shot, mark })
        else if (n_pictures == 1)
            allocator.dupe(u8, "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video.")
        else
            std.fmt.allocPrint(allocator, "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {d}) aligns with the {d:.2}-second mark of the target video.", .{ last_shot, mark }),
    };
}

pub fn startsWithInstruction(line: []const u8) bool {
    for (instruction_openings) |open| {
        if (std.mem.startsWith(u8, line, open)) return true;
    }
    return false;
}
