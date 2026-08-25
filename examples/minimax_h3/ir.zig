const std = @import("std");

const config = @import("config.zig");

const log = std.log.scoped(.minimax_h3_ir);

/// How the text that reaches the H3 encoder is produced.
pub const Mode = enum {
    off,
    prompt,
    h3ir,
    auto,

    pub fn parse(text: []const u8) ?Mode {
        return std.meta.stringToEnum(Mode, text);
    }
};

pub const Request = struct {
    prompt: []const u8,
    variant: config.Variant = .t2va,
    duration_s: f32 = 5.0,
    mode: Mode = .auto,
    llm_url: ?[]const u8 = null,
};

pub const Brief = struct {
    text: []u8,
    source: enum { raw, prompting_guidance, openh3_ir },

    pub fn deinit(self: Brief, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
    }
};

pub fn alreadyCompiled(prompt: []const u8) bool {
    return std.mem.indexOf(u8, prompt, "integrated_multimodal_description:") != null or
        std.mem.indexOf(u8, prompt, "detailed_description:") != null;
}

/// Official T2VA / I2VA / FL2VA field order from `skills/h3-prompt-writing/references/base-en.txt`.
pub fn promptingGuidance(allocator: std.mem.Allocator, req: Request) ![]u8 {
    if (alreadyCompiled(req.prompt)) return allocator.dupe(u8, req.prompt);

    const body = std.mem.trim(u8, req.prompt, " \t\r\n");
    const duration = @round(req.duration_s * 100.0) / 100.0;

    return switch (req.variant) {
        .t2va => std.fmt.allocPrint(allocator,
            \\integrated_multimodal_description: [Shot 1] Live-action, cinematic, {s}
            \\
            \\overall_soundscape: Ambient sound follows the scene described above for the full {d:.2}-second clip, including room tone and physical action.
            \\
            \\non_diegetic_music: N/A
            \\
        , .{ body, duration }),
        .fl2va => std.fmt.allocPrint(allocator,
            \\How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the {d:.2}-second mark of the target video.
            \\
            \\integrated_multimodal_description: [Shot 1] Live-action, cinematic, the opening matches Picture 1 and develops continuously until the composition of Picture 2 at {d:.2} seconds. {s}
            \\
            \\overall_soundscape: Ambient sound follows the scene described above for the full {d:.2}-second clip, including room tone and physical action.
            \\
            \\non_diegetic_music: N/A
            \\
        , .{ duration, duration, body, duration }),
        .ref2va => std.fmt.allocPrint(allocator,
            \\subject_definitions:
            \\<Picture 1> is the first supplied reference used for appearance and style of the target video.
            \\
            \\summary:
            \\[reference generation] The target video follows the request: {s}
            \\
            \\retention_analysis:
            \\<Picture 1> (appears in [Shot 1]): fully_preserved — appearance, clothing, and visual style stay consistent.
            \\
            \\detailed_description:
            \\[Shot 1] Live-action, cinematic, {s}
            \\
            \\overall_soundscape: Ambient sound follows the scene described above for the full {d:.2}-second clip, including room tone and physical action.
            \\
            \\non_diegetic_music: N/A
            \\
        , .{ body, body, duration }),
    };
}

pub fn compile(allocator: std.mem.Allocator, io: std.Io, req: Request) !Brief {
    const mode = switch (req.mode) {
        .auto => if (h3irAvailable(allocator, io, req.llm_url)) Mode.h3ir else Mode.prompt,
        else => req.mode,
    };
    if (req.mode == .auto and mode != .h3ir) {
        log.info("ir auto: OpenH3-IR unavailable, using prompting guidance", .{});
    }
    const brief: Brief = switch (mode) {
        .off => .{ .text = try allocator.dupe(u8, req.prompt), .source = .raw },
        .prompt => .{ .text = try promptingGuidance(allocator, req), .source = .prompting_guidance },
        .h3ir => .{ .text = try compileH3ir(allocator, io, req), .source = .openh3_ir },
        .auto => unreachable,
    };
    log.info("ir source={s} chars={d} variant={s}", .{ @tagName(brief.source), brief.text.len, @tagName(req.variant) });
    return brief;
}

fn h3irAvailable(allocator: std.mem.Allocator, io: std.Io, llm_url: ?[]const u8) bool {
    const url = llm_url orelse return false;
    if (url.len == 0) return false;
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "h3ir", "--help" },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(4096),
    }) catch return false;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    return switch (result.term) {
        .exited => |code| code == 0,
        else => false,
    };
}

fn compileH3ir(allocator: std.mem.Allocator, io: std.Io, req: Request) ![]u8 {
    var seconds_buf: [16]u8 = undefined;
    const seconds = try std.fmt.bufPrint(&seconds_buf, "{d:.3}", .{req.duration_s});
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "h3ir", "compile", req.prompt, "--seconds", seconds },
        .stdout_limit = .limited(64 * 1024),
        .stderr_limit = .limited(16 * 1024),
    }) catch return error.H3irMissing;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.H3irFailed,
        else => return error.H3irFailed,
    }
    const out = std.mem.trim(u8, result.stdout, " \t\r\n");
    if (out.len == 0) return error.H3irEmpty;
    return allocator.dupe(u8, out);
}
