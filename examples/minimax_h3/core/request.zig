const std = @import("std");

const config = @import("config.zig");
const packing = @import("../model/packing.zig");

pub const Reference = struct {
    kind: packing.ReferenceKind,
    path: []const u8,
    soundtrack: []const u8 = "",
};

pub const Request = struct {
    variant: config.Variant = .t2va,
    prompt: []const u8,
    first_image: []const u8 = "",
    last_image: []const u8 = "",
    refs: []const Reference = &.{},
};

pub fn inferVariant(first_image: []const u8, last_image: []const u8, refs: []const Reference) !config.Variant {
    const has_keyframes = first_image.len != 0 or last_image.len != 0;
    if (refs.len != 0) {
        if (has_keyframes) return error.Ref2vaRejectsKeyframes;
        return .ref2va;
    }
    if (has_keyframes) return .fl2va;
    return .t2va;
}

pub fn splitComma(allocator: std.mem.Allocator, text: []const u8) ![][]const u8 {
    if (text.len == 0) return &.{};
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(allocator);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t");
        if (trimmed.len == 0) continue;
        try out.append(allocator, trimmed);
    }
    return out.toOwnedSlice(allocator);
}

pub fn refsFromComma(allocator: std.mem.Allocator, text: []const u8) ![]Reference {
    const paths = try splitComma(allocator, text);
    defer allocator.free(paths);
    return refsFromPaths(allocator, paths);
}

pub fn refsFromPaths(allocator: std.mem.Allocator, paths: []const []const u8) ![]Reference {
    var out: std.ArrayList(Reference) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < paths.len) : (i += 1) {
        const kind = guessKind(paths[i]);
        if (kind == .video and i + 1 < paths.len and guessKind(paths[i + 1]) == .audio) {
            try out.append(allocator, .{
                .kind = .video_audio,
                .path = paths[i],
                .soundtrack = paths[i + 1],
            });
            i += 1;
            continue;
        }
        try out.append(allocator, .{ .kind = kind, .path = paths[i] });
    }
    return out.toOwnedSlice(allocator);
}

fn guessKind(path: []const u8) packing.ReferenceKind {
    const ext = std.fs.path.extension(path);
    if (std.ascii.eqlIgnoreCase(ext, ".wav") or std.ascii.eqlIgnoreCase(ext, ".mp3") or std.ascii.eqlIgnoreCase(ext, ".flac") or std.ascii.eqlIgnoreCase(ext, ".m4a"))
        return .audio;
    if (std.ascii.eqlIgnoreCase(ext, ".mp4") or std.ascii.eqlIgnoreCase(ext, ".mov") or std.ascii.eqlIgnoreCase(ext, ".mkv") or std.ascii.eqlIgnoreCase(ext, ".webm"))
        return .video;
    return .image;
}

pub fn freeRefs(allocator: std.mem.Allocator, refs: []Reference, owned_strings: bool) void {
    if (owned_strings) {
        for (refs) |r| {
            allocator.free(r.path);
            if (r.soundtrack.len != 0) allocator.free(r.soundtrack);
        }
    }
    allocator.free(refs);
}

pub fn validate(req: Request) !void {
    switch (req.variant) {
        .t2va => {
            if (req.first_image.len != 0 or req.last_image.len != 0 or req.refs.len != 0)
                return error.T2vaRejectsMedia;
        },
        .fl2va => {
            if (req.first_image.len == 0 and req.last_image.len == 0) return error.Fl2vaNeedsImage;
            if (req.refs.len != 0) return error.Fl2vaRejectsRefs;
        },
        .ref2va => {
            if (req.refs.len == 0) return error.Ref2vaNeedsRefs;
            if (req.first_image.len != 0 or req.last_image.len != 0) return error.Ref2vaRejectsKeyframes;
        },
    }
    if (std.mem.trim(u8, req.prompt, " \t\r\n").len == 0) return error.IntentEmpty;
    try validateRefs(req.refs);
}

pub fn validateRefs(refs: []const Reference) !void {
    if (refs.len > config.max_ref_files) return error.TooManyRefs;
    var n_img: u32 = 0;
    var n_vid: u32 = 0;
    var n_aud: u32 = 0;
    for (refs) |r| {
        switch (r.kind) {
            .image => {
                n_img += 1;
                if (n_img > config.max_ref_images) return error.TooManyRefImages;
            },
            .video, .video_audio => {
                n_vid += 1;
                if (n_vid > config.max_ref_videos) return error.TooManyRefVideos;
                if (r.kind == .video_audio) {
                    n_aud += 1;
                    if (n_aud > config.max_ref_audios) return error.TooManyRefAudios;
                }
            },
            .audio => {
                n_aud += 1;
                if (n_aud > config.max_ref_audios) return error.TooManyRefAudios;
            },
        }
    }
    if (n_aud != 0 and n_img == 0 and n_vid == 0) return error.AudioRefNeedsVisual;
}
