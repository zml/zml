const std = @import("std");

const config = @import("../core/config.zig");
const geom = @import("geometry.zig");
const packing = @import("../model/packing.zig");
const vision = @import("../model/vision.zig");

pub const VisionSpan = struct {
    start: u32,
    tokens: u32,
    grid_h: u32,
    grid_w: u32,
    temporal: u32,
};

pub fn fillEncoderPositions(pos: []f32, seq: u32, spans: []const VisionSpan) void {
    var cursor: f32 = 0;
    var i: u32 = 0;
    var span_i: usize = 0;
    while (i < seq) {
        if (span_i < spans.len and i == spans[span_i].start) {
            const span = spans[span_i];
            vision.applyVisionPositions(pos, span.start, span.tokens, span.grid_h, span.grid_w, span.temporal, &cursor);
            i += span.tokens;
            span_i += 1;
        } else {
            pos[i * 3 + 0] = cursor;
            pos[i * 3 + 1] = cursor;
            pos[i * 3 + 2] = cursor;
            cursor += 1;
            i += 1;
        }
    }
}

pub const VisualSpec = struct {
    kind: packing.ReferenceKind,
    merged: u32,
    grid_h: u32,
    grid_w: u32,
    temporal: u32 = 1,
    timestamps: []const f32 = &.{},
    has_audio: bool = false,
};

pub const Assembled = struct {
    tokens: []u32,
    tags: []u8,
    spans: []VisionSpan,

    pub fn deinit(self: Assembled, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.tags);
        allocator.free(self.spans);
    }
};

const Builder = struct {
    allocator: std.mem.Allocator,
    tokens: std.ArrayList(u32),
    tags: std.ArrayList(u8),
    spans: std.ArrayList(VisionSpan),

    fn init(allocator: std.mem.Allocator) Builder {
        return .{
            .allocator = allocator,
            .tokens = .empty,
            .tags = .empty,
            .spans = .empty,
        };
    }

    fn deinit(self: *Builder) void {
        self.tokens.deinit(self.allocator);
        self.tags.deinit(self.allocator);
        self.spans.deinit(self.allocator);
    }

    fn emitText(self: *Builder, ids: []const u32) !void {
        try self.tokens.appendSlice(self.allocator, ids);
        try self.tags.appendNTimes(self.allocator, @intFromEnum(packing.Modality.text), ids.len);
    }

    fn emitVision(self: *Builder, pad: u32, count: u32, grid_h: u32, grid_w: u32, temporal: u32) !void {
        try self.tokens.append(self.allocator, vision.VISION_START);
        try self.tags.append(self.allocator, @intFromEnum(packing.Modality.video));
        const start: u32 = @intCast(self.tokens.items.len);
        var i: u32 = 0;
        while (i < count) : (i += 1) try self.tokens.append(self.allocator, pad);
        try self.tags.appendNTimes(self.allocator, @intFromEnum(packing.Modality.video), count);
        try self.tokens.append(self.allocator, vision.VISION_END);
        try self.tags.append(self.allocator, @intFromEnum(packing.Modality.video));
        try self.spans.append(self.allocator, .{
            .start = start,
            .tokens = count,
            .grid_h = grid_h,
            .grid_w = grid_w,
            .temporal = temporal,
        });
    }

    fn finish(self: *Builder) !Assembled {
        return .{
            .tokens = try self.tokens.toOwnedSlice(self.allocator),
            .tags = try self.tags.toOwnedSlice(self.allocator),
            .spans = try self.spans.toOwnedSlice(self.allocator),
        };
    }
};

fn encodeLabel(allocator: std.mem.Allocator, encode_text: anytype, comptime fmt: []const u8, args: anytype) ![]u32 {
    var buf: [64]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, fmt, args);
    return encode_text.encodeAlloc(allocator, text);
}

pub fn assembleT2va(allocator: std.mem.Allocator, encode_text: anytype, prompt: []const u8) !Assembled {
    var b = Builder.init(allocator);
    errdefer b.deinit();
    const ids = try encode_text.encodeAlloc(allocator, prompt);
    defer allocator.free(ids);
    try b.emitText(ids);
    return b.finish();
}

pub fn assembleFl2va(allocator: std.mem.Allocator, encode_text: anytype, visuals: []const VisualSpec, prompt: []const u8) !Assembled {
    var b = Builder.init(allocator);
    errdefer b.deinit();
    for (visuals, 0..) |vis, i| {
        const label = try encodeLabel(allocator, encode_text, "<Picture {d}>: ", .{i + 1});
        defer allocator.free(label);
        try b.emitText(label);
        try b.emitVision(vision.IMAGE_PAD, vis.merged, vis.grid_h, vis.grid_w, vis.temporal);
    }
    const ids = try encode_text.encodeAlloc(allocator, prompt);
    defer allocator.free(ids);
    try b.emitText(ids);
    return b.finish();
}

pub fn assembleRef2va(allocator: std.mem.Allocator, encode_text: anytype, visuals: []const VisualSpec, prompt: []const u8) !Assembled {
    var b = Builder.init(allocator);
    errdefer b.deinit();
    var n_pic: u32 = 0;
    var n_vid: u32 = 0;
    var n_aud: u32 = 0;
    for (visuals) |vis| {
        if (vis.has_audio or vis.kind == .audio or vis.kind == .video_audio) {
            n_aud += 1;
            const label = try encodeLabel(allocator, encode_text, "<Audio {d}>: ", .{n_aud});
            defer allocator.free(label);
            try b.emitText(label);
        }
        if (vis.kind == .image) {
            n_pic += 1;
            const label = try encodeLabel(allocator, encode_text, "<Picture {d}>: ", .{n_pic});
            defer allocator.free(label);
            try b.emitText(label);
            try b.emitVision(vision.IMAGE_PAD, vis.merged, vis.grid_h, vis.grid_w, vis.temporal);
        } else if (vis.kind == .video or vis.kind == .video_audio) {
            n_vid += 1;
            const label = try encodeLabel(allocator, encode_text, "<Video {d}>: ", .{n_vid});
            defer allocator.free(label);
            try b.emitText(label);
            for (vis.timestamps) |ts| {
                var tbuf: [32]u8 = undefined;
                const rendered = geom.formatSeconds1(ts, &tbuf);
                var sbuf: [48]u8 = undefined;
                const stamp = try std.fmt.bufPrint(&sbuf, "<{s} seconds>", .{rendered});
                const ids = try encode_text.encodeAlloc(allocator, stamp);
                defer allocator.free(ids);
                try b.emitText(ids);
                try b.emitVision(vision.VIDEO_PAD, vis.merged, vis.grid_h, vis.grid_w, 1);
            }
        }
    }
    const ids = try encode_text.encodeAlloc(allocator, prompt);
    defer allocator.free(ids);
    try b.emitText(ids);
    return b.finish();
}

pub fn assemble(
    allocator: std.mem.Allocator,
    encode_text: anytype,
    variant: config.Variant,
    visuals: []const VisualSpec,
    prompt: []const u8,
) !Assembled {
    return switch (variant) {
        .t2va => assembleT2va(allocator, encode_text, prompt),
        .fl2va => assembleFl2va(allocator, encode_text, visuals, prompt),
        .ref2va => assembleRef2va(allocator, encode_text, visuals, prompt),
    };
}
