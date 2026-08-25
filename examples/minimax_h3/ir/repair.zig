const std = @import("std");

const config = @import("../core/config.zig");
const grid = @import("grid.zig");

const aliases = [_]struct { wrong: []const u8, right: []const u8 }{
    .{ .wrong = "Image", .right = "Picture" },
    .{ .wrong = "Img", .right = "Picture" },
    .{ .wrong = "Photo", .right = "Picture" },
    .{ .wrong = "Frame", .right = "Picture" },
    .{ .wrong = "Ref", .right = "Picture" },
    .{ .wrong = "Reference", .right = "Picture" },
    .{ .wrong = "Clip", .right = "Video" },
    .{ .wrong = "Footage", .right = "Video" },
    .{ .wrong = "Sound", .right = "Audio" },
    .{ .wrong = "Track", .right = "Audio" },
};

const quote_map = [_]struct { from: u21, to: []const u8 }{
    .{ .from = '‘', .to = "'" },
    .{ .from = '’', .to = "'" },
    .{ .from = '‚', .to = "'" },
    .{ .from = '‛', .to = "'" },
    .{ .from = '“', .to = "\"" },
    .{ .from = '”', .to = "\"" },
    .{ .from = '„', .to = "\"" },
    .{ .from = 0x00A0, .to = " " },
    .{ .from = '〈', .to = "<" },
    .{ .from = '〉', .to = ">" },
    .{ .from = '＜', .to = "<" },
    .{ .from = '＞', .to = ">" },
    .{ .from = '（', .to = "(" },
    .{ .from = '）', .to = ")" },
    .{ .from = '［', .to = "[" },
    .{ .from = '］', .to = "]" },
};

pub fn stripFences(text: []const u8) []const u8 {
    var out = std.mem.trim(u8, text, " \t\r\n");
    if (std.mem.startsWith(u8, out, "```")) {
        if (std.mem.indexOfScalar(u8, out, '\n')) |nl| out = out[nl + 1 ..];
        if (std.mem.endsWith(u8, out, "```")) out = out[0 .. out.len - 3];
        out = std.mem.trim(u8, out, " \t\r\n");
    }
    if (std.mem.indexOf(u8, out, "\nsubject_definitions:")) |pos| {
        if (!std.mem.startsWith(u8, out, "subject_definitions:")) out = out[pos + 1 ..];
    }
    return out;
}

pub fn repair(
    allocator: std.mem.Allocator,
    text: []const u8,
    variant: config.Variant,
    n_pictures: u32,
    n_videos: u32,
    n_audios: u32,
    last_only: bool,
    duration_s: f32,
    task_types: []const []const u8,
    dialogue: []const []const u8,
) ![]u8 {
    var cur = try allocator.dupe(u8, stripFences(text));
    cur = try replaceOwned(allocator, cur, try normalizeUnicode(allocator, cur));
    cur = try replaceOwned(allocator, cur, try fixAliases(allocator, cur));
    cur = try replaceOwned(allocator, cur, try clampOrdinals(allocator, cur, n_pictures, n_videos, n_audios));
    if (variant == .fl2va) {
        cur = try replaceOwned(allocator, cur, try fixFl2vaNotation(allocator, cur));
        cur = try replaceOwned(allocator, cur, try fixInstruction(allocator, cur, variant, n_pictures, last_only, duration_s));
    }
    cur = try replaceOwned(allocator, cur, try fixTaskPrefix(allocator, cur, task_types));
    cur = try replaceOwned(allocator, cur, try fixDialogue(allocator, cur, dialogue));
    if (cur.len != 0 and cur[cur.len - 1] != '\n') {
        const grown = try allocator.realloc(cur, cur.len + 1);
        grown[grown.len - 1] = '\n';
        cur = grown;
    }
    return cur;
}

fn replaceOwned(allocator: std.mem.Allocator, old: []u8, new: []u8) ![]u8 {
    if (old.ptr != new.ptr) allocator.free(old);
    return new;
}

fn normalizeUnicode(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    var it = std.unicode.Utf8Iterator{ .bytes = text, .i = 0 };
    var i: usize = 0;
    while (it.nextCodepoint()) |cp| {
        const start = i;
        i = it.i;
        if (insideVerbatim(text, start)) {
            try aw.writer.writeAll(text[start..i]);
            continue;
        }
        var mapped: ?[]const u8 = null;
        for (quote_map) |m| {
            if (cp == m.from) mapped = m.to;
        }
        if (mapped) |s| try aw.writer.writeAll(s) else try aw.writer.writeAll(text[start..i]);
    }
    return aw.toOwnedSlice();
}

fn insideVerbatim(text: []const u8, pos: usize) bool {
    var i: usize = 0;
    var in_d = false;
    var in_q = false;
    while (i < pos) : (i += 1) {
        if (!in_q and i + 3 <= text.len and std.mem.eql(u8, text[i .. i + 3], "<d>")) {
            in_d = true;
            i += 2;
            continue;
        }
        if (in_d and i + 4 <= text.len and std.mem.eql(u8, text[i .. i + 4], "</d>")) {
            in_d = false;
            i += 3;
            continue;
        }
        if (!in_d and text[i] == '"') in_q = !in_q;
    }
    return in_d or in_q;
}

fn fixAliases(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var cur = try allocator.dupe(u8, text);
    for (aliases) |a| {
        const next = try rewriteKind(allocator, cur, a.wrong, a.right);
        if (next.ptr != cur.ptr) allocator.free(cur);
        cur = next;
    }
    return cur;
}

fn rewriteKind(allocator: std.mem.Allocator, text: []const u8, wrong: []const u8, right: []const u8) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    var i: usize = 0;
    var changed = false;
    while (i < text.len) {
        if (text[i] == '<' and i + 1 + wrong.len < text.len and std.mem.startsWith(u8, text[i + 1 ..], wrong)) {
            var j = i + 1 + wrong.len;
            while (j < text.len and (text[j] == ' ' or text[j] == '\t')) j += 1;
            const n_start = j;
            while (j < text.len and text[j] >= '0' and text[j] <= '9') j += 1;
            var k = j;
            while (k < text.len and (text[k] == ' ' or text[k] == '\t')) k += 1;
            if (n_start < j and k < text.len and text[k] == '>') {
                try aw.writer.print("<{s} {s}>", .{ right, text[n_start..j] });
                i = k + 1;
                changed = true;
                continue;
            }
        }
        try aw.writer.writeByte(text[i]);
        i += 1;
    }
    if (!changed) {
        aw.deinit();
        return allocator.dupe(u8, text);
    }
    return aw.toOwnedSlice();
}

fn clampOrdinals(allocator: std.mem.Allocator, text: []const u8, n_pic: u32, n_vid: u32, n_aud: u32) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    var i: usize = 0;
    var changed = false;
    while (i < text.len) {
        if (text[i] == '<') {
            if (clampAt(text, i, "Picture", n_pic)) |c| {
                try aw.writer.print("<Picture {d}>", .{c.n});
                i = c.end;
                changed = changed or c.changed;
                continue;
            }
            if (clampAt(text, i, "Video", n_vid)) |c| {
                try aw.writer.print("<Video {d}>", .{c.n});
                i = c.end;
                changed = changed or c.changed;
                continue;
            }
            if (clampAt(text, i, "Audio", n_aud)) |c| {
                try aw.writer.print("<Audio {d}>", .{c.n});
                i = c.end;
                changed = changed or c.changed;
                continue;
            }
        }
        try aw.writer.writeByte(text[i]);
        i += 1;
    }
    if (!changed) {
        aw.deinit();
        return allocator.dupe(u8, text);
    }
    return aw.toOwnedSlice();
}

fn clampAt(text: []const u8, i: usize, kind: []const u8, top: u32) ?struct { n: u32, end: usize, changed: bool } {
    if (top == 0) return null;
    if (i + 1 + kind.len >= text.len) return null;
    if (text[i] != '<' or !std.mem.startsWith(u8, text[i + 1 ..], kind)) return null;
    var j = i + 1 + kind.len;
    while (j < text.len and (text[j] == ' ' or text[j] == '\t')) j += 1;
    const n_start = j;
    var n: u32 = 0;
    while (j < text.len and text[j] >= '0' and text[j] <= '9') : (j += 1) {
        n = n * 10 + (text[j] - '0');
    }
    var k = j;
    while (k < text.len and (text[k] == ' ' or text[k] == '\t')) k += 1;
    if (n_start == j or k >= text.len or text[k] != '>') return null;
    const clamped = if (n > top) top else n;
    return .{ .n = clamped, .end = k + 1, .changed = clamped != n };
}

fn fixFl2vaNotation(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    var i: usize = 0;
    var changed = false;
    while (i < text.len) {
        if (i + 9 < text.len and std.mem.startsWith(u8, text[i..], "<Picture ")) {
            var j = i + 9;
            while (j < text.len and text[j] >= '0' and text[j] <= '9') j += 1;
            if (j < text.len and text[j] == '>') {
                try aw.writer.writeAll(text[i + 1 .. j]);
                i = j + 1;
                changed = true;
                continue;
            }
        }
        if (std.mem.startsWith(u8, text[i..], "(from [Shot ")) {
            var j = i + 12;
            const n_start = j;
            while (j < text.len and text[j] >= '0' and text[j] <= '9') j += 1;
            if (n_start < j and j + 1 < text.len and text[j] == ']' and text[j + 1] == ')') {
                try aw.writer.print("(from Shot {s})", .{text[n_start..j]});
                i = j + 2;
                changed = true;
                continue;
            }
        }
        try aw.writer.writeByte(text[i]);
        i += 1;
    }
    if (!changed) {
        aw.deinit();
        return allocator.dupe(u8, text);
    }
    return aw.toOwnedSlice();
}

fn fixInstruction(allocator: std.mem.Allocator, text: []const u8, variant: config.Variant, n_pictures: u32, last_only: bool, duration_s: f32) ![]u8 {
    const last_shot = lastShotIn(text);
    const n_pic = if (n_pictures == 0) 2 else n_pictures;
    const want = try grid.instructionLine(allocator, variant, last_shot, duration_s, n_pic, last_only);
    defer allocator.free(want);
    if (want.len == 0) return allocator.dupe(u8, text);
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const nl = std.mem.indexOfScalar(u8, trimmed, '\n') orelse {
        if (grid.startsWithInstruction(trimmed)) return allocator.dupe(u8, want);
        return std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ want, trimmed });
    };
    const first = std.mem.trimEnd(u8, trimmed[0..nl], " \t\r");
    var rest = trimmed[nl + 1 ..];
    if (grid.startsWithInstruction(first)) {
        if (rest.len != 0 and rest[0] == '\n') rest = rest[1..];
        rest = std.mem.trimStart(u8, rest, "\r\n");
        return std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ want, rest });
    }
    return std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ want, trimmed });
}

fn fixTaskPrefix(allocator: std.mem.Allocator, text: []const u8, task_types: []const []const u8) ![]u8 {
    if (task_types.len == 0) return allocator.dupe(u8, text);
    const head = headerPos(text, "summary") orelse return allocator.dupe(u8, text);
    var i = head + "summary".len + 1;
    while (i < text.len and (text[i] == ' ' or text[i] == '\t' or text[i] == '\n' or text[i] == '\r')) i += 1;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    try aw.writer.writeAll(text[0..i]);
    try aw.writer.writeByte('[');
    for (task_types, 0..) |t, n| {
        if (n != 0) try aw.writer.writeAll(" + ");
        try aw.writer.writeAll(t);
    }
    try aw.writer.writeAll("] ");
    if (i < text.len and text[i] == '[') {
        const close = std.mem.indexOfScalarPos(u8, text, i, ']') orelse i;
        i = close + 1;
        while (i < text.len and text[i] == ' ') i += 1;
    }
    try aw.writer.writeAll(text[i..]);
    return aw.toOwnedSlice();
}

fn fixDialogue(allocator: std.mem.Allocator, text: []const u8, dialogue: []const []const u8) ![]u8 {
    if (dialogue.len == 0) return allocator.dupe(u8, text);
    var cur = try allocator.dupe(u8, text);
    for (dialogue) |want| {
        if (std.mem.indexOf(u8, cur, want) != null) continue;
        const open = std.mem.indexOf(u8, cur, "<d>") orelse continue;
        const close = std.mem.indexOfPos(u8, cur, open + 3, "</d>") orelse continue;
        const blk = cur[open + 3 .. close];
        const lang = if (std.mem.indexOfScalar(u8, blk, ']')) |p| blk[0 .. p + 1] else "[English]";
        const next = try std.fmt.allocPrint(allocator, "{s}<d>{s} {s}</d>{s}", .{ cur[0..open], std.mem.trim(u8, lang, " "), want, cur[close + 4 ..] });
        allocator.free(cur);
        cur = next;
    }
    return cur;
}

fn lastShotIn(text: []const u8) u32 {
    var max_n: u32 = 1;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, text, i, "[Shot ")) |pos| {
        const rest = text[pos + 6 ..];
        var n: u32 = 0;
        var j: usize = 0;
        while (j < rest.len and rest[j] >= '0' and rest[j] <= '9') : (j += 1) {
            n = n * 10 + (rest[j] - '0');
        }
        if (j != 0 and j < rest.len and rest[j] == ']' and n > max_n) max_n = n;
        i = pos + 6;
    }
    return max_n;
}

fn headerPos(text: []const u8, name: []const u8) ?usize {
    var i: usize = 0;
    while (i + name.len < text.len) : (i += 1) {
        if (i != 0 and text[i - 1] != '\n') continue;
        if (!std.mem.startsWith(u8, text[i..], name)) continue;
        if (text[i + name.len] == ':') return i;
    }
    return null;
}
