//! Bare port of `difflib.unified_diff(a, b, n=2, lineterm="")` for the
//! harness's two-blob compare stage. Implements Ratcliff-Obershelp
//! longest-common-subsequence (rather than Myers) so output and counts
//! match what the previous Python round-trip produced — including the
//! `b.len >= 200` "popular line" pruning that difflib calls autojunk.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Result = struct {
    is_match: bool,
    added: u32,
    removed: u32,
    /// Empty unless `capture_body` was true. Lines joined with '\n', no
    /// trailing newline — same shape as `"\n".join(unified_diff(...))`.
    body: []const u8,
};

const Tag = enum { equal, replace, delete, insert };
const Op = struct { tag: Tag, i1: usize, i2: usize, j1: usize, j2: usize };
const Block = struct { i: usize, j: usize, k: usize };

pub fn diffText(
    arena: Allocator,
    a: []const u8,
    b: []const u8,
    n: usize,
    capture_body: bool,
) !Result {
    if (std.mem.eql(u8, a, b)) return .{ .is_match = true, .added = 0, .removed = 0, .body = "" };
    return diffLines(arena, try splitLines(arena, a), try splitLines(arena, b), n, capture_body);
}

fn splitLines(arena: Allocator, text: []const u8) ![]const []const u8 {
    if (text.len == 0) return &.{};
    var out: std.ArrayList([]const u8) = .empty;
    var iter = std.mem.splitScalar(u8, text, '\n');
    while (iter.next()) |line| try out.append(arena, line);
    return out.toOwnedSlice(arena);
}

fn diffLines(
    arena: Allocator,
    a: []const []const u8,
    b: []const []const u8,
    n: usize,
    capture_body: bool,
) !Result {
    const b2j = try buildIndex(arena, b);
    const blocks = try matchingBlocks(arena, a, b, b2j);
    const ops = try opcodesFromBlocks(arena, blocks);
    const groups = try groupedOpcodes(arena, ops, n);

    var added: u32 = 0;
    var removed: u32 = 0;
    for (groups) |g| for (g) |op| {
        switch (op.tag) {
            .equal => {},
            .delete => removed += @intCast(op.i2 - op.i1),
            .insert => added += @intCast(op.j2 - op.j1),
            .replace => {
                removed += @intCast(op.i2 - op.i1);
                added += @intCast(op.j2 - op.j1);
            },
        }
    };

    const body: []const u8 = if (capture_body and groups.len > 0)
        try formatBody(arena, a, b, groups)
    else
        "";

    return .{
        .is_match = groups.len == 0,
        .added = added,
        .removed = removed,
        .body = body,
    };
}

const LineMap = std.StringHashMapUnmanaged(std.ArrayList(usize));

fn buildIndex(arena: Allocator, b: []const []const u8) !LineMap {
    var b2j: LineMap = .empty;
    for (b, 0..) |line, j| {
        const gop = try b2j.getOrPut(arena, line);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(arena, j);
    }
    if (b.len >= 200) {
        const ntest = b.len / 100 + 1;
        var rm: std.ArrayList([]const u8) = .empty;
        var it = b2j.iterator();
        while (it.next()) |e| if (e.value_ptr.items.len > ntest) try rm.append(arena, e.key_ptr.*);
        for (rm.items) |k| _ = b2j.remove(k);
    }
    return b2j;
}

fn matchingBlocks(
    arena: Allocator,
    a: []const []const u8,
    b: []const []const u8,
    b2j: LineMap,
) ![]const Block {
    const Frame = struct { alo: usize, ahi: usize, blo: usize, bhi: usize };
    var queue: std.ArrayList(Frame) = .empty;
    try queue.append(arena, .{ .alo = 0, .ahi = a.len, .blo = 0, .bhi = b.len });

    var blocks: std.ArrayList(Block) = .empty;

    var j2_a: std.AutoHashMapUnmanaged(usize, usize) = .empty;
    var j2_b: std.AutoHashMapUnmanaged(usize, usize) = .empty;

    while (queue.pop()) |seg| {
        const m = try findLongestMatch(arena, a, b, b2j, &j2_a, &j2_b, seg.alo, seg.ahi, seg.blo, seg.bhi);
        if (m.k == 0) continue;
        try blocks.append(arena, m);
        if (seg.alo < m.i and seg.blo < m.j) try queue.append(arena, .{ .alo = seg.alo, .ahi = m.i, .blo = seg.blo, .bhi = m.j });
        if (m.i + m.k < seg.ahi and m.j + m.k < seg.bhi) try queue.append(arena, .{ .alo = m.i + m.k, .ahi = seg.ahi, .blo = m.j + m.k, .bhi = seg.bhi });
    }

    std.mem.sort(Block, blocks.items, {}, struct {
        fn lt(_: void, x: Block, y: Block) bool {
            if (x.i != y.i) return x.i < y.i;
            return x.j < y.j;
        }
    }.lt);

    var collapsed: std.ArrayList(Block) = .empty;
    var ci: usize = 0;
    var cj: usize = 0;
    var ck: usize = 0;
    for (blocks.items) |bk| {
        if (ci + ck == bk.i and cj + ck == bk.j) {
            ck += bk.k;
        } else {
            if (ck != 0) try collapsed.append(arena, .{ .i = ci, .j = cj, .k = ck });
            ci = bk.i;
            cj = bk.j;
            ck = bk.k;
        }
    }
    if (ck != 0) try collapsed.append(arena, .{ .i = ci, .j = cj, .k = ck });
    try collapsed.append(arena, .{ .i = a.len, .j = b.len, .k = 0 });
    return collapsed.toOwnedSlice(arena);
}

fn findLongestMatch(
    arena: Allocator,
    a: []const []const u8,
    b: []const []const u8,
    b2j: LineMap,
    cur_in: *std.AutoHashMapUnmanaged(usize, usize),
    nxt_in: *std.AutoHashMapUnmanaged(usize, usize),
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) !Block {
    var besti: usize = alo;
    var bestj: usize = blo;
    var bestsize: usize = 0;

    cur_in.clearRetainingCapacity();
    var cur = cur_in;
    var nxt = nxt_in;

    var i: usize = alo;
    while (i < ahi) : (i += 1) {
        nxt.clearRetainingCapacity();
        if (b2j.get(a[i])) |list| {
            for (list.items) |j| {
                if (j < blo) continue;
                if (j >= bhi) break;
                const prev = if (j > 0) (cur.get(j - 1) orelse 0) else 0;
                const k = prev + 1;
                try nxt.put(arena, j, k);
                if (k > bestsize) {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        std.mem.swap(*std.AutoHashMapUnmanaged(usize, usize), &cur, &nxt);
    }

    // Re-attach popular-pruned matches that fall on either side of the
    // best block — they were intentionally held out of `b2j` to keep the
    // inner loop fast.
    while (besti > alo and bestj > blo and std.mem.eql(u8, a[besti - 1], b[bestj - 1])) {
        besti -= 1;
        bestj -= 1;
        bestsize += 1;
    }
    while (besti + bestsize < ahi and bestj + bestsize < bhi and std.mem.eql(u8, a[besti + bestsize], b[bestj + bestsize])) {
        bestsize += 1;
    }
    return .{ .i = besti, .j = bestj, .k = bestsize };
}

fn opcodesFromBlocks(arena: Allocator, blocks: []const Block) ![]const Op {
    var ops: std.ArrayList(Op) = .empty;
    var i: usize = 0;
    var j: usize = 0;
    for (blocks) |blk| {
        if (i < blk.i and j < blk.j) {
            try ops.append(arena, .{ .tag = .replace, .i1 = i, .i2 = blk.i, .j1 = j, .j2 = blk.j });
        } else if (i < blk.i) {
            try ops.append(arena, .{ .tag = .delete, .i1 = i, .i2 = blk.i, .j1 = j, .j2 = blk.j });
        } else if (j < blk.j) {
            try ops.append(arena, .{ .tag = .insert, .i1 = i, .i2 = blk.i, .j1 = j, .j2 = blk.j });
        }
        i = blk.i + blk.k;
        j = blk.j + blk.k;
        if (blk.k > 0) try ops.append(arena, .{ .tag = .equal, .i1 = blk.i, .i2 = i, .j1 = blk.j, .j2 = j });
    }
    return ops.toOwnedSlice(arena);
}

fn groupedOpcodes(arena: Allocator, codes_in: []const Op, n: usize) ![]const []const Op {
    if (codes_in.len == 0) return &.{};
    var codes = try arena.dupe(Op, codes_in);

    if (codes[0].tag == .equal and codes[0].i2 - codes[0].i1 > n) {
        codes[0].i1 = codes[0].i2 - n;
        codes[0].j1 = codes[0].j2 - n;
    }
    const last = codes.len - 1;
    if (codes[last].tag == .equal and codes[last].i2 - codes[last].i1 > n) {
        codes[last].i2 = codes[last].i1 + n;
        codes[last].j2 = codes[last].j1 + n;
    }

    const nn = n + n;
    var groups: std.ArrayList([]const Op) = .empty;
    var cur: std.ArrayList(Op) = .empty;
    for (codes) |op_in| {
        var op = op_in;
        if (op.tag == .equal and op.i2 - op.i1 > nn) {
            try cur.append(arena, .{ .tag = .equal, .i1 = op.i1, .i2 = op.i1 + n, .j1 = op.j1, .j2 = op.j1 + n });
            try groups.append(arena, try cur.toOwnedSlice(arena));
            cur = .empty;
            op.i1 = op.i2 - n;
            op.j1 = op.j2 - n;
        }
        try cur.append(arena, op);
    }
    if (cur.items.len > 0 and !(cur.items.len == 1 and cur.items[0].tag == .equal)) {
        try groups.append(arena, try cur.toOwnedSlice(arena));
    }
    return groups.toOwnedSlice(arena);
}

fn formatBody(
    arena: Allocator,
    a: []const []const u8,
    b: []const []const u8,
    groups: []const []const Op,
) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(arena, "--- \n+++ ");
    for (groups) |group| {
        try buf.append(arena, '\n');
        try buf.appendSlice(arena, "@@ -");
        try formatRange(arena, &buf, group[0].i1, group[group.len - 1].i2);
        try buf.appendSlice(arena, " +");
        try formatRange(arena, &buf, group[0].j1, group[group.len - 1].j2);
        try buf.appendSlice(arena, " @@");
        for (group) |op| switch (op.tag) {
            .equal => for (a[op.i1..op.i2]) |line| {
                try buf.append(arena, '\n');
                try buf.append(arena, ' ');
                try buf.appendSlice(arena, line);
            },
            .replace => {
                for (a[op.i1..op.i2]) |line| {
                    try buf.append(arena, '\n');
                    try buf.append(arena, '-');
                    try buf.appendSlice(arena, line);
                }
                for (b[op.j1..op.j2]) |line| {
                    try buf.append(arena, '\n');
                    try buf.append(arena, '+');
                    try buf.appendSlice(arena, line);
                }
            },
            .delete => for (a[op.i1..op.i2]) |line| {
                try buf.append(arena, '\n');
                try buf.append(arena, '-');
                try buf.appendSlice(arena, line);
            },
            .insert => for (b[op.j1..op.j2]) |line| {
                try buf.append(arena, '\n');
                try buf.append(arena, '+');
                try buf.appendSlice(arena, line);
            },
        };
    }
    return buf.toOwnedSlice(arena);
}

fn formatRange(arena: Allocator, buf: *std.ArrayList(u8), start: usize, stop: usize) !void {
    var beginning = start + 1;
    const length = stop - start;
    if (length == 1) {
        try buf.print(arena, "{}", .{beginning});
        return;
    }
    if (length == 0) beginning -= 1;
    try buf.print(arena, "{},{}", .{ beginning, length });
}
