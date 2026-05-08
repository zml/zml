//! Strip XLA pretty-debug-info from post-XLA TTIR/TTGIR so the Zig and
//! Python sides line up. For Triton MLIR, also reorder the leading
//! `arith.constant` block by *body-use order* — both frontends emit the
//! same constants but in different source order, and once the body uses
//! them in a consistent sequence we can pin down a canonical ordering
//! that kills the SSA cascade through the rest of the function.
//! Non-constant SSA names pass through verbatim so operand-identity
//! divergence (different register routing, different upstream value
//! consumed) stays visible in the diff.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Lang = enum { triton_mlir, mosaic_mlir, llvm_ir, ptx };

/// Stateless — kept as a struct so call sites can hold a pointer,
/// matching the previous regex-cache shape.
pub const Normalizer = struct {
    pub fn init(_: Allocator) !Normalizer {
        return .{};
    }
    pub fn deinit(_: *Normalizer) void {}

    /// Output lives in `arena`; compare passes use a fresh arena per sweep.
    pub fn normalize(_: *const Normalizer, arena: Allocator, text: []const u8, lang: Lang) ![]const u8 {
        const stripped = try normalizeInto(arena, text, lang);
        return switch (lang) {
            .triton_mlir, .mosaic_mlir => try renumberConstants(arena, stripped),
            .llvm_ir, .ptx => stripped,
        };
    }
};

fn normalizeInto(arena: Allocator, text: []const u8, lang: Lang) ![]const u8 {
    var out: std.ArrayList(u8) = try .initCapacity(arena, text.len);

    const triton = (lang == .triton_mlir);
    var i: usize = 0;
    var line_start: usize = 0;
    var at_line_start: bool = true;

    while (i < text.len) {
        if (at_line_start and triton) if (matchLocDefLine(text, i)) |line_end| {
            i = line_end;
            continue;
        };
        at_line_start = false;

        const c = text[i];

        if (c == '\n') {
            if (isBlankRun(out.items[line_start..])) {
                out.items.len = line_start;
            } else {
                try out.append(arena, '\n');
                line_start = out.items.len;
            }
            i += 1;
            at_line_start = true;
            continue;
        }

        if (triton and (c == ' ' or c == '\t')) if (matchSpacePrefixDrop(text, i)) |new_i| {
            i = new_i;
            continue;
        };

        try out.append(arena, c);
        i += 1;
    }

    if (isBlankRun(out.items[line_start..])) out.items.len = line_start;
    if (out.items.len > 0 and out.items[out.items.len - 1] == '\n') out.items.len -= 1;

    return out.toOwnedSlice(arena);
}

/// Find the first contiguous run of `arith.constant` defs in the function
/// body and renumber them by *body-use order*: each constant's canonical
/// id `%K<n>` is assigned the first time its source name appears in the
/// body. Both Zig and Python end up with the constants block emitted in
/// the same order, which kills the SSA cascade through every downstream
/// op. Constants defined but never used in the body get appended in
/// source order. The `%K` prefix avoids colliding with MLIR's natural
/// `%c<int>` (index const) and `%cst[_<n>]` names.
fn renumberConstants(arena: Allocator, text: []const u8) ![]const u8 {
    const ConstEntry = struct {
        src_id: []const u8,
        line: []const u8,
    };

    var consts: std.ArrayList(ConstEntry) = .empty;
    var const_set: std.StringHashMapUnmanaged(usize) = .empty;
    var block_start: ?usize = null;
    var block_end: usize = text.len;

    var pos: usize = 0;
    while (pos < text.len) {
        const eol = std.mem.indexOfScalarPos(u8, text, pos, '\n') orelse text.len;
        const line = text[pos..eol];
        if (parseConstantDefLine(line)) |src_id| {
            if (block_start == null) block_start = pos;
            try consts.append(arena, .{ .src_id = src_id, .line = line });
            try const_set.put(arena, src_id, consts.items.len - 1);
        } else if (block_start != null) {
            block_end = pos;
            break;
        }
        pos = if (eol < text.len) eol + 1 else eol;
    }

    if (block_start == null) return text;

    var canonical_to_src: std.ArrayList(usize) = .empty;
    var assigned: std.StringHashMapUnmanaged(usize) = .empty;

    var bp = block_end;
    while (bp < text.len) {
        const c = text[bp];
        if (c == '%' and bp + 1 < text.len and isIdentChar(text[bp + 1])) {
            const id_start = bp + 1;
            var id_end = id_start;
            while (id_end < text.len and isIdentChar(text[id_end])) id_end += 1;
            const id = text[id_start..id_end];
            if (const_set.get(id)) |src_idx| {
                if (!assigned.contains(id)) {
                    try assigned.put(arena, id, canonical_to_src.items.len);
                    try canonical_to_src.append(arena, src_idx);
                }
            }
            bp = id_end;
        } else {
            bp += 1;
        }
    }

    for (consts.items, 0..) |c, i| {
        if (!assigned.contains(c.src_id)) {
            try assigned.put(arena, c.src_id, canonical_to_src.items.len);
            try canonical_to_src.append(arena, i);
        }
    }

    var out: std.ArrayList(u8) = try .initCapacity(arena, text.len);
    try out.appendSlice(arena, text[0..block_start.?]);

    var buf: [32]u8 = undefined;
    for (canonical_to_src.items, 0..) |src_idx, canon_idx| {
        const c = consts.items[src_idx];
        const pct = std.mem.indexOfScalar(u8, c.line, '%').?;
        try out.appendSlice(arena, c.line[0..pct]);
        try out.appendSlice(arena, try std.fmt.bufPrint(&buf, "%K{d}", .{canon_idx}));
        var skip = pct + 1;
        while (skip < c.line.len and isIdentChar(c.line[skip])) skip += 1;
        try out.appendSlice(arena, c.line[skip..]);
        try out.append(arena, '\n');
    }

    var rp = block_end;
    while (rp < text.len) {
        const c = text[rp];
        if (c == '%' and rp + 1 < text.len and isIdentChar(text[rp + 1])) {
            const id_start = rp + 1;
            var id_end = id_start;
            while (id_end < text.len and isIdentChar(text[id_end])) id_end += 1;
            const id = text[id_start..id_end];
            if (assigned.get(id)) |canon_idx| {
                try out.appendSlice(arena, try std.fmt.bufPrint(&buf, "%K{d}", .{canon_idx}));
                rp = id_end;
                continue;
            }
            try out.appendSlice(arena, text[rp..id_end]);
            rp = id_end;
        } else {
            try out.append(arena, c);
            rp += 1;
        }
    }

    return out.toOwnedSlice(arena);
}

/// `<ws>%<id> = "arith.constant"...` (generic, XLA TTIR/TTGIR dumps) or
/// `<ws>%<id> = arith.constant ...` (custom, MLIR's default printer used
/// by Mosaic) — returns the SSA id (no `%`) on match.
fn parseConstantDefLine(line: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < line.len and (line[i] == ' ' or line[i] == '\t')) i += 1;
    if (i >= line.len or line[i] != '%') return null;
    i += 1;
    const id_start = i;
    while (i < line.len and isIdentChar(line[i])) i += 1;
    if (i == id_start) return null;
    const id = line[id_start..i];
    while (i < line.len and (line[i] == ' ' or line[i] == '\t')) i += 1;
    if (i >= line.len or line[i] != '=') return null;
    i += 1;
    while (i < line.len and (line[i] == ' ' or line[i] == '\t')) i += 1;

    const generic = "\"arith.constant\"";
    if (i + generic.len <= line.len and std.mem.eql(u8, line[i .. i + generic.len], generic)) return id;

    const custom = "arith.constant";
    if (i + custom.len < line.len and
        std.mem.eql(u8, line[i .. i + custom.len], custom) and
        (line[i + custom.len] == ' ' or line[i + custom.len] == '\t')) return id;

    return null;
}

fn isIdentChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn isBlankRun(s: []const u8) bool {
    for (s) |c| if (c != ' ' and c != '\t' and c != '\r') return false;
    return true;
}

/// Approximation of `^\s*#loc\d*\s*=.*$`. Returns the position just past
/// the terminating '\n' (or `text.len`) on match. We treat any
/// `#loc...`-prefixed line as a def — MLIR's pretty-debug-info form
/// never emits `#loc` at line start in any other context.
fn matchLocDefLine(text: []const u8, i: usize) ?usize {
    var j = i;
    while (j < text.len and (text[j] == ' ' or text[j] == '\t')) j += 1;
    if (j + 4 > text.len) return null;
    if (!std.mem.eql(u8, text[j .. j + 4], "#loc")) return null;
    const eol = std.mem.indexOfScalarPos(u8, text, j, '\n') orelse return text.len;
    return eol + 1;
}

/// `text[i]` is space or tab. Tries the four whitespace-prefixed strip
/// patterns; returns the post-match position (excluding any trailing
/// boundary char that should remain in output).
fn matchSpacePrefixDrop(text: []const u8, i: usize) ?usize {
    if (i + 1 >= text.len) return null;
    return switch (text[i + 1]) {
        '[' => matchUnknown(text, i),
        '"' => matchNameSfx(text, i),
        'l' => matchLocWrapped(text, i),
        else => matchFileLineCol(text, i),
    };
}

/// ` [unknown]` followed by `,`, `)`, ` `, `\n`, or EOF.
fn matchUnknown(text: []const u8, i: usize) ?usize {
    const lit = "[unknown]";
    const end = i + 1 + lit.len;
    if (end > text.len) return null;
    if (!std.mem.eql(u8, text[i + 1 .. end], lit)) return null;
    if (end == text.len) return end;
    return switch (text[end]) {
        ',', ')', ' ', '\n' => end,
        else => null,
    };
}

/// ` "name"(\(#loc\d+\))?<boundary>` — boundary in {`,`, `)`, ` `, `\n`, EOF}.
fn matchNameSfx(text: []const u8, i: usize) ?usize {
    var k: usize = i + 2;
    while (k < text.len and text[k] != '"' and text[k] != '\n') k += 1;
    if (k >= text.len or text[k] != '"' or k == i + 2) return null;
    var end: usize = k + 1;
    if (end + 5 <= text.len and text[end] == '(' and std.mem.eql(u8, text[end + 1 .. end + 5], "#loc")) {
        var n = end + 5;
        while (n < text.len and std.ascii.isDigit(text[n])) n += 1;
        if (n > end + 5 and n < text.len and text[n] == ')') end = n + 1;
    }
    if (end == text.len) return end;
    return switch (text[end]) {
        ',', ')', ' ', '\n' => end,
        else => null,
    };
}

/// ` <prefix>:\d+:\d+`. `\S+?` is non-greedy, so we take the leftmost
/// `:\d+:\d+` within the non-space run.
fn matchFileLineCol(text: []const u8, i: usize) ?usize {
    var run_end: usize = i + 1;
    while (run_end < text.len and !isWs(text[run_end])) run_end += 1;
    var k: usize = i + 2;
    while (k < run_end) : (k += 1) {
        if (text[k] != ':') continue;
        var d1 = k + 1;
        while (d1 < text.len and std.ascii.isDigit(text[d1])) d1 += 1;
        if (d1 == k + 1 or d1 >= text.len or text[d1] != ':') continue;
        var d2 = d1 + 1;
        while (d2 < text.len and std.ascii.isDigit(text[d2])) d2 += 1;
        if (d2 > d1 + 1) return d2;
    }
    return null;
}

/// ` loc(...)` — single-level balanced parens. XLA never emits this on
/// post-pass TTIR but Mosaic occasionally does, so kept for parity.
fn matchLocWrapped(text: []const u8, i: usize) ?usize {
    if (i + 5 > text.len) return null;
    if (!std.mem.eql(u8, text[i + 1 .. i + 5], "loc(")) return null;
    var depth: usize = 1;
    var k: usize = i + 5;
    while (k < text.len) : (k += 1) {
        switch (text[k]) {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if (depth == 0) return k + 1;
            },
            '\n' => return null,
            else => {},
        }
    }
    return null;
}

fn isWs(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r' or c == '\n';
}

test "strip locs from triton-mlir" {
    const input =
        "#loc1 = \"foo\"\n" ++
        "#loc2 = \"bar\"\n" ++
        "module {\n" ++
        "  %0 = \"tt.load\"(%arg1) : () -> i32 [unknown]\n" ++
        "  %1 = \"tt.add\"(%0, %0) \"add\"(#loc1) : (i32, i32) -> i32\n" ++
        "  ^bb0(%arg2: i32 -:55:24, %arg3: i32 -:55:42):\n" ++
        "}\n";
    const expected =
        "module {\n" ++
        "  %0 = \"tt.load\"(%arg1) : () -> i32\n" ++
        "  %1 = \"tt.add\"(%0, %0) : (i32, i32) -> i32\n" ++
        "  ^bb0(%arg2: i32, %arg3: i32):\n" ++
        "}";

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const got = try normalizeInto(arena.allocator(), input, .triton_mlir);
    try std.testing.expectEqualStrings(expected, got);
}

test "preserve ssa names on mosaic-mlir" {
    const input = "  %5 = arith.addi %arg0, %2 : i32\n";
    const expected = "  %5 = arith.addi %arg0, %2 : i32";

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const got = try normalizeInto(arena.allocator(), input, .mosaic_mlir);
    try std.testing.expectEqualStrings(expected, got);
}

test "preserve ssa names on llvm-ir" {
    const input = "  %5 = add i32 %arg0, %2\n  ret i32 %5\n";
    const expected = "  %5 = add i32 %arg0, %2\n  ret i32 %5";

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const got = try normalizeInto(arena.allocator(), input, .llvm_ir);
    try std.testing.expectEqualStrings(expected, got);
}

test "preserve register names on ptx" {
    const input = "  add.s32 %r1, %r2, %r3;\n  ret;\n";
    const expected = "  add.s32 %r1, %r2, %r3;\n  ret;";

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const got = try normalizeInto(arena.allocator(), input, .ptx);
    try std.testing.expectEqualStrings(expected, got);
}

test "drop blank lines" {
    const input = "alpha\n\n  \nbeta\n   \n";
    const expected = "alpha\nbeta";

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const got = try normalizeInto(arena.allocator(), input, .llvm_ir);
    try std.testing.expectEqualStrings(expected, got);
}

test "renumber constants by body-use order" {
    // Same constants in different declaration order on the two sides;
    // the body uses them in the same logical sequence. After renumber
    // both inputs should converge to the same canonical text.
    const input_a =
        "module {\n" ++
        "  ^bb0(%arg0: i32):\n" ++
        "    %0 = \"arith.constant\"() <{value = 64 : i32}> : () -> i32\n" ++
        "    %1 = \"arith.constant\"() <{value = 0 : i32}> : () -> i32\n" ++
        "    %2 = \"arith.constant\"() <{value = 1024 : i32}> : () -> i32\n" ++
        "    %3 = \"foo\"(%2, %0, %1) : (i32, i32, i32) -> i32\n" ++
        "}\n";
    const input_b =
        "module {\n" ++
        "  ^bb0(%arg0: i32):\n" ++
        "    %0 = \"arith.constant\"() <{value = 1024 : i32}> : () -> i32\n" ++
        "    %1 = \"arith.constant\"() <{value = 64 : i32}> : () -> i32\n" ++
        "    %2 = \"arith.constant\"() <{value = 0 : i32}> : () -> i32\n" ++
        "    %3 = \"foo\"(%0, %1, %2) : (i32, i32, i32) -> i32\n" ++
        "}\n";
    const expected =
        "module {\n" ++
        "  ^bb0(%arg0: i32):\n" ++
        "    %K0 = \"arith.constant\"() <{value = 1024 : i32}> : () -> i32\n" ++
        "    %K1 = \"arith.constant\"() <{value = 64 : i32}> : () -> i32\n" ++
        "    %K2 = \"arith.constant\"() <{value = 0 : i32}> : () -> i32\n" ++
        "    %3 = \"foo\"(%K0, %K1, %K2) : (i32, i32, i32) -> i32\n" ++
        "}";

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var nz: Normalizer = try .init(std.testing.allocator);
    defer nz.deinit();

    const got_a = try nz.normalize(arena.allocator(), input_a, .triton_mlir);
    const got_b = try nz.normalize(arena.allocator(), input_b, .triton_mlir);
    try std.testing.expectEqualStrings(expected, got_a);
    try std.testing.expectEqualStrings(expected, got_b);
}

test "renumber constants on custom-syntax mosaic mlir" {
    // Mosaic prints with MLIR's default pretty-printer, which uses
    // custom syntax for arith.constant: `%N = arith.constant V : T`.
    const input_a =
        "func.func @kernel(%arg0: i32) {\n" ++
        "  ^bb0(%arg0: i32):\n" ++
        "    %0 = arith.constant 64 : i32\n" ++
        "    %1 = arith.constant 0 : i32\n" ++
        "    %2 = arith.constant 1024 : i32\n" ++
        "    %3 = arith.muli %2, %0 : i32\n" ++
        "    %4 = arith.addi %3, %1 : i32\n" ++
        "}\n";
    const input_b =
        "func.func @kernel(%arg0: i32) {\n" ++
        "  ^bb0(%arg0: i32):\n" ++
        "    %0 = arith.constant 1024 : i32\n" ++
        "    %1 = arith.constant 64 : i32\n" ++
        "    %2 = arith.constant 0 : i32\n" ++
        "    %3 = arith.muli %0, %1 : i32\n" ++
        "    %4 = arith.addi %3, %2 : i32\n" ++
        "}\n";
    const expected =
        "func.func @kernel(%arg0: i32) {\n" ++
        "  ^bb0(%arg0: i32):\n" ++
        "    %K0 = arith.constant 1024 : i32\n" ++
        "    %K1 = arith.constant 64 : i32\n" ++
        "    %K2 = arith.constant 0 : i32\n" ++
        "    %3 = arith.muli %K0, %K1 : i32\n" ++
        "    %4 = arith.addi %3, %K2 : i32\n" ++
        "}";

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var nz: Normalizer = try .init(std.testing.allocator);
    defer nz.deinit();

    const got_a = try nz.normalize(arena.allocator(), input_a, .mosaic_mlir);
    const got_b = try nz.normalize(arena.allocator(), input_b, .mosaic_mlir);
    try std.testing.expectEqualStrings(expected, got_a);
    try std.testing.expectEqualStrings(expected, got_b);
}

test "% literal stays put when not followed by ident" {
    const input = "a %% b\n%= c";
    const expected = "a %% b\n%= c";

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const got = try normalizeInto(arena.allocator(), input, .ptx);
    try std.testing.expectEqualStrings(expected, got);
}
