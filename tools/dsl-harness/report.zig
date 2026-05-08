//! Output formatting for the `compare` and `list` subcommands. Domain
//! structs are stringified directly via `std.json.Stringify` reflection.

const std = @import("std");

const Allocator = std.mem.Allocator;

const harness = @import("harness.zig");

pub const Format = enum { human, json };

pub const StageVerdict = enum {
    match,
    diff,
    missing_zig,
    missing_py,
    /// Filtered out by `--stage=...`; rendering skips these.
    skipped,
};

pub const StageRow = struct {
    name: []const u8,
    verdict: StageVerdict,
    added: u32 = 0,
    removed: u32 = 0,
    /// Empty unless `--show-diffs` is set.
    diff_blob: []const u8 = "",
    /// True for stages that XLA's lowering canonicalizes downstream
    /// (TTIR/TTGIR), so a diff there alone collapses to
    /// `codegen_equivalent` when the post-XLA stages still match.
    is_cosmetic: bool = false,
};

/// `codegen_equivalent` = LLIR+PTX matched but TTIR/TTGIR diverged
/// (XLA's lowering canonicalizes the cosmetic divergence away).
pub const KernelClassification = enum {
    all_match,
    codegen_equivalent,
    diff,
    pipeline_error,
};

pub const KernelRow = struct {
    kernel: []const u8,
    sweep: []const u8,
    kind: harness.KernelKind,
    classification: KernelClassification,
    /// Populated only for `pipeline_error`.
    error_reason: []const u8 = "",
    stages: []const StageRow = &.{},
};

/// Deep-copies `stages` and `diff_blob` into `arena` so the row can
/// outlive its source per-sweep arena. `kernel`/`sweep`/`name`/
/// `error_reason` are static (comptime strings or `@errorName`) — no
/// copy needed.
pub fn cloneKernelRow(arena: Allocator, src: KernelRow) !KernelRow {
    var dst = src;
    if (src.stages.len > 0) {
        const stages = try arena.alloc(StageRow, src.stages.len);
        for (src.stages, 0..) |s, i| {
            stages[i] = s;
            if (s.diff_blob.len > 0) stages[i].diff_blob = try arena.dupe(u8, s.diff_blob);
        }
        dst.stages = stages;
    }
    return dst;
}

pub fn errorRow(entry: *const harness.KernelEntry, sweep: []const u8, reason: []const u8) KernelRow {
    return .{
        .kernel = entry.name,
        .sweep = sweep,
        .kind = entry.kind,
        .classification = .pipeline_error,
        .error_reason = reason,
    };
}

pub const Aggregate = struct {
    all_match: u32 = 0,
    codegen_equivalent: u32 = 0,
    diff: u32 = 0,
    pipeline_error: u32 = 0,

    pub fn add(self: *Aggregate, c: KernelClassification) void {
        switch (c) {
            .all_match => self.all_match += 1,
            .codegen_equivalent => self.codegen_equivalent += 1,
            .diff => self.diff += 1,
            .pipeline_error => self.pipeline_error += 1,
        }
    }

    pub fn anyFailures(self: Aggregate) bool {
        return self.diff > 0 or self.pipeline_error > 0;
    }
};

pub fn classify(stages: []const StageRow) KernelClassification {
    var any_diff = false;
    var any_critical_diff = false;
    var any_match = false;
    for (stages) |row| {
        switch (row.verdict) {
            .match => any_match = true,
            .diff => {
                any_diff = true;
                if (!row.is_cosmetic) any_critical_diff = true;
            },
            .missing_zig, .missing_py => return .pipeline_error,
            .skipped => {},
        }
    }
    if (!any_match and !any_diff) return .pipeline_error;
    if (!any_diff) return .all_match;
    if (any_critical_diff) return .diff;
    return .codegen_equivalent;
}

pub const Colors = struct {
    use: bool,

    pub fn init(stream: std.Io.File, io: std.Io) Colors {
        const is_tty = stream.isTty(io) catch false;
        if (!is_tty) return .{ .use = false };
        const term_c = std.c.getenv("TERM") orelse return .{ .use = false };
        const term = std.mem.span(term_c);
        if (std.mem.eql(u8, term, "dumb")) return .{ .use = false };
        return .{ .use = true };
    }

    pub fn fmt(self: Colors, code: []const u8) []const u8 {
        return if (self.use) code else "";
    }

    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";
    pub const green = "\x1b[32m";
    pub const red = "\x1b[31m";
    pub const yellow = "\x1b[33m";
    pub const cyan = "\x1b[36m";
    pub const gray = "\x1b[90m";
};

pub fn renderKernelHuman(
    w: *std.Io.Writer,
    c: Colors,
    row: KernelRow,
    show_diffs: bool,
) !void {
    const header_color = switch (row.classification) {
        .all_match => c.fmt(Colors.green),
        .codegen_equivalent => c.fmt(Colors.yellow),
        .diff => c.fmt(Colors.red),
        .pipeline_error => c.fmt(Colors.yellow),
    };
    try w.print("\n{s}{s}{s} :: {s} [{s}]{s}\n", .{
        c.fmt(Colors.bold),
        header_color,
        row.kernel,
        row.sweep,
        @tagName(row.kind),
        c.fmt(Colors.reset),
    });
    if (row.classification == .pipeline_error and row.error_reason.len > 0) {
        try w.print("  {s}{s}{s}\n", .{ c.fmt(Colors.yellow), row.error_reason, c.fmt(Colors.reset) });
        return;
    }
    for (row.stages) |s| {
        try renderStageHuman(w, c, s);
        if (show_diffs and s.diff_blob.len > 0) try writeColoredDiffBody(w, c, s.diff_blob);
    }
}

fn writeColoredDiffBody(w: *std.Io.Writer, c: Colors, body: []const u8) !void {
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line| {
        const color = pickDiffColor(c, line);
        if (color.len > 0) {
            try w.print("{s}{s}{s}\n", .{ color, line, c.fmt(Colors.reset) });
        } else {
            try w.print("{s}\n", .{line});
        }
    }
    // Force the terminal back to default state and add a trailing blank
    // line. Without this, std.Progress's next redraw can inherit color
    // from the last colored line (mid-buffer flush leaves no boundary
    // between our output and progress's), and the next stage row sticks
    // to the diff body with no visual gap.
    try w.print("{s}\n", .{c.fmt(Colors.reset)});
}

fn pickDiffColor(c: Colors, line: []const u8) []const u8 {
    if (line.len == 0) return "";
    if (std.mem.startsWith(u8, line, "+++") or std.mem.startsWith(u8, line, "---")) {
        return c.fmt(Colors.bold);
    }
    return switch (line[0]) {
        '+' => c.fmt(Colors.green),
        '-' => c.fmt(Colors.red),
        '@' => c.fmt(Colors.cyan),
        else => "",
    };
}

fn renderStageHuman(w: *std.Io.Writer, c: Colors, s: StageRow) !void {
    switch (s.verdict) {
        .skipped => return,
        .match => try w.print("  {s:<5} {s}✓{s}\n", .{ s.name, c.fmt(Colors.green), c.fmt(Colors.reset) }),
        .diff => try w.print("  {s:<5} {s}+{d}/-{d}{s}\n", .{ s.name, c.fmt(Colors.red), s.added, s.removed, c.fmt(Colors.reset) }),
        .missing_zig => try w.print("  {s:<5} {s}missing-zig{s}\n", .{ s.name, c.fmt(Colors.yellow), c.fmt(Colors.reset) }),
        .missing_py => try w.print("  {s:<5} {s}missing-py{s}\n", .{ s.name, c.fmt(Colors.yellow), c.fmt(Colors.reset) }),
    }
}

pub fn renderSummaryHuman(w: *std.Io.Writer, c: Colors, agg: Aggregate) !void {
    try w.writeAll("\n");
    var first = true;
    if (agg.all_match > 0) {
        try writePart(w, c, &first, Colors.green, "{d} matching", .{agg.all_match});
    }
    if (agg.codegen_equivalent > 0) {
        try writePart(w, c, &first, Colors.yellow, "{d} codegen-equivalent", .{agg.codegen_equivalent});
    }
    if (agg.diff > 0) {
        try writePart(w, c, &first, Colors.red, "{d} differing", .{agg.diff});
    }
    if (agg.pipeline_error > 0) {
        try writePart(w, c, &first, Colors.yellow, "{d} pipeline-error", .{agg.pipeline_error});
    }
    if (first) {
        try w.writeAll("(no kernels compared)");
    }
    try w.writeAll("\n");
}

fn writePart(w: *std.Io.Writer, c: Colors, first: *bool, color: []const u8, comptime fmt: []const u8, args: anytype) !void {
    if (!first.*) try w.writeAll("  ");
    first.* = false;
    try w.writeAll(c.fmt(color));
    try w.print(fmt, args);
    try w.writeAll(c.fmt(Colors.reset));
}

// JSON output is newline-delimited so consumers can stream row-by-row.

pub fn renderKernelJson(w: *std.Io.Writer, row: KernelRow) !void {
    var jw: std.json.Stringify = .{ .writer = w };
    try jw.write(row);
    try w.writeAll("\n");
}

pub fn renderSummaryJson(w: *std.Io.Writer, agg: Aggregate) !void {
    var jw: std.json.Stringify = .{ .writer = w };
    try jw.write(agg);
    try w.writeAll("\n");
}

// `KernelEntry` carries function pointers that can't auto-serialize;
// `list` mirrors it through this flat shape first.

const ListEntry = struct {
    name: []const u8,
    sweeps: []const []const u8,
};

const KernelList = struct {
    triton: []const ListEntry,
    mosaic_tpu: []const ListEntry,
};

pub fn renderListJson(
    arena: Allocator,
    w: *std.Io.Writer,
    triton: []const *const harness.KernelEntry,
    mosaic_tpu: []const *const harness.KernelEntry,
) !void {
    const t_entries = try sweepRefsToList(arena, triton);
    const m_entries = try sweepRefsToList(arena, mosaic_tpu);
    var jw: std.json.Stringify = .{ .writer = w };
    try jw.write(KernelList{ .triton = t_entries, .mosaic_tpu = m_entries });
    try w.writeAll("\n");
}

fn sweepRefsToList(arena: Allocator, entries: []const *const harness.KernelEntry) ![]const ListEntry {
    var out = try arena.alloc(ListEntry, entries.len);
    for (entries, 0..) |e, i| {
        var sweeps = try arena.alloc([]const u8, e.sweeps.len);
        for (e.sweeps, 0..) |s, j| sweeps[j] = s.label;
        out[i] = .{ .name = e.name, .sweeps = sweeps };
    }
    return out;
}

pub fn renderListHuman(
    w: *std.Io.Writer,
    c: Colors,
    triton: []const *const harness.KernelEntry,
    mosaic_tpu: []const *const harness.KernelEntry,
) !void {
    try w.print("{s}Triton kernels{s} ({d}):\n", .{ c.fmt(Colors.bold), c.fmt(Colors.reset), triton.len });
    for (triton) |e| try renderListEntry(w, c, e);
    try w.print("\n{s}Mosaic-TPU kernels{s} ({d}):\n", .{ c.fmt(Colors.bold), c.fmt(Colors.reset), mosaic_tpu.len });
    for (mosaic_tpu) |e| try renderListEntry(w, c, e);
}

fn renderListEntry(w: *std.Io.Writer, c: Colors, e: *const harness.KernelEntry) !void {
    try w.print("  {s}\n", .{e.name});
    for (e.sweeps) |s| try w.print("    {s}-{s} {s}\n", .{ c.fmt(Colors.dim), c.fmt(Colors.reset), s.label });
}
