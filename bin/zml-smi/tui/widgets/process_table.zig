const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const TitledBorder = @import("titled_border.zig");
const ColumnLayout = @import("column_layout.zig");
const Selection = @import("../selection.zig").Selection;
const pi = @import("zml-smi/info").process_info;
const ProcessInfo = pi.ProcessInfo;

const RichText = vxfw.RichText;
const Segment = vaxis.Cell.Segment;

const ProcessTable = @This();

const max_visible_rows: u16 = 20;

scroll_bars: vxfw.ScrollBars = .{
    .scroll_view = .{
        .children = .{ .builder = .{ .userdata = undefined, .buildFn = buildRow } },
        .draw_cursor = false,
    },
    .draw_horizontal_scrollbar = false,
},
merged: std.ArrayList(ProcessInfo) = .empty,
rows: std.ArrayList(Row) = .empty,
selection: *Selection = undefined,

last_click_at: std.Io.Timestamp = .zero,
last_click_idx: ?u32 = null,

io: ?std.Io = null,

cmd_scroll: u16 = 0,
cmd_scroll_pid: u32 = 0,
cmd_scroll_max: u16 = 0,

/// Highlight background for the selected row.
const sel_bg: vaxis.Color = .{ .rgb = .{ 45, 55, 80 } };

const Row = struct {
    table: *ProcessTable,
    idx: u32,

    pub fn handleEvent(self: *Row, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
        const table = self.table;

        switch (event) {
            .mouse => |mouse| {
                if (mouse.type == .press and mouse.button == .left) {
                    if (self.idx >= table.merged.items.len) {
                        return;
                    }

                    table.scroll_bars.scroll_view.cursor = self.idx;
                    _ = table.selection.setProcess(table.merged.items[self.idx].pid);

                    // Double-click — the same row clicked again within 400ms — kills.
                    var dbl = false;
                    if (table.io) |io| {
                        const now = std.Io.Timestamp.now(io, .awake);
                        dbl = table.last_click_idx == self.idx and
                            table.last_click_at.durationTo(now).toMilliseconds() < 400;
                        table.last_click_at = now;
                    }
                    table.last_click_idx = self.idx;

                    if (dbl) {
                        table.killSelected();
                    }

                    return ctx.consumeAndRedraw();
                }
            },
            else => {},
        }
    }

    pub fn draw(self: *Row, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
        const table = self.table;
        const vw: u16 = ctx.min.width;
        if (self.idx >= table.merged.items.len) {
            var empty_surf = try (RichText{ .text = &.{}, .softwrap = false, .overflow = .clip }).draw(ui.fixedSize(ctx, vw, 1));
            empty_surf.widget = ui.widget(self);

            return empty_surf;
        }
        const info = &table.merged.items[self.idx];
        const selected = table.selection.processEq(info.pid);
        const row_bg: vaxis.Color = if (selected) sel_bg else .default;
        const val: vaxis.Style = .{ .fg = theme.text_primary, .bg = row_bg };
        const dim: vaxis.Style = .{ .fg = theme.dim, .bg = row_bg };

        const segs = try ctx.arena.alloc(Segment, 8);
        segs[0] = .{ .style = val, .text = try col(ctx.arena, col_fmt.pid, "{d}", .{info.pid}) };
        segs[1] = .{ .style = val, .text = try col(ctx.arena, col_fmt.user, "{s}", .{info.username[0..@min(info.username.len, 10)]}) };
        segs[2] = .{ .style = val, .text = try col(ctx.arena, col_fmt.dev, "{d}", .{info.device_idx}) };
        segs[3] = if (info.dev_util_percent) |util|
            .{ .style = val, .text = try col(ctx.arena, col_fmt.util, "{d}%", .{util}) }
        else
            .{ .style = dim, .text = try col(ctx.arena, col_fmt.util, "-", .{}) };
        segs[4] = if (info.dev_mem_kib) |mem|
            .{ .style = val, .text = try col(ctx.arena, col_fmt.mem, "{s}", .{try fmtMem(ctx.arena, mem)}) }
        else
            .{ .style = dim, .text = try col(ctx.arena, col_fmt.mem, "-", .{}) };
        segs[5] = .{ .style = val, .text = try col(ctx.arena, col_fmt.cpu, "{d}.{d}%", .{ info.cpu_percent / 10, info.cpu_percent % 10 }) };
        segs[6] = .{ .style = val, .text = try col(ctx.arena, col_fmt.host_mem, "{s}", .{try fmtMem(ctx.arena, info.rss_kib)}) };
        segs[7] = .{ .style = val, .text = info.comm };

        const rich: RichText = .{ .text = segs, .softwrap = false, .overflow = .clip, .base_style = .{ .bg = row_bg } };

        if (!selected) { // make it clipped
            var surf = try rich.draw(ui.fixedSize(ctx, vw, 1));
            surf.widget = ui.widget(self);
            return surf;
        }

        const full = try rich.draw(ctx.withConstraints(
            .{ .width = vw, .height = 1 },
            .{ .width = null, .height = 1 },
        ));
        table.cmd_scroll_max = full.size.width -| vw;
        const off: u16 = @min(table.cmd_scroll, table.cmd_scroll_max);

        var sb = compose.surfaceBuilder(ctx.arena);
        try sb.add(0, -@as(i17, off), full);
        return sb.finish(.{ .width = vw, .height = 1 }, ui.widget(self));
    }
};

pub fn deinit(self: *ProcessTable, allocator: std.mem.Allocator) void {
    self.merged.deinit(allocator);
    self.rows.deinit(allocator);
}

pub fn rowIndexOfPid(self: *const ProcessTable, pid: u32) ?usize {
    for (self.merged.items, 0..) |p, i| {
        if (p.pid == pid) return i;
    }

    return null;
}

pub fn killableSelected(self: *const ProcessTable) bool {
    if (self.selection.kind != .process) {
        return false;
    }

    const idx = self.rowIndexOfPid(self.selection.pid) orelse return false;
    const p = self.merged.items[idx];

    return !p.remote and p.pid != 0;
}

pub fn killSelected(self: *ProcessTable) void {
    if (!self.killableSelected()) {
        return;
    }

    const idx = self.rowIndexOfPid(self.selection.pid).?;
    std.posix.kill(@intCast(self.merged.items[idx].pid), std.posix.SIG.KILL) catch {};
}

pub fn scrollCommand(self: *ProcessTable, dir: i8) bool {
    if (self.selection.kind != .process or self.cmd_scroll_max == 0) {
        return false;
    }

    const step: u16 = 2;
    const old = self.cmd_scroll;
    if (dir > 0) {
        self.cmd_scroll = @min(self.cmd_scroll + step, self.cmd_scroll_max);
    } else {
        self.cmd_scroll -|= step;
    }

    return self.cmd_scroll != old;
}

pub fn prepare(self: *ProcessTable, state: *data.SystemState, device_id: ?u16) void {
    self.scroll_bars.scroll_view.children.builder.userdata = self;
    self.io = state.io;
    self.merged.clearRetainingCapacity();

    for (state.process_lists) |pl| {
        const pl_items = pl.front().items;
        if (device_id) |did| {
            for (pl_items) |entry| {
                if (entry.device_idx != did) {
                    continue;
                }
                self.merged.append(state.gpa, entry) catch break;
            }
        } else {
            self.merged.appendSlice(state.gpa, pl_items) catch break;
        }
    }

    // Enrich with host data (uid, username, cpu%, rss, cmdline)
    state.enricher.enrich(state.io, self.merged.items);

    // Sort by dev_mem descending
    std.sort.insertion(ProcessInfo, self.merged.items, {}, struct {
        fn cmp(_: void, a: ProcessInfo, b: ProcessInfo) bool {
            return (a.dev_mem_kib orelse 0) > (b.dev_mem_kib orelse 0);
        }
    }.cmp);

    const count: u32 = @intCast(self.merged.items.len);
    self.scroll_bars.estimated_content_height = count;
    self.scroll_bars.scroll_view.item_count = count;

    if (self.rows.items.len != self.merged.items.len) {
        self.rows.clearRetainingCapacity();
        for (0..self.merged.items.len) |i| {
            self.rows.append(state.gpa, .{ .table = self, .idx = @intCast(i) }) catch break;
        }
    }

    if (self.selection.kind == .process) {
        if (self.selection.pid != self.cmd_scroll_pid) {
            self.cmd_scroll = 0;
            self.cmd_scroll_pid = self.selection.pid;
        }
    } else {
        self.cmd_scroll = 0;
        self.cmd_scroll_pid = 0;
    }
}

pub fn resetScroll(self: *ProcessTable) void {
    self.scroll_bars.scroll_view.scroll = .{};
    self.scroll_bars.scroll_view.cursor = 0;
}

pub fn draw(self: *ProcessTable, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const row_count = self.merged.items.len;

    var header_rt: RichText = .{ .text = &header_segs, .softwrap = false, .overflow = .clip };
    const empty_rt: RichText = .{
        .text = &.{.{ .text = "  No process data available", .style = theme.dim_style }},
        .softwrap = false,
        .overflow = .clip,
    };

    var children: std.ArrayList(vxfw.Widget) = .empty;
    try children.append(ctx.arena, header_rt.widget());

    if (row_count == 0) {
        try children.append(ctx.arena, try compose.sized(
            ctx.arena,
            try compose.pad(ctx.arena, empty_rt.widget(), .{ .top = 1 }),
            .{ .width = w, .height = 2 },
        ));
    } else {
        const scroll_h: u16 = @min(@as(u16, @intCast(row_count)), max_visible_rows);
        try children.append(ctx.arena, try compose.sized(ctx.arena, self.scroll_bars.widget(), .{ .width = w, .height = scroll_h }));
    }

    const content: ColumnLayout = .{ .children = children.items };
    const tb: TitledBorder = .{
        .child = ui.widget(&content),
        .title = "Processes",
    };
    return tb.draw(ui.fixedWidth(ctx, w));
}

/// Builds only data rows (no header) for the scroll view.
fn buildRow(ptr: *const anyopaque, idx: usize, _: usize) ?vxfw.Widget {
    const self: *ProcessTable = @ptrCast(@alignCast(@constCast(ptr)));
    if (idx >= self.merged.items.len or idx >= self.rows.items.len) {
        return null;
    }
    return ui.widget(&self.rows.items[idx]);
}

// Column format strings — shared between header (comptime) and row (runtime).
const col_fmt = struct {
    const pid = " {s: >7} ";
    const user = "{s: <10} ";
    const dev = "{s: <5} ";
    const util = "{s: >6} ";
    const mem = "{s: >8} ";
    const cpu = "{s: >6} ";
    const host_mem = "{s: >8}  ";
};

const header_segs = blk: {
    const s = theme.label_style;
    break :blk [_]Segment{
        .{ .style = s, .text = std.fmt.comptimePrint(col_fmt.pid, .{"PID"}) },
        .{ .style = s, .text = std.fmt.comptimePrint(col_fmt.user, .{"USER"}) },
        .{ .style = s, .text = std.fmt.comptimePrint(col_fmt.dev, .{"DEV"}) },
        .{ .style = s, .text = std.fmt.comptimePrint(col_fmt.util, .{"UTIL"}) },
        .{ .style = s, .text = std.fmt.comptimePrint(col_fmt.mem, .{"DEV MEM"}) },
        .{ .style = s, .text = std.fmt.comptimePrint(col_fmt.cpu, .{"CPU"}) },
        .{ .style = s, .text = std.fmt.comptimePrint(col_fmt.host_mem, .{"HOST MEM"}) },
        .{ .style = s, .text = "Command" },
    };
};

/// Format a value into a column: col(arena, col_fmt.pid, "{d}", .{pid})
fn col(arena: std.mem.Allocator, comptime column: []const u8, comptime fmt: []const u8, args: anytype) std.mem.Allocator.Error![]const u8 {
    return std.fmt.allocPrint(arena, column, .{try std.fmt.allocPrint(arena, fmt, args)});
}

fn fmtMem(arena: std.mem.Allocator, kib: u64) std.mem.Allocator.Error![]const u8 {
    if (kib >= 1024 * 1024) {
        const whole = kib / (1024 * 1024);
        const frac = (kib % (1024 * 1024)) * 10 / (1024 * 1024);
        return std.fmt.allocPrint(arena, "{d}.{d}G", .{ whole, frac });
    } else if (kib >= 1024) {
        const whole = kib / 1024;
        const frac = (kib % 1024) * 10 / 1024;
        return std.fmt.allocPrint(arena, "{d}.{d}M", .{ whole, frac });
    } else {
        return std.fmt.allocPrint(arena, "{d}K", .{kib});
    }
}
