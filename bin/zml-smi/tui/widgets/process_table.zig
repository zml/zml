const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const TitledBorder = @import("titled_border.zig");
const ColumnLayout = @import("column_layout.zig");
const ProcessInfo = @import("zml-smi").ProcessInfo;

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
processes: std.ArrayList(ProcessInfo) = .empty,

pub fn deinit(self: *ProcessTable, allocator: std.mem.Allocator) void {
    self.processes.deinit(allocator);
}

pub fn prepare(self: *ProcessTable, state: *data.SystemState, device_id: ?u16, frame: std.mem.Allocator) void {
    self.scroll_bars.scroll_view.children.builder.userdata = self;
    self.processes.clearRetainingCapacity();

    const merged = state.collector.processes.lock(state.io);
    defer state.collector.processes.unlock(state.io);
    for (merged.rows.items) |entry| {
        if (device_id) |did| {
            if (entry.device_idx != did) {
                continue;
            }
        }
        var row = entry;
        row.username = frame.dupe(u8, entry.username) catch "";
        row.comm = frame.dupe(u8, entry.comm) catch "";
        self.processes.append(state.gpa, row) catch break;
    }

    // Sort by dev_mem descending
    std.sort.insertion(ProcessInfo, self.processes.items, {}, struct {
        fn cmp(_: void, a: ProcessInfo, b: ProcessInfo) bool {
            return (a.dev_mem_kib orelse 0) > (b.dev_mem_kib orelse 0);
        }
    }.cmp);

    const count: u32 = @intCast(self.processes.items.len);
    self.scroll_bars.estimated_content_height = count;
    self.scroll_bars.scroll_view.item_count = count;

    if (count == 0) {
        self.resetScroll();
    } else {
        const max_top: u32 = count -| max_visible_rows;
        if (self.scroll_bars.scroll_view.scroll.top > max_top) {
            self.scroll_bars.scroll_view.scroll.top = max_top;
            self.scroll_bars.scroll_view.scroll.vertical_offset = 0;
            self.scroll_bars.scroll_view.scroll.pending_lines = 0;
        }
        self.scroll_bars.scroll_view.cursor = @min(self.scroll_bars.scroll_view.cursor, count - 1);
    }
}

pub fn resetScroll(self: *ProcessTable) void {
    self.scroll_bars.scroll_view.scroll = .{};
    self.scroll_bars.scroll_view.cursor = 0;
    self.scroll_bars.is_dragging_vertical_thumb = false;
}

pub fn draw(self: *ProcessTable, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const row_count = self.processes.items.len;

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
        const scroll_h: u16 = @intCast(@min(row_count, max_visible_rows));
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
    if (idx >= self.processes.items.len) {
        return null;
    }
    return .{ .userdata = @ptrCast(&self.processes.items[idx]), .drawFn = drawProcessRow };
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

fn drawProcessRow(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const info: *const ProcessInfo = @ptrCast(@alignCast(ptr));
    const val: vaxis.Style = .{ .fg = theme.text_primary };
    const dim: vaxis.Style = .{ .fg = theme.dim };

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

    const rich: RichText = .{ .text = segs, .softwrap = false, .overflow = .clip };
    return rich.draw(ctx.withConstraints(
        .{ .width = ctx.min.width, .height = 1 },
        .{ .width = ctx.max.width, .height = 1 },
    ));
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

test "ProcessTable.prepare clamps scroll offset when list shrinks" {
    const smi = @import("zml-smi");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var collector: smi.Collector = .init(arena.allocator(), std.testing.allocator, std.testing.io, .{});
    defer collector.deinit();

    var state: data.SystemState = try .init(.{
        .collector = &collector,
        .tui_refresh_rate = 100,
        .gpa = std.testing.allocator,
        .arena = arena.allocator(),
        .io = std.testing.io,
    });
    defer state.deinit();

    var table: ProcessTable = .{};
    defer table.deinit(std.testing.allocator);

    for (0..25) |i| {
        try collector.processes.back().append(.{ .pid = @intCast(i + 1) });
    }
    collector.processes.swap(std.testing.io);
    table.prepare(&state, null, arena.allocator());
    table.scroll_bars.scroll_view.scroll.top = 15;
    table.scroll_bars.scroll_view.scroll.vertical_offset = 1;
    table.scroll_bars.scroll_view.scroll.pending_lines = -2;
    table.scroll_bars.scroll_view.cursor = 24;

    collector.processes.back().clear();
    try collector.processes.back().append(.{ .pid = 1 });
    collector.processes.swap(std.testing.io);
    table.prepare(&state, null, arena.allocator());
    try std.testing.expectEqual(@as(u32, 0), table.scroll_bars.scroll_view.scroll.top);
    try std.testing.expectEqual(@as(i17, 0), table.scroll_bars.scroll_view.scroll.vertical_offset);
    try std.testing.expectEqual(@as(i17, 0), table.scroll_bars.scroll_view.scroll.pending_lines);
    try std.testing.expectEqual(@as(u32, 0), table.scroll_bars.scroll_view.cursor);
}
