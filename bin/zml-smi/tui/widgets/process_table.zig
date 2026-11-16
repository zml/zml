const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const TitledBorder = @import("titled_border.zig");
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
processes: []ProcessInfo = &.{},

/// Selects and sorts rows whose strings are owned by the captured frame.
pub fn prepare(
    self: *ProcessTable,
    frame: std.mem.Allocator,
    state: *const data.SystemState,
    device_id: ?u16,
) std.mem.Allocator.Error!void {
    self.scroll_bars.scroll_view.children.builder.userdata = self;
    const processes = state.processes();
    const rows = try frame.alloc(ProcessInfo, processes.len);
    var kept: usize = 0;
    for (processes) |entry| {
        if (device_id != null and entry.device_idx != device_id.?) continue;
        rows[kept] = entry;
        kept += 1;
    }

    // Sort by dev_mem descending
    std.sort.insertion(ProcessInfo, rows[0..kept], {}, struct {
        fn cmp(_: void, a: ProcessInfo, b: ProcessInfo) bool {
            return (a.dev_mem_kib orelse 0) > (b.dev_mem_kib orelse 0);
        }
    }.cmp);

    self.processes = rows[0..kept];
    const count: u32 = @intCast(kept);
    self.scroll_bars.estimated_content_height = count;
    self.scroll_bars.scroll_view.item_count = count;
}

pub fn resetScroll(self: *ProcessTable) void {
    self.scroll_bars.scroll_view.scroll = .{};
    self.scroll_bars.scroll_view.cursor = 0;
}

pub fn draw(self: *ProcessTable, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const border: TitledBorder = .{
        .child = ui.drawWidget(self, drawContent),
        .title = "Processes",
    };
    return border.draw(ui.fixedWidth(ctx, ctx.max.width orelse 80));
}

fn drawContent(self: *ProcessTable, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const width = ctx.max.width orelse 80;
    const header = try ui.drawRichLine(ctx, &header_segs, width);
    var layout = compose.surfaceBuilder(ctx.arena);
    try layout.add(0, 0, header);
    var height = header.size.height;

    if (self.processes.len == 0) {
        const empty: []const Segment = &.{.{
            .text = "  No process data available",
            .style = theme.dim_style,
        }};
        try layout.add(height + 1, 0, try ui.drawRichLine(ctx, empty, width));
        height += 2;
    } else {
        const rows: u16 = @intCast(@min(self.processes.len, max_visible_rows));
        const body = try self.scroll_bars.draw(ui.fixedSize(ctx, width, rows));
        try layout.add(height, 0, body);
        height += body.size.height;
    }
    return layout.finish(.{ .width = width, .height = height }, ui.drawWidget(self, drawContent));
}

/// Builds only data rows (no header) for the scroll view.
fn buildRow(ptr: *const anyopaque, idx: usize, _: usize) ?vxfw.Widget {
    const self: *ProcessTable = @ptrCast(@alignCast(@constCast(ptr)));
    if (idx >= self.processes.len) {
        return null;
    }
    return .{ .userdata = @ptrCast(&self.processes[idx]), .drawFn = drawProcessRow };
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
    if (kib < 1024) return std.fmt.allocPrint(arena, "{d}K", .{kib});
    const gib = kib >= 1024 * 1024;
    const unit: u64 = if (gib) 1024 * 1024 else 1024;
    return std.fmt.allocPrint(arena, "{d}.{d}{c}", .{ kib / unit, kib % unit * 10 / unit, @as(u8, if (gib) 'G' else 'M') });
}
