const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const TitledBorder = @import("titled_border.zig");
const ProcessInfo = @import("zml-smi-2/contract").ProcessInfo;

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
    arena: std.mem.Allocator,
    processes: []const ProcessInfo,
    device_id: ?u16,
) std.mem.Allocator.Error!void {
    self.scroll_bars.scroll_view.children.builder.userdata = self;
    const rows = try arena.alloc(ProcessInfo, processes.len);
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
    const empty = self.processes.len == 0;
    const rows: u16 = @intCast(@min(self.processes.len, max_visible_rows));
    const body = if (empty)
        try ui.drawRichLine(ctx, &.{.{
            .text = "  No process data available",
            .style = theme.dim_style,
        }}, width)
    else
        try self.scroll_bars.draw(ui.fixedSize(ctx, width, rows));
    const row = header.size.height + @intFromBool(empty);
    try layout.add(row, 0, body);
    const height = row + if (empty) @as(u16, 1) else body.size.height;
    return layout.finish(.{ .width = width, .height = height }, ui.drawWidget(self, drawContent));
}

/// Builds only data rows (no header) for the scroll view.
fn buildRow(pointer: *const anyopaque, index: usize, _: usize) ?vxfw.Widget {
    const self: *const ProcessTable = @ptrCast(@alignCast(pointer));
    if (index >= self.processes.len) return null;
    return ui.drawWidget(&self.processes[index], drawProcessRow);
}

const columns = [_]struct { label: []const u8, format: ?[]const u8 = null }{
    .{ .label = "PID", .format = " {s: >7} " },
    .{ .label = "USER", .format = "{s: <10} " },
    .{ .label = "DEV", .format = "{s: <5} " },
    .{ .label = "UTIL", .format = "{s: >6} " },
    .{ .label = "DEV MEM", .format = "{s: >8} " },
    .{ .label = "CPU", .format = "{s: >6} " },
    .{ .label = "HOST MEM", .format = "{s: >8}  " },
    .{ .label = "Command" },
};

const header_segs = header: {
    var segments: [columns.len]Segment = undefined;
    for (columns, &segments) |column, *segment| segment.* = .{
        .style = theme.label_style,
        .text = if (column.format) |format| std.fmt.comptimePrint(format, .{column.label}) else column.label,
    };
    break :header segments;
};

fn drawProcessRow(info: *const ProcessInfo, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const values = [columns.len]?[]const u8{
        try std.fmt.allocPrint(ctx.arena, "{d}", .{info.pid}),
        info.username[0..@min(info.username.len, 10)],
        try std.fmt.allocPrint(ctx.arena, "{d}", .{info.device_idx}),
        if (info.dev_util_percent) |util| try std.fmt.allocPrint(ctx.arena, "{d}%", .{util}) else null,
        if (info.dev_mem_kib) |memory| try fmtMem(ctx.arena, memory) else null,
        try std.fmt.allocPrint(ctx.arena, "{d}.{d}%", .{ info.cpu_percent / 10, info.cpu_percent % 10 }),
        try fmtMem(ctx.arena, info.rss_kib),
        info.comm,
    };
    const segs = try ctx.arena.alloc(Segment, columns.len);
    inline for (columns, values, segs) |column, value, *segment| {
        const text = value orelse "-";
        segment.* = .{
            .style = .{ .fg = if (value != null) theme.text_primary else theme.dim },
            .text = if (column.format) |format| try std.fmt.allocPrint(ctx.arena, format, .{text}) else text,
        };
    }

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
