const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const theme = @import("../theme.zig");
const TitledBorder = @import("components/titled_border.zig");
const ColumnLayout = @import("components/column_layout.zig");
const pi = @import("../../info/process_info.zig");
const ProcessInfo = pi.ProcessInfo;

const RichText = vxfw.RichText;
const Segment = vaxis.Cell.Segment;

const ProcessTable = @This();

const max_expanded_rows: u16 = 20;

scroll_bars: vxfw.ScrollBars,
row_count: u32 = 0,
collapsed: bool = true,
collapsed_visible: u32 = 3,
rows: [pi.max_processes]ProcessRow = [_]ProcessRow{.{}} ** pi.max_processes,

pub fn init(self: *ProcessTable) void {
    self.scroll_bars = .{
        .scroll_view = .{
            .children = .{
                .builder = .{
                    .userdata = self,
                    .buildFn = buildRow,
                },
            },
            .draw_cursor = false,
        },
        .draw_horizontal_scrollbar = false,
    };
}

pub fn prepare(self: *ProcessTable, state: *const data.SystemState, device_id: ?u8) void {
    if (state.getProcesses()) |pl| {
        if (device_id) |did| {
            // Detail view: only show processes associated with this device
            var n: u32 = 0;
            for (0..pl.count) |i| {
                if (pl.entries[i].device_idx) |d| {
                    if (d == did) {
                        self.rows[n].info = pl.entries[i];
                        n += 1;
                    }
                }
            }
            self.row_count = n;
            // Sort by GPU memory descending
            std.sort.insertion(ProcessRow, self.rows[0..n], {}, struct {
                fn cmp(_: void, a: ProcessRow, b: ProcessRow) bool {
                    return (a.info.gpu_mem_kib orelse 0) > (b.info.gpu_mem_kib orelse 0);
                }
            }.cmp);
        } else {
            // Overview: all processes, GPU processes pinned at top
            self.row_count = pl.count;
            for (0..pl.count) |i| {
                self.rows[i].info = pl.entries[i];
            }
            std.sort.insertion(ProcessRow, self.rows[0..pl.count], {}, struct {
                fn cmp(_: void, a: ProcessRow, b: ProcessRow) bool {
                    const a_gpu: u1 = if (a.info.device_idx != null) 1 else 0;
                    const b_gpu: u1 = if (b.info.device_idx != null) 1 else 0;
                    if (a_gpu != b_gpu) return a_gpu > b_gpu;
                    if (a_gpu == 1) return (a.info.gpu_mem_kib orelse 0) > (b.info.gpu_mem_kib orelse 0);
                    return false;
                }
            }.cmp);
        }
    } else {
        self.row_count = 0;
    }

    self.scroll_bars.estimated_content_height = self.row_count;
    self.scroll_bars.scroll_view.item_count = self.row_count;
}

pub fn toggleCollapsed(self: *ProcessTable) void {
    self.collapsed = !self.collapsed;
}

pub fn resetScroll(self: *ProcessTable) void {
    self.scroll_bars.scroll_view.scroll = .{};
    self.scroll_bars.scroll_view.cursor = 0;
}

pub fn widget(self: *ProcessTable) vxfw.Widget {
    return .{
        .userdata = self,
        .drawFn = typeErasedDrawFn,
    };
}

fn typeErasedDrawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const self: *ProcessTable = @ptrCast(@alignCast(ptr));
    return self.draw(ctx);
}

pub fn draw(self: *ProcessTable, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const show_footer = self.collapsed and self.row_count > self.collapsed_visible;

    var header_rt = try headerRichText(ctx.arena);
    const empty_rt: RichText = .{
        .text = &.{.{ .text = "  No process data available", .style = theme.dim_style }},
        .softwrap = false,
        .overflow = .clip,
    };
    const footer_text = if (show_footer)
        try std.fmt.allocPrint(ctx.arena, "  + {d} more — press 'p' to expand", .{self.row_count - self.collapsed_visible})
    else
        "";
    var footer_rt: RichText = .{
        .text = if (show_footer) try ctx.arena.dupe(Segment, &.{.{ .text = footer_text, .style = theme.dim_style }}) else &.{},
        .softwrap = false,
        .overflow = .clip,
    };
    var spacer_rt: RichText = .{ .text = &.{.{ .text = " " }}, .softwrap = false };
    const scroll_h: u16 = @min(@as(u16, @intCast(self.row_count)), max_expanded_rows);
    const scroll_box: vxfw.SizedBox = .{
        .child = self.scroll_bars.widget(),
        .size = .{ .width = w, .height = scroll_h },
    };

    // Build children list, conditionally picking from the stack widgets above.
    var children: std.ArrayList(vxfw.Widget) = .empty;
    try children.append(ctx.arena, header_rt.widget());

    if (self.row_count == 0) {
        try children.append(ctx.arena, empty_rt.widget());
    } else if (self.collapsed) {
        const visible = @min(self.row_count, self.collapsed_visible);
        for (0..visible) |i| {
            try children.append(ctx.arena, self.rows[i].widget());
        }
        if (show_footer) {
            try children.append(ctx.arena, spacer_rt.widget());
            try children.append(ctx.arena, footer_rt.widget());
        }
    } else {
        try children.append(ctx.arena, scroll_box.widget());
    }

    const content: ColumnLayout = .{ .children = children.items };
    const content_h: u16 = @intCast(children.items.len);
    const tb: TitledBorder = .{
        .child = content.widget(),
        .title = "Processes",
    };
    return tb.draw(ctx.withConstraints(
        .{ .width = w },
        .{ .width = w, .height = content_h + max_expanded_rows + 4 },
    ));
}

/// Builds only data rows (no header) for the scroll view in expanded mode.
fn buildRow(ptr: *const anyopaque, idx: usize, _: usize) ?vxfw.Widget {
    const self: *ProcessTable = @ptrCast(@alignCast(@constCast(ptr)));
    if (idx >= self.row_count) return null;
    return self.rows[idx].widget();
}

fn headerRichText(arena: std.mem.Allocator) !RichText {
    const s = theme.label_style;
    const segs = try arena.alloc(Segment, 8);
    segs[0] = .{ .style = s, .text = try std.fmt.allocPrint(arena, " {s: >7} ", .{"PID"}) };
    segs[1] = .{ .style = s, .text = try std.fmt.allocPrint(arena, "{s: <10} ", .{"USER"}) };
    segs[2] = .{ .style = s, .text = try std.fmt.allocPrint(arena, "{s: <5} ", .{"DEV"}) };
    segs[3] = .{ .style = s, .text = try std.fmt.allocPrint(arena, "{s: >6} ", .{"UTIL"}) };
    segs[4] = .{ .style = s, .text = try std.fmt.allocPrint(arena, "{s: >8} ", .{"DEV MEM"}) };
    segs[5] = .{ .style = s, .text = try std.fmt.allocPrint(arena, "{s: >6} ", .{"CPU"}) };
    segs[6] = .{ .style = s, .text = try std.fmt.allocPrint(arena, "{s: >8}  ", .{"HOST MEM"}) };
    segs[7] = .{ .style = s, .text = "Command" };
    return .{ .text = segs, .softwrap = false, .overflow = .clip };
}

const ProcessRow = struct {
    info: ProcessInfo = .{},

    fn widget(self: *const ProcessRow) vxfw.Widget {
        return .{ .userdata = @constCast(self), .drawFn = drawFn };
    }

    fn drawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
        const self: *const ProcessRow = @ptrCast(@alignCast(ptr));
        const info = &self.info;
        const val: vaxis.Style = .{ .fg = theme.text_primary };
        const dim: vaxis.Style = .{ .fg = theme.dim };

        const segs = try ctx.arena.alloc(Segment, 8);
        segs[0] = .{ .style = val, .text = try std.fmt.allocPrint(ctx.arena, " {d: >7} ", .{info.pid}) };
        segs[1] = .{ .style = val, .text = try std.fmt.allocPrint(ctx.arena, "{s: <10} ", .{trunc(strZ(&info.username), 10)}) };
        segs[2] = if (info.device_idx) |idx|
            .{ .style = val, .text = try std.fmt.allocPrint(ctx.arena, "{d: <5} ", .{idx}) }
        else
            .{ .style = dim, .text = try std.fmt.allocPrint(ctx.arena, "{s: <5} ", .{@as([]const u8, "-")}) };
        segs[3] = if (info.gpu_util_percent) |util|
            .{ .style = val, .text = try std.fmt.allocPrint(ctx.arena, "{d: >4}% ", .{util}) }
        else
            .{ .style = dim, .text = try std.fmt.allocPrint(ctx.arena, "{s: >6} ", .{@as([]const u8, "-")}) };
        segs[4] = if (info.gpu_mem_kib) |mem|
            .{ .style = val, .text = try std.fmt.allocPrint(ctx.arena, "{s: >8} ", .{try fmtMem(ctx.arena, mem)}) }
        else
            .{ .style = dim, .text = try std.fmt.allocPrint(ctx.arena, "{s: >8} ", .{@as([]const u8, "-")}) };
        segs[5] = .{ .style = val, .text = try std.fmt.allocPrint(ctx.arena, "{d: >3}.{d}% ", .{ info.cpu_percent / 10, info.cpu_percent % 10 }) };
        segs[6] = .{ .style = val, .text = try std.fmt.allocPrint(ctx.arena, "{s: >8}  ", .{try fmtMem(ctx.arena, info.rss_kib)}) };
        segs[7] = .{ .style = val, .text = strZ(&info.comm) };

        const rich: RichText = .{ .text = segs, .softwrap = false, .overflow = .clip };
        return rich.draw(ctx.withConstraints(
            .{ .width = ctx.min.width, .height = 1 },
            .{ .width = ctx.max.width, .height = 1 },
        ));
    }
};

fn strZ(buf: anytype) []const u8 {
    return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(buf)), 0);
}

fn trunc(s: []const u8, max: usize) []const u8 {
    return s[0..@min(s.len, max)];
}

fn fmtMem(arena: std.mem.Allocator, kib: u64) ![]const u8 {
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
