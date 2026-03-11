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

const max_visible_rows: u16 = 20;

scroll_bars: vxfw.ScrollBars,
merged: std.ArrayList(ProcessInfo) = .{},

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

pub fn prepare(self: *ProcessTable, state: *data.SystemState, device_id: ?u8) void {
    self.merged.clearRetainingCapacity();

    for (state.process_lists) |pl| {
        for (pl.items) |entry| {
            if (device_id) |did| {
                if (entry.device_idx != did) continue;
            }
            self.merged.append(state.allocator, entry) catch break;
        }
    }

    // Enrich with host data (uid, username, cpu%, rss, cmdline)
    state.enricher.enrich(state.io, self.merged.items);

    // Sort by gpu_mem descending
    std.sort.insertion(ProcessInfo, self.merged.items, {}, struct {
        fn cmp(_: void, a: ProcessInfo, b: ProcessInfo) bool {
            return (a.gpu_mem_kib orelse 0) > (b.gpu_mem_kib orelse 0);
        }
    }.cmp);

    const count: u32 = @intCast(self.merged.items.len);
    self.scroll_bars.estimated_content_height = count;
    self.scroll_bars.scroll_view.item_count = count;
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
    const row_count = self.merged.items.len;

    var header_rt = try headerRichText(ctx.arena);
    const empty_rt: RichText = .{
        .text = &.{.{ .text = "  No process data available", .style = theme.dim_style }},
        .softwrap = false,
        .overflow = .clip,
    };

    var children: std.ArrayList(vxfw.Widget) = .empty;
    try children.append(ctx.arena, header_rt.widget());

    if (row_count == 0) {
        try children.append(ctx.arena, empty_rt.widget());
    } else {
        const scroll_h: u16 = @min(@as(u16, @intCast(row_count)), max_visible_rows);
        const scroll_box: vxfw.SizedBox = .{
            .child = self.scroll_bars.widget(),
            .size = .{ .width = w, .height = scroll_h },
        };
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
        .{ .width = w, .height = content_h + max_visible_rows + 4 },
    ));
}

/// Builds only data rows (no header) for the scroll view.
fn buildRow(ptr: *const anyopaque, idx: usize, _: usize) ?vxfw.Widget {
    const self: *ProcessTable = @ptrCast(@alignCast(@constCast(ptr)));
    if (idx >= self.merged.items.len) return null;
    return .{ .userdata = @ptrCast(&self.merged.items[idx]), .drawFn = drawProcessRow };
}

fn drawProcessRow(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const info: *const ProcessInfo = @ptrCast(@alignCast(ptr));
    const val: vaxis.Style = .{ .fg = theme.text_primary };
    const dim: vaxis.Style = .{ .fg = theme.dim };

    const segs = try ctx.arena.alloc(Segment, 8);
    segs[0] = .{ .style = val, .text = try std.fmt.allocPrint(ctx.arena, " {d: >7} ", .{info.pid}) };
    segs[1] = .{ .style = val, .text = try std.fmt.allocPrint(ctx.arena, "{s: <10} ", .{trunc(strZ(&info.username), 10)}) };
    segs[2] = .{ .style = val, .text = try std.fmt.allocPrint(ctx.arena, "{d: <5} ", .{info.device_idx}) };
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
