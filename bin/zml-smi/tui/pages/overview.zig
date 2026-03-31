const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const image_cache = @import("../image_cache.zig");
const ColumnLayout = @import("../widgets/column_layout.zig");
const common = @import("detail/detail_common.zig");
const DeviceCard = @import("../widgets/device_card.zig");
const Logo = @import("../widgets/logo.zig");
const InfoLines = @import("../widgets/info_lines.zig");
const ProcessTable = @import("../widgets/process_table.zig");

const Overview = @This();

const two_col_breakpoint: u16 = 120;
const narrow_breakpoint: u16 = 80;
const max_info_width: u16 = 50;
/// Total width of the wide banner: logo box (incl. left margin) + info lines + right margin.
pub const host_line_width: u16 = (Logo.logo_width + 6) + max_info_width + 2;

state: *const data.SystemState,
device_cards: []DeviceCard = &.{},
process_table: ?*ProcessTable = null, // maybe null if we decide not to print the process table in print mode
viewing_device: *?u8 = undefined,
use_braille: bool = false,

pub fn init(allocator: std.mem.Allocator, state: *const data.SystemState, process_table: ?*ProcessTable, viewing_device: *?u8) !Overview {
    const count = state.deviceCount();
    const cards = try allocator.alloc(DeviceCard, count);
    for (cards, 0..) |*card, i| {
        card.* = .{
            .device_id = @intCast(i),
            .state = state,
        };
    }
    return .{
        .state = state,
        .device_cards = cards,
        .process_table = process_table,
        .viewing_device = viewing_device,
    };
}

pub fn deinit(self: *Overview, allocator: std.mem.Allocator) void {
    allocator.free(self.device_cards);
}

fn drawNarrowBanner(self: *const Overview, ctx: vxfw.DrawContext, content_w: u16) !vxfw.Surface {
    const logo: Logo = .{ .image = image_cache.global.get("logo"), .compact = true };
    const info_lines: InfoLines = .{ .state = self.state };

    const logo_h: u16 = if (logo.image != null) Logo.compact_image_height else Logo.compact_height;
    const children = [2]vxfw.Widget{
        try compose.sized(ctx.arena, try compose.center(ctx.arena, ui.widget(&logo)), .{ .width = content_w, .height = logo_h }),
        ui.widget(&info_lines),
    };
    const layout: ColumnLayout = .{ .children = &children, .gap = 1 };
    return ui.widget(&layout).draw(ui.fixedWidth(ctx, content_w));
}

fn drawWideBanner(self: *const Overview, ctx: vxfw.DrawContext, content_w: u16) !vxfw.Surface {
    const logo: Logo = .{ .image = image_cache.global.get("logo"), .compact = false };
    const info_lines: InfoLines = .{ .state = self.state };

    const logo_h: u16 = if (logo.image != null) Logo.image_height else Logo.logo_height;
    const logo_box_w = Logo.logo_width + 6; // +4 centering + 2 absorbed page margin
    const info_max_w = @min(content_w -| logo_box_w, max_info_width);
    const banner_h = @max(logo_h, InfoLines.entry_count);

    const flex_items = [2]vxfw.FlexItem{
        .{ .widget = try compose.sized(ctx.arena, try compose.center(ctx.arena, ui.widget(&logo)), .{ .width = logo_box_w, .height = banner_h }), .flex = 0 },
        .{ .widget = try compose.sized(ctx.arena, try compose.center(ctx.arena, ui.widget(&info_lines)), .{ .width = info_max_w, .height = banner_h }), .flex = 0 },
    };
    const sized_banner = try compose.sized(ctx.arena, (vxfw.FlexRow{ .children = &flex_items }).widget(), .{ .width = content_w, .height = banner_h });
    return sized_banner.draw(ui.fixedWidth(ctx, content_w));
}

pub fn draw(self: *Overview, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const narrow = w < narrow_breakpoint;
    const content_w: u16 = w -| 4;
    const content_ctx = ui.fixedWidth(ctx, content_w);

    var sb = compose.surfaceBuilder(ctx.arena);
    var row: i17 = 1; // top margin

    // ── Banner (Logo + Info Lines) ─────────────────────────
    const banner_surf = try if (narrow) self.drawNarrowBanner(ctx, content_w) else self.drawWideBanner(ctx, content_w + 2);
    try sb.add(row, if (narrow) @as(i17, 2) else 0, banner_surf);
    row += @intCast(banner_surf.size.height + 1);

    // ── Device Grid ─────────────────────────────────────────
    for (self.device_cards) |*card| {
        card.use_braille = self.use_braille;
        card.viewing_device = self.viewing_device;
    }
    const card_widgets = try ctx.arena.alloc(vxfw.Widget, self.device_cards.len);
    for (self.device_cards, card_widgets) |*card, *cw| {
        cw.* = ui.widget(card);
    }
    const device_grid: ColumnLayout = .{
        .children = card_widgets,
        .min_child_width = two_col_breakpoint / 2,
        .col_gap = 2,
        .gap = 1,
    };
    const grid_surf = try ui.widget(&device_grid).draw(content_ctx);
    try sb.add(row, 2, grid_surf);
    row += @intCast(grid_surf.size.height + 1);

    // ── Process Table ────────────────────────────────────────
    if (self.process_table) |pt| {
        const pt_surf = try ui.widget(pt).draw(content_ctx);
        try sb.add(row, 2, pt_surf);
        row += @intCast(pt_surf.size.height);
    }

    return sb.finish(.{ .width = w, .height = @intCast(row) }, ui.widget(self));
}
