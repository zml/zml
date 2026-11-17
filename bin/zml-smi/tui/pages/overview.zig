const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const image_cache = @import("../image_cache.zig");
const ColumnLayout = @import("../widgets/column_layout.zig");
const DeviceCard = @import("../widgets/device_card.zig");
const Logo = @import("../widgets/logo.zig");
const InfoLines = @import("../widgets/info_lines.zig");
const ProcessTable = @import("../widgets/process_table.zig");

const Overview = @This();

const two_col_breakpoint: u16 = 140;
const narrow_breakpoint: u16 = 80;
const max_info_width: u16 = 50;
/// Total width of the wide banner: logo box (incl. left margin) + info lines + right margin.
pub const host_line_width: u16 = (Logo.logo_width + 6) + max_info_width + 2;

state: *const data.SystemState,
device_cards: ?[]DeviceCard = null,
process_table: ?*ProcessTable = null,
viewing_device: *?u16,
use_braille: bool = false,

pub fn prepare(self: *Overview, allocator: std.mem.Allocator) std.mem.Allocator.Error!void {
    if (self.device_cards != null) return;
    const cards = try allocator.alloc(DeviceCard, self.state.deviceCount());
    for (cards, 0..) |*card, i| {
        card.* = .{
            .device_id = @intCast(i),
            .state = self.state,
            .viewing_device = self.viewing_device,
        };
    }
    self.device_cards = cards;
}

pub fn deinit(self: *Overview, allocator: std.mem.Allocator) void {
    if (self.device_cards) |cards| allocator.free(cards);
}

fn drawBanner(
    self: *const Overview,
    ctx: vxfw.DrawContext,
    content_width: u16,
    narrow: bool,
    layout: *compose.SurfaceBuilder,
) !u16 {
    const logo: Logo = .{ .image = image_cache.global.zml };
    const info: InfoLines = .{ .state = self.state };
    const logo_width = if (narrow) content_width else Logo.logo_width + 6;
    const banner_height = if (narrow) Logo.logo_height else @max(Logo.logo_height, InfoLines.entry_count);
    const logo_surface = try logo.draw(ui.maxSize(ctx, logo_width, banner_height));
    const logo_column = (logo_width -| logo_surface.size.width) / 2;
    const logo_row = 1 + (banner_height -| logo_surface.size.height) / 2;
    try layout.add(logo_row, logo_column + @as(u16, if (narrow) 2 else 0), logo_surface);

    if (narrow) {
        const info_surface = try info.draw(ui.fixedWidth(ctx, content_width));
        try layout.add(banner_height + 2, 2, info_surface);
        return banner_height + 1 + info_surface.size.height;
    }
    const info_width = @min(content_width + 2 -| logo_width, max_info_width);
    const info_surface = try info.draw(ui.maxSize(ctx, info_width, banner_height));
    const info_row = 1 + (banner_height -| info_surface.size.height) / 2;
    const info_column = logo_width + (info_width -| info_surface.size.width) / 2;
    try layout.add(info_row, info_column, info_surface);
    return banner_height;
}

pub fn draw(self: *Overview, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const narrow = w < narrow_breakpoint;
    const content_w: u16 = w -| 4;
    const content_ctx = ui.fixedWidth(ctx, content_w);

    var sb = compose.surfaceBuilder(ctx.arena);
    var row: i17 = @as(i17, try self.drawBanner(ctx, content_w, narrow, &sb)) + 2;

    const cards = self.device_cards.?;
    const card_widgets = try ctx.arena.alloc(vxfw.Widget, cards.len);
    for (cards, card_widgets) |*card, *cw| {
        card.use_braille = self.use_braille;
        cw.* = ui.widget(card);
    }
    const device_grid: ColumnLayout = .{
        .children = card_widgets,
        .min_child_width = two_col_breakpoint / 2,
        .col_gap = 2,
        .gap = 1,
    };
    const grid_surf = try device_grid.draw(content_ctx);
    try sb.add(row, 2, grid_surf);
    row += @intCast(grid_surf.size.height + 1);

    if (self.process_table) |pt| {
        const pt_surf = try pt.draw(content_ctx);
        try sb.add(row, 2, pt_surf);
        row += @intCast(pt_surf.size.height);
    }

    return sb.finish(.{ .width = w, .height = @intCast(row) }, ui.widget(self));
}
