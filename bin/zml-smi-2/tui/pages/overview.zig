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

state: *data.SystemState,
device_cards: []DeviceCard = &.{},
process_table: ?*ProcessTable = null,
viewing_device: ?u16 = null,
use_braille: bool = false,

/// Stable device widgets preserve hover state across frames.
pub fn init(self: *Overview, allocator: std.mem.Allocator) std.mem.Allocator.Error!void {
    self.device_cards = try allocator.alloc(DeviceCard, self.state.devices.len);
    for (self.device_cards, 0..) |*card, index| {
        card.* = .{
            .device_id = @intCast(index),
            .state = self.state,
            .viewing_device = &self.viewing_device,
            .use_braille = &self.use_braille,
        };
    }
}

pub fn deinit(self: *Overview, allocator: std.mem.Allocator) void {
    allocator.free(self.device_cards);
}

pub fn draw(self: *Overview, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const width = ctx.max.width orelse 80;
    const narrow = width < narrow_breakpoint;
    const content_width = width -| 4;
    const content_ctx = ui.fixedWidth(ctx, content_width);
    var layout = compose.surfaceBuilder(ctx.arena);

    const logo: Logo = .{ .image = image_cache.get(0) };
    const info: InfoLines = .{ .state = self.state };
    const logo_width = if (narrow) content_width else Logo.logo_width + 6;
    const banner_height = if (narrow) Logo.logo_height else @max(Logo.logo_height, InfoLines.entry_count);
    const logo_surface = try logo.draw(ui.maxSize(ctx, logo_width, banner_height));
    const logo_column = (logo_width -| logo_surface.size.width) / 2;
    const logo_row = 1 + (banner_height -| logo_surface.size.height) / 2;
    try layout.add(logo_row, logo_column + @as(u16, if (narrow) 2 else 0), logo_surface);

    const info_width = @min(content_width + 2 -| logo_width, max_info_width);
    const info_surface = try info.draw(if (narrow)
        content_ctx
    else
        ui.maxSize(ctx, info_width, banner_height));
    const info_row = if (narrow) banner_height + 2 else 1 + (banner_height -| info_surface.size.height) / 2;
    const info_column = if (narrow) 2 else logo_width + (info_width -| info_surface.size.width) / 2;
    try layout.add(info_row, info_column, info_surface);
    var row: i17 = @as(i17, banner_height) + 2;
    if (narrow) row += @as(i17, info_surface.size.height) + 1;

    const card_widgets = try ctx.arena.alloc(vxfw.Widget, self.device_cards.len);
    for (self.device_cards, card_widgets) |*card, *widget| widget.* = ui.widget(card);
    const device_grid: ColumnLayout = .{
        .children = card_widgets,
        .min_child_width = two_col_breakpoint / 2,
        .col_gap = 2,
        .gap = 1,
    };
    const grid_surf = try device_grid.draw(content_ctx);
    try layout.add(row, 2, grid_surf);
    row += @intCast(grid_surf.size.height + 1);

    if (self.process_table) |pt| {
        const pt_surf = try pt.draw(content_ctx);
        try layout.add(row, 2, pt_surf);
        row += @intCast(pt_surf.size.height);
    }

    return layout.finish(.{ .width = width, .height = @intCast(row) }, ui.widget(self));
}
