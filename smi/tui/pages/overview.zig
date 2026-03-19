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

state: *const data.SystemState,
device_cards: []DeviceCard = &.{},
process_table: ?*ProcessTable = null,
viewing_device: *?u8 = undefined,
use_braille: bool = false,

pub fn init(self: *Overview, allocator: std.mem.Allocator) !void {
    const count = self.state.deviceCount();
    const cards = try allocator.alloc(DeviceCard, count);
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
    allocator.free(self.device_cards);
}

pub fn handleEvent(self: *Overview, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
    switch (event) {
        .key_press => |key| {
            if (key.matches('v', .{})) {
                self.use_braille = !self.use_braille;
                return ctx.consumeAndRedraw();
            }
        },
        else => {},
    }
}

pub fn draw(self: *Overview, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const narrow = w < narrow_breakpoint;
    const content_w: u16 = w -| 4;

    // We use an ArrayList because the number of widgets depends on device count.
    var widgets: std.ArrayList(vxfw.Widget) = .empty;

    // ── Logo + Info Lines ────────────────────────────────────
    const logo: Logo = .{ .image = image_cache.global.get("logo"), .compact = narrow };
    const info_lines: InfoLines = .{ .state = self.state };

    // Compute logo height (differs for image vs ASCII)
    const has_image = logo.image != null;
    const logo_h: u16 = if (has_image)
        (if (narrow) Logo.compact_image_height else Logo.image_height)
    else
        (if (narrow) Logo.compact_height else Logo.logo_height);

    // Narrow: logo centered, info lines below.
    const narrow_children = [2]vxfw.Widget{
        try compose.sized(ctx.arena, try compose.center(ctx.arena, ui.widget(&logo)), .{ .width = content_w, .height = logo_h }),
        ui.widget(&info_lines),
    };
    const narrow_layout: ColumnLayout = .{ .children = &narrow_children, .gap = 1 };

    // Wide: logo left, info lines vertically centered right.
    const info_max_w = @min(w -| (Logo.logo_width + 6), max_info_width);
    const banner_h = @max(logo_h, 9); // info lines need at least 9 rows
    const wide_flex_items = [2]vxfw.FlexItem{
        .{ .widget = try compose.sized(ctx.arena, ui.widget(&logo), .{ .width = Logo.logo_width + 4, .height = banner_h }), .flex = 0 },
        .{ .widget = try compose.sized(ctx.arena, try compose.center(ctx.arena, ui.widget(&info_lines)), .{ .width = info_max_w, .height = banner_h }), .flex = 0 },
    };
    const wide_flex_row: vxfw.FlexRow = .{ .children = &wide_flex_items };

    const banner = if (narrow) ui.widget(&narrow_layout) else try compose.sized(ctx.arena, wide_flex_row.widget(), .{ .width = content_w, .height = banner_h });
    try widgets.append(ctx.arena, banner);

    // ── Device Grid ─────────────────────────────────────────
    for (self.device_cards) |*card| {
        card.use_braille = self.use_braille;
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
    try widgets.append(ctx.arena, ui.widget(&device_grid));

    // ── Process Table ────────────────────────────────────────
    if (self.process_table) |pt| {
        try widgets.append(ctx.arena, ui.widget(pt));
    }

    // ── Compose with ColumnLayout ────────────────────────────
    return common.pageFrame(ctx, .{
        .children = widgets.items,
        .gap = 1,
    }, w, content_w, ui.widget(self));
}
