const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const image_cache = @import("../image_cache.zig");
const ColumnLayout = @import("../widgets/components/column_layout.zig");
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

pub fn widget(self: *Overview) vxfw.Widget {
    return .{
        .userdata = self,
        .eventHandler = typeErasedEventHandler,
        .drawFn = typeErasedDrawFn,
    };
}

fn typeErasedEventHandler(ptr: *anyopaque, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
    const self: *Overview = @ptrCast(@alignCast(ptr));
    return self.handleEvent(ctx, event);
}

fn typeErasedDrawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const self: *Overview = @ptrCast(@alignCast(ptr));
    return self.draw(ctx);
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

    // Both layout variants live on the stack; only the selected one is drawn.
    // Narrow: logo centered horizontally, info lines below.
    const centered_logo: vxfw.Center = .{ .child = logo.widget() };
    const logo_box: vxfw.SizedBox = .{ .child = centered_logo.widget(), .size = .{ .width = content_w, .height = logo_h } };
    const narrow_children = [2]vxfw.Widget{ logo_box.widget(), info_lines.widget() };
    const narrow_layout: ColumnLayout = .{ .children = &narrow_children, .gap = 1 };

    // Wide: logo left, info lines vertically centered right.
    const info_max_w = @min(w -| (Logo.logo_width + 6), max_info_width);
    const banner_h = @max(logo_h, 9); // info lines need at least 9 rows
    const logo_sized: vxfw.SizedBox = .{ .child = logo.widget(), .size = .{ .width = Logo.logo_width + 4, .height = banner_h } };
    const info_centered: vxfw.Center = .{ .child = info_lines.widget() };
    const info_sized: vxfw.SizedBox = .{ .child = info_centered.widget(), .size = .{ .width = info_max_w, .height = banner_h } };
    const wide_flex_items = [2]vxfw.FlexItem{
        .{ .widget = logo_sized.widget(), .flex = 0 },
        .{ .widget = info_sized.widget(), .flex = 0 },
    };
    const wide_flex_row: vxfw.FlexRow = .{ .children = &wide_flex_items };
    const wide_sized: vxfw.SizedBox = .{ .child = wide_flex_row.widget(), .size = .{ .width = content_w, .height = banner_h } };

    const banner = if (narrow) narrow_layout.widget() else wide_sized.widget();
    try widgets.append(ctx.arena, banner);

    // ── Device Grid ─────────────────────────────────────────
    for (self.device_cards) |*card| {
        card.use_braille = self.use_braille;
    }

    const use_two_cols = w >= two_col_breakpoint;
    var i: usize = 0;
    while (i < self.device_cards.len) {
        if (use_two_cols and i + 1 < self.device_cards.len) {
            // Arena-allocate because loop locals die each iteration.
            const left_padded = try ctx.arena.create(vxfw.Padding);
            left_padded.* = .{ .child = self.device_cards[i].widget(), .padding = .{ .right = 1 } };
            const right_padded = try ctx.arena.create(vxfw.Padding);
            right_padded.* = .{ .child = self.device_cards[i + 1].widget(), .padding = .{ .left = 1 } };
            const flex_items = try ctx.arena.alloc(vxfw.FlexItem, 2);
            flex_items[0] = .{ .widget = left_padded.widget(), .flex = 1 };
            flex_items[1] = .{ .widget = right_padded.widget(), .flex = 1 };
            const flex_row = try ctx.arena.create(vxfw.FlexRow);
            flex_row.* = .{ .children = flex_items };
            const sized = try ctx.arena.create(vxfw.SizedBox);
            sized.* = .{ .child = flex_row.widget(), .size = .{ .width = content_w, .height = 100 } };
            try widgets.append(ctx.arena, sized.widget());
            i += 2;
        } else {
            const sized = try ctx.arena.create(vxfw.SizedBox);
            sized.* = .{ .child = self.device_cards[i].widget(), .size = .{ .width = content_w, .height = 100 } };
            try widgets.append(ctx.arena, sized.widget());
            i += 1;
        }
    }

    // ── Process Table ──────────────────────────────────────────
    if (self.process_table) |pt| {
        try widgets.append(ctx.arena, pt.widget());
    }

    // ── Compose with ColumnLayout ───────────────────────────
    const layout: ColumnLayout = .{
        .children = widgets.items,
        .gap = 1,
    };

    const layout_surf = try layout.widget().draw(ctx.withConstraints(
        .{ .width = content_w },
        .{ .width = content_w, .height = null },
    ));

    const children = try ctx.arena.alloc(vxfw.SubSurface, 1);
    children[0] = .{ .origin = .{ .row = 1, .col = 2 }, .surface = layout_surf };

    return .{
        .size = .{ .width = w, .height = layout_surf.size.height + 1 },
        .widget = self.widget(),
        .buffer = &.{},
        .children = children,
    };
}
