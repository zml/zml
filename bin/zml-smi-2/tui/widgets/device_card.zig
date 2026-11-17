const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const presentation = @import("../presentation.zig");
const utils = @import("../lib/utils.zig");
const ui = @import("../lib/ui.zig");

const DeviceCard = @This();

device_id: u16,
highlighted: bool = false,

state: *const data.SystemState,
use_braille: *const bool,
viewing_device: *?u16,

pub fn handleEvent(self: *DeviceCard, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
    switch (event) {
        .mouse => |mouse| {
            if (mouse.type == .press and mouse.button == .left) {
                self.viewing_device.* = self.device_id;
                return ctx.consumeAndRedraw();
            }
        },
        .mouse_enter, .mouse_leave => {
            self.highlighted = event == .mouse_enter;
            try ctx.setMouseShape(if (self.highlighted) .pointer else .default);
            if (self.highlighted) return ctx.consumeAndRedraw();
            ctx.redraw = true;
        },
        else => {},
    }
}

pub fn draw(self: *DeviceCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    var view = try presentation.overviewDevice(ctx.arena, self.state, self.device_id);
    if (self.use_braille.*) {
        for (&view.gauges, [_]data.Metric{ .util, .memory }) |*gauge, metric| {
            gauge.history = try utils.normalizeRange(
                ctx.arena,
                try self.state.history(ctx.arena, self.device_id, metric),
                0,
                100,
            );
        }
    }

    view.card.gauges = &view.gauges;
    view.card.highlighted = self.highlighted;
    var card_surf = try view.card.draw(ctx);
    card_surf.widget = ui.widget(self);
    return card_surf;
}
