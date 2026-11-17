const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const presentation = @import("../presentation.zig");
const utils = @import("../lib/utils.zig");
const ui = @import("../lib/ui.zig");

const DeviceCard = @This();

device_id: u16 = 0,
highlighted: bool = false,

state: *const data.SystemState = undefined,
use_braille: bool = false,
viewing_device: *?u16 = undefined,

pub fn handleEvent(self: *DeviceCard, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
    switch (event) {
        .mouse => |mouse| {
            if (mouse.type == .press and mouse.button == .left) {
                self.viewing_device.* = self.device_id;
                return ctx.consumeAndRedraw();
            }
        },
        .mouse_enter => {
            self.highlighted = true;
            try ctx.setMouseShape(.pointer);
            return ctx.consumeAndRedraw();
        },
        .mouse_leave => {
            self.highlighted = false;
            try ctx.setMouseShape(.default);
            ctx.redraw = true;
        },
        else => {},
    }
}

pub fn draw(self: *DeviceCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    var view = try presentation.overviewDevice(ctx.arena, self.state.frame.?, self.device_id);
    if (self.use_braille) {
        for (&view.gauges, [_]data.Metric{ .util, .memory }) |*gauge, metric| {
            gauge.history = try utils.normalizeRange(
                ctx.arena,
                try self.state.history(ctx.arena, self.device_id, metric),
                0,
                100,
            );
        }
    }

    view.card.cell_size = ctx.cell_size;
    view.card.gauges = &view.gauges;
    view.card.highlighted = self.highlighted;
    var card_surf = try view.card.draw(ctx);
    card_surf.widget = ui.widget(self);
    return card_surf;
}
