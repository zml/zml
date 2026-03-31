const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const ui = @import("../lib/ui.zig");
const ProcessTable = @import("../widgets/process_table.zig");
const gpu_detail = @import("detail/gpu_detail.zig");
const neuron_detail = @import("detail/neuron_detail.zig");
const tpu_detail = @import("detail/tpu_detail.zig");

const Detail = @This();

state: *const data.SystemState,
device_id: u8,
process_table: *ProcessTable = undefined,

pub fn draw(self: *const Detail, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const id = self.device_id;
    const dev = self.state.devices[id];
    const wgt = ui.widget(self);

    return switch (dev.*) {
        .cuda, .rocm => |*sv| try gpu_detail.draw(self.state, self.process_table, ctx, w, id, sv.front().*, wgt),
        .neuron => |*sv| try neuron_detail.draw(self.state, self.process_table, ctx, w, id, sv.front().*, wgt),
        .tpu => |*sv| try tpu_detail.draw(self.state, self.process_table, ctx, w, id, sv.front().*, wgt),
    };
}
