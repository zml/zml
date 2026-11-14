const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const ui = @import("../lib/ui.zig");
const ProcessTable = @import("../widgets/process_table.zig");
const gpu_detail = @import("detail/gpu_detail.zig");
const neuron_detail = @import("detail/neuron_detail.zig");
const tpu_detail = @import("detail/tpu_detail.zig");
const furiosa_detail = @import("detail/furiosa_detail.zig");

const Detail = @This();

state: *const data.SystemState,
device_id: u16,
process_table: *ProcessTable = undefined,

pub fn draw(self: *const Detail, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const id = self.device_id;
    const device = self.state.device(id);
    const wgt = ui.widget(self);

    return switch (self.state.sample(id)) {
        .cuda, .rocm, .oneapi => |gpu| try gpu_detail.draw(self.state, self.process_table, ctx, w, id, device, gpu, wgt),
        .neuron => |core| try neuron_detail.draw(self.state, self.process_table, ctx, w, id, device, core, wgt),
        .tpu => |chip| try tpu_detail.draw(self.state, self.process_table, ctx, w, id, device, chip, wgt),
        .furiosa => |chip| try furiosa_detail.draw(self.state, self.process_table, ctx, w, id, device, chip, wgt),
    };
}
