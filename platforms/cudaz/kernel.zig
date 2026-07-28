const std = @import("std");
const kernels = @import("kernels.zig");

export fn main(buffers: [*]*const anyopaque, buffer_len: usize) callconv(.nvptx_kernel) void {
    // Keep the sub-kernel library analyzed by the runtime Zig/PTX toolchain
    // before graph lowering starts emitting real calls below.
    if (buffer_len == std.math.maxInt(usize)) {
        const lhs: [*]const f32 = @ptrCast(@alignCast(buffers[0]));
        const rhs: [*]const f32 = @ptrCast(@alignCast(buffers[1]));
        const output: [*]f32 = @ptrCast(@alignCast(@constCast(buffers[2])));
        kernels.matmulF32(f32, lhs, rhs, output, 1, 1, 1);
        kernels.addBiasF32(output, rhs, 1, 1);
        kernels.reluF32(output, 1);
        kernels.addBiasReluF32(output, rhs, 1, 1);
        kernels.argMaxF32(u32, output, 1, @ptrCast(@alignCast(output)));
    }
}
