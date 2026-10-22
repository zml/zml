//! Kernel-side building blocks of the MegaMoE port (DeepGEMM's `sm100_fp8_fp4_mega_moe`):
//! no dependency on `zml`, only on the CuTe builder. `layout` is pure Zig (host and comptime);
//! `ptx` emits DeepGEMM's PTX primitives into a `kernels/cute/builder` kernel; `kernel` emits the
//! MegaMoE kernel itself (with `scheduler`, `device` and `flow`).
pub const layout = @import("layout.zig");
pub const ptx = @import("ptx.zig");
pub const tma = @import("tma.zig");
pub const flow = @import("flow.zig");
pub const device = @import("device.zig");
pub const scheduler = @import("scheduler.zig");
pub const kernel = @import("kernel.zig");
pub const staging = @import("staging.zig");
pub const program = @import("program.zig");
/// Test support: `emitEveryPrimitive` (every `ptx` primitive in one kernel).
pub const ptx_test = @import("ptx_test.zig");
/// Test support: `Instance` (instantiations of the kernel for tests).
pub const kernel_test = @import("kernel_test.zig");

test {
    _ = layout;
    _ = ptx;
    _ = tma;
    _ = flow;
    _ = device;
    _ = scheduler;
    _ = kernel;
    _ = staging;
    _ = program;
    _ = ptx_test;
    _ = kernel_test;
}
