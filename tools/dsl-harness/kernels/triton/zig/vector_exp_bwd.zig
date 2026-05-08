//! Registration of `triton_exp_backward_kernel` (gradient of vector exp).

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const tri = zml.kernel.triton;
const Tensor = zml.Tensor;

const Cfg = struct { BLOCK_SIZE: i32 = 1024 };

pub const Kernel = tri.Kernel(Cfg, .{
    .name = "triton_exp_backward_kernel",
    .inputs = &.{ "grad_output_ptr", "output_ptr", "n_elements" },
    .outputs = &.{"grad_input"},
    .run = run,
});

fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .grad_output_ptr = .{ .ptr = .f32 },
        .output_ptr = .{ .ptr = .f32 },
        .grad_input_ptr = .{ .ptr = .f32 },
        .n_elements = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
    });

    const pid = b.programId(.x);
    const offsets = pid.mul(cfg.BLOCK_SIZE).add(b.arange(0, cfg.BLOCK_SIZE, .i32));
    const mask = offsets.lt(a.n_elements);

    const grad_output = b.loadOpts(a.grad_output_ptr.addPtr(offsets), .{ .mask = mask });
    const output = b.loadOpts(a.output_ptr.addPtr(offsets), .{ .mask = mask });
    const grad_input = grad_output.mul(output);
    b.storeOpts(a.grad_input_ptr.addPtr(offsets), grad_input, .{ .mask = mask });
}

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{ .BLOCK_SIZE = 1024 } },
};

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(grad_out: Tensor, out: Tensor, _: Tensor, n: Tensor) Tensor {
    return ops.triton(
        .{ grad_out, out, n },
        .{grad_out.shape()},
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ 1, 1, 1 },
            .num_warps = 4,
            .num_stages = 1,
        },
    )[0];
}

pub fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
    return .{
        Tensor.init(.{1024}, .f32), // grad_output_ptr
        Tensor.init(.{1024}, .f32), // output_ptr
        Tensor.init(.{1024}, .f32), // grad_input_ptr placeholder
        Tensor.init(.{}, .i32),     // n_elements
    };
}
