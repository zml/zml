//! Zig ports of `kernels_py/vector_exp.py` (`triton_exp_kernel` +
//! `triton_exp_backward_kernel`).

const tri = @import("zml").kernel.triton;
const zml = @import("zml");

pub const VectorExpFwd = struct {
    pub const Cfg = struct { BLOCK_SIZE: i32 = 1024 };
    pub const Kernel = tri.Kernel(Cfg, .{
        .name = "triton_exp_kernel",
        .inputs = &.{ "x_ptr", "n_elements" },
        .outputs = &.{"output"},
        .run = run,
    });
    fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
        const a = try b.declareArgs(.{
            .x_ptr = .{ .ptr = .f32 },
            .output_ptr = .{ .ptr = .f32 },
            .n_elements = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
        });

        const pid = b.programId(.x);
        const offsets = pid.mul(cfg.BLOCK_SIZE).add(b.arange(0, cfg.BLOCK_SIZE, .i32));
        const mask = offsets.lt(a.n_elements);

        const x = b.loadOpts(a.x_ptr.addPtr(offsets), .{ .mask = mask });
        const out = b.exp(x);
        b.storeOpts(a.output_ptr.addPtr(offsets), out, .{ .mask = mask });
    }
};

pub const VectorExpBwd = struct {
    pub const Cfg = struct { BLOCK_SIZE: i32 = 1024 };
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
};
