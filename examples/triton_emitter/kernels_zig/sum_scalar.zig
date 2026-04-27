//! Zig port of `kernels_py/sum_scalar.py:triton_sum_kernel_scalar_result`.

const ttir_dialect = @import("mlir/dialects/ttir");

const tri = @import("zml").kernel.triton;
const zml = @import("zml");

pub const SumScalar = struct {
    pub const Cfg = struct { BLOCK_SIZE_M: i32 = 1024 };
    pub const Kernel = tri.Kernel(Cfg, .{
        .name = "triton_sum_kernel_scalar_result",
        .inputs = &.{ "input_ptr", "output_ptr", "M" },
        .outputs = &.{},
        .run = run,
    });
    fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
        const a = try b.declareArgs(.{
            .input_ptr = .{ .ptr = .f32 },
            .output_ptr = .{ .ptr = .f32 },
            .M = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
        });

        const pid = b.programId(.x);
        const offsets = pid.mul(cfg.BLOCK_SIZE_M).add(b.arange(0, cfg.BLOCK_SIZE_M, .i32));
        const mask = offsets.lt(a.M);

        // `other` is the i1 mask cast to f32 via uitofp.
        const x = b.loadOpts(a.input_ptr.addPtr(offsets), .{ .mask = mask, .other = mask.to(.f32) });
        const total = b.sum(x);

        // `tl.atomic_add` always emits an i1 true-mask (3-operand form).
        const out_ptrs = a.output_ptr.addPtr(b.arange(0, 1, .i32));
        const true_mask = b.full(&.{1}, 1, .i1);
        _ = b.atomicRmwOpts(ttir_dialect.RMWOp.fadd, out_ptrs, total.splatTo(&.{1}), .{ .mask = true_mask });
    }
};
