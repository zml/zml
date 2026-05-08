//! Registration of `triton_sum_kernel_scalar_result`. Each program sums a
//! BLOCK_SIZE_M slice of `input` and atomically adds the partial into a
//! single scalar output. Self-contained migration of the legacy
//! `examples/triton_emitter/kernels_zig/sum_scalar.zig`.

const std = @import("std");

const ttir_dialect = @import("mlir/dialects/ttir");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const tri = zml.kernel.triton;
const Tensor = zml.Tensor;
const Shape = zml.Shape;

const Cfg = struct { BLOCK_SIZE_M: i32 = 1024 };

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

    const x = b.loadOpts(a.input_ptr.addPtr(offsets), .{ .mask = mask, .other = mask.to(.f32) });
    const total = b.sum(x);

    const out_ptrs = a.output_ptr.addPtr(b.arange(0, 1, .i32));
    const true_mask = b.full(&.{1}, 1, .i1);
    _ = b.atomicRmwOpts(ttir_dialect.RMWOp.fadd, out_ptrs, total.splatTo(&.{1}), .{ .mask = true_mask });
}

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{ .BLOCK_SIZE_M = 1024 } },
};

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(input: Tensor, _: Tensor, m: Tensor) Tensor {
    return ops.triton(
        .{ input, m },
        .{Shape.init(.{}, .f32)},
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
        Tensor.init(.{1024}, .f32), // input_ptr
        Tensor.init(.{1}, .f32),    // output_ptr placeholder
        Tensor.init(.{}, .i32),     // M
    };
}
