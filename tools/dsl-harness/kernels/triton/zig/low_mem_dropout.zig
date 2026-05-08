//! Registration of `_triton_dropout` (low-memory variant — externally
//! supplied keep-mask, no `tl.rand`). Self-contained migration of the
//! legacy `examples/triton_emitter/kernels_zig/low_mem_dropout.zig`.

const std = @import("std");

const arith = @import("mlir/dialects").arith;

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const tri = zml.kernel.triton;
const Tensor = zml.Tensor;

const Cfg = struct { BLOCK_SIZE: i32 = 1024 };

pub const Kernel = tri.Kernel(Cfg, .{
    .name = "_triton_dropout",
    .inputs = &.{ "x_ptr", "x_keep_ptr", "n_elements", "p" },
    .outputs = &.{"output"},
    .run = run,
});

fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .x_ptr = .{ .ptr = .f32 },
        .x_keep_ptr = .{ .ptr = .f32 },
        .output_ptr = .{ .ptr = .f32 },
        .n_elements = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
        .p = .{ .scalar = .f32 },
    });

    const pid = b.programId(.x);
    const offsets = pid.mul(cfg.BLOCK_SIZE).add(b.arange(0, cfg.BLOCK_SIZE, .i32));
    const mask = offsets.lt(a.n_elements);

    const x = b.loadOpts(a.x_ptr.addPtr(offsets), .{ .mask = mask });
    const x_keep = b.loadOpts(a.x_keep_ptr.addPtr(offsets), .{ .mask = mask });

    const one_minus_p = b.liftAs(@as(f32, 1.0), .f32).sub(a.p);
    const scaled = x.div(one_minus_p.splatTo(&.{cfg.BLOCK_SIZE}));

    const zero_tensor = b.zeros(&.{cfg.BLOCK_SIZE}, .f32);
    const keep_bool = b.cmpf(arith.CmpFPredicate.une, x_keep, zero_tensor);
    const out = b.where(keep_bool, scaled, zero_tensor);

    b.storeOpts(a.output_ptr.addPtr(offsets), out, .{ .mask = mask });
}

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{ .BLOCK_SIZE = 1024 } },
};

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(x: Tensor, x_keep: Tensor, _: Tensor, n: Tensor, p: Tensor) Tensor {
    return ops.triton(
        .{ x, x_keep, n, p },
        .{x.shape()},
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
        Tensor.init(.{1024}, .f32), // x_ptr
        Tensor.init(.{1024}, .f32), // x_keep_ptr
        Tensor.init(.{1024}, .f32), // output_ptr placeholder
        Tensor.init(.{}, .i32),     // n_elements
        Tensor.init(.{}, .f32),     // p
    };
}
