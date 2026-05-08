//! Smoke-test kernel for `tools/dsl-harness`. Mirrors
//! `examples/triton_emitter/kernels_zig/vector_add.zig` and is paired with
//! the `@triton.jit triton_add_kernel` in `vector_add.py` for the
//! Python-side reference.

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const tri = zml.kernel.triton;
const ops = zml.ops;
const Tensor = zml.Tensor;

// =============================================================================
// Kernel definition + sweeps (consumed by the macro-generated entry shim).
// =============================================================================

const Cfg = struct { BLOCK_SIZE: i32 = 1024 };

pub const Kernel = tri.Kernel(Cfg, .{
    .name = "triton_add_kernel",
    .inputs = &.{ "x_ptr", "y_ptr", "n_elements" },
    .outputs = &.{"output"},
    .run = run,
});

fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .x_ptr = .{ .ptr = .f32 },
        .y_ptr = .{ .ptr = .f32 },
        .output_ptr = .{ .ptr = .f32 },
        .n_elements = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
    });

    const pid = b.programId(.x);
    const offsets = pid.mul(cfg.BLOCK_SIZE).add(b.arange(0, cfg.BLOCK_SIZE, .i32));
    const mask = offsets.lt(a.n_elements);

    const x = b.loadOpts(a.x_ptr.addPtr(offsets), .{ .mask = mask });
    const y = b.loadOpts(a.y_ptr.addPtr(offsets), .{ .mask = mask });
    const out = x.add(y);
    b.storeOpts(a.output_ptr.addPtr(offsets), out, .{ .mask = mask });
}

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{ .BLOCK_SIZE = 1024 } },
};

// =============================================================================
// XLA-pipeline driver: `forward`, `args`, and `setActiveTtir` for the harness
// to pump synthetic Tensors through `zml.module.compile`. Mirrors the pattern
// in `examples/triton_emitter/dump_via_xla.zig:VectorAdd`. The synthetic
// shapes here don't have to be runtime-correct — XLA never launches the
// kernel, it just runs the codegen pipeline.
// =============================================================================

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(x: Tensor, y: Tensor, _: Tensor, n: Tensor) Tensor {
    return ops.triton(.{ x, y, n }, .{x.shape()}, .{
        .name = Kernel.name,
        .ir = active_ttir,
        .grid = .{ 1, 1, 1 },
        .num_warps = 4,
        .num_stages = 1,
    })[0];
}

pub fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
    return .{
        Tensor.init(.{1024}, .f32), // x_ptr
        Tensor.init(.{1024}, .f32), // y_ptr
        Tensor.init(.{1024}, .f32), // output_ptr placeholder (XLA passes inputs first, then outputs)
        Tensor.init(.{}, .i32), // n_elements
    };
}
