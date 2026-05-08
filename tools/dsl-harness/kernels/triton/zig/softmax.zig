//! Registration of `softmax_kernel` (row-softmax). Self-contained
//! migration of `examples/triton_emitter/kernels_zig/softmax.zig`.

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const tri = zml.kernel.triton;
const Tensor = zml.Tensor;

const Cfg = struct { BLOCK_SIZE: i32 = 1024 };

pub const Kernel = tri.Kernel(Cfg, .{
    .name = "softmax_kernel",
    // NOTE: `output_ptr` is declared first in this kernel's `tt.func`;
    // doesn't follow the inputs-then-outputs convention. Declared here
    // as all-inputs since this kernel is emit-only via the harness
    // (not invoked through `Kernel.call`).
    .inputs = &.{ "output_ptr", "input_ptr", "input_row_stride", "output_row_stride", "n_cols" },
    .outputs = &.{},
    .run = run,
});

fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .output_ptr = .{ .ptr = .f32 },
        .input_ptr = .{ .ptr = .f32 },
        .input_row_stride = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
        .output_row_stride = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
        .n_cols = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
    });

    const row_idx = b.programId(.x);
    const row_start_in = a.input_ptr.addPtr(row_idx.mul(a.input_row_stride));

    const col_offsets = b.arange(0, cfg.BLOCK_SIZE, .i32);
    const input_ptrs = row_start_in.addPtr(col_offsets);
    const mask = col_offsets.lt(a.n_cols);

    const neg_inf = b.full(&.{cfg.BLOCK_SIZE}, -std.math.inf(f32), .f32);
    const row = b.loadOpts(input_ptrs, .{ .mask = mask, .other = neg_inf });

    const row_max = b.maxOpts(row, .{ .axis = 0 });
    const row_minus_max = row.sub(row_max.splatTo(&.{cfg.BLOCK_SIZE}));
    const numerator = b.exp(row_minus_max);
    const denominator = b.sumOpts(numerator, .{ .axis = 0 });
    const softmax_out = numerator.div(denominator.splatTo(&.{cfg.BLOCK_SIZE}));

    const row_start_out = a.output_ptr.addPtr(row_idx.mul(a.output_row_stride));
    const output_ptrs = row_start_out.addPtr(col_offsets);
    b.storeOpts(output_ptrs, softmax_out, .{ .mask = mask });
}

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{ .BLOCK_SIZE = 1024 } },
};

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(input: Tensor, in_stride: Tensor, out_stride: Tensor, n_cols: Tensor, _: Tensor) Tensor {
    return ops.triton(
        .{ input, in_stride, out_stride, n_cols },
        .{input.shape()},
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
        Tensor.init(.{ 1024 * 64 }, .f32), // input_ptr
        Tensor.init(.{}, .i32),            // input_row_stride
        Tensor.init(.{}, .i32),            // output_row_stride
        Tensor.init(.{}, .i32),            // n_cols
        Tensor.init(.{ 1024 * 64 }, .f32), // output_ptr placeholder
    };
}
