//! Registration of `per_token_group_quant_fp8` for the harness.
//!
//! Re-exports the production Kernel from `zml.moe.triton_kernels`
//! (`PerTokenGroupQuantFp8`) — no fork, no local copy. One default sweep
//! mirroring `examples/triton_emitter/kernels_zig.zig`: the fp8 output is
//! `e5m2` (not `e4m3fn`) because some Triton+GPU combos reject `e4m3fn`,
//! and we want apples-to-apples TTIR between the Zig DSL and Python sides.

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const Tensor = zml.Tensor;
const Shape = zml.Shape;

pub const Kernel = zml.moe.triton_kernels.PerTokenGroupQuantFp8.Kernel;

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{
        .input_dtype = .bf16,
        .output_dtype = .f8e5m2,
        .scale_dtype = .bf16,
        .block = 128,
        .fp8_min = -57344.0,
        .fp8_max = 57344.0,
        .use_ue8m0 = false,
    } },
};

// =============================================================================
// XLA-pipeline driver. 5 inputs + 2 outputs (`y_q`, `y_s`) → 7 `tt.func`
// params. Shapes are synthetic; XLA only runs the codegen pipeline.
// =============================================================================

const NUM_TOKENS: i64 = 64;
const NUM_COLS: i64 = 1024;
const BLOCK: i64 = 128;
const GROUPS_PER_ROW: i64 = NUM_COLS / BLOCK;
const NUM_GROUPS: i64 = NUM_TOKENS * GROUPS_PER_ROW;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(
    y: Tensor,
    group_size: Tensor,
    y_num_columns: Tensor,
    y_row_stride: Tensor,
    eps: Tensor,
) struct { Tensor, Tensor } {
    const out = ops.triton(
        .{ y, group_size, y_num_columns, y_row_stride, eps },
        .{
            Shape.init(.{NUM_GROUPS * BLOCK}, .f8e5m2),
            Shape.init(.{NUM_GROUPS}, .bf16),
        },
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ 1, 1, 1 },
            .num_warps = 4,
            .num_stages = 1,
        },
    );
    return .{ out[0], out[1] };
}

pub fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
    return .{
        Tensor.init(.{NUM_TOKENS * NUM_COLS}, .bf16), // y_ptr
        Tensor.init(.{1}, .i64), // group_size_ptr
        Tensor.init(.{1}, .i64), // y_num_columns_ptr
        Tensor.init(.{1}, .i64), // y_row_stride_ptr
        Tensor.init(.{1}, .f32), // eps_ptr
    };
}
