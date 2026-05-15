//! Registration of `count_and_sort_expert_tokens_kernel` for the harness.
//!
//! Re-exports the production Kernel from `zml.moe.triton_kernels`
//! (`CountAndSortExpertTokens`). One default sweep mirroring the dump
//! config in `examples/triton_emitter/kernels_zig.zig`.
//!
//! In-place kernel: the 2 `tt.func` "output" pointers alias the mutable
//! `sorted_token_ids` and `cumsum` input buffers.

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const Tensor = zml.Tensor;

pub const Kernel = zml.moe.triton_kernels.CountAndSortExpertTokens.Kernel;

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{
        .numel = 1024,
        .num_experts = 8,
        .sort_block_size = 256,
    } },
};

// =============================================================================
// XLA-pipeline driver. 3 inputs + 2 outputs (out0/out1 aliasing inputs 1/2)
// → 5 `tt.func` params. Shapes match the default sweep.
// =============================================================================

const NUMEL: i64 = 1024;
const NUM_EXPERTS: i64 = 8;
const MAX_NUM_TOKENS_PADDED: i64 = 2048;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(
    topk_ids: Tensor,
    sorted_token_ids: Tensor,
    cumsum: Tensor,
) struct { Tensor, Tensor } {
    const out = ops.triton(
        .{ topk_ids, sorted_token_ids, cumsum },
        .{ sorted_token_ids.shape(), cumsum.shape() },
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ 4, 1, 1 },
            .num_warps = 4,
            .num_stages = 1,
            .output_operand_aliases = &.{
                .{ .output_index = 0, .operand_index = 1 },
                .{ .output_index = 1, .operand_index = 2 },
            },
        },
    );
    return .{ out[0], out[1] };
}

pub fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
    return .{
        Tensor.init(.{NUMEL}, .i32), // topk_ids_ptr
        Tensor.init(.{MAX_NUM_TOKENS_PADDED}, .i32), // sorted_token_ids_ptr
        Tensor.init(.{NUM_EXPERTS + 1}, .i32), // cumsum_ptr
    };
}
