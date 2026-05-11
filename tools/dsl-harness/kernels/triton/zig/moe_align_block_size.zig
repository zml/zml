//! Registration of `moe_align_block_size_kernel` for the harness.
//!
//! Re-exports the production Kernel from `zml.moe.triton_kernels`
//! (`MoeAlignBlockSize`). One default sweep mirroring the dump config in
//! `examples/triton_emitter/kernels_zig.zig`.
//!
//! In-place kernel: the 4 `tt.func` "output" pointers alias the 4 mutable
//! input buffers (`sorted_token_ids`, `expert_ids`, `num_tokens_post_pad`,
//! `cumsum`), so the XLA driver passes `output_operand_aliases` accordingly.

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const Tensor = zml.Tensor;

pub const Kernel = zml.moe.triton_kernels.MoeAlignBlockSize.Kernel;

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "default", .cfg = .{
        .numel = 1024,
        .num_experts = 8,
        .padded_num_experts = 8,
        .max_num_tokens_padded = 2048,
        .max_num_m_blocks = 32,
        .block_size_m = 64,
        .hist_block = 64,
    } },
};

// =============================================================================
// XLA-pipeline driver. 5 inputs + 4 outputs (out0..out3 aliasing inputs 1..4)
// → 9 `tt.func` params. Shapes match the default sweep; XLA only runs the
// codegen pipeline.
// =============================================================================

const NUMEL: i64 = 1024;
const NUM_EXPERTS: i64 = 8;
const MAX_NUM_TOKENS_PADDED: i64 = 2048;
const MAX_NUM_M_BLOCKS: i64 = 32;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(
    topk_ids: Tensor,
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_pad: Tensor,
    cumsum: Tensor,
) struct { Tensor, Tensor, Tensor, Tensor } {
    const out = ops.triton(
        .{ topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum },
        .{ sorted_token_ids.shape(), expert_ids.shape(), num_tokens_post_pad.shape(), cumsum.shape() },
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ 2, 1, 1 },
            .num_warps = 8,
            .num_stages = 1,
            .output_operand_aliases = &.{
                .{ .output_index = 0, .operand_index = 1 },
                .{ .output_index = 1, .operand_index = 2 },
                .{ .output_index = 2, .operand_index = 3 },
                .{ .output_index = 3, .operand_index = 4 },
            },
        },
    );
    return .{ out[0], out[1], out[2], out[3] };
}

pub fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
    return .{
        Tensor.init(.{NUMEL}, .i32), // topk_ids_ptr
        Tensor.init(.{MAX_NUM_TOKENS_PADDED}, .i32), // sorted_token_ids_ptr
        Tensor.init(.{MAX_NUM_M_BLOCKS}, .i32), // expert_ids_ptr
        Tensor.init(.{1}, .i32), // num_tokens_post_pad_ptr
        Tensor.init(.{NUM_EXPERTS + 1}, .i32), // cumsum_ptr
    };
}
