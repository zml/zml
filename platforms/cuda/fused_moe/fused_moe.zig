const std = @import("std");
const c = @import("c");

const log = std.log.scoped(.@"platforms/cuda/fused_moe");

// External CUDA kernel function
pub extern fn fused_moe_kernel_launch(
    stream: ?*anyopaque,
    // Inputs (all on device)
    input: *const anyopaque,                    // [seq_len, hidden_dim] bf16
    gate_up_blocks: *const anyopaque,           // [num_experts, ffn_dim*2, hidden_dim] quantized f4e2m1
    gate_up_scales: *const anyopaque,           // [num_experts, ffn_dim*2, hidden_dim] f8e8m0
    gate_up_bias: ?*const anyopaque,            // [num_experts, ffn_dim*2] bf16 or nullptr
    down_blocks: *const anyopaque,              // [num_experts, hidden_dim, ffn_dim] quantized f4e2m1
    down_scales: *const anyopaque,              // [num_experts, hidden_dim, ffn_dim] f8e8m0
    down_bias: ?*const anyopaque,               // [num_experts, hidden_dim] bf16 or nullptr
    expert_indices: [*]const c_int,             // [seq_len, top_k] - expert IDs
    routing_scores: [*]const f32,               // [seq_len, top_k] - routing scores
    token_mask: ?[*]const c_int,                // [seq_len] - 1 if valid, 0 otherwise, or nullptr
    workspace: *anyopaque,                      // Device workspace for intermediate arrays
    output: *anyopaque,                         // [seq_len, hidden_dim] bf16 (initialized to zero)
    seq_len: c_int,
    num_experts: c_int,
    top_k: c_int,
    hidden_dim: c_int,
    ffn_dim: c_int,
    block_size: c_int,
) callconv(.c) c_int;

pub fn launchFusedMoEKernel(
    stream: c.CUstream,
    input_ptr: *const anyopaque,
    gate_up_blocks_ptr: *const anyopaque,
    gate_up_scales_ptr: *const anyopaque,
    gate_up_bias_ptr: ?*const anyopaque,
    down_blocks_ptr: *const anyopaque,
    down_scales_ptr: *const anyopaque,
    down_bias_ptr: ?*const anyopaque,
    expert_indices_ptr: [*]const c_int,
    routing_scores_ptr: [*]const f32,
    token_mask_ptr: ?[*]const c_int,
    workspace_ptr: *anyopaque,
    output_ptr: *anyopaque,
    params: struct {
        seq_len: c_int,
        num_experts: c_int,
        top_k: c_int,
        hidden_dim: c_int,
        ffn_dim: c_int,
        block_size: c_int,
    },
) !void {
    const result = fused_moe_kernel_launch(
        stream,
        input_ptr,
        gate_up_blocks_ptr,
        gate_up_scales_ptr,
        gate_up_bias_ptr,
        down_blocks_ptr,
        down_scales_ptr,
        down_bias_ptr,
        expert_indices_ptr,
        routing_scores_ptr,
        token_mask_ptr,
        workspace_ptr,
        output_ptr,
        params.seq_len,
        params.num_experts,
        params.top_k,
        params.hidden_dim,
        params.ffn_dim,
        params.block_size,
    );

    if (result != 0) {
        log.err("fused_moe_kernel_launch failed with error code: {}", .{result});
        return error.KernelLaunchFailed;
    }
}

// Calculate workspace size needed (all on device)
pub fn calculateWorkspaceSize(
    seq_len: usize,
    num_experts: usize,
    top_k: usize,
) usize {
    // Workspace layout:
    // - sorted_token_indices: [seq_len] * top_k
    // - expert_offsets: [num_experts + 1] * top_k
    // - restore_indices: [seq_len] * top_k
    // - expert_indices_1d: [seq_len]
    // - routing_scores_1d: [seq_len]
    const sorted_token_indices_size = seq_len * top_k * @sizeOf(c_int);
    const expert_offsets_size = (num_experts + 1) * top_k * @sizeOf(c_int);
    const restore_indices_size = seq_len * top_k * @sizeOf(c_int);
    const expert_indices_1d_size = seq_len * @sizeOf(c_int);
    const routing_scores_1d_size = seq_len * @sizeOf(f32);
    
    return sorted_token_indices_size + expert_offsets_size + restore_indices_size + 
           expert_indices_1d_size + routing_scores_1d_size;
}
