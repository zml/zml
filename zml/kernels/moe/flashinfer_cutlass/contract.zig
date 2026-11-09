pub fn Inputs(comptime T: type) type {
    return struct {
        hidden_states: T,
        fc1_weights: T,
        fc2_weights: T,
        topk_weights: T,
        topk_ids: T,
        fc1_act_global: T,
        fc1_weight_block: T,
        fc1_global: T,
        fc2_act_global: T,
        fc2_weight_block: T,
        fc2_global: T,
    };
}

pub fn Outputs(comptime T: type) type {
    return struct {
        output: T,
        workspace: T,
    };
}

pub const Attributes = struct {
    runners: u64 = 0,
    num_tokens: i64,
    hidden_size: i64,
    intermediate_size: i64,
    num_experts: i32,
    top_k: i32,
    activation: i32,
    enable_pdl: bool,
    fc1_act_per_expert: bool,
    fc2_act_per_expert: bool,
    gemm1_tactic: i32,
    gemm2_tactic: i32,
};

pub const compute_on_host: bool = false;
pub const name = "flashinfer_cutlass_nvfp4_routed_moe";
