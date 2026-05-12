//! Harness adapter for GDN P4 (`fused_decoding_gdn`). The kernel implementation
//! lives at `zml/attention/mosaic_tpu_kernels/gdn_decode.zig` and is re-exported
//! via `zml.attention.gdn_decode`. This file declares only the sweep configs
//! that the harness diffs against the Pallas reference at
//! `kernels/mosaic_tpu/py/gdn_decode_fast.py`.

const zml = @import("zml");
const harness = @import("harness");

const gdn_decode = zml.attention.gdn_decode;

pub const Kernel = gdn_decode.Kernel; // alias for KernelFull
const Cfg = gdn_decode.Cfg;

// =============================================================================
// Sweeps — same matrix as before the consolidation. Each entry is a
// `(label, cfg)` pair the harness sends to both sides for diffing.
// `bt=8, n_v=128, bf16` is the baseline; everything else is a single-axis
// variant relative to it.
// =============================================================================

const baseline = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476, // 1/sqrt(128)
    .bt = 8,
};

const bt32_nv32 = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 32,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 32,
};

const bt16_nv64 = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 64,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 16,
};

const bt4_state_f32 = Cfg{
    .dtype = .bf16,
    .state_dtype = .f32,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 4,
};

const rf1_nv32 = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 32,
    .num_v_heads = 32,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 32,
};

const sigmoid_gate = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 8,
    .lower_bound = 0.5,
};

const no_l2norm = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 8,
    .use_qk_l2norm = false,
};

const k256 = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 256,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0625, // 1/sqrt(256)
    .bt = 4,
};

const v256 = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 256,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 4,
};

const no_dt_bias = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 8,
    .has_dt_bias = false,
};

const no_gate = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 8,
    .use_gate_in_kernel = false,
    .has_dt_bias = false,
};

const no_b = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 8,
    .has_b = false,
};

const mamba2_style = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 8,
    .use_qk_l2norm = false,
    .has_b = false,
    .lower_bound = 0.7,
};

const minimal = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 8,
    .use_gate_in_kernel = false,
    .has_dt_bias = false,
    .has_b = false,
};

const state_f16 = Cfg{
    .dtype = .bf16,
    .state_dtype = .f16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 8,
};

const f16_inputs = Cfg{
    .dtype = .f16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 8,
};

const f32_inputs = Cfg{
    .dtype = .f32,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 8,
};

const f32_all = Cfg{
    .dtype = .f32,
    .state_dtype = .f32,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 4,
};

const f16_state_f32 = Cfg{
    .dtype = .f16,
    .state_dtype = .f32,
    .num_k_heads = 16,
    .num_v_heads = 128,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .num_decode_tokens = 4,
    .num_states = 5,
    .scale = 0.0883883476,
    .bt = 4,
};

pub const SWEEPS: []const harness.Sweep(Cfg) = &.{
    .{ .label = "bt8_nv128", .cfg = baseline },
    .{ .label = "bt16_nv64", .cfg = bt16_nv64 },
    .{ .label = "bt32_nv32", .cfg = bt32_nv32 },
    .{ .label = "bt4_state_f32", .cfg = bt4_state_f32 },
    .{ .label = "rf1_nv32", .cfg = rf1_nv32 },
    .{ .label = "sigmoid_gate", .cfg = sigmoid_gate },
    .{ .label = "no_l2norm", .cfg = no_l2norm },
    .{ .label = "k256", .cfg = k256 },
    .{ .label = "v256", .cfg = v256 },
    .{ .label = "no_dt_bias", .cfg = no_dt_bias },
    .{ .label = "no_gate", .cfg = no_gate },
    .{ .label = "no_b", .cfg = no_b },
    .{ .label = "mamba2_style", .cfg = mamba2_style },
    .{ .label = "minimal", .cfg = minimal },
    .{ .label = "state_f16", .cfg = state_f16 },
    .{ .label = "f16_inputs", .cfg = f16_inputs },
    .{ .label = "f32_inputs", .cfg = f32_inputs },
    .{ .label = "f32_all", .cfg = f32_all },
    .{ .label = "f16_state_f32", .cfg = f16_state_f32 },
};
