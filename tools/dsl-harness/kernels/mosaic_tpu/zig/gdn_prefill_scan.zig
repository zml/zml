//! Harness adapter for GDN P5 (`recurrent_scan`). The kernel implementation
//! lives at `zml/attention/mosaic_tpu_kernels/gdn_prefill.zig` and is re-exported
//! via `zml.attention.gdn_prefill`. This file declares only the sweep configs
//! that the harness diffs against the Pallas reference at
//! `kernels/mosaic_tpu/py/gdn_prefill_scan.py` (which monkeypatches the
//! pallas_call into `name="recurrent_scan"` to match `Kernel.name`).

const zml = @import("zml");
const harness = @import("harness");

const gdn_prefill = zml.attention.gdn_prefill;

pub const Kernel = gdn_prefill.Kernel;
const Cfg = gdn_prefill.Cfg;

// =============================================================================
// Sweeps — same matrix as before the consolidation. Each `(label, cfg)` pair
// is sent to both the Zig kernel (`gdn_prefill.Kernel.emit`) and the Python
// reference for IR-diff. Add new entries here without touching the kernel.
// =============================================================================

const baseline = Cfg{
    // Small dev config that still exercises every path: 2 decode reqs (1 tok each)
    // + 2 prefill reqs (32, 16 toks) ⇒ num_tokens=50, num_seqs=4; chunk_size=16
    // ⇒ a start-transition block (decode↔prefill boundary at token 2, not
    // sublane-aligned), an end-transition block, regular prefill chunks, 1 decode
    // batch, and a stitch step on block 0.
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 1,
    .num_v_heads = 2,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .chunk_size = 16,
    .num_tokens = 50,
    .num_seqs = 4,
};

const qwen_realistic = Cfg{
    // Qwen3.5-9B config: n_kq=16, n_v=32 (GQA factor 2), chunk_size=64 (4× the
    // small case ⇒ `invert_triangular_matrix` runs 4 size-16 block iterations
    // instead of 1). Tests the GQA-repeat path.
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 32,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .chunk_size = 64,
    .num_tokens = 388,
    .num_seqs = 6,
};

const qwen_2b = Cfg{
    // Qwen3.5-2B config: n_kq=16, n_v=16, repeat_factor=1 (no GQA repeat),
    // chunk_size=64. Exercises the n_v=16 arm of `SUPPORTED_N_V` in the decode
    // arm's per-head matmul unroll, and the `rf==1` fast path in the prefill
    // arm's GQA-repeat code (shape-cast + broadcast become no-ops).
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 16,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .chunk_size = 64,
    .num_tokens = 388,
    .num_seqs = 6,
};

// Bisect intermediates — kept in `SWEEPS` while we shake out fresh divergences
// from the n_v=2 → n_v=32 jump and the chunk_size=16 → 64 jump.
const small_nv32 = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 16,
    .num_v_heads = 32,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .chunk_size = 16,
    .num_tokens = 50,
    .num_seqs = 4,
};
const small_chunk32 = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 1,
    .num_v_heads = 2,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .chunk_size = 32,
    .num_tokens = 100,
    .num_seqs = 4,
};
const small_chunk64 = Cfg{
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 1,
    .num_v_heads = 2,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .chunk_size = 64,
    .num_tokens = 200,
    .num_seqs = 4,
};

const decode_only = Cfg{
    // Every seq is 1 decode token (num_tokens == num_seqs).  decode_tokens =
    //   num_tokens, prefill_valid = 0 for every schedule row — exercises the
    //   decode path alone.
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 1,
    .num_v_heads = 2,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .chunk_size = 16,
    .num_tokens = 4,
    .num_seqs = 4,
};

const prefill_only = Cfg{
    // No decode tokens at all (decode_tokens=0 set via harness).  Exercises
    //   regular+transition prefill alone; the decode body's `pl.when(decode_valid>0)`
    //   never fires.
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 1,
    .num_v_heads = 2,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .chunk_size = 16,
    .num_tokens = 64,
    .num_seqs = 2,
};

const small_chunk128 = Cfg{
    // chunk_size=128 — max supported.  Exercises 8 blocks of
    //   `invert_triangular_matrix`, 128-iter cumsum unroll, hierarchical
    //   `tpu.concatenate` with 8 groups.
    .dtype = .bf16,
    .state_dtype = .bf16,
    .num_k_heads = 1,
    .num_v_heads = 2,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .chunk_size = 128,
    .num_tokens = 400,
    .num_seqs = 4,
};

pub const SWEEPS: []const harness.Sweep(Cfg) = &.{
    .{ .label = "small", .cfg = baseline },
    .{ .label = "small_nv32", .cfg = small_nv32 },
    .{ .label = "small_chunk32", .cfg = small_chunk32 },
    .{ .label = "small_chunk64", .cfg = small_chunk64 },
    .{ .label = "small_chunk128", .cfg = small_chunk128 },
    .{ .label = "decode_only", .cfg = decode_only },
    .{ .label = "prefill_only", .cfg = prefill_only },
    .{ .label = "qwen_realistic", .cfg = qwen_realistic },
    .{ .label = "qwen_2b", .cfg = qwen_2b },
};
