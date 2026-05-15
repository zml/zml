//! Registration of the `ragged_paged_attention_kernel` Mosaic-TPU kernel.
//!
//! `platforms/tpu/ragged_paged.zig` exposes the lower-layer
//! `Builder.build(...)` form (`Cfg` + `buildIr`) rather than the high-layer
//! `Kernel(Config, spec)` declarative form, so this file wraps it into a
//! `Kernel`-shaped struct that the harness's entry shim can consume.
//!
//! Sweeps mirror the `DEFAULT_COMBOS` matrix from
//! `examples/mosaic_ragged_paged/compare.sh`: dtype + head_dim variants
//! plus the codegen-branch knobs (sliding window, soft cap, k/v scales,
//! GQA ratios).

const std = @import("std");

const harness = @import("harness");
const ragged_paged = @import("platforms/tpu/ragged_paged");
const zml = @import("zml");

// =============================================================================
// Kernel adapter — projects `ragged_paged.{Cfg, buildIr}` into the
// `Kernel`-shaped struct the harness's entry shim consumes.
// =============================================================================

pub const Kernel = struct {
    pub const name: [:0]const u8 = "ragged_paged_attention_kernel";
    pub const Config = ragged_paged.Cfg;

    pub fn emit(allocator: std.mem.Allocator, cfg: Config) ![:0]const u8 {
        const ctx = try zml.kernel.mosaic_tpu.newContext();
        defer ctx.deinit();
        return ragged_paged.buildIr(allocator, ctx, cfg);
    }
};

// =============================================================================
// Sweeps — direct port of `compare.sh:DEFAULT_COMBOS`. Each entry is a
// `(label, cfg)` pair that the harness sends to both sides for diffing.
// `kv_dtype = .bf16, head_dim = 128` is the baseline; everything else is
// a single-axis variant relative to it.
// =============================================================================

const baseline = ragged_paged.Cfg{
    .num_q_tokens = 8,
    .num_q_heads = 8,
    .num_kv_heads = 2,
    .head_dim = 128,
    .total_num_pages = 8,
    .page_size = 16,
    .max_num_seqs = 2,
    .pages_per_seq = 2,
    .num_kv_pages_per_block = 1,
    .num_queries_per_block = 8,
    .sm_scale = 0.0883883461,
    .q_dtype = .bf16,
    .kv_dtype = .bf16,
};

fn override(comptime patches: anytype) ragged_paged.Cfg {
    var c = baseline;
    inline for (std.meta.fields(@TypeOf(patches))) |f| {
        @field(c, f.name) = @field(patches, f.name);
    }
    return c;
}

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "baseline", .cfg = baseline },

    // dtype / head_dim sweep.
    .{ .label = "head_dim_256", .cfg = override(.{ .head_dim = 256 }) },
    .{ .label = "kv_f16", .cfg = override(.{ .kv_dtype = .f16 }) },
    .{ .label = "kv_f32", .cfg = override(.{ .kv_dtype = .f32 }) },
    .{ .label = "kv_i32", .cfg = override(.{ .kv_dtype = .i32 }) },
    .{ .label = "kv_i8", .cfg = override(.{ .kv_dtype = .i8 }) },
    .{ .label = "kv_f8e4m3fn", .cfg = override(.{ .kv_dtype = .f8e4m3fn }) },
    .{ .label = "kv_f16_hd256", .cfg = override(.{ .kv_dtype = .f16, .head_dim = 256 }) },
    .{ .label = "kv_f32_hd256", .cfg = override(.{ .kv_dtype = .f32, .head_dim = 256 }) },
    .{ .label = "kv_i8_hd256", .cfg = override(.{ .kv_dtype = .i8, .head_dim = 256 }) },

    // Codegen-branch knobs.
    .{ .label = "sliding_16", .cfg = override(.{ .sliding_window = @as(?i64, 16) }) },
    .{ .label = "sliding_8", .cfg = override(.{ .sliding_window = @as(?i64, 8) }) },
    .{ .label = "softcap_30", .cfg = override(.{ .soft_cap = @as(?f32, 30.0) }) },
    .{ .label = "k_scale_05", .cfg = override(.{ .k_scale = @as(?f32, 0.5) }) },
    .{ .label = "v_scale_025", .cfg = override(.{ .v_scale = @as(?f32, 0.25) }) },
    .{ .label = "kv_scales", .cfg = override(.{ .k_scale = @as(?f32, 0.5), .v_scale = @as(?f32, 0.25) }) },
    .{ .label = "sliding_softcap", .cfg = override(.{ .sliding_window = @as(?i64, 16), .soft_cap = @as(?f32, 30.0) }) },

    // q_dtype variants — Pallas's `fold_on_2nd_minor` rejects q=f16, so we
    // only sweep f32 here.
    .{ .label = "q_f32", .cfg = override(.{ .q_dtype = .f32 }) },

    // GQA-ratio sweep (must satisfy num_q_heads % num_kv_heads == 0).
    .{ .label = "gqa_8x8", .cfg = override(.{ .num_q_heads = 8, .num_kv_heads = 8 }) },
    .{ .label = "gqa_8x1", .cfg = override(.{ .num_q_heads = 8, .num_kv_heads = 1 }) },
};
