const std = @import("std");

const zml = @import("zml");
const config = @import("config.zig");
const packing = @import("../model/packing.zig");

pub const Decision = struct {
    attention: zml.attention.Backend,
    resident_blocks: u32,
    group_size: u32,
    score_bytes: u64,
    fa2_scratch_bytes: u64,
    adaln_table_bytes: u64,
    fixed_bytes: u64,
    resident_core_bytes: u64,
    transient_core_bytes: u64,
    denoise_live_bytes: u64,
    denoise_peak_bytes: u64,
    budget_bytes: u64,
};

pub const Query = struct {
    target: zml.Target,
    dtype: zml.DataType,
    head_dim: i64,
    heads: i64,
    seq: u64,
    causal: bool,
    tp: u32,
    flash: zml.attention.Backend = .cuda_fa2,
};

pub fn isFlash(backend: zml.attention.Backend) bool {
    return backend == .cuda_fa2 or backend == .cuda_fa3;
}

pub fn selectAttention(q: Query) zml.attention.Backend {
    if (q.target != .cuda) return .vanilla;
    if (q.dtype != .bf16 and q.dtype != .f16) return .vanilla;
    if (q.head_dim < 16 or q.head_dim > 256 or @rem(q.head_dim, 8) != 0) return .vanilla;
    if (q.heads <= 0 or @rem(q.heads, @as(i64, @max(1, q.tp))) != 0) return .vanilla;
    if (isFlash(q.flash) and !q.flash.supportsHeadDim(q.head_dim)) return .vanilla;
    if (q.causal and q.seq < 2) return .vanilla;
    const heads_local: u64 = @intCast(@divExact(q.heads, @as(i64, @max(1, q.tp))));
    const quadratic = q.seq * q.seq * 4 * heads_local;
    const linear = q.seq * @as(u64, @intCast(q.head_dim)) * heads_local * 8;
    if (quadratic <= linear * 4) return .vanilla;
    return if (isFlash(q.flash)) q.flash else .cuda_fa2;
}

pub fn sdpaScoreBytes(seq: u64, heads: i64, tp: u32) u64 {
    const heads_local: u64 = @intCast(@divExact(@max(heads, 1), @as(i64, @max(1, tp))));
    return seq * seq * 4 * heads_local;
}

pub fn fa2ScratchBytes(seq: u64, heads: i64, head_dim: i64, tp: u32) u64 {
    const heads_local: u64 = @intCast(@divExact(@max(heads, 1), @as(i64, @max(1, tp))));
    const hd: u64 = @intCast(@max(head_dim, 1));
    const lse = seq * heads_local * 4;
    const lse_accum = heads_local * hd * 4;
    const out_accum = seq * heads_local * hd * 4;
    return lse + lse_accum + out_accum;
}

pub fn adalnTableBytes(steps: u32, hidden: i64, layers: i64, dtype_bytes: u32) u64 {
    const slots = packing.timestep_slot_count;
    const mods: u64 = @intCast(config.modality_count);
    const hid: u64 = @intCast(@max(hidden, 1));
    const per_block = @as(u64, steps) * slots * mods * 6 * hid * dtype_bytes;
    const final = @as(u64, steps) * slots * 2 * hid * dtype_bytes;
    return per_block * @as(u64, @intCast(@max(layers, 0))) + final;
}

pub const compile_group_cap: u32 = 4;
/// Encode is a single pass: extra resident layers do not help when weights
/// come from disk. Hide H2D behind compute on the stream path.
pub const enc_prefetch: u32 = 4;
/// Visual VAE keeps every block after the first fill; only the fill is parallel.
pub const vae_load_window: u32 = 8;

pub fn groupSize(resident: u32) u32 {
    if (resident <= 1) return 1;
    return @min(resident, compile_group_cap);
}

pub fn encPrefetch(layers: u32) u32 {
    if (layers <= 1) return 1;
    return @min(layers, enc_prefetch);
}

pub fn vaeLoadWindow(blocks: u32) u32 {
    if (blocks <= 1) return 1;
    return @min(blocks, vae_load_window);
}

pub const safety_numer: u64 = 85;
pub const safety_denom: u64 = 100;

/// PJRT's non-preallocated BFC allocator grows in chunks and retains those
/// chunks between phases. Measured HBM is about 1.5x live tensor bytes for H3,
/// so policy decisions reserve that allocator high-water overhead explicitly.
pub fn allocatorPeak(live_bytes: u64) u64 {
    return live_bytes +| live_bytes / 2;
}

pub fn ditKeepBlocks(resident: u32, layers: u32) u32 {
    return @min(resident, layers);
}

pub fn transientCoreBytes(resident: u32, layers: u32, per_core: u64) u64 {
    const streamed = layers - @min(resident, layers);
    return @as(u64, @min(streamed, 2)) * per_core;
}

pub fn chooseResidentBlocks(budget: u64, fixed: u64, per_core: u64, layers: u32) u32 {
    if (per_core == 0 or layers == 0 or budget <= fixed) return 0;
    var best: u32 = 0;
    var resident: u32 = 0;
    while (resident <= layers) : (resident += 1) {
        const live = fixed +| @as(u64, resident) *| per_core +| transientCoreBytes(resident, layers, per_core);
        const peak = allocatorPeak(live);
        if (peak <= budget) best = resident;
    }
    return best;
}

pub fn canPrefetchVae(device_bytes: u64, budget: u64, denoise_live: u64, vae_cache_bytes: u64) bool {
    if (device_bytes == 0 or vae_cache_bytes == 0) return false;
    return allocatorPeak(denoise_live +| vae_cache_bytes) <= budget;
}

pub fn tileBatch(tile_count: u32, tile_act_bytes: u64, headroom: u64, partitions: u32) u32 {
    if (tile_count == 0) return 1;
    const fit: u32 = if (tile_act_bytes == 0)
        tile_count
    else
        @intCast(@min(@as(u64, tile_count), @max(@as(u64, 1), headroom / tile_act_bytes)));
    const part = @max(1, partitions);
    if (part == 1) return @max(1, fit);
    if (fit < part) return 1;
    return (fit / part) * part;
}

pub fn decide(args: struct {
    target: zml.Target,
    seq: u64,
    hidden: i64,
    heads: i64,
    head_dim: i64,
    layers: u32,
    steps: u32,
    dtype: zml.DataType,
    device_bytes: u64,
    tp: u32,
    block_core_bytes: u64,
    dtype_bytes: u32,
    flash: zml.attention.Backend = .cuda_fa2,
    fixed_bytes: u64 = 0,
}) Decision {
    const attn = selectAttention(.{
        .target = args.target,
        .dtype = args.dtype,
        .head_dim = args.head_dim,
        .heads = args.heads,
        .seq = args.seq,
        .causal = false,
        .tp = args.tp,
        .flash = args.flash,
    });
    const hid: u64 = @intCast(@max(args.hidden, 1));
    const act = args.seq * hid * args.dtype_bytes * 8;
    const scores = sdpaScoreBytes(args.seq, args.heads, args.tp);
    const scratch = if (isFlash(attn))
        fa2ScratchBytes(args.seq, args.heads, args.head_dim, args.tp)
    else
        scores;
    const tables = adalnTableBytes(args.steps, args.hidden, args.layers, args.dtype_bytes);
    const collective = act / 4;
    const fixed = act +| scratch +| collective +| tables +| args.fixed_bytes;
    const budget = if (args.device_bytes == 0)
        std.math.maxInt(u64)
    else
        args.device_bytes * safety_numer / safety_denom;
    const per_core = if (args.block_core_bytes == 0) 0 else args.block_core_bytes;
    const resident = if (args.device_bytes == 0)
        @as(u32, 0)
    else
        chooseResidentBlocks(budget, fixed, per_core, args.layers);
    const resident_bytes = @as(u64, resident) *| per_core;
    const transient_bytes = transientCoreBytes(resident, args.layers, per_core);
    const denoise_live = fixed +| resident_bytes +| transient_bytes;
    const denoise_peak = allocatorPeak(denoise_live);
    const group = groupSize(resident);
    return .{
        .attention = attn,
        .resident_blocks = resident,
        .group_size = group,
        .score_bytes = scores,
        .fa2_scratch_bytes = if (isFlash(attn)) scratch else 0,
        .adaln_table_bytes = tables,
        .fixed_bytes = fixed,
        .resident_core_bytes = resident_bytes,
        .transient_core_bytes = transient_bytes,
        .denoise_live_bytes = denoise_live,
        .denoise_peak_bytes = denoise_peak,
        .budget_bytes = budget,
    };
}

pub fn dtypeBytes(dt: zml.DataType) u32 {
    return @intCast(dt.sizeOf());
}
