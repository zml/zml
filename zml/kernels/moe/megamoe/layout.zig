//! Host and comptime replica of DeepGEMM MegaMoE's static layouts (`sm100_fp8_fp4_mega_moe`,
//! vLLM's vendored DeepGEMM, equal to upstream 1fc9c66 for these files):
//!
//! * `layout/mega_moe.cuh`: the symmetric buffer (`MegaMoESignals`, `Workspace`,
//!   `MegaMoEBuffer` regions) and the pool/ring sizing helpers;
//! * `scheduler/mega_moe.cuh`: `TaskInfo` and the ring capacity bound
//!   (`get_num_max_live_pool_blocks`);
//! * `csrc/apis/mega_moe.hpp`: `build_symm_buffer_layout` (ring sizes over every candidate
//!   BLOCK_M, the buffer byte count of `get_symm_buffer_size_for_mega_moe`) and the token
//!   alignment;
//! * `csrc/jit_kernels/heuristics/mega_moe.hpp`: block config, pipeline config (stages and
//!   shared memory), `get_mega_moe_config`;
//! * the kernel's derived constants (`impls/sm100_fp8_fp4_mega_moe.cuh`: TMEM columns, register
//!   split, barrier ids, descriptors, `SharedStorage` offsets);
//! * `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` + `runtime_utils.hpp`: the
//!   parameters of the 18 `cuTensorMapEncodeTiled` calls.
//!
//! Every function mirrors the DeepGEMM function named in its doc comment, with the same
//! integer types and operation order (the BLOCK_M heuristic is evaluated in `f32` like the
//! C++). Pure Zig: usable at comptime and on the host. The ground truth for the tests in
//! `layout_test.zig` comes from DeepGEMM's own code (`gt/layout_gt.cu`).
const std = @import("std");

// ==================== layout/mega_moe.cuh, layout/sym_buffer.cuh ====================

/// `kCandidateBlockM`: every BLOCK_M a MegaMoE instantiation may use.
pub const candidate_block_ms = [_]u32{ 8, 16, 32, 64, 96, 128, 192, 240 };
pub const max_candidate_block_m: u32 = 240;
pub const min_candidate_block_m: u32 = 8;
/// `kLCMCandidateBlockM`: the token alignment of `num_max_tokens_per_rank`.
pub const lcm_candidate_block_m: u32 = 1920;
/// `layout::kNumMaxRanks`: `SymBuffer` always carries 72 offsets.
pub const max_ranks: u32 = 72;
/// `kNumDeviceLocalityDomains` (`common/types.cuh`), hard-coded 2 in DeepGEMM.
pub const device_locality_domains: u32 = 2;

pub fn ceilDiv(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return @divFloor(a + b - 1, b);
}

pub fn alignUp(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return ceilDiv(a, b) * b;
}

/// `get_num_max_pool_tokens`: worst-case received tokens plus per-expert BLOCK_M padding,
/// aligned to 1920.
pub fn numMaxPoolTokens(num_ranks: u32, num_max_tokens_per_rank: u32, num_topk: u32, num_experts_per_rank: u32) u32 {
    const num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const num_max_experts_per_token = @min(num_topk, num_experts_per_rank);
    return alignUp(num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (max_candidate_block_m - 1), lcm_candidate_block_m);
}

/// `get_num_sf_ring_tokens`: SF rows of a ring of `num_ring_tokens` at `block_m` (each block
/// padded to 128 rows for UTCCP).
pub fn numSfRingTokens(num_ring_tokens: u32, block_m: u32) u32 {
    return (num_ring_tokens / block_m) * alignUp(block_m, 128);
}

/// `get_num_max_shared_sf_tokens`: shared-expert SF rows over every candidate BLOCK_M.
pub fn numMaxSharedSfTokens(num_max_tokens_per_rank: u32) u32 {
    return ceilDiv(num_max_tokens_per_rank, min_candidate_block_m) * 128;
}

/// `layout::TokenSrcMetadata` (12 bytes).
pub const TokenSrcMetadata = extern struct {
    rank_idx: u32,
    token_idx: u32,
    topk_idx: u32,
};

/// `layout::MegaMoESignals` (`alignas(128)`, 21140864 bytes). Field offsets are `@offsetOf`.
pub const Signals = extern struct {
    pub const max_grid_sync_counters = 4;
    pub const max_experts = 2048;
    pub const max_experts_per_rank = 512;
    pub const max_ring_blocks = 1 << 20;
    pub const max_shared_l2_blocks = 1 << 15;

    grid_sync_count: [max_grid_sync_counters]u32 align(128),
    nvl_barrier_counter: u32,
    nvl_barrier_signals: [2]i32,
    l1_task_count: [device_locality_domains]u32,
    l2_task_count: [device_locality_domains]u32,
    shared_l1_task_count: [device_locality_domains]u32,
    shared_l2_task_count: [device_locality_domains]u32,
    combine_ready_grid_idx: [max_ranks]u64 align(128),
    peer_grid_idx: [max_ranks]u64 align(128),
    expert_send_count: [max_experts]u64 align(128),
    expert_recv_count: [max_experts]u64,
    expert_recv_count_sum: [max_experts_per_rank]u64,
    l1_full_count: [max_ring_blocks]u32,
    l1_empty_count: [max_ring_blocks]u32,
    l2_full_mask: [max_ring_blocks]u64,
    l2_empty_count: [max_ring_blocks]u32,
    shared_l2_full_count: [max_shared_l2_blocks]u32,

    /// Byte offset of `field` (`@offsetOf`).
    pub fn offset(comptime field: []const u8) u64 {
        return @offsetOf(Signals, field);
    }
};

comptime {
    std.debug.assert(@offsetOf(Signals, "combine_ready_grid_idx") == 128);
    std.debug.assert(@sizeOf(TokenSrcMetadata) == 12);
}

/// `layout::Workspace`: `MegaMoESignals` then `src_token_topk_idx` u32
/// `[experts_per_rank][ranks][num_max_tokens_per_rank]`, then `token_src_metadata`
/// `[num_max_pool_tokens]`, aligned to 16 bytes. Offsets are relative to the symmetric
/// buffer base.
pub const Workspace = struct {
    num_ranks: u32,
    num_experts: u32,
    num_experts_per_rank: u32,
    num_max_tokens_per_rank: u32,
    num_shared_l2_pool_blocks: u32,
    /// Full-pool span used by non-ring token metadata.
    num_max_pool_tokens: u32,

    pub fn init(num_ranks: u32, num_experts: u32, num_max_tokens_per_rank: u32, num_topk: u32, num_ring_tokens: u32) Workspace {
        const num_experts_per_rank = num_experts / num_ranks;
        const self: Workspace = .{
            .num_ranks = num_ranks,
            .num_experts = num_experts,
            .num_experts_per_rank = num_experts_per_rank,
            .num_max_tokens_per_rank = num_max_tokens_per_rank,
            .num_shared_l2_pool_blocks = ceilDiv(num_max_tokens_per_rank, min_candidate_block_m),
            .num_max_pool_tokens = numMaxPoolTokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank),
        };
        std.debug.assert(num_ranks > 0);
        std.debug.assert(num_ranks <= max_ranks);
        std.debug.assert(num_experts % num_ranks == 0);
        std.debug.assert(num_experts <= Signals.max_experts);
        std.debug.assert(num_experts_per_rank <= Signals.max_experts_per_rank);
        std.debug.assert(num_ring_tokens <= Signals.max_ring_blocks * min_candidate_block_m);
        std.debug.assert(self.num_shared_l2_pool_blocks <= Signals.max_shared_l2_blocks);
        return self;
    }

    /// `Workspace::get_num_bytes`.
    pub fn numBytes(self: Workspace) u64 {
        var num_bytes: u64 = @sizeOf(Signals);
        num_bytes += @as(u64, self.num_experts) * self.num_max_tokens_per_rank * @sizeOf(u32);
        num_bytes += @as(u64, self.num_max_pool_tokens) * @sizeOf(TokenSrcMetadata);
        return alignUp(num_bytes, 16);
    }

    /// `get_src_token_topk_idx_ptr(expert, rank, token)`.
    pub fn srcTokenTopkIdxOffset(self: Workspace, expert_idx: u64, rank_idx: u64, token_idx: u64) u64 {
        return @sizeOf(Signals) + ((expert_idx * self.num_ranks + rank_idx) * self.num_max_tokens_per_rank + token_idx) * @sizeOf(u32);
    }

    /// `get_token_src_metadata_ptr(pool_token)`.
    pub fn tokenSrcMetadataOffset(self: Workspace, pool_token_idx: u64) u64 {
        return self.srcTokenTopkIdxOffset(self.num_experts_per_rank, 0, 0) + pool_token_idx * @sizeOf(TokenSrcMetadata);
    }
};

/// `layout::Buffer`: `num_slots` (DeepGEMM's `num_ranks`) x `num_tokens` items of `item_bytes`
/// at byte `offset` of the symmetric buffer. `Data.require_tma_alignment` is checked by `init`.
pub const Region = struct {
    offset: u64,
    item_bytes: u64,
    num_slots: u64 = 1,
    num_tokens: u64,

    pub fn bytesPerSlot(self: Region) u64 {
        return self.num_tokens * self.item_bytes;
    }
    pub fn numBytes(self: Region) u64 {
        return self.bytesPerSlot() * self.num_slots;
    }
    pub fn end(self: Region) u64 {
        return self.offset + self.numBytes();
    }
    /// `get_rank_buffer(slot).get_data_buffer(token)`.
    pub fn itemOffset(self: Region, slot: u64, token: u64) u64 {
        return self.offset + slot * self.bytesPerSlot() + token * self.item_bytes;
    }
};

/// `layout::MegaMoEBuffer` with `with_sf = true` (FP8 x FP4): region offsets relative to the
/// symmetric buffer base, in address order.
pub const MegaMoEBuffer = struct {
    workspace: Workspace,
    /// `[T][H]` FP8 (also the shared expert's L1 activations).
    input_token: Region,
    /// `[T][H/128]` i32, K-major packed UE8M0.
    input_sf: Region,
    /// `[T][topk]` i64.
    input_topk_idx: Region,
    /// `[T][topk]` f32.
    input_topk_weights: Region,
    /// MN-major i32 `[H/128][num_max_shared_sf_tokens]` (rows of `H/32` bytes per SF token).
    shared_l1_sf: Region,
    /// `[T][I * shared]` FP8.
    shared_l2_token: Region,
    shared_l2_sf: Region,
    /// Ring `[num_ring_tokens][H]` FP8.
    l1_token: Region,
    /// MN-major i32 `[H/128][num_sf_ring_tokens]`.
    l1_sf: Region,
    /// `[num_ring_tokens]` f32.
    l1_topk_weights: Region,
    /// Ring `[num_ring_tokens][I]` FP8.
    l2_token: Region,
    l2_sf: Region,
    /// bf16 `[topk + (shared > 0)][T][H]`.
    combine_token: Region,

    pub fn init(
        hidden: u32,
        intermediate_hidden: u32,
        num_ranks: u32,
        num_experts: u32,
        num_max_tokens_per_rank: u32,
        num_topk: u32,
        num_ring_tokens: u32,
        num_sf_ring_tokens: u32,
        num_shared_experts: u32,
    ) MegaMoEBuffer {
        const workspace: Workspace = .init(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_ring_tokens);
        const shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const num_max_shared_sf_tokens = numMaxSharedSfTokens(num_max_tokens_per_rank);
        const has_shared = num_shared_experts > 0;
        const T: u64 = num_max_tokens_per_rank;

        // `layout::Data(n)` requires TMA alignment, `layout::Data(n, false)` does not.
        std.debug.assert(hidden % 16 == 0 and intermediate_hidden % 16 == 0 and shared_intermediate_hidden % 16 == 0);

        var self: MegaMoEBuffer = undefined;
        self.workspace = workspace;
        self.input_token = .{ .offset = workspace.numBytes(), .item_bytes = hidden, .num_tokens = T };
        self.input_sf = .{ .offset = self.input_token.end(), .item_bytes = hidden / 32, .num_tokens = T };
        self.input_topk_idx = .{ .offset = self.input_sf.end(), .item_bytes = num_topk * 8, .num_tokens = T };
        self.input_topk_weights = .{ .offset = self.input_topk_idx.end(), .item_bytes = num_topk * 4, .num_tokens = T };
        self.shared_l1_sf = .{ .offset = self.input_topk_weights.end(), .item_bytes = hidden / 32, .num_tokens = if (has_shared) num_max_shared_sf_tokens else 0 };
        self.shared_l2_token = .{ .offset = self.shared_l1_sf.end(), .item_bytes = shared_intermediate_hidden, .num_tokens = if (has_shared) T else 0 };
        self.shared_l2_sf = .{ .offset = self.shared_l2_token.end(), .item_bytes = shared_intermediate_hidden / 32, .num_tokens = if (has_shared) num_max_shared_sf_tokens else 0 };
        self.l1_token = .{ .offset = if (has_shared) self.shared_l2_sf.end() else self.input_topk_weights.end(), .item_bytes = hidden, .num_tokens = num_ring_tokens };
        self.l1_sf = .{ .offset = self.l1_token.end(), .item_bytes = hidden / 32, .num_tokens = num_sf_ring_tokens };
        self.l1_topk_weights = .{ .offset = self.l1_sf.end(), .item_bytes = 4, .num_tokens = num_ring_tokens };
        self.l2_token = .{ .offset = self.l1_topk_weights.end(), .item_bytes = intermediate_hidden, .num_tokens = num_ring_tokens };
        self.l2_sf = .{ .offset = self.l2_token.end(), .item_bytes = intermediate_hidden / 32, .num_tokens = num_sf_ring_tokens };
        self.combine_token = .{ .offset = self.l2_sf.end(), .item_bytes = @as(u64, hidden) * 2, .num_slots = num_topk + @as(u32, @intFromBool(has_shared)), .num_tokens = T };
        return self;
    }

    /// `MegaMoEBuffer::get_num_bytes`.
    pub fn numBytes(self: MegaMoEBuffer) u64 {
        return self.combine_token.end();
    }
};

// ==================== scheduler/mega_moe.cuh ====================

/// `sched::BlockPhase`.
pub const BlockPhase = enum(u32) { none = 0, linear1 = 1, linear2 = 2, shared_linear1 = 3, shared_linear2 = 4 };

/// `sched::TaskInfo` (`alignas(16)`, 32 bytes), as published to both CTAs with `st.async`.
pub const TaskInfo = extern struct {
    block_phase: u32 align(16),
    local_expert_idx: u32,
    m_block_idx: u32,
    n_cluster_idx: u32,
    pool_block_idx: u32,
    valid_m: u32,
    shape_n: u32,
    shape_k: u32,
};

comptime {
    std.debug.assert(@sizeOf(TaskInfo) == 32 and @alignOf(TaskInfo) == 16);
}

/// `sched::get_num_l1_warmup_waves` (`int` arithmetic).
pub fn numL1WarmupWaves(num_total_m_blocks: i32, num_clusters: i32, num_l1_n_clusters: i32, num_l2_n_clusters: i32) i32 {
    const num_first_l2_wave_m_blocks = ceilDiv(num_clusters, num_l2_n_clusters);
    const num_l1_warmup_clusters_for_first_l2_wave = ceilDiv(num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters);
    const num_interleave_cluster_diff_per_m_block = if (num_l1_n_clusters > num_l2_n_clusters) num_l1_n_clusters - num_l2_n_clusters else 0;
    const num_warmup_waves_for_interleave_schedule = ceilDiv(num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block, num_clusters) + 1;
    return @max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule);
}

/// `sched::get_num_max_live_pool_blocks`: the ring capacity, in pool blocks, the L1/L2
/// interleaved schedule can keep alive.
pub fn numMaxLivePoolBlocks(num_total_m_blocks: i32, num_sms: i32, hidden: i32, intermediate_hidden: i32) i32 {
    const block_n = 128;
    const ctas_per_cluster = 2;
    std.debug.assert(@mod(intermediate_hidden * 2, ctas_per_cluster * block_n) == 0);
    std.debug.assert(@mod(hidden, ctas_per_cluster * block_n) == 0);
    const num_clusters = @divTrunc(num_sms, ctas_per_cluster);
    const num_l1_n_clusters = @divTrunc(intermediate_hidden * 2, ctas_per_cluster * block_n);
    const num_l2_n_clusters = @divTrunc(hidden, ctas_per_cluster * block_n);
    const num_l1_clusters = num_total_m_blocks * num_l1_n_clusters;
    const num_l1_waves = ceilDiv(num_l1_clusters, num_clusters);
    const num_min_l1_warmup_waves = numL1WarmupWaves(num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters);
    const num_l1_warmup_waves = @min(num_min_l1_warmup_waves, num_l1_waves);
    const num_l1_warmup_clusters = @min(num_l1_warmup_waves * num_clusters, num_l1_clusters);
    const num_live_blocks_after_warmup = ceilDiv(num_l1_warmup_clusters, num_l1_n_clusters);
    const frontier_growth = if (num_l2_n_clusters > num_l1_n_clusters)
        ceilDiv(num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters), num_l2_n_clusters)
    else
        0;
    const wave_margin = ceilDiv(num_clusters, @min(num_l1_n_clusters, num_l2_n_clusters));
    return @min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin);
}

// ==================== csrc/apis/mega_moe.hpp ====================

/// The model-level shape of one MegaMoE instantiation (the API's `Problem`, with
/// `num_max_tokens_per_rank` already aligned by `alignTokens`).
pub const Shape = struct {
    num_ranks: u32,
    num_experts: u32,
    num_topk: u32,
    hidden: u32,
    intermediate_hidden: u32,
    num_shared_experts: u32,
    /// Aligned to `lcm_candidate_block_m` (`alignTokens`).
    num_max_tokens_per_rank: u32,

    pub fn numExpertsPerRank(self: Shape) u32 {
        return self.num_experts / self.num_ranks;
    }
    pub fn sharedIntermediateHidden(self: Shape) u32 {
        return self.intermediate_hidden * self.num_shared_experts;
    }
};

/// `get_token_alignment_for_mega_moe`.
pub const token_alignment = lcm_candidate_block_m;

/// What `deep_gemm.mega.SymmBuffer` does to the requested token count.
pub fn alignTokens(num_max_tokens_per_rank: u32) u32 {
    return alignUp(num_max_tokens_per_rank, token_alignment);
}

/// `build_symm_buffer_layout` (FP8 x FP4, swiglu): ring sizes over every candidate BLOCK_M,
/// the buffer regions and the byte count `get_symm_buffer_size_for_mega_moe` returns.
pub const SymmBufferLayout = struct {
    num_ring_tokens: u32,
    num_sf_ring_tokens: u32,
    buffer: MegaMoEBuffer,

    pub fn init(shape: Shape, num_sms: u32) SymmBufferLayout {
        std.debug.assert(shape.num_experts % shape.num_ranks == 0);
        std.debug.assert(shape.num_max_tokens_per_rank % token_alignment == 0);
        const num_experts_per_rank: i32 = @intCast(shape.numExpertsPerRank());
        const num_active_topk = @min(@as(i32, @intCast(shape.num_topk)), num_experts_per_rank);
        const num_max_routed_tokens = @as(i32, @intCast(shape.num_max_tokens_per_rank)) * @as(i32, @intCast(shape.num_ranks)) * num_active_topk;

        var num_ring_tokens: i32 = 0;
        for (candidate_block_ms) |block_m_| {
            const block_m: i32 = @intCast(block_m_);
            const num_pool_blocks = ceilDiv(num_max_routed_tokens, block_m) + num_experts_per_rank;
            const num_live_pool_blocks = numMaxLivePoolBlocks(num_pool_blocks, @intCast(num_sms), @intCast(shape.hidden), @intCast(shape.intermediate_hidden));
            num_ring_tokens = @max(num_ring_tokens, num_live_pool_blocks * block_m);
        }
        num_ring_tokens = alignUp(num_ring_tokens, @as(i32, lcm_candidate_block_m));

        var num_sf_ring_tokens: u32 = 0;
        for (candidate_block_ms) |block_m| num_sf_ring_tokens = @max(num_sf_ring_tokens, numSfRingTokens(@intCast(num_ring_tokens), block_m));

        std.debug.assert(shape.hidden % 128 == 0 and shape.intermediate_hidden % 128 == 0);
        std.debug.assert(shape.sharedIntermediateHidden() % 128 == 0);
        std.debug.assert(num_sf_ring_tokens % 4 == 0);
        return .{
            .num_ring_tokens = @intCast(num_ring_tokens),
            .num_sf_ring_tokens = num_sf_ring_tokens,
            .buffer = .init(shape.hidden, shape.intermediate_hidden, shape.num_ranks, shape.num_experts, shape.num_max_tokens_per_rank, shape.num_topk, @intCast(num_ring_tokens), num_sf_ring_tokens, shape.num_shared_experts),
        };
    }

    /// `get_symm_buffer_size_for_mega_moe(...)[0]`.
    pub fn numBytes(self: SymmBufferLayout) u64 {
        return self.buffer.numBytes();
    }
};

// ==================== csrc/jit_kernels/heuristics/mega_moe.hpp ====================

pub const MmaKind = enum { bf16, mxfp8fp4 };

/// `SM100ArchSpec::smem_capacity`.
pub const smem_capacity: u32 = 232448;

/// `get_block_config_for_mega_moe`'s result: `{cluster, block_m, store_block_m_l1,
/// store_block_m_l2, block_k, num_epilogue_threads}`.
pub const BlockConfig = struct {
    cluster_size: u32 = 2,
    block_m: u32,
    store_block_m_l1: u32,
    store_block_m_l2: u32,
    block_k: u32,
    num_epilogue_threads: u32 = 2 * 128,
};

fn numMmaElemBytes(kind: MmaKind) u32 {
    return if (kind == .bf16) 2 else 1;
}

/// `get_block_config_for_mega_moe`. The expected tokens per expert and its one-sigma margin
/// are `float`s in DeepGEMM; this evaluates the same `f32` operations in the same order.
pub fn blockConfig(num_ranks: u32, num_experts: u32, num_topk: u32, num_tokens: u32, kind: MmaKind) BlockConfig {
    const expected: f32 = @as(f32, @floatFromInt(num_tokens)) * @as(f32, @floatFromInt(num_ranks)) * @as(f32, @floatFromInt(num_topk)) / @as(f32, @floatFromInt(num_experts));
    const covered: f32 = expected + @sqrt(expected);

    var block_m: u32 = if (expected <= 10) 16 else 32;
    if (expected > 24) {
        var num_blocks: i32 = @intFromFloat(@ceil(covered / 240.0));
        if (num_blocks == 1 and covered > 192 and num_experts / num_ranks < 14) num_blocks = 2;
        for ([_]u32{ 64, 128, 192, 240 }) |candidate| {
            block_m = candidate;
            if (@as(f32, @floatFromInt(num_blocks * @as(i32, @intCast(candidate)))) >= covered) break;
        }
    }
    const store_block_m_l1: u32 = if (block_m <= 16) 8 else if (block_m <= 64) 16 else if (block_m <= 192) 32 else if (kind == .mxfp8fp4) 24 else 40;
    const store_block_m_l2: u32 = if (block_m == 240 and kind == .mxfp8fp4) 8 else store_block_m_l1;
    return .{
        .block_m = block_m,
        .store_block_m_l1 = store_block_m_l1,
        .store_block_m_l2 = store_block_m_l2,
        .block_k = 128 / numMmaElemBytes(kind),
    };
}

pub const PipelineConfig = struct { num_stages: u32, smem_size: u32 };

/// `get_pipeline_config_for_mega_moe`: the deepest A/B/SF ring that fits next to the fixed
/// shared memory, and the dynamic shared memory size.
pub fn pipelineConfig(
    capacity: u32,
    num_experts: u32,
    block_m: u32,
    block_n: u32,
    block_k: u32,
    num_bytes_per_pull: u32,
    store_block_m_l1: u32,
    store_block_m_l2: u32,
    sf_block_m: u32,
    sf_block_n: u32,
    gran_k: u32,
    num_dispatch_warps: u32,
    num_epilogue_warps: u32,
    kind: MmaKind,
) PipelineConfig {
    const smem_alignment = 1024;
    const num_epilogue_stages = 2;
    const num_tma_store_stages = 2;
    const elem_bytes = numMmaElemBytes(kind);
    const with_sf = kind == .mxfp8fp4;
    const load_block_m = block_m / 2;

    const smem_expert_count_size = alignUp(num_experts * 4, smem_alignment);
    const smem_send_buffers_size = alignUp(num_bytes_per_pull * num_dispatch_warps, smem_alignment);
    const smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    const num_epilogue_warpgroups = num_epilogue_warps / 4;
    const smem_cd_l1 = num_epilogue_warpgroups * store_block_m_l1 * (block_n / 2) * num_tma_store_stages * elem_bytes;
    const smem_cd_l2 = num_epilogue_warpgroups * store_block_m_l2 * block_n * 2;
    const smem_cd = alignUp(@max(smem_cd_l1, smem_cd_l2), smem_alignment);

    const num_schedule_stages = 2;
    const smem_task_info = num_schedule_stages * @sizeOf(TaskInfo);
    const smem_barriers = (num_dispatch_warps + num_epilogue_stages * 2 + num_epilogue_warps * 2 + num_schedule_stages * 2) * 8;
    const smem_amax_reduction = if (with_sf) store_block_m_l1 * num_epilogue_warps * 4 else 0;
    const smem_tmem_ptr = 4;

    const smem_sfa_per_stage = if (with_sf) sf_block_m * (block_k / gran_k) else 0;
    const smem_sfb_per_stage = if (with_sf) sf_block_n * (block_k / gran_k) else 0;
    const smem_a_size_per_stage = load_block_m * block_k * elem_bytes;
    const smem_b_size_per_stage = block_n * block_k * elem_bytes;
    std.debug.assert(smem_a_size_per_stage % smem_alignment == 0);
    std.debug.assert(smem_b_size_per_stage % smem_alignment == 0);
    const smem_stage_barriers = 2 * 8;
    const smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + smem_stage_barriers;

    const smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers + smem_task_info + smem_tmem_ptr;
    const num_stages = (capacity - smem_fixed) / smem_size_per_stage;
    std.debug.assert(num_stages >= 2);
    return .{ .num_stages = num_stages, .smem_size = smem_fixed + num_stages * smem_size_per_stage };
}

/// `MegaMoEConfig` as `get_mega_moe_config` builds it for the FP8 x FP4 kernel.
pub const Config = struct {
    block_m: u32,
    block_n: u32 = 128,
    block_k: u32,
    load_block_m: u32,
    load_block_n: u32 = 128,
    store_block_m_l1: u32,
    store_block_m_l2: u32,
    sf_block_m: u32,
    sf_block_n: u32,
    num_ring_tokens: u32,
    num_sf_ring_tokens: u32,
    swizzle_acts_mode: u32 = 128,
    swizzle_weights_mode: u32 = 128,
    num_stages: u32,
    smem_size: u32,
    num_dispatch_threads: u32 = 128,
    num_non_epilogue_threads: u32 = 128,
    num_epilogue_threads: u32 = 256,
    num_bytes_per_pull: u32,

    pub fn numThreads(self: Config) u32 {
        return self.num_dispatch_threads + self.num_non_epilogue_threads + self.num_epilogue_threads;
    }
};

/// `get_mega_moe_config(num_ranks, num_experts, ..., num_tokens, ...)` with
/// `MmaKind::MXFP8FP4` (DeepGEMM maps both `fp8xfp4` and `fp8xfp8` to it). `num_tokens` is
/// the per-call token count that selects BLOCK_M.
pub fn config(shape: Shape, num_tokens: u32, ring: SymmBufferLayout) Config {
    const kind: MmaKind = .mxfp8fp4;
    const bc = blockConfig(shape.num_ranks, shape.num_experts, shape.num_topk, num_tokens, kind);
    std.debug.assert(ring.num_ring_tokens % bc.block_m == 0);
    const block_n = 128;
    const sf_block_m = alignUp(bc.block_m, 128);
    const sf_block_n = alignUp(@as(u32, block_n), 128);
    const max_pull_bytes: u32 = 8192;
    var num_bytes_per_pull = shape.hidden * numMmaElemBytes(kind);
    while (num_bytes_per_pull > max_pull_bytes) {
        std.debug.assert(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }
    const pipeline = pipelineConfig(smem_capacity, shape.num_experts, bc.block_m, block_n, bc.block_k, num_bytes_per_pull, bc.store_block_m_l1, bc.store_block_m_l2, sf_block_m, sf_block_n, 32, 4, bc.num_epilogue_threads / 32, kind);
    return .{
        .block_m = bc.block_m,
        .block_k = bc.block_k,
        .load_block_m = bc.block_m / 2,
        .store_block_m_l1 = bc.store_block_m_l1,
        .store_block_m_l2 = bc.store_block_m_l2,
        .sf_block_m = sf_block_m,
        .sf_block_n = sf_block_n,
        .num_ring_tokens = ring.num_ring_tokens,
        .num_sf_ring_tokens = ring.num_sf_ring_tokens,
        .num_stages = pipeline.num_stages,
        .smem_size = pipeline.smem_size,
        .num_epilogue_threads = bc.num_epilogue_threads,
        .num_bytes_per_pull = num_bytes_per_pull,
    };
}

// ==================== impls/sm100_fp8_fp4_mega_moe.cuh: derived constants ====================

/// Element type of the routed weights (`weight_dtype_t`).
pub const WeightType = enum {
    /// `cutlass::detail::float_e2m1_unpacksmem_t`: packed FP4 in global memory, one byte per
    /// element in shared memory (TMA `16U4_ALIGN16B`).
    fp4_unpacked,
    /// `cutlass::float_e4m3_t`.
    fp8,
};

/// `cute::UMMA::MXF8F6F4Format`.
pub const MmaFormat = enum(u32) { e4m3 = 0, e5m2 = 1, e2m3 = 3, e3m2 = 4, e2m1 = 5 };

/// `cute::UMMA::InstrDescriptorBlockScaled` bit layout.
pub const InstrDescBlockScaled = struct {
    a_format: MmaFormat,
    b_format: MmaFormat,
    m: u32,
    n: u32,
    a_sf_id: u32 = 0,
    b_sf_id: u32 = 0,

    /// `make_instr_desc_block_scaled<a, b, float, ue8m0, M, N, K, K>()` with the scale-factor
    /// ids of `make_runtime_instr_desc_with_sf_id` (dense, K-major, no negate, UE8M0 scale).
    pub fn bits(self: InstrDescBlockScaled) u32 {
        return (self.b_sf_id << 4) | (@backingInt(self.a_format) << 7) | (@backingInt(self.b_format) << 10) |
            ((self.n >> 3) << 17) | (1 << 23) | ((self.m >> 4) << 24) | (self.a_sf_id << 29);
    }
};

/// `cute::UMMA::LayoutType` (shared-memory descriptor layout field).
pub const SmemLayoutType = enum(u64) { swizzle_none = 0, swizzle_128b_base32b = 1, swizzle_128b = 2, swizzle_64b = 4, swizzle_32b = 6 };

/// High 32 bits of a `cute::UMMA::SmemDescriptor` (`mma::sm100::make_smem_desc`): SBO,
/// version 1, base offset 0, LBO mode 0, layout. The low word is `start >> 4 | (LBO >> 4) << 16`.
pub fn smemDescHi(layout: SmemLayoutType, stride_byte_offset: u32) u32 {
    return (stride_byte_offset >> 4) | (1 << 14) | @as(u32, @intCast(@backingInt(layout) << 29));
}

/// Every compile-time constant of `sm100_fp8_fp4_mega_moe_impl` for one instantiation.
pub const Kernel = struct {
    shape: Shape,
    cfg: Config,
    num_sms: u32,
    weight_type: WeightType = .fp4_unpacked,

    // Thread layout.
    pub const num_dispatch_warps = 4;
    pub const num_mma_non_epilogue_warps = 4;
    pub const num_epilogue_warps = 8;
    pub const num_epilogue_warpgroups = 2;
    pub const num_threads = 512;

    // MMA (always swap A/B, 2-CTA MMA, K-major).
    pub const layout_ad_m = 128;
    pub const umma_m = layout_ad_m * 2;
    pub const umma_block_k = 128;
    pub const umma_k = 32;
    pub const swizzle_a_mode = 128;
    pub const swizzle_b_mode = 128;
    pub const swizzle_cd_mode = 128;
    pub const gran_k = 32;
    pub const num_utccp_aligned_elems = 128;

    // Epilogue.
    pub const num_epilogue_stages = 2;
    pub const num_tma_store_stages = 2;
    pub const atom_m = 8;
    pub const num_bank_group_bytes = 16;

    // Scheduler.
    pub const num_schedule_stages = 2;

    // Named barriers (`bar.sync` / `barrier.sync` ids).
    pub const dispatch_barrier_idx = 0;
    pub const dispatch_with_epilogue_barrier_idx = 1;
    pub const epilogue_full_barrier_idx = 2;
    pub const epilogue_wg_barrier_start_idx = 3;

    // NVLink barrier tags and grid-sync counters.
    pub const before_dispatch_pull_barrier_tag = 1;
    pub const after_workspace_clean_barrier_tag = 2;
    pub const dispatch_grid_sync_index = 0;
    pub const epilogue_grid_sync_index = 1;

    /// `TmaDescriptor` load cache hint (`CacheHintSm100::EVICT_NORMAL`).
    pub const evict_normal: u64 = 0x1000000000000000;
    /// `CacheHintSm100::EVICT_FIRST` (dispatch pulls, combine loads).
    pub const evict_first: u64 = 0x12F0000000000000;
    /// `cute::Sm100MmaPeerBitMask`: 2-CTA TMA loads complete on the leader CTA's mbarrier.
    pub const peer_bit_mask: u32 = 0xFEFFFFFF;
    /// `comm::grid_sync`'s `kFinishSumTag`.
    pub const finish_sum_tag: u32 = 0x80000000;

    pub fn init(shape: Shape, cfg: Config, num_sms: u32, weight_type: WeightType) Kernel {
        const k: Kernel = .{ .shape = shape, .cfg = cfg, .num_sms = num_sms, .weight_type = weight_type };
        std.debug.assert(cfg.block_n == layout_ad_m and cfg.block_k == 128);
        std.debug.assert(cfg.block_m % 16 == 0);
        std.debug.assert(cfg.sf_block_m == alignUp(cfg.block_m, num_utccp_aligned_elems) and cfg.sf_block_n == cfg.block_n);
        std.debug.assert(cfg.num_dispatch_threads == 128 and cfg.num_non_epilogue_threads == 128 and cfg.num_epilogue_threads == 256);
        std.debug.assert(shape.num_experts % shape.num_ranks == 0);
        std.debug.assert(num_sms % 2 == 0 and num_sms > 1);
        std.debug.assert(shape.num_topk <= 32 and shape.num_topk + @intFromBool(shape.num_shared_experts > 0) <= 32);
        std.debug.assert(k.l1ShapeN() % (cfg.block_n * 2) == 0 and k.l2ShapeN() % (cfg.block_n * 2) == 0);
        std.debug.assert(k.registerSum() <= 64512);
        std.debug.assert(32 <= k.numTmemCols() and k.numTmemCols() <= 512);
        std.debug.assert(k.numL1BlockNs() <= 64);
        return k;
    }

    pub fn hasShared(self: Kernel) bool {
        return self.shape.num_shared_experts > 0;
    }
    pub fn l1ShapeN(self: Kernel) u32 {
        return self.shape.intermediate_hidden * 2;
    }
    pub fn l1ShapeK(self: Kernel) u32 {
        return self.shape.hidden;
    }
    pub fn l2ShapeN(self: Kernel) u32 {
        return self.shape.hidden;
    }
    pub fn l2ShapeK(self: Kernel) u32 {
        return self.shape.intermediate_hidden;
    }
    pub fn sharedL2ShapeK(self: Kernel) u32 {
        return self.l2ShapeK() * self.shape.num_shared_experts;
    }
    pub fn numExpertsPerRank(self: Kernel) u32 {
        return self.shape.numExpertsPerRank();
    }
    pub fn numTokensPerWarp(self: Kernel) u32 {
        return 32 / self.shape.num_topk;
    }
    /// `kNumActivateLanes` of the dispatch warps.
    pub fn numActivateLanes(self: Kernel) u32 {
        return self.numTokensPerWarp() * self.shape.num_topk;
    }
    pub fn numRingBlocks(self: Kernel) u32 {
        return self.cfg.num_ring_tokens / self.cfg.block_m;
    }
    pub fn numSharedSfTokens(self: Kernel) u32 {
        return numMaxSharedSfTokens(self.shape.num_max_tokens_per_rank);
    }
    /// `kNumLocalityDomains`: 2 when both GEMMs split into whole clusters per domain.
    pub fn numLocalityDomains(self: Kernel) u32 {
        const unit = self.cfg.block_n * 2 * device_locality_domains;
        return if (self.l1ShapeN() % unit == 0 and self.l2ShapeN() % unit == 0) device_locality_domains else 1;
    }

    // MMA shape.
    pub fn ummaN(self: Kernel) u32 {
        return self.cfg.block_m;
    }
    pub fn loadBlockM(self: Kernel) u32 {
        return self.cfg.block_m / 2;
    }
    pub fn loadBlockN(self: Kernel) u32 {
        return self.cfg.block_n;
    }

    /// `kNumEpiPrefetchStages`.
    pub fn numEpiPrefetchStages(self: Kernel) u32 {
        return if (self.cfg.block_m == 240) 1 else 0;
    }
    pub fn wgBlockM(self: Kernel) u32 {
        return self.cfg.block_m / num_epilogue_warpgroups;
    }
    pub fn l1OutBlockN(self: Kernel) u32 {
        return self.cfg.block_n / 2;
    }
    /// `AMAX_REDUCTION_WARP_BUFFER_SIZE` (float2 per warp).
    pub fn amaxReductionWarpBufferSize(self: Kernel) u32 {
        return self.cfg.store_block_m_l1 / 2;
    }
    pub fn numScheduleConsumerThreads(self: Kernel) u32 {
        return 2 * self.cfg.num_epilogue_threads;
    }

    // Tensor memory.
    pub fn numAccumTmemCols(self: Kernel) u32 {
        return self.ummaN() * num_epilogue_stages;
    }
    pub fn numSfaTmemCols(self: Kernel) u32 {
        return self.cfg.sf_block_m / 32;
    }
    pub fn numSfbTmemCols(self: Kernel) u32 {
        return self.cfg.sf_block_n / 32;
    }
    /// `utils::get_num_aligned_tmem_cols`.
    pub fn numTmemCols(self: Kernel) u32 {
        const n = self.numAccumTmemCols() + self.numSfaTmemCols() + self.numSfbTmemCols();
        std.debug.assert(n <= 512);
        return if (n <= 32) 32 else if (n <= 64) 64 else if (n <= 128) 128 else if (n <= 256) 256 else 512;
    }
    pub fn tmemStartColOfSfa(self: Kernel) u32 {
        return self.numAccumTmemCols();
    }
    pub fn tmemStartColOfSfb(self: Kernel) u32 {
        return self.numAccumTmemCols() + self.numSfaTmemCols();
    }

    // Registers (`setmaxnreg`).
    fn useMoreEpilogueRegisters(self: Kernel) bool {
        return self.numExpertsPerRank() <= 64;
    }
    pub fn numDispatchRegisters(self: Kernel) u32 {
        return if (self.useMoreEpilogueRegisters()) 48 else 96;
    }
    pub fn numNonEpilogueRegisters(self: Kernel) u32 {
        return if (self.useMoreEpilogueRegisters()) 40 else 72;
    }
    pub fn numEpilogueRegisters(self: Kernel) u32 {
        return if (self.useMoreEpilogueRegisters()) 208 else 168;
    }
    pub fn registerSum(self: Kernel) u32 {
        return self.numDispatchRegisters() * self.cfg.num_dispatch_threads + self.numNonEpilogueRegisters() * self.cfg.num_non_epilogue_threads + self.numEpilogueRegisters() * self.cfg.num_epilogue_threads;
    }

    // L1 -> L2 dependency (`sched::L2KBlockDependency`).
    pub fn numL1BlockNs(self: Kernel) u32 {
        return self.l1ShapeN() / self.cfg.block_n;
    }
    pub fn numL2BlockNs(self: Kernel) u32 {
        return self.l2ShapeN() / self.cfg.block_n;
    }
    pub fn numL1Clusters(self: Kernel) u32 {
        return self.numL1BlockNs() / 2;
    }
    pub fn numL2Clusters(self: Kernel) u32 {
        return self.numL2BlockNs() / 2;
    }
    pub fn numL1BlockNsPerL2KBlock(self: Kernel) u32 {
        return self.cfg.block_k / (self.cfg.block_n / 2);
    }
    pub fn l2FullMask(self: Kernel) u64 {
        const n = self.numL1BlockNs();
        return if (n == 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(n)) - 1;
    }
    /// `(SHARED_L2_SHAPE_K / BLOCK_N) * 2`: shared L2 waits for every shared L1 N block.
    pub fn sharedL2ExpectedBlocks(self: Kernel) u32 {
        return (self.sharedL2ShapeK() / self.cfg.block_n) * 2;
    }

    /// The impl's `transform_sf_token_idx`: UTCCP 4x32 transpose within each 128-group.
    pub fn transformSfTokenIdx(self: Kernel, token_idx_in_expert: u32) u32 {
        const idx = token_idx_in_expert % self.cfg.block_m;
        return token_idx_in_expert / self.cfg.block_m * self.cfg.sf_block_m + (idx & ~@as(u32, 127)) + (idx & 31) * 4 + ((idx >> 5) & 3);
    }

    // Descriptors.
    pub fn routedFormat(self: Kernel) MmaFormat {
        return if (self.weight_type == .fp8) .e4m3 else .e2m1;
    }
    /// Instruction descriptor of the routed (FP4 or FP8 weights) MMA, before
    /// `update_instr_desc_with_umma_n` and the per-k SF ids.
    pub fn routedInstrDesc(self: Kernel) InstrDescBlockScaled {
        return .{ .a_format = self.routedFormat(), .b_format = .e4m3, .m = umma_m, .n = self.ummaN() };
    }
    pub fn sharedInstrDesc(self: Kernel) InstrDescBlockScaled {
        return .{ .a_format = .e4m3, .b_format = .e4m3, .m = umma_m, .n = self.ummaN() };
    }
    /// `make_umma_desc<K, LOAD_BLOCK_{M,N}, 128, 128>`: SW128, SBO 8 x 128 B, LBO 0.
    pub fn abSmemDescHi() u32 {
        return smemDescHi(.swizzle_128b, 8 * 128);
    }
    /// `make_sf_desc`: SWIZZLE_NONE, SBO 8 x 16 B, LBO 0.
    pub fn sfSmemDescHi() u32 {
        return smemDescHi(.swizzle_none, 8 * 16);
    }

    // Transaction bytes completing on the leader's full barrier per stage.
    pub fn smemABytes(self: Kernel) u32 {
        return self.loadBlockM() * self.cfg.block_k;
    }
    pub fn smemBBytes(self: Kernel) u32 {
        return self.loadBlockN() * self.cfg.block_k;
    }
    pub fn smemSfaBytes(self: Kernel) u32 {
        return self.cfg.sf_block_m * (self.cfg.block_k / 128) * 4;
    }
    pub fn smemSfbBytes(self: Kernel) u32 {
        return self.cfg.sf_block_n * (self.cfg.block_k / 128) * 4;
    }
    /// TMA-A warp leader `arrive_and_expect_tx`.
    pub fn expectTxA(self: Kernel) u32 {
        return self.smemABytes() * 2 + self.smemSfaBytes() * 2;
    }
    /// TMA-B warp leader `arrive_and_expect_tx` (routed FP4 counts packed global bytes).
    pub fn expectTxRoutedB(self: Kernel) u32 {
        return self.smemBBytes() * 2 / (if (self.weight_type == .fp8) @as(u32, 1) else 2) + self.smemSfbBytes() * 2;
    }
    pub fn expectTxSharedB(self: Kernel) u32 {
        return self.smemBBytes() * 2 + self.smemSfbBytes() * 2;
    }

    /// The combine's chunking (`kNumChunks`): 1 when 3 chunk slots x 8 warps fit in the
    /// reusable shared memory and H fits the per-lane register budget, else 2.
    pub fn numCombineChunks(self: Kernel) u32 {
        const hidden_bytes = self.shape.hidden * 2;
        const reusable = self.smem().dispatch_barriers;
        return if (3 * num_epilogue_warps * hidden_bytes <= reusable and self.shape.hidden <= 32 * 128) 1 else 2;
    }
    pub fn combineChunkBytes(self: Kernel) u32 {
        return self.shape.hidden * 2 / self.numCombineChunks();
    }

    pub fn smem(self: Kernel) SharedStorage {
        return .init(self);
    }
};

/// Byte offsets of the kernel's `SharedStorage` fields (C++ layout rules: the first four
/// members and the `smem_d` union are `alignas(1024)`, `task_infos` 16, barriers 8).
pub const SharedStorage = struct {
    expert_token_count: u32,
    dispatch_send_buffer: u32,
    smem_d: u32,
    /// `sizeof(smem_d)`: max(L1 FP8 staging, L2 BF16 staging), rounded to 1024.
    smem_d_bytes: u32,
    smem_a: u32,
    smem_b: u32,
    smem_sfa: u32,
    smem_sfb: u32,
    amax_reduction: u32,
    task_infos: u32,
    dispatch_barriers: u32,
    full_barriers: u32,
    empty_barriers: u32,
    tmem_full_barriers: u32,
    tmem_empty_barriers: u32,
    combine_barriers: u32,
    task_info_full_barriers: u32,
    task_info_empty_barriers: u32,
    tmem_ptr_in_smem: u32,
    /// Bytes up to the end of `tmem_ptr_in_smem` (= the host's `smem_size`).
    used_bytes: u32,
    /// `sizeof(SharedStorage)` (rounded to 1024).
    size: u32,

    pub fn init(k: Kernel) SharedStorage {
        const c = k.cfg;
        const stages = c.num_stages;
        var s: SharedStorage = undefined;
        var off: u32 = 0;
        s.expert_token_count = off;
        off += k.shape.num_experts * 4;
        off = alignUp(off, 1024);
        s.dispatch_send_buffer = off;
        off += Kernel.num_dispatch_warps * c.num_bytes_per_pull;
        off = alignUp(off, 1024);
        s.smem_d = off;
        const l1_bytes = Kernel.num_epilogue_warpgroups * Kernel.num_tma_store_stages * c.store_block_m_l1 * k.l1OutBlockN();
        const l2_bytes = Kernel.num_epilogue_warpgroups * c.store_block_m_l2 * c.block_n * 2;
        s.smem_d_bytes = alignUp(@max(l1_bytes, l2_bytes), 1024);
        off += s.smem_d_bytes;
        s.smem_a = off;
        off += stages * k.smemABytes();
        off = alignUp(off, 1024);
        s.smem_b = off;
        off += stages * k.smemBBytes();
        s.smem_sfa = off;
        off += stages * k.smemSfaBytes();
        s.smem_sfb = off;
        off += stages * k.smemSfbBytes();
        off = alignUp(off, 8);
        s.amax_reduction = off;
        off += Kernel.num_epilogue_warps * k.amaxReductionWarpBufferSize() * 8;
        off = alignUp(off, 16);
        s.task_infos = off;
        off += Kernel.num_schedule_stages * @sizeOf(TaskInfo);
        off = alignUp(off, 8);
        s.dispatch_barriers = off;
        off += Kernel.num_dispatch_warps * 8;
        s.full_barriers = off;
        off += stages * 8;
        s.empty_barriers = off;
        off += stages * 8;
        s.tmem_full_barriers = off;
        off += Kernel.num_epilogue_stages * 8;
        s.tmem_empty_barriers = off;
        off += Kernel.num_epilogue_stages * 8;
        s.combine_barriers = off;
        off += Kernel.num_epilogue_warps * 2 * 8;
        s.task_info_full_barriers = off;
        off += Kernel.num_schedule_stages * 8;
        s.task_info_empty_barriers = off;
        off += Kernel.num_schedule_stages * 8;
        s.tmem_ptr_in_smem = off;
        off += 4;
        s.used_bytes = off;
        s.size = alignUp(off, 1024);
        return s;
    }

    /// `kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers)` (combine reuse).
    pub fn reusableBytes(self: SharedStorage) u32 {
        return self.dispatch_barriers;
    }
};

// ==================== TMA descriptors (runtime_utils.hpp) ====================

/// `CUtensorMapDataType`.
pub const TensorMapDataType = enum(u32) {
    uint8 = 0,
    uint16 = 1,
    uint32 = 2,
    int32 = 3,
    uint64 = 4,
    int64 = 5,
    float16 = 6,
    float32 = 7,
    float64 = 8,
    bfloat16 = 9,
    float32_ftz = 10,
    tfloat32 = 11,
    tfloat32_ftz = 12,
    @"16u4_align8b" = 13,
    @"16u4_align16b" = 14,
    @"16u6_align16b" = 15,
};

/// `CUtensorMapSwizzle`.
pub const TensorMapSwizzle = enum(u32) { none = 0, @"32b" = 1, @"64b" = 2, @"128b" = 3, @"128b_atom_32b" = 4 };
/// `CUtensorMapL2promotion`.
pub const TensorMapL2Promotion = enum(u32) { none = 0, @"64b" = 1, @"128b" = 2, @"256b" = 3 };
/// `CUtensorMapInterleave`.
pub const TensorMapInterleave = enum(u32) { none = 0, @"16b" = 1, @"32b" = 2 };
/// `CUtensorMapFloatOOBfill`.
pub const TensorMapOobFill = enum(u32) { none = 0, nan_request_zero_fma = 1 };

/// Element type of a tensor DeepGEMM builds a map over (`aten_dtype_to_tensor_map_dtype`
/// with `fp4_unpacked_smem = true`).
pub const TensorDType = enum {
    /// `torch.float8_e4m3fn` -> UINT8.
    fp8,
    /// `torch.int` -> INT32 (packed UE8M0 scale factors).
    int32,
    /// `kPackedFP4` (`torch.int8`, two FP4 per byte) -> 16U4_ALIGN16B.
    packed_fp4,

    pub fn elemSize(self: TensorDType) u32 {
        return switch (self) {
            .fp8, .packed_fp4 => 1,
            .int32 => 4,
        };
    }
    pub fn mapType(self: TensorDType) TensorMapDataType {
        return switch (self) {
            .fp8 => .uint8,
            .int32 => .int32,
            .packed_fp4 => .@"16u4_align16b",
        };
    }
};

/// `mode_into_tensor_map_swizzle(mode, 0)`.
pub fn swizzleOfMode(mode: u32) TensorMapSwizzle {
    return switch (mode) {
        0, 16 => .none,
        32 => .@"32b",
        64 => .@"64b",
        128 => .@"128b",
        else => unreachable,
    };
}

/// Every parameter of one `cuTensorMapEncodeTiled` call, as DeepGEMM passes it, except the
/// global address: `base` names the tensor and `base_offset` the byte offset within it.
pub const TensorMapSpec = struct {
    pub const Base = enum {
        /// The local symmetric buffer (`SymBuffer::base`).
        symm_buffer,
        l1_weights,
        l1_weights_sf,
        l2_weights,
        l2_weights_sf,
        shared_l1_weights,
        shared_l1_weights_sf,
        shared_l2_weights,
        shared_l2_weights_sf,
    };

    base: Base,
    base_offset: u64 = 0,
    data_type: TensorMapDataType,
    rank: u32,
    global_dims: [5]u64 = @splat(1),
    /// Byte strides of dimensions 1..rank-1.
    global_strides: [4]u64 = @splat(0),
    box_dims: [5]u32 = @splat(1),
    element_strides: [5]u32 = @splat(1),
    interleave: TensorMapInterleave = .none,
    swizzle: TensorMapSwizzle,
    l2_promotion: TensorMapL2Promotion = .@"256b",
    oob_fill: TensorMapOobFill = .none,
    /// How the kernel issues copies through this map (`cta_group::2` loads or 1-CTA stores);
    /// not part of the descriptor.
    two_cta: bool,

    /// Bytes one box moves in shared memory (FP4 is unpacked to a byte per element).
    pub fn boxSmemBytes(self: TensorMapSpec) u32 {
        var n: u32 = switch (self.data_type) {
            .int32, .uint32, .float32 => 4,
            else => 1,
        };
        for (self.box_dims[0..self.rank]) |d| n *= d;
        return n;
    }
};

/// A global tensor as DeepGEMM sees it: logical sizes and element strides (`t.size(i)`,
/// `t.stride(i)`, packed-FP4 sizes in bytes like the int8 storage), outermost first.
pub const TensorDesc = struct {
    dtype: TensorDType,
    sizes: []const u64,
    strides: []const u64,

    pub fn dim(self: TensorDesc) usize {
        return self.sizes.len;
    }
    fn size(self: TensorDesc, i: isize) u64 {
        return self.sizes[@intCast(if (i < 0) @as(isize, @intCast(self.sizes.len)) + i else i)];
    }
    fn stride(self: TensorDesc, i: isize) u64 {
        return self.strides[@intCast(if (i < 0) @as(isize, @intCast(self.strides.len)) + i else i)];
    }
};

/// `make_tma_2d_desc`.
pub fn tma2d(base: TensorMapSpec.Base, base_offset: u64, dtype: TensorDType, gmem_inner_dim: u64, gmem_outer_dim: u64, smem_inner_dim_: u32, smem_outer_dim: u32, gmem_outer_stride: u64, swizzle_mode: u32, two_cta: bool) TensorMapSpec {
    const elem_size = dtype.elemSize();
    std.debug.assert(gmem_outer_stride * elem_size % 16 == 0);
    var smem_inner_dim = smem_inner_dim_;
    if (swizzle_mode != 0) smem_inner_dim = swizzle_mode / elem_size;
    if (dtype == .packed_fp4) std.debug.assert(gmem_inner_dim % 128 == 0);
    return .{
        .base = base,
        .base_offset = base_offset,
        .data_type = dtype.mapType(),
        .rank = 2,
        .global_dims = .{ gmem_inner_dim, gmem_outer_dim, 1, 1, 1 },
        .global_strides = .{ gmem_outer_stride * elem_size, 0, 0, 0 },
        .box_dims = .{ smem_inner_dim, smem_outer_dim, 1, 1, 1 },
        .swizzle = swizzleOfMode(swizzle_mode),
        .two_cta = two_cta,
    };
}

/// `make_tma_4d_desc`.
pub fn tma4d(base: TensorMapSpec.Base, dtype: TensorDType, gmem_dims: [4]u64, smem_dims_: [4]u32, gmem_strides: [3]u64, swizzle_mode: u32, two_cta: bool) TensorMapSpec {
    const elem_size = dtype.elemSize();
    var smem_dims = smem_dims_;
    for (gmem_strides) |s| std.debug.assert(s * elem_size % 16 == 0);
    if (swizzle_mode != 0) smem_dims[0] = swizzle_mode / elem_size;
    if (dtype == .packed_fp4) std.debug.assert(gmem_dims[0] % 128 == 0);
    return .{
        .base = base,
        .data_type = dtype.mapType(),
        .rank = 4,
        .global_dims = .{ gmem_dims[0], gmem_dims[1], gmem_dims[2], gmem_dims[3], 1 },
        .global_strides = .{ gmem_strides[0] * elem_size, gmem_strides[1] * elem_size, gmem_strides[2] * elem_size, 0 },
        .box_dims = .{ smem_dims[0], smem_dims[1], smem_dims[2], smem_dims[3], 1 },
        .swizzle = swizzleOfMode(swizzle_mode),
        .two_cta = two_cta,
    };
}

/// `make_tma_sf_desc(MN, t, shape_mn, shape_k, block_mn, gran_k, num_groups, 0, 0, false,
/// smem_outer_dim, sf_k_stride)` over an int32 (4 packed UE8M0 per element) MN-major tensor.
pub fn tmaSf(base: TensorMapSpec.Base, base_offset: u64, shape_mn_: u64, shape_k: u64, block_mn: u32, gran_k_: u32, num_groups: u64, smem_outer_dim: u32, sf_k_stride: u64, two_cta: bool) TensorMapSpec {
    const shape_mn = alignUp(shape_mn_, 16 / 4);
    const gmem_outer_stride = if (sf_k_stride == 0) shape_mn else sf_k_stride;
    std.debug.assert(gmem_outer_stride >= shape_mn);
    return tma2d(base, base_offset, .int32, shape_mn, ceilDiv(shape_k, @as(u64, gran_k_) * 4) * num_groups, block_mn, smem_outer_dim, gmem_outer_stride, 0, two_cta);
}

/// `LocalityDomainAllocator::get_num_locality_domains()` (hard-coded 2).
pub const num_locality_domains: u64 = 2;

/// `make_tma_weights_3d_desc`: `(B, N, K)` or localized `(2, B, N/2, K)` -> 4D `(K, N/2, 2, B)`.
pub fn tmaWeights3d(base: TensorMapSpec.Base, t: TensorDesc, block_k: u32, load_block_n: u32, swizzle_mode: u32) TensorMapSpec {
    const localized = t.dim() == 4;
    const num_batches = t.size(if (localized) 1 else 0);
    const n_per_domain = if (localized) t.size(2) else t.size(1) / num_locality_domains;
    const k = t.size(-1) * (if (t.dtype == .packed_fp4) @as(u64, 2) else 1);
    const slice_stride = if (localized) t.stride(0) else n_per_domain * t.stride(-2);
    return tma4d(base, t.dtype, .{ k, n_per_domain, num_locality_domains, num_batches }, .{ block_k, load_block_n, 1, 1 }, .{ t.stride(-2), slice_stride, t.stride(if (localized) 1 else 0) }, swizzle_mode, true);
}

/// `make_tma_weights_2d_desc`: `(N, K)` or localized `(2, N/2, K)` -> 4D `(K, N/2, 2, 1)`.
pub fn tmaWeights2d(base: TensorMapSpec.Base, t: TensorDesc, block_k: u32, load_block_n: u32, swizzle_mode: u32) TensorMapSpec {
    const localized = t.dim() == 3;
    const n_per_domain = if (localized) t.size(1) else t.size(0) / num_locality_domains;
    const k = t.size(-1) * (if (t.dtype == .packed_fp4) @as(u64, 2) else 1);
    const slice_stride = if (localized) t.stride(0) else n_per_domain * t.stride(-2);
    return tma4d(base, t.dtype, .{ k, n_per_domain, num_locality_domains, 1 }, .{ block_k, load_block_n, 1, 1 }, .{ t.stride(-2), slice_stride, slice_stride * num_locality_domains }, swizzle_mode, true);
}

/// The weights the kernel reads, as tensor descriptors (DeepGEMM's post-transform layout).
pub const WeightTensors = struct {
    /// `(E_local, 2I, H/2)` packed FP4 (or `(E_local, 2I, H)` FP8).
    l1: TensorDesc,
    /// `(E_local, 2I, H/128)` int32, MN-major (`stride(-2) == 1`).
    l1_sf: TensorDesc,
    l2: TensorDesc,
    l2_sf: TensorDesc,
    /// `(2 sI, H)` FP8; null without a shared expert.
    shared_l1: ?TensorDesc = null,
    shared_l1_sf: ?TensorDesc = null,
    shared_l2: ?TensorDesc = null,
    shared_l2_sf: ?TensorDesc = null,
};

/// The 18 tensor maps of `sm100_fp8_fp4_mega_moe`, in kernel-parameter order.
pub const TensorMaps = struct {
    l1_acts: TensorMapSpec,
    l1_acts_sf: TensorMapSpec,
    l1_weights: TensorMapSpec,
    l1_weights_sf: TensorMapSpec,
    l1_output: TensorMapSpec,
    l2_acts: TensorMapSpec,
    l2_acts_sf: TensorMapSpec,
    l2_weights: TensorMapSpec,
    l2_weights_sf: TensorMapSpec,
    shared_l1_acts: TensorMapSpec,
    shared_l1_acts_sf: TensorMapSpec,
    shared_l1_weights: TensorMapSpec,
    shared_l1_weights_sf: TensorMapSpec,
    shared_l1_output: TensorMapSpec,
    shared_l2_acts: TensorMapSpec,
    shared_l2_acts_sf: TensorMapSpec,
    shared_l2_weights: TensorMapSpec,
    shared_l2_weights_sf: TensorMapSpec,

    pub const names = @typeInfo(TensorMaps).@"struct".field_names;

    /// The host side of `sm100_fp8_fp4_mega_moe` (vLLM fork, `sm100_fp8_fp4_mega_moe.hpp`):
    /// the activation maps over the symmetric buffer views `slice_symm_buffer_from_layout`
    /// returns, the weight maps over `w`. Without a shared expert, each shared map is a copy
    /// of the routed one.
    pub fn init(k: Kernel, buf: MegaMoEBuffer, w: WeightTensors) TensorMaps {
        const c = k.cfg;
        const H: u64 = k.shape.hidden;
        const I: u64 = k.shape.intermediate_hidden;
        const sI: u64 = k.shape.sharedIntermediateHidden();
        const T: u64 = k.shape.num_max_tokens_per_rank;
        const R: u64 = c.num_ring_tokens;
        const S: u64 = c.num_sf_ring_tokens;
        const Ssh: u64 = numMaxSharedSfTokens(k.shape.num_max_tokens_per_rank);
        const sf_smem_outer_dim = c.block_k / (Kernel.gran_k * 4);
        const E: u64 = k.numExpertsPerRank();

        var m: TensorMaps = undefined;
        // Views: `l1_acts` [R, H] (stride H), `l1_acts_sf` [S, H/128] strides (1, S), ...
        m.l1_acts = tma2d(.symm_buffer, buf.l1_token.offset, .fp8, H, R, c.block_k, c.load_block_m, H, c.swizzle_acts_mode, true);
        m.l1_acts_sf = tmaSf(.symm_buffer, buf.l1_sf.offset, S, H, c.sf_block_m, Kernel.gran_k, 1, sf_smem_outer_dim, S, true);
        m.l1_weights = tmaWeights3d(.l1_weights, w.l1, c.block_k, c.load_block_n, c.swizzle_weights_mode);
        m.l1_weights_sf = tmaSf(.l1_weights_sf, 0, I * 2, H, c.block_n, Kernel.gran_k, E, sf_smem_outer_dim, w.l1_sf.stride(-1), true);
        m.l1_output = tma2d(.symm_buffer, buf.l2_token.offset, .fp8, I, R, c.block_n / 2, c.store_block_m_l1, I, c.swizzle_acts_mode / 2, false);
        m.l2_acts = tma2d(.symm_buffer, buf.l2_token.offset, .fp8, I, R, c.block_k, c.load_block_m, I, c.swizzle_acts_mode, true);
        m.l2_acts_sf = tmaSf(.symm_buffer, buf.l2_sf.offset, S, I, c.sf_block_m, Kernel.gran_k, 1, sf_smem_outer_dim, S, true);
        m.l2_weights = tmaWeights3d(.l2_weights, w.l2, c.block_k, c.load_block_n, c.swizzle_weights_mode);
        m.l2_weights_sf = tmaSf(.l2_weights_sf, 0, H, I, c.block_n, Kernel.gran_k, E, sf_smem_outer_dim, w.l2_sf.stride(-1), true);
        if (k.hasShared()) {
            m.shared_l1_acts = tma2d(.symm_buffer, buf.input_token.offset, .fp8, H, T, c.block_k, c.load_block_m, H, c.swizzle_acts_mode, true);
            m.shared_l1_acts_sf = tmaSf(.symm_buffer, buf.shared_l1_sf.offset, Ssh, H, c.sf_block_m, Kernel.gran_k, 1, sf_smem_outer_dim, Ssh, true);
            m.shared_l1_weights = tmaWeights2d(.shared_l1_weights, w.shared_l1.?, c.block_k, c.load_block_n, c.swizzle_weights_mode);
            m.shared_l1_weights_sf = tmaSf(.shared_l1_weights_sf, 0, sI * 2, H, c.block_n, Kernel.gran_k, 1, sf_smem_outer_dim, w.shared_l1_sf.?.stride(-1), true);
            m.shared_l1_output = tma2d(.symm_buffer, buf.shared_l2_token.offset, .fp8, sI, T, c.block_n / 2, c.store_block_m_l1, sI, c.swizzle_acts_mode / 2, false);
            m.shared_l2_acts = tma2d(.symm_buffer, buf.shared_l2_token.offset, .fp8, sI, T, c.block_k, c.load_block_m, sI, c.swizzle_acts_mode, true);
            m.shared_l2_acts_sf = tmaSf(.symm_buffer, buf.shared_l2_sf.offset, Ssh, sI, c.sf_block_m, Kernel.gran_k, 1, sf_smem_outer_dim, Ssh, true);
            m.shared_l2_weights = tmaWeights2d(.shared_l2_weights, w.shared_l2.?, c.block_k, c.load_block_n, c.swizzle_weights_mode);
            m.shared_l2_weights_sf = tmaSf(.shared_l2_weights_sf, 0, H, sI, c.block_n, Kernel.gran_k, 1, sf_smem_outer_dim, w.shared_l2_sf.?.stride(-1), true);
        } else {
            m.shared_l1_acts = m.l1_acts;
            m.shared_l1_acts_sf = m.l1_acts_sf;
            m.shared_l1_weights = m.l1_weights;
            m.shared_l1_weights_sf = m.l1_weights_sf;
            m.shared_l1_output = m.l1_output;
            m.shared_l2_acts = m.l2_acts;
            m.shared_l2_acts_sf = m.l2_acts_sf;
            m.shared_l2_weights = m.l2_weights;
            m.shared_l2_weights_sf = m.l2_weights_sf;
        }
        return m;
    }
};

/// DeepGEMM's post-transform weight layout for non-localized weights with the scale factors
/// stored `(E, K/128, N)` (MN contiguous), as vLLM registers them: routed FP4 `(E, N, K/2)`,
/// shared FP8 `(N, K)` and shared SF `(K/128, N)`. Sizes and strides outermost first.
pub const DefaultWeights = struct {
    l1_sizes: [3]u64,
    l1_strides: [3]u64,
    l1_sf_sizes: [3]u64,
    l1_sf_strides: [3]u64,
    l2_sizes: [3]u64,
    l2_strides: [3]u64,
    l2_sf_sizes: [3]u64,
    l2_sf_strides: [3]u64,
    shared_l1_sizes: [2]u64,
    shared_l1_strides: [2]u64,
    shared_l1_sf_sizes: [2]u64,
    shared_l1_sf_strides: [2]u64,
    shared_l2_sizes: [2]u64,
    shared_l2_strides: [2]u64,
    shared_l2_sf_sizes: [2]u64,
    shared_l2_sf_strides: [2]u64,
    weight_dtype: TensorDType,

    pub fn init(shape: Shape, weight_type: WeightType) DefaultWeights {
        const E: u64 = shape.numExpertsPerRank();
        const H: u64 = shape.hidden;
        const I: u64 = shape.intermediate_hidden;
        const sI: u64 = shape.sharedIntermediateHidden();
        const pack: u64 = if (weight_type == .fp4_unpacked) 2 else 1;
        return .{
            .weight_dtype = if (weight_type == .fp4_unpacked) .packed_fp4 else .fp8,
            .l1_sizes = .{ E, 2 * I, H / pack },
            .l1_strides = .{ 2 * I * H / pack, H / pack, 1 },
            .l1_sf_sizes = .{ E, 2 * I, H / 128 },
            .l1_sf_strides = .{ 2 * I * (H / 128), 1, 2 * I },
            .l2_sizes = .{ E, H, I / pack },
            .l2_strides = .{ H * I / pack, I / pack, 1 },
            .l2_sf_sizes = .{ E, H, I / 128 },
            .l2_sf_strides = .{ H * (I / 128), 1, H },
            .shared_l1_sizes = .{ 2 * sI, H },
            .shared_l1_strides = .{ H, 1 },
            .shared_l1_sf_sizes = .{ 2 * sI, H / 128 },
            .shared_l1_sf_strides = .{ 1, 2 * sI },
            .shared_l2_sizes = .{ H, sI },
            .shared_l2_strides = .{ sI, 1 },
            .shared_l2_sf_sizes = .{ H, sI / 128 },
            .shared_l2_sf_strides = .{ 1, H },
        };
    }

    pub fn tensors(self: *const DefaultWeights, has_shared: bool) WeightTensors {
        return .{
            .l1 = .{ .dtype = self.weight_dtype, .sizes = &self.l1_sizes, .strides = &self.l1_strides },
            .l1_sf = .{ .dtype = .int32, .sizes = &self.l1_sf_sizes, .strides = &self.l1_sf_strides },
            .l2 = .{ .dtype = self.weight_dtype, .sizes = &self.l2_sizes, .strides = &self.l2_strides },
            .l2_sf = .{ .dtype = .int32, .sizes = &self.l2_sf_sizes, .strides = &self.l2_sf_strides },
            .shared_l1 = if (has_shared) .{ .dtype = .fp8, .sizes = &self.shared_l1_sizes, .strides = &self.shared_l1_strides } else null,
            .shared_l1_sf = if (has_shared) .{ .dtype = .int32, .sizes = &self.shared_l1_sf_sizes, .strides = &self.shared_l1_sf_strides } else null,
            .shared_l2 = if (has_shared) .{ .dtype = .fp8, .sizes = &self.shared_l2_sizes, .strides = &self.shared_l2_strides } else null,
            .shared_l2_sf = if (has_shared) .{ .dtype = .int32, .sizes = &self.shared_l2_sf_sizes, .strides = &self.shared_l2_sf_strides } else null,
        };
    }
};

/// The balanced even SM-to-locality-domain map DeepGEMM uses without MLOPart
/// (`get_even_sm_locality_domains`: `(sm / 2) % 2`; already balanced for an even TPC count).
pub fn evenSmLocalityDomain(sm_idx: u32) u8 {
    return @intCast((sm_idx / 2) % num_locality_domains);
}

test {
    _ = @import("layout_test.zig");
}

/// vLLM's DSv4.1 instantiation (EP2, `max_num_batched_tokens` 8192, 152 SMs, 4096 tokens per
/// rank): the template arguments of its JIT cache entry, and DeepGEMM's own `DG_PRINT_CONFIGS`
/// output of the 18 `cuTensorMapEncodeTiled` calls (`dumps/r2_t4096_s1/rank0_dg_stdout.txt`).
fn dsv41() struct { shape: Shape, symm: SymmBufferLayout, kernel: Kernel } {
    const shape: Shape = .{ .num_ranks = 2, .num_experts = 384, .num_topk = 6, .hidden = 5120, .intermediate_hidden = 2304, .num_shared_experts = 1, .num_max_tokens_per_rank = alignTokens(8192) };
    const symm: SymmBufferLayout = .init(shape, 152);
    return .{ .shape = shape, .symm = symm, .kernel = .init(shape, config(shape, 4096, symm), 152, .fp4_unpacked) };
}

test "vLLM's DSv4.1 instantiation" {
    const d = dsv41();
    const k = d.kernel;
    const c = k.cfg;
    // sm100_fp8_fp4_mega_moe_impl<9600, 5120, 2304, 384, 1, 6, 192, 128, 128, 32, 32, 256, 128,
    //   21120, 337920, 6, 5120, 128, 128, 256, 152, 2, ...>
    try std.testing.expectEqual(@as(u32, 9600), d.shape.num_max_tokens_per_rank);
    try std.testing.expectEqualSlices(u32, &.{ 192, 128, 128, 32, 32, 256, 128, 21120, 337920, 6, 5120, 128, 128, 256 }, &.{
        c.block_m,         c.block_n,            c.block_k,    c.store_block_m_l1,   c.store_block_m_l2,     c.sf_block_m,               c.sf_block_n,
        c.num_ring_tokens, c.num_sf_ring_tokens, c.num_stages, c.num_bytes_per_pull, c.num_dispatch_threads, c.num_non_epilogue_threads, c.num_epilogue_threads,
    });
    try std.testing.expectEqual(@as(u32, 221572), c.smem_size);
    try std.testing.expectEqual(@as(u32, 221572), k.smem().used_bytes);
    try std.testing.expectEqual(@as(u32, 221248), k.smem().reusableBytes());
    try std.testing.expectEqual(@as(u64, 1070359424), d.symm.numBytes());
    // Derived kernel constants.
    try std.testing.expectEqual(@as(u32, 2), k.numLocalityDomains());
    try std.testing.expectEqual(@as(u32, 110), k.numRingBlocks());
    try std.testing.expectEqual(@as(u32, 153600), k.numSharedSfTokens());
    try std.testing.expectEqual(@as(u32, 512), k.numTmemCols());
    try std.testing.expectEqual(@as(u32, 384), k.tmemStartColOfSfa());
    try std.testing.expectEqual(@as(u32, 392), k.tmemStartColOfSfb());
    try std.testing.expectEqualSlices(u32, &.{ 96, 72, 168 }, &.{ k.numDispatchRegisters(), k.numNonEpilogueRegisters(), k.numEpilogueRegisters() });
    try std.testing.expectEqual(@as(u32, 36), k.numL1BlockNs());
    try std.testing.expectEqual(@as(u32, 18), k.numL1Clusters());
    try std.testing.expectEqual(@as(u32, 20), k.numL2Clusters());
    try std.testing.expectEqual(@as(u64, (1 << 36) - 1), k.l2FullMask());
    try std.testing.expectEqual(@as(u32, 36), k.sharedL2ExpectedBlocks());
    try std.testing.expectEqual(@as(u32, 5), k.numTokensPerWarp());
    try std.testing.expectEqual(@as(u32, 44032), k.expectTxA() + k.expectTxRoutedB());
    try std.testing.expectEqual(@as(u32, 2), k.numCombineChunks());
    try std.testing.expectEqual(@as(u32, 0), k.numEpiPrefetchStages());
    // Instruction / shared-memory descriptors (cute::UMMA bitfields).
    try std.testing.expectEqual(@as(u32, 0x10b00280), k.routedInstrDesc().bits());
    try std.testing.expectEqual(@as(u32, 0x70b002b0), (InstrDescBlockScaled{ .a_format = .e2m1, .b_format = .e4m3, .m = 256, .n = 192, .a_sf_id = 3, .b_sf_id = 3 }).bits());
    try std.testing.expectEqual(@as(u32, 0x10b00000), k.sharedInstrDesc().bits());
    try std.testing.expectEqual(@as(u32, 0x40004040), Kernel.abSmemDescHi());
    try std.testing.expectEqual(@as(u32, 0x4008), Kernel.sfSmemDescHi());
    // UTCCP transposition of the SF token index.
    try std.testing.expectEqual(@as(u32, 0), k.transformSfTokenIdx(0));
    try std.testing.expectEqual(@as(u32, 1), k.transformSfTokenIdx(32));
    try std.testing.expectEqual(@as(u32, 4), k.transformSfTokenIdx(1));
    try std.testing.expectEqual(@as(u32, 128 + 4 * 31 + 1), k.transformSfTokenIdx(191));
    try std.testing.expectEqual(@as(u32, 256 + 4), k.transformSfTokenIdx(192 + 1));
    try std.testing.expectEqual(@as(u32, 256), k.transformSfTokenIdx(192));
}

test "tensor maps match DeepGEMM's cuTensorMapEncodeTiled parameters" {
    const d = dsv41();
    const weights: DefaultWeights = .init(d.shape, .fp4_unpacked);
    const maps: TensorMaps = .init(d.kernel, d.symm.buffer, weights.tensors(true));
    // DG_PRINT_CONFIGS: global dims, box, outer strides (elements of the int8/fp8/int32 tensor),
    // swizzle, and for maps over the symmetric buffer the pointer minus `shared_l1_acts`'s.
    const Expected = struct { dims: []const u64, box: []const u32, strides: []const u64, swizzle: u32, elem: u64, ptr: ?i64 };
    const shared_l1_acts_ptr: i64 = 1206923641216;
    const expected = [_]Expected{
        .{ .dims = &.{ 5120, 21120 }, .box = &.{ 128, 96 }, .strides = &.{5120}, .swizzle = 128, .elem = 1, .ptr = 1207032774016 },
        .{ .dims = &.{ 337920, 40 }, .box = &.{ 256, 1 }, .strides = &.{337920}, .swizzle = 0, .elem = 4, .ptr = 1207140908416 },
        .{ .dims = &.{ 5120, 2304, 2, 192 }, .box = &.{ 128, 128, 1, 1 }, .strides = &.{ 2560, 5898240, 11796480 }, .swizzle = 128, .elem = 1, .ptr = null },
        .{ .dims = &.{ 4608, 7680 }, .box = &.{ 128, 1 }, .strides = &.{4608}, .swizzle = 0, .elem = 4, .ptr = null },
        .{ .dims = &.{ 2304, 21120 }, .box = &.{ 64, 32 }, .strides = &.{2304}, .swizzle = 64, .elem = 1, .ptr = 1207195060096 },
        .{ .dims = &.{ 2304, 21120 }, .box = &.{ 128, 96 }, .strides = &.{2304}, .swizzle = 128, .elem = 1, .ptr = 1207195060096 },
        .{ .dims = &.{ 337920, 18 }, .box = &.{ 256, 1 }, .strides = &.{337920}, .swizzle = 0, .elem = 4, .ptr = 1207243720576 },
        .{ .dims = &.{ 2304, 2560, 2, 192 }, .box = &.{ 128, 128, 1, 1 }, .strides = &.{ 1152, 2949120, 5898240 }, .swizzle = 128, .elem = 1, .ptr = null },
        .{ .dims = &.{ 5120, 3456 }, .box = &.{ 128, 1 }, .strides = &.{5120}, .swizzle = 0, .elem = 4, .ptr = null },
        .{ .dims = &.{ 5120, 9600 }, .box = &.{ 128, 96 }, .strides = &.{5120}, .swizzle = 128, .elem = 1, .ptr = 1206923641216 },
        .{ .dims = &.{ 153600, 40 }, .box = &.{ 256, 1 }, .strides = &.{153600}, .swizzle = 0, .elem = 4, .ptr = 1206975020416 },
        .{ .dims = &.{ 5120, 2304, 2, 1 }, .box = &.{ 128, 128, 1, 1 }, .strides = &.{ 5120, 11796480, 23592960 }, .swizzle = 128, .elem = 1, .ptr = null },
        .{ .dims = &.{ 4608, 40 }, .box = &.{ 128, 1 }, .strides = &.{4608}, .swizzle = 0, .elem = 4, .ptr = null },
        .{ .dims = &.{ 2304, 9600 }, .box = &.{ 64, 32 }, .strides = &.{2304}, .swizzle = 64, .elem = 1, .ptr = 1206999596416 },
        .{ .dims = &.{ 2304, 9600 }, .box = &.{ 128, 96 }, .strides = &.{2304}, .swizzle = 128, .elem = 1, .ptr = 1206999596416 },
        .{ .dims = &.{ 153600, 18 }, .box = &.{ 256, 1 }, .strides = &.{153600}, .swizzle = 0, .elem = 4, .ptr = 1207021714816 },
        .{ .dims = &.{ 2304, 2560, 2, 1 }, .box = &.{ 128, 128, 1, 1 }, .strides = &.{ 2304, 5898240, 11796480 }, .swizzle = 128, .elem = 1, .ptr = null },
        .{ .dims = &.{ 5120, 18 }, .box = &.{ 128, 1 }, .strides = &.{5120}, .swizzle = 0, .elem = 4, .ptr = null },
    };
    const shared_l1_acts_offset: i64 = @intCast(maps.shared_l1_acts.base_offset);
    inline for (TensorMaps.names, 0..) |name, i| {
        const m = @field(maps, name);
        const e = expected[i];
        try std.testing.expectEqual(e.dims.len, m.rank);
        try std.testing.expectEqualSlices(u64, e.dims, m.global_dims[0..m.rank]);
        try std.testing.expectEqualSlices(u32, e.box, m.box_dims[0..m.rank]);
        for (e.strides, 0..) |stride, j| try std.testing.expectEqual(stride * e.elem, m.global_strides[j]);
        try std.testing.expectEqual(swizzleOfMode(e.swizzle), m.swizzle);
        try std.testing.expectEqual(TensorMapL2Promotion.@"256b", m.l2_promotion);
        try std.testing.expectEqual(if (e.elem == 4) TensorMapDataType.int32 else if (e.ptr == null and e.dims[3] != 1) TensorMapDataType.@"16u4_align16b" else TensorMapDataType.uint8, m.data_type);
        if (e.ptr) |ptr| {
            try std.testing.expectEqual(TensorMapSpec.Base.symm_buffer, m.base);
            try std.testing.expectEqual(ptr - shared_l1_acts_ptr, @as(i64, @intCast(m.base_offset)) - shared_l1_acts_offset);
        } else try std.testing.expect(m.base != .symm_buffer);
        try std.testing.expectEqual(!std.mem.endsWith(u8, name, "_output"), m.two_cta);
    }
}
