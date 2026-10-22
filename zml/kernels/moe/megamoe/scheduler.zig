//! Port of DeepGEMM's MegaMoE task scheduler (`scheduler/mega_moe.cuh`, vLLM's vendored copy),
//! in the same structure and order:
//!
//! * `TaskInfo` (`sched::TaskInfo<kHasSharedExperts>`, as eight `u32` values);
//! * `L2KBlockDependency` (`sched::L2KBlockDependency`: the L1 N-block readiness bit mask the
//!   L2 TMA-A warp waits on);
//! * `Scheduler` (`sched::MegaMoEScheduler`):
//!   - consumer side, every role of both CTAs: `getNextTask` / `releaseTaskInfo`;
//!   - expert-count side (dispatch warps and the scheduler warp): `fetchExpertRecvCount`,
//!     `getNumTokens`, `getPoolBlockOffset`;
//!   - producer side, the leader CTA's scheduler warp: `mainloop` (`create_task`,
//!     `get_next_task_idx`, `get_next_task`, `shared_mainloop`, `publish_task`).
//!
//! C++ members that the scheduler mutates (`sched_stage_idx`/`sched_phase`, the per-lane token
//! counts, `num_sched_l1_waves`) are explicit values here (`Pipe`, `Counts`) threaded through
//! the callers' loops. `break`/`continue`/early `return` are rewritten as `scf.while` loops with
//! the same evaluation order; each rewrite is commented.
const std = @import("std");

const cute = @import("kernels/cute/builder");

const device = @import("device.zig");
const flow = @import("flow.zig");
const layout = @import("layout.zig");
const ptx = @import("ptx.zig");

const B = cute.Builder;
const V = cute.Value;
const BlockPhase = layout.BlockPhase;

/// `kNumScheduleStages`.
pub const num_schedule_stages = 2;
/// `kNumSchedL1WavesDone`.
pub const sched_l1_waves_done: u32 = 0xffffffff;
/// `kNumMinStealWaves`.
pub const num_min_steal_waves = 2;

/// The template parameters of `MegaMoEScheduler<BLOCK_M, BLOCK_N, BLOCK_K, L1_SHAPE_N,
/// L1_SHAPE_K, L2_SHAPE_N, L2_SHAPE_K, kNumExpertsPerRank, kNumSMs, kNumRanks, kNumRingBlocks,
/// kNumLocalityDomains, kNumSharedExperts>` (`kDecodeShaped` is `false` in MegaMoE).
pub const Params = struct {
    block_m: u32,
    block_n: u32,
    block_k: u32,
    l1_shape_n: u32,
    l1_shape_k: u32,
    l2_shape_n: u32,
    l2_shape_k: u32,
    num_experts_per_rank: u32,
    num_sms: u32,
    num_ranks: u32,
    num_ring_blocks: u32,
    num_locality_domains: u32,
    num_shared_experts: u32,

    pub fn fromKernel(k: layout.Kernel) Params {
        const p: Params = .{
            .block_m = k.cfg.block_m,
            .block_n = k.cfg.block_n,
            .block_k = k.cfg.block_k,
            .l1_shape_n = k.l1ShapeN(),
            .l1_shape_k = k.l1ShapeK(),
            .l2_shape_n = k.l2ShapeN(),
            .l2_shape_k = k.l2ShapeK(),
            .num_experts_per_rank = k.numExpertsPerRank(),
            .num_sms = k.num_sms,
            .num_ranks = k.shape.num_ranks,
            .num_ring_blocks = k.numRingBlocks(),
            .num_locality_domains = k.numLocalityDomains(),
            .num_shared_experts = k.shape.num_shared_experts,
        };
        p.check();
        return p;
    }

    /// The scheduler's `DG_STATIC_ASSERT`s.
    pub fn check(p: Params) void {
        std.debug.assert(p.l1_shape_n % (p.block_n * 2) == 0 and p.l2_shape_n % (p.block_n * 2) == 0);
        std.debug.assert(p.l1_shape_k % p.block_k == 0 and p.l2_shape_k % p.block_k == 0);
        std.debug.assert(p.sharedL1ShapeN() % (p.block_n * 2) == 0 and p.sharedL2ShapeN() % (p.block_n * 2) == 0);
        std.debug.assert(p.sharedL1ShapeK() % p.block_k == 0 and p.sharedL2ShapeK() % p.block_k == 0);
        std.debug.assert(p.num_sms % 2 == 0);
        std.debug.assert(p.num_ring_blocks > 0);
        std.debug.assert(p.num_locality_domains == 1 or p.num_locality_domains == layout.device_locality_domains);
        std.debug.assert(p.numL1Clusters() % p.num_locality_domains == 0 and p.numL2Clusters() % p.num_locality_domains == 0);
    }

    pub fn hasShared(p: Params) bool {
        return p.num_shared_experts > 0;
    }
    pub fn sharedL1ShapeN(p: Params) u32 {
        return p.l1_shape_n * p.num_shared_experts;
    }
    pub fn sharedL1ShapeK(p: Params) u32 {
        return p.l1_shape_k;
    }
    pub fn sharedL2ShapeN(p: Params) u32 {
        return p.l2_shape_n;
    }
    pub fn sharedL2ShapeK(p: Params) u32 {
        return p.l2_shape_k * p.num_shared_experts;
    }
    /// `kNumExpertsPerLane`.
    pub fn numExpertsPerLane(p: Params) u32 {
        return layout.ceilDiv(p.num_experts_per_rank, 32);
    }
    pub fn numL1BlockNs(p: Params) u32 {
        return p.l1_shape_n / p.block_n;
    }
    pub fn numL2BlockNs(p: Params) u32 {
        return p.l2_shape_n / p.block_n;
    }
    /// `kNumL1Clusters`.
    pub fn numL1Clusters(p: Params) u32 {
        return p.numL1BlockNs() / 2;
    }
    /// `kNumL2Clusters`.
    pub fn numL2Clusters(p: Params) u32 {
        return p.numL2BlockNs() / 2;
    }
};

// ==================== TaskInfo ====================

/// `sched::TaskInfo` (32 bytes, `alignas(16)`): eight `u32` (`i32`) values.
pub const TaskInfo = struct {
    block_phase: V,
    local_expert_idx: V,
    m_block_idx: V,
    n_cluster_idx: V,
    pool_block_idx: V,
    valid_m: V,
    shape_n: V,
    shape_k: V,

    /// `TaskInfo(block_phase, local_expert_idx, m_block_idx, pair_n_block_idx, pool_block_idx,
    /// valid_m, shape_n, shape_k)`.
    pub fn make(b: *B, block_phase: BlockPhase, local_expert_idx: anytype, m_block_idx: anytype, n_cluster_idx: anytype, pool_block_idx: anytype, valid_m: anytype, shape_n: anytype, shape_k: anytype) TaskInfo {
        return .{
            .block_phase = b.cst(.i32, @as(i32, @intCast(@backingInt(block_phase)))),
            .local_expert_idx = b.liftAs(local_expert_idx, .i32),
            .m_block_idx = b.liftAs(m_block_idx, .i32),
            .n_cluster_idx = b.liftAs(n_cluster_idx, .i32),
            .pool_block_idx = b.liftAs(pool_block_idx, .i32),
            .valid_m = b.liftAs(valid_m, .i32),
            .shape_n = b.liftAs(shape_n, .i32),
            .shape_k = b.liftAs(shape_k, .i32),
        };
    }

    /// `TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0)`.
    pub fn none(b: *B) TaskInfo {
        return make(b, .none, 0, 0, 0, 0, 0, 0, 0);
    }

    pub fn words(self: TaskInfo) [8]V {
        return .{ self.block_phase, self.local_expert_idx, self.m_block_idx, self.n_cluster_idx, self.pool_block_idx, self.valid_m, self.shape_n, self.shape_k };
    }

    pub fn fromWords(w: [8]V) TaskInfo {
        return .{ .block_phase = w[0], .local_expert_idx = w[1], .m_block_idx = w[2], .n_cluster_idx = w[3], .pool_block_idx = w[4], .valid_m = w[5], .shape_n = w[6], .shape_k = w[7] };
    }

    /// `is_valid()`.
    pub fn isValid(self: TaskInfo) V {
        return self.block_phase.ne(@as(i32, @intCast(@backingInt(BlockPhase.none))));
    }

    /// `is_phase(p)` (`block_phase == p`).
    pub fn isPhase(self: TaskInfo, p: BlockPhase) V {
        return self.block_phase.eq(@as(i32, @intCast(@backingInt(p))));
    }

    /// `get_umma_aligned_valid_m()`: `math::align(valid_m, 16)`.
    pub fn getUmmaAlignedValidM(self: TaskInfo, b: *B) V {
        return ptx.ceilDivU(b, self.valid_m, @as(i32, 16)).mul(16);
    }

    /// `is_shared()`: `kHasSharedExperts ? block_phase > Linear2 : false` (i1).
    pub fn isShared(self: TaskInfo, b: *B, has_shared: bool) V {
        return if (has_shared) ptx.gtU(b, self.block_phase, @as(i32, @intCast(@backingInt(BlockPhase.linear2)))) else b.cst(.i1, false);
    }
};

// ==================== L2KBlockDependency ====================

/// `sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>`: each finished L1 N block toggles
/// its bit in `l2_full_mask`; an L2 K block can start once its `BLOCK_K / (BLOCK_N / 2)` L1 N
/// blocks are fed. `pending_mask` is the state the TMA-A warp carries over its K loop.
pub const L2KBlockDependency = struct {
    mask_ptr: V,
    expected_mask: V,
    pending_mask: V,

    pub fn numL1BlockNsPerL2KBlock(p: Params) u32 {
        std.debug.assert(p.block_k % (p.block_n / 2) == 0);
        return p.block_k / (p.block_n / 2);
    }

    /// `kFullMask`.
    pub fn fullMask(p: Params) u64 {
        const n = p.numL1BlockNs();
        std.debug.assert(n <= 64);
        return if (n == 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(n)) - 1;
    }

    /// `L2KBlockDependency(mask_ptr, generation_idx)`: `expected_mask = generation_idx & 1 ? 0 :
    /// kFullMask`, `pending_mask = kFullMask`.
    pub fn init(b: *B, p: Params, mask_ptr: V, generation_idx: V) L2KBlockDependency {
        const full = ptx.constU64(b, fullMask(p));
        return .{
            .mask_ptr = mask_ptr,
            .expected_mask = b.select(generation_idx.bitAnd(1).ne(0), ptx.constU64(b, 0), full),
            .pending_mask = full,
        };
    }

    /// `arrive(mask_ptr, n_block_idx)`: `red.release.gpu.global.xor.b64` of the block's bit.
    pub fn arrive(b: *B, mask_ptr: V, n_block_idx: V) void {
        ptx.redXorRel(b, mask_ptr, ptx.constU64(b, 1).shl(ptx.zext(b, n_block_idx, .i64)));
    }

    /// `wait(k_block_idx)`: re-read the mask only while the K block is not fed yet. Returns the
    /// dependency with the updated `pending_mask`.
    pub fn wait(self: L2KBlockDependency, b: *B, p: Params, k_block_idx: V) L2KBlockDependency {
        const per = numL1BlockNsPerL2KBlock(p);
        const ones = ptx.constU64(b, (@as(u64, 1) << @intCast(per)) - 1);
        const k_block_mask = ones.shl(ptx.zext(b, k_block_idx.mul(@as(i32, @intCast(per))), .i64));
        // while (pending_mask & k_block_mask) pending_mask = ld_acq_gpu(mask_ptr) ^ expected_mask;
        var loop = flow.While(V, V).open(b, self.pending_mask);
        const pending = loop.state;
        const after = loop.cond(pending.bitAnd(k_block_mask).ne(0), pending);
        _ = after;
        const pending_out = loop.end(ptx.ldAcqGpuU64(b, self.mask_ptr).bitXor(self.expected_mask));
        return .{ .mask_ptr = self.mask_ptr, .expected_mask = self.expected_mask, .pending_mask = pending_out };
    }
};

// ==================== MegaMoEScheduler ====================

/// The consumer-side pipeline (`sched_stage_idx`, `sched_phase`).
pub const Pipe = struct {
    stage: V,
    phase: V,

    pub fn init(b: *B) Pipe {
        return .{ .stage = b.cst(.i32, 0), .phase = b.cst(.i32, 0) };
    }

    /// `advance_sched_pipeline()`: `stage ^= 1; phase ^= stage == 0`.
    pub fn advance(self: Pipe) Pipe {
        const stage = self.stage.bitXor(1);
        return .{ .stage = stage, .phase = self.phase.bitXor(stage.eq(0).to(.i32)) };
    }
};

/// What `fetch_expert_recv_count()` caches: `stored_num_tokens_per_expert[i]` holds expert
/// `i * 32 + lane_idx`'s token count; `num_total_m_blocks`; `num_sched_l1_waves`.
pub const Counts = struct {
    stored_num_tokens_per_expert: []V,
    num_total_m_blocks: V,
    num_sched_l1_waves: V,
};

/// `sched::MegaMoEScheduler`: the shared-memory slots it publishes to and the workspace.
pub const Scheduler = struct {
    b: *B,
    p: Params,
    workspace: device.Workspace,
    /// `task_info_full_barriers` / `task_info_empty_barriers` (`[2]` mbarriers) and
    /// `task_infos` (`[2]` TaskInfo), `shared::cta` addresses.
    task_info_full_barriers: V,
    task_info_empty_barriers: V,
    task_infos: V,

    fn fullBarrier(self: Scheduler, stage: V) V {
        return self.task_info_full_barriers.add(stage.mul(8));
    }
    fn emptyBarrier(self: Scheduler, stage: V) V {
        return self.task_info_empty_barriers.add(stage.mul(8));
    }
    fn taskInfoSlot(self: Scheduler, stage: V) V {
        return self.task_infos.add(stage.mul(@sizeOf(layout.TaskInfo)));
    }

    // ---------- consumers ----------

    pub const NextTask = struct { valid: V, task: TaskInfo, pipe: Pipe };

    /// `get_next_task(task_info)`: wait the stage's full barrier, copy the slot, advance.
    pub fn getNextTask(self: Scheduler, pipe: Pipe) NextTask {
        const b = self.b;
        ptx.mbarrierWait(b, self.fullBarrier(pipe.stage), pipe.phase);
        const slot = self.taskInfoSlot(pipe.stage);
        const lo = b.loadWords(4, slot, .shared, .{});
        const hi = b.loadWords(4, slot.add(16), .shared, .{});
        const task = TaskInfo.fromWords(.{ lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3] });
        return .{ .valid = task.isValid(), .task = task, .pipe = pipe.advance() };
    }

    /// `release_task_info()` (after `get_next_task` advanced `pipe`): complete the metadata
    /// reads, then arrive on the leader CTA's empty barrier of the consumed stage.
    pub fn releaseTaskInfo(self: Scheduler, pipe: Pipe) void {
        const b = self.b;
        ptx.fenceAcqRelCta(b);
        ptx.mbarrierArriveCluster(b, self.emptyBarrier(pipe.stage.bitXor(1)), b.cst(.i32, 0));
    }

    // ---------- expert counts ----------

    /// `get_num_tokens(expert_idx)`.
    pub fn getNumTokens(self: Scheduler, counts: Counts, expert_idx: V) V {
        const b = self.b;
        const lane = ptx.getLaneIdx(b);
        var valid_value = b.cst(.i32, 0);
        for (counts.stored_num_tokens_per_expert, 0..) |stored, i| {
            valid_value = b.select(expert_idx.eq(lane.add(@as(i32, @intCast(i * 32)))), stored, valid_value);
        }
        return ptx.shflSync(b, valid_value, ptx.remU(b, expert_idx, @as(i32, 32)));
    }

    /// `get_pool_block_offset(expert_idx)`: pool blocks of the experts before `expert_idx`.
    pub fn getPoolBlockOffset(self: Scheduler, stored: []const V, expert_idx: V) V {
        const b = self.b;
        const lane = ptx.getLaneIdx(b);
        var num_blocks = b.cst(.i32, 0);
        for (stored, 0..) |s, i| {
            const add = ptx.ltU(b, lane.add(@as(i32, @intCast(i * 32))), expert_idx);
            num_blocks = b.select(add, num_blocks.add(ptx.ceilDivU(b, s, @as(i32, @intCast(self.p.block_m)))), num_blocks);
        }
        return ptx.reduceAddSync(b, num_blocks);
    }

    /// `get_num_total_pool_blocks()`.
    pub fn getNumTotalPoolBlocks(self: Scheduler, stored: []const V) V {
        return self.getPoolBlockOffset(stored, self.b.cst(.i32, @as(i32, @intCast(self.p.num_experts_per_rank))));
    }

    /// `get_num_l1_warmup_waves(num_total_m_blocks, num_clusters, num_l1_n_clusters,
    /// num_l2_n_clusters)` on the device (`int` arithmetic; only `num_total_m_blocks` is a
    /// run-time value).
    fn getNumL1WarmupWaves(b: *B, num_total_m_blocks: V, num_clusters: i32, num_l1_n_clusters: i32, num_l2_n_clusters: i32) V {
        const num_first_l2_wave_m_blocks = @divTrunc(num_clusters + num_l2_n_clusters - 1, num_l2_n_clusters);
        const num_l1_warmup_clusters_for_first_l2_wave = @divTrunc(num_first_l2_wave_m_blocks * num_l1_n_clusters + num_clusters - 1, num_clusters);
        const num_interleave_cluster_diff_per_m_block: i32 = if (num_l1_n_clusters > num_l2_n_clusters) num_l1_n_clusters - num_l2_n_clusters else 0;
        // ceil_div(num_l1_n_clusters + (m - 1) * diff, num_clusters) + 1 (signed, truncating)
        const numerator = num_total_m_blocks.sub(1).mul(num_interleave_cluster_diff_per_m_block).add(num_l1_n_clusters);
        const num_warmup_waves_for_interleave_schedule = numerator.add(num_clusters - 1).div(num_clusters).add(1);
        return b.cst(.i32, num_l1_warmup_clusters_for_first_l2_wave).maximum(num_warmup_waves_for_interleave_schedule);
    }

    /// `fetch_expert_recv_count()`: each lane spins on its experts' `expert_recv_count_sum`
    /// until all `kNumSMs * kNumRanks` senders have added (count in the high word), then
    /// derives the pool size and the L1 warmup waves.
    pub fn fetchExpertRecvCount(self: Scheduler) Counts {
        const b = self.b;
        const p = self.p;
        const lane = ptx.getLaneIdx(b);
        const stored = b.arena.allocator().alloc(V, p.numExpertsPerLane()) catch @panic("OOM");
        for (stored, 0..) |*s, i| {
            const expert_idx = lane.add(@as(i32, @intCast(i * 32)));
            var in_range = flow.IfElse(V).open(b, ptx.ltU(b, expert_idx, @as(i32, @intCast(p.num_experts_per_rank))));
            {
                // do { value = ld_volatile(ptr); } while ((uint32_t)(value >> 32) != kNumSMs * kNumRanks);
                const ptr = self.workspace.expertRecvCountSumPtr(expert_idx);
                var loop = flow.While(void, V).open(b, {});
                const value = ptx.ldVolatileU64(b, ptr);
                const high = ptx.hi32(b, value);
                _ = loop.cond(high.ne(@as(i32, @intCast(p.num_sms * p.num_ranks))), value);
                in_range.orElse(loop.end({}));
            }
            const value = in_range.end(ptx.constU64(b, 0));
            s.* = ptx.lo32(b, value);
        }
        ptx.syncWarp(b);

        const num_total_m_blocks = self.getNumTotalPoolBlocks(stored);
        const num_total_l1_tasks = num_total_m_blocks.mul(@as(i32, @intCast(p.numL1Clusters())));
        const num_total_l1_waves = ptx.ceilDivU(b, num_total_l1_tasks, @as(i32, @intCast(p.num_sms / 2)));
        const min_l1_warmup_waves = getNumL1WarmupWaves(b, num_total_m_blocks, @intCast(p.num_sms / 2), @intCast(p.numL1Clusters()), @intCast(p.numL2Clusters()));
        // cute::min(int, uint32_t) compares as unsigned
        return .{
            .stored_num_tokens_per_expert = stored,
            .num_total_m_blocks = num_total_m_blocks,
            .num_sched_l1_waves = ptx.minU(b, min_l1_warmup_waves, num_total_l1_waves),
        };
    }

    // ---------- producer (leader CTA's scheduler warp) ----------

    /// `create_task(block_phase, m_block_idx, n_cluster_idx, shape_n, shape_k)`: find the
    /// local expert owning pool block `m_block_idx` (warp prefix sums of per-expert M blocks).
    fn createTask(self: Scheduler, counts: Counts, block_phase: BlockPhase, m_block_idx: V, n_cluster_idx: V, shape_n: u32, shape_k: u32) TaskInfo {
        const b = self.b;
        const p = self.p;
        const lane = ptx.getLaneIdx(b);
        var result = TaskInfo.make(b, block_phase, 0, 0, n_cluster_idx, m_block_idx, 0, @as(i32, @intCast(shape_n)), @as(i32, @intCast(shape_k)));
        var block_offset = b.cst(.i32, 0);
        for (counts.stored_num_tokens_per_expert, 0..) |num_tokens, i| {
            // Reduce whether task fall in the expert
            const expert_idx = lane.add(@as(i32, @intCast(i * 32)));
            const num_m_blocks = ptx.ceilDivU(b, num_tokens, @as(i32, @intCast(p.block_m)));
            const inclusive_num_m_blocks = ptx.warpInclusiveSum(b, num_m_blocks, lane);
            const lane_pool_block_offset = block_offset.add(inclusive_num_m_blocks).sub(num_m_blocks);
            const is_owner = ptx.ltU(b, expert_idx, @as(i32, @intCast(p.num_experts_per_rank)))
                .bitAnd(ptx.geU(b, m_block_idx, lane_pool_block_offset))
                .bitAnd(ptx.ltU(b, m_block_idx, lane_pool_block_offset.add(num_m_blocks)));
            const owner_mask = ptx.ballotSync(b, is_owner);

            // Exchange the expert info
            const Owner = struct { local_expert_idx: V, m_block_idx: V, valid_m: V };
            var owned = flow.IfElse(Owner).open(b, owner_mask.ne(0));
            {
                const owner_lane_idx = ptx.ffs(b, owner_mask).sub(1);
                const owner_m_block_idx = m_block_idx.sub(lane_pool_block_offset);
                const owner_valid_m = ptx.minU(b, num_tokens.sub(owner_m_block_idx.mul(@as(i32, @intCast(p.block_m)))), @as(i32, @intCast(p.block_m)));
                owned.orElse(.{
                    .local_expert_idx = ptx.shflSync(b, expert_idx, owner_lane_idx),
                    .m_block_idx = ptx.shflSync(b, owner_m_block_idx, owner_lane_idx),
                    .valid_m = ptx.shflSync(b, owner_valid_m, owner_lane_idx),
                });
            }
            const o = owned.end(.{ .local_expert_idx = result.local_expert_idx, .m_block_idx = result.m_block_idx, .valid_m = result.valid_m });
            result.local_expert_idx = o.local_expert_idx;
            result.m_block_idx = o.m_block_idx;
            result.valid_m = o.valid_m;
            block_offset = block_offset.add(ptx.shflSync(b, inclusive_num_m_blocks, @as(i32, 31)));
        }
        return result;
    }

    /// `get_next_local_task_idx(task_count_ptr, locality_domain_idx)`: one elected lane claims
    /// a task index; lane 0's value is broadcast.
    fn getNextLocalTaskIdx(b: *B, task_count_ptr: V, locality_domain_idx: V) V {
        var elected = flow.IfElse(V).open(b, ptx.electOneSync(b));
        elected.orElse(ptx.atomicAddU32(b, device.index(b, task_count_ptr, locality_domain_idx, 4), @as(i32, 1)));
        const result = elected.end(b.cst(.i32, 0));
        return ptx.shflSync(b, result, @as(i32, 0));
    }

    pub const TaskIdx = struct { ok: V, m_block_idx: V, n_cluster_idx: V };

    /// The static `get_next_task_idx<kNumClusters>(task_count_ptr, locality_domain_idx,
    /// num_m_blocks, m_block_idx, n_cluster_idx)`. On failure the outputs are unspecified
    /// (DeepGEMM leaves them untouched; callers ignore them).
    fn getNextTaskIdxIn(self: Scheduler, num_clusters: u32, task_count_ptr: V, locality_domain_idx: V, num_m_blocks: V) TaskIdx {
        const b = self.b;
        const num_local_clusters = num_clusters / self.p.num_locality_domains;
        const task_idx = getNextLocalTaskIdx(b, task_count_ptr, locality_domain_idx);
        return .{
            .ok = ptx.ltU(b, task_idx, num_m_blocks.mul(@as(i32, @intCast(num_local_clusters)))),
            .m_block_idx = ptx.divU(b, task_idx, @as(i32, @intCast(num_local_clusters))),
            .n_cluster_idx = locality_domain_idx.mul(@as(i32, @intCast(num_local_clusters))).add(ptx.remU(b, task_idx, @as(i32, @intCast(num_local_clusters)))),
        };
    }

    /// The member `get_next_task_idx<kNumClusters>(task_count_ptr, num_m_blocks, ...)`: the
    /// SM's own locality domain first; with enough tasks (`kNumMinStealWaves` waves), steal
    /// from the other domain (only evaluated when the first claim failed, as `or`/`and`
    /// short-circuit in C++).
    fn getNextTaskIdx(self: Scheduler, sm_locality_domain_idx: V, num_clusters: u32, task_count_ptr: V, num_m_blocks: V) TaskIdx {
        const b = self.b;
        const p = self.p;
        const own = self.getNextTaskIdxIn(num_clusters, task_count_ptr, sm_locality_domain_idx, num_m_blocks);
        if (p.num_locality_domains <= 1) return own;
        const steal_allowed = ptx.geU(b, num_m_blocks.mul(@as(i32, @intCast(num_clusters))), @as(i32, @intCast(num_min_steal_waves * (p.num_sms / 2))));
        var steal = flow.IfElse(TaskIdx).open(b, own.ok.bitXor(b.cst(.i1, true)).bitAnd(steal_allowed));
        steal.orElse(self.getNextTaskIdxIn(num_clusters, task_count_ptr, sm_locality_domain_idx.bitXor(1), num_m_blocks));
        return steal.end(own);
    }

    pub const Produced = struct { task: TaskInfo, num_sched_l1_waves: V };

    /// The producer's `get_next_task()`: L1 tasks during the warmup waves, then L2 tasks
    /// interleaved with one L1 wave each; an L2 task waits until every L1 task of its pool
    /// block has been claimed. Returns `TaskInfo::None` when both queues are drained.
    ///
    /// The C++ `while (true)` runs at most twice: the only `continue` (no L1 task left) sets
    /// `num_sched_l1_waves = kNumSchedL1WavesDone`, so the next pass takes the L2 branch. Here
    /// the L1 attempt and the L2 attempt are two successive `scf.if`s with that same order.
    fn getNextTaskProducer(self: Scheduler, counts: Counts, sm_locality_domain_idx: V, num_sched_l1_waves: V) Produced {
        const b = self.b;
        const p = self.p;
        const done = ptx.constU32(b, sched_l1_waves_done);
        const ws = self.workspace;

        // if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) { ... }
        const L1Try = struct { found: V, task: TaskInfo, waves: V };
        var l1 = flow.IfElse(L1Try).open(b, num_sched_l1_waves.ne(done).bitAnd(num_sched_l1_waves.ne(0)));
        {
            // One local L1 task per scheduler; globally this is one CTA-pair wave.
            const waves = num_sched_l1_waves.sub(1);
            const idx = self.getNextTaskIdx(sm_locality_domain_idx, p.numL1Clusters(), ws.l1TaskCountPtr(), counts.num_total_m_blocks);
            var created = flow.IfElse(TaskInfo).open(b, idx.ok);
            created.orElse(self.createTask(counts, .linear1, idx.m_block_idx, idx.n_cluster_idx, p.l1_shape_n, p.l1_shape_k));
            const task = created.end(TaskInfo.none(b));
            // No more L1 tasks: `num_sched_l1_waves = kNumSchedL1WavesDone; continue;`
            l1.orElse(.{ .found = idx.ok, .task = task, .waves = b.select(idx.ok, waves, done) });
        }
        const tried = l1.end(.{ .found = b.cst(.i1, false), .task = TaskInfo.none(b), .waves = num_sched_l1_waves });

        var l2 = flow.IfElse(Produced).open(b, tried.found.bitXor(b.cst(.i1, true)));
        {
            const idx = self.getNextTaskIdx(sm_locality_domain_idx, p.numL2Clusters(), ws.l2TaskCountPtr(), counts.num_total_m_blocks);
            var created = flow.IfElse(Produced).open(b, idx.ok);
            {
                // The next task should be L1
                const waves = b.select(tried.waves.ne(done), b.cst(.i32, 1), tried.waves);
                const task = self.createTask(counts, .linear2, idx.m_block_idx, idx.n_cluster_idx, p.l2_shape_n, p.l2_shape_k);
                // Wait until all required L1 tasks are fetched from all scheduling queues
                const num_required_l1_tasks = task.pool_block_idx.add(1).mul(@as(i32, @intCast(p.numL1Clusters() / p.num_locality_domains)));
                for (0..p.num_locality_domains) |i| {
                    flow.spin(b, struct {
                        ptr: V,
                        required: V,
                        pub fn pending(s: @This(), bb: *B) V {
                            return ptx.ltU(bb, ptx.ldVolatileU32(bb, s.ptr), s.required);
                        }
                    }{ .ptr = device.offsetBy(b, ws.l1TaskCountPtr(), 4 * i), .required = num_required_l1_tasks });
                }
                created.orElse(.{ .task = task, .num_sched_l1_waves = waves });
            }
            // `break` out of the loop: TaskInfo::None
            l2.orElse(created.end(.{ .task = TaskInfo.none(b), .num_sched_l1_waves = tried.waves }));
        }
        return l2.end(.{ .task = tried.task, .num_sched_l1_waves = tried.waves });
    }

    /// `publish_task(task_info, lane_idx)`: lanes 0 and 1 each expect 32 bytes on CTA `lane`'s
    /// full barrier of the stage and `st.async` the task there; then advance.
    fn publishTask(self: Scheduler, pipe: Pipe, task: TaskInfo, lane_idx: V) Pipe {
        const b = self.b;
        var two = b.openIf(ptx.ltU(b, lane_idx, @as(i32, 2)));
        {
            const bar = self.fullBarrier(pipe.stage);
            ptx.mbarrierArriveAndExpectTxCluster(b, bar, @as(i32, @sizeOf(layout.TaskInfo)), lane_idx, b.cst(.i32, 1));
            const words = task.words();
            ptx.stAsyncCluster(b, self.taskInfoSlot(pipe.stage), &words, lane_idx, bar);
            two.yieldThen(.{});
        }
        ptx.syncWarp(b);
        return pipe.advance();
    }

    /// `shared_mainloop<kBlockPhase, kShapeN, kShapeK>(num_tokens, lane_idx, task_count_ptr)`:
    /// dynamic scheduling of the shared expert's M x N-cluster tasks.
    ///
    /// `while (true) { wait empty; if (not claim) break; publish; }` becomes an `scf.while`
    /// whose before region waits and claims and whose after region publishes.
    fn sharedMainloop(self: Scheduler, pipe: Pipe, sm_locality_domain_idx: V, block_phase: BlockPhase, shape_n: u32, shape_k: u32, num_tokens: V, lane_idx: V, task_count_ptr: V) Pipe {
        const b = self.b;
        const p = self.p;
        const num_n_clusters = shape_n / p.block_n / 2;
        std.debug.assert(num_n_clusters % p.num_locality_domains == 0);
        const num_m_blocks = ptx.ceilDivU(b, num_tokens, @as(i32, @intCast(p.block_m)));

        const Fwd = struct { pipe: Pipe, m_block_idx: V, n_cluster_idx: V };
        var loop = flow.While(Pipe, Fwd).open(b, pipe);
        const before = loop.state;
        ptx.mbarrierWait(b, self.emptyBarrier(before.stage), before.phase.bitXor(1));
        // Use dynamic scheduling to reduce tailing across shared L1/L2 tile shapes.
        const idx = self.getNextTaskIdx(sm_locality_domain_idx, num_n_clusters, task_count_ptr, num_m_blocks);
        const after = loop.cond(idx.ok, .{ .pipe = before, .m_block_idx = idx.m_block_idx, .n_cluster_idx = idx.n_cluster_idx });
        const valid_m = ptx.minU(b, num_tokens.sub(after.m_block_idx.mul(@as(i32, @intCast(p.block_m)))), @as(i32, @intCast(p.block_m)));
        const task = TaskInfo.make(b, block_phase, 0, after.m_block_idx, after.n_cluster_idx, after.m_block_idx, valid_m, @as(i32, @intCast(shape_n)), @as(i32, @intCast(shape_k)));
        const next = self.publishTask(after.pipe, task, lane_idx);
        return loop.end(next).pipe;
    }

    /// `mainloop(num_tokens, sm_locality_domains)`: the leader CTA's scheduler warp.
    pub fn mainloop(self: Scheduler, num_tokens: V, sm_locality_domains: V) void {
        const b = self.b;
        const p = self.p;
        const ws = self.workspace;
        const lane_idx = ptx.getLaneIdx(b);
        const sm_locality_domain_idx = if (p.num_locality_domains == 1)
            b.cst(.i32, 0)
        else
            ptx.ldGlobalU8(b, device.index(b, sm_locality_domains, ptx.getSmIdx(b), 1));

        // Shared L2 tasks go before the routed tasks if one wave fits the dispatch gap, else at the tail
        const num_shared_l2_clusters = p.sharedL2ShapeN() / p.block_n / 2;
        const is_shared_l2_early = ptx.leU(b, ptx.ceilDivU(b, num_tokens, @as(i32, @intCast(p.block_m))).mul(@as(i32, @intCast(num_shared_l2_clusters))), @as(i32, @intCast(p.num_sms / 2)));
        var pipe = Pipe.init(b);
        if (p.hasShared()) {
            // Shared expert L1 tasks do not depend on dispatch.
            pipe = self.sharedMainloop(pipe, sm_locality_domain_idx, .shared_linear1, p.sharedL1ShapeN(), p.sharedL1ShapeK(), num_tokens, lane_idx, ws.sharedL1TaskCountPtr());
            var early = flow.IfElse(Pipe).open(b, is_shared_l2_early);
            early.orElse(self.sharedMainloop(pipe, sm_locality_domain_idx, .shared_linear2, p.sharedL2ShapeN(), p.sharedL2ShapeK(), num_tokens, lane_idx, ws.sharedL2TaskCountPtr()));
            pipe = early.end(pipe);
        }

        // Wait dispatch's results
        const counts = self.fetchExpertRecvCount();

        // Generate routed tasks. Keep the original wait -> claim -> publish ordering:
        // `get_next_task()` advances global task counters and must not run before the
        // schedule slot is released by consumers.
        // do { wait; task = get_next_task(); if (valid) publish; } while (valid);
        const State = struct { pipe: Pipe, waves: V };
        var loop = flow.While(State, State).open(b, .{ .pipe = pipe, .waves = counts.num_sched_l1_waves });
        {
            const s = loop.state;
            ptx.mbarrierWait(b, self.emptyBarrier(s.pipe.stage), s.pipe.phase.bitXor(1));
            const produced = self.getNextTaskProducer(counts, sm_locality_domain_idx, s.waves);
            const valid = produced.task.isValid();
            var publish = flow.IfElse(Pipe).open(b, valid);
            publish.orElse(self.publishTask(s.pipe, produced.task, lane_idx));
            const next_pipe = publish.end(s.pipe);
            const after = loop.cond(valid, .{ .pipe = next_pipe, .waves = produced.num_sched_l1_waves });
            pipe = loop.end(after).pipe;
        }

        if (p.hasShared()) {
            var late = flow.IfElse(Pipe).open(b, is_shared_l2_early.bitXor(b.cst(.i1, true)));
            late.orElse(self.sharedMainloop(pipe, sm_locality_domain_idx, .shared_linear2, p.sharedL2ShapeN(), p.sharedL2ShapeK(), num_tokens, lane_idx, ws.sharedL2TaskCountPtr()));
            pipe = late.end(pipe);
        }

        // Sentinel.
        ptx.mbarrierWait(b, self.emptyBarrier(pipe.stage), pipe.phase.bitXor(1));
        _ = self.publishTask(pipe, TaskInfo.none(b), lane_idx);
    }
};
