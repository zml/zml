//! Device code of DeepGEMM's MegaMoE kernel, `sm100_fp8_fp4_mega_moe_impl` (vLLM's vendored
//! `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`), for the Zig CuTe builder.
//!
//! Same structure and order as the `.cuh`: the prologue (descriptor prefetch, shared-memory
//! plan, barrier initialization, cluster syncs, 2-CTA TMEM allocation, PDL wait), then one
//! function per warp role:
//!
//!   warps 0-3   `dispatchRole`   (token counting, slot reservation, source-index push, grid
//!                                 sync, count publish, NVLink barriers, pull loop, cleanup)
//!   warp 4      `tmaARole`       (tokens + SFA, 2-CTA TMA)
//!   warp 5      `tmaBRole`       (weights + SFB, 2-CTA TMA)
//!   warp 6      `mmaRole`        (leader CTA: UTCCP SF copies, block-scaled 2-CTA UMMA, commits)
//!   warp 7      `schedulerRole`  (leader CTA: `MegaMoEScheduler::mainloop`, `scheduler.zig`)
//!   warps 8-15  `epilogueRole`   (L1 epilogue: SwiGLU, amax, UE8M0, FP8 `stmatrix` + TMA
//!                                 store; L2 epilogue: BF16 `stmatrix`, NVLink writes into the
//!                                 combine slots; then `combine`: top-k reduction into `y`)
//!
//! Every `ptx::`/`cute::`/`cutlass::arch::` primitive goes through `ptx.zig` (DeepGEMM's exact
//! PTX). DeepGEMM's `uint32_t` arithmetic is `i32` with unsigned division, remainder and
//! comparisons (`ptx.divU`, `ptx.ltU`, ...); its `int current_expert_idx` stays signed.
//! Mutable C++ locals carried by loops (`stage_idx`/`phase`, the scheduler pipeline, the
//! dispatch cursor, the pull barrier phase) are `flow` loop states; every rewrite of a
//! `break`/`continue` into `scf` is commented where it happens.
const std = @import("std");

const mlir = @import("mlir");
const cute = @import("kernels/cute/builder");

const device = @import("device.zig");
const flow = @import("flow.zig");
const layout = @import("layout.zig");
const ptx = @import("ptx.zig");
const scheduler = @import("scheduler.zig");
const tma = @import("tma.zig");

const B = cute.Builder;
const V = cute.Value;
const BlockPhase = layout.BlockPhase;
const Kc = layout.Kernel;

/// Every template parameter of `sm100_fp8_fp4_mega_moe_impl`. `kernel` carries
/// `kNumMaxTokensPerRank`, `kHidden`, `kIntermediateHidden`, `kNumExperts`,
/// `kNumSharedExperts`, `kNumTopk` (its `shape`), `BLOCK_M/N/K`, `STORE_BLOCK_M_L1/L2`,
/// `SF_BLOCK_M/N`, `kNumRingTokens`, `kNumSFRingTokens`, `kNumStages`, `kNumBytesPerPull`, the
/// thread counts (its `cfg`), `kNumSMs`, `kNumRanks` (`shape.num_ranks`) and `weight_dtype_t`;
/// `maps` the 18 tensor maps the kernel receives (their parameters are part of the argument
/// types). Values are folded into the emitted code as constants, like DeepGEMM's JIT.
pub const Config = struct {
    kernel: layout.Kernel,
    maps: layout.TensorMaps,
    /// `kActivationClamp` (`inf`: no clamp).
    activation_clamp: f32 = std.math.inf(f32),
    /// `kClampAfterSilu`.
    clamp_after_silu: bool = false,
    /// `kActivationAlpha`.
    activation_alpha: f32 = 1.0,
    /// `kActivationBeta`.
    activation_beta: f32 = 0.0,
    /// `kFastMath`.
    fast_math: bool = true,
    /// `comm::kNumTimeoutCycles` of `wait_until` (grid sync, NVLink barrier, combine): after this
    /// many `clock64` cycles the kernel `trap`s (DeepGEMM prints, then asserts). 0: no timeout.
    timeout_cycles: i64 = device.default_timeout_cycles,

    /// The DeepGEMM static asserts of the kernel body.
    pub fn check(self: Config) void {
        const k = self.kernel;
        const c = k.cfg;
        std.debug.assert(c.num_dispatch_threads % 128 == 0);
        std.debug.assert(c.num_non_epilogue_threads == 128);
        std.debug.assert(c.num_epilogue_threads % 128 == 0);
        std.debug.assert(k.shape.num_experts % k.shape.num_ranks == 0);
        std.debug.assert(c.sf_block_m == layout.alignUp(c.block_m, Kc.num_utccp_aligned_elems));
        std.debug.assert(c.sf_block_n == c.block_n);
        std.debug.assert(c.block_m % 16 == 0);
        std.debug.assert(c.block_n == Kc.layout_ad_m);
        std.debug.assert(c.block_n % Kc.swizzle_cd_mode == 0);
        std.debug.assert(32 <= k.numTmemCols() and k.numTmemCols() <= 512);
        std.debug.assert(k.registerSum() <= 64512);
        std.debug.assert(k.shape.num_topk <= 32);
        std.debug.assert(k.shape.hidden % c.num_bytes_per_pull == 0);
        std.debug.assert(k.shape.hidden % 128 == 0);
        std.debug.assert(k.num_sms > 1);
        std.debug.assert(numDispatchWarps(self) >= 2);
        std.debug.assert(k.shape.num_ranks <= c.num_dispatch_threads and k.shape.num_ranks <= c.num_epilogue_threads);
        std.debug.assert((numDispatchWarps(self) + numMmaNonEpilogueWarps(self)) % 4 == 0 and numEpilogueWarps(self) % 4 == 0);
        std.debug.assert(c.block_k / Kc.umma_block_k >= 1 and c.block_k % Kc.umma_block_k == 0);
        std.debug.assert(c.num_stages <= 32);
        std.debug.assert(c.smem_size == k.smem().used_bytes);
        // Not ported: the `kFastMath == false` SwiGLU (`expf` and IEEE division)
        std.debug.assert(self.fast_math);
    }
};

fn numDispatchWarps(cfg: Config) u32 {
    return cfg.kernel.cfg.num_dispatch_threads / 32;
}
fn numMmaNonEpilogueWarps(cfg: Config) u32 {
    return cfg.kernel.cfg.num_non_epilogue_threads / 32;
}
fn numEpilogueWarps(cfg: Config) u32 {
    return cfg.kernel.cfg.num_epilogue_threads / 32;
}

/// `kNumThreads` (the block size).
pub fn numThreads(cfg: Config) u32 {
    return cfg.kernel.cfg.numThreads();
}

// Intra-SM named barrier indices and NVLink barrier tags.
const dispatch_barrier_idx = Kc.dispatch_barrier_idx;
const dispatch_with_epilogue_barrier_idx = Kc.dispatch_with_epilogue_barrier_idx;
const epilogue_full_barrier_idx = Kc.epilogue_full_barrier_idx;
const epilogue_wg_barrier_start_idx = Kc.epilogue_wg_barrier_start_idx;
const dispatch_grid_sync_index = Kc.dispatch_grid_sync_index;
const epilogue_grid_sync_index = Kc.epilogue_grid_sync_index;

/// Kernel argument positions (after `declareArgList`).
pub const Args = struct {
    pub const y = 0;
    pub const cumulative_local_expert_recv_stats = 1;
    pub const num_tokens = 2;
    pub const sym_base = 3;
    pub const sym_offsets = 4;
    /// `sym_rank` is at `sym_offsets + num_ranks`, the 18 maps follow, then
    /// `sm_locality_domains`.
    pub fn symRank(num_ranks: u32) usize {
        return sym_offsets + num_ranks;
    }
    pub fn map(num_ranks: u32, i: usize) usize {
        return symRank(num_ranks) + 1 + i;
    }
    pub fn smLocalityDomains(num_ranks: u32) usize {
        return map(num_ranks, layout.TensorMaps.names.len);
    }
    pub fn count(num_ranks: u32) usize {
        return smLocalityDomains(num_ranks) + 1;
    }
};

/// The 18 `tma.Map`s of `cfg.maps`, in kernel-parameter order (stable storage for `raw()`).
pub fn tmaMaps(b: *B, cfg: Config) []tma.Map {
    const out = b.arena.allocator().alloc(tma.Map, layout.TensorMaps.names.len) catch @panic("OOM");
    inline for (layout.TensorMaps.names, 0..) |name, i| out[i] = .init(@field(cfg.maps, name));
    return out;
}

/// The kernel's parameter list, in DeepGEMM's order: `void* y`,
/// `int* cumulative_local_expert_recv_stats` (an `i64` address, 0 for `nullptr`),
/// `uint32_t num_tokens`, the `SymBuffer` (`base: i64`, `offsets[kNumRanks]: i64`,
/// `rank_idx: i32`), the 18 `__grid_constant__` tensor maps and
/// `const uint8_t* sm_locality_domains`.
pub fn argSpecs(b: *B, cfg: Config, maps: []const tma.Map) []cute.ArgSpec {
    const R = cfg.kernel.shape.num_ranks;
    const a = b.arena.allocator();
    const specs = a.alloc(cute.ArgSpec, Args.count(R)) catch @panic("OOM");
    const i64t = cute.DType.i64.toMlir(b.ctx);
    const i32t = cute.DType.i32.toMlir(b.ctx);
    specs[Args.y] = .{ .name = "y", .kind = .{ .ptr = .bf16 } };
    specs[Args.cumulative_local_expert_recv_stats] = .{ .name = "cumulative_local_expert_recv_stats", .kind = .{ .mlir_type = .{ .type_ = i64t } } };
    specs[Args.num_tokens] = .{ .name = "num_tokens", .kind = .{ .mlir_type = .{ .type_ = i32t } } };
    specs[Args.sym_base] = .{ .name = "sym_base", .kind = .{ .mlir_type = .{ .type_ = i64t } } };
    for (0..R) |r| specs[Args.sym_offsets + r] = .{ .name = "sym_offset", .kind = .{ .mlir_type = .{ .type_ = i64t } } };
    specs[Args.symRank(R)] = .{ .name = "sym_rank", .kind = .{ .mlir_type = .{ .type_ = i32t } } };
    for (maps, 0..) |*m, i| specs[Args.map(R, i)] = .{ .name = layout.TensorMaps.names[i], .kind = .{ .mlir_type = .{ .type_ = b.rawTmaArgType(m.raw()), .grid_constant = true } } };
    specs[Args.smLocalityDomains(R)] = .{ .name = "sm_locality_domains", .kind = .{ .ptr = .i8 } };
    return specs;
}

/// Emits `sm100_fp8_fp4_mega_moe_impl` as the `cuda.kernel` `name` of the open program.
pub fn emit(b: *B, cfg: Config, name: []const u8) cute.FinishError!void {
    cfg.check();
    const maps = tmaMaps(b, cfg);
    b.beginFunction(name, .cuda_kernel);
    try b.declareArgList(argSpecs(b, cfg, maps));
    // `__launch_bounds__(kNumThreads, 1)`, opt-in shared memory, 2-CTA clusters.
    b.setFunctionAttribute("nvvm.minctasm", .int(b.ctx, .i32, 1));
    b.setFunctionAttribute("cu_attrs", b.parseAttribute("{max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}"));
    var k = Kernel.init(b, cfg, maps);
    k.body();
    b.endFunction(.{ @as(i32, @intCast(numThreads(cfg))), 1, 1 });
}

/// Run-time values and compile-time constants of one kernel instance.
const Kernel = struct {
    b: *B,
    cfg: Config,
    k: layout.Kernel,
    sp: scheduler.Params,
    so: layout.SharedStorage,
    maps: []const tma.Map,

    // Arguments.
    y: V,
    cumulative_local_expert_recv_stats: V,
    num_tokens: V,
    sym: device.Sym,
    descs: [18]V,
    sm_locality_domains: V,

    // Thread indices.
    is_leader_cta: V,
    sm_idx: V,
    thread_idx: V,
    warp_idx: V,
    lane_idx: V,

    smem: V,
    buffer: device.Buffer,
    workspace: device.Workspace,
    sched: scheduler.Scheduler,

    fn init(b: *B, cfg: Config, maps: []const tma.Map) Kernel {
        const k = cfg.kernel;
        const R = k.shape.num_ranks;
        var self: Kernel = undefined;
        self.b = b;
        self.cfg = cfg;
        self.k = k;
        self.sp = .fromKernel(k);
        self.so = k.smem();
        self.maps = maps;
        self.y = b.ptrToInt(b.arg(Args.y), .i64);
        self.cumulative_local_expert_recv_stats = b.arg(Args.cumulative_local_expert_recv_stats);
        self.num_tokens = b.arg(Args.num_tokens);
        const offsets = b.arena.allocator().alloc(V, R) catch @panic("OOM");
        for (offsets, 0..) |*o, r| o.* = b.arg(Args.sym_offsets + r);
        self.sym = .{ .b = b, .num_ranks = R, .base = b.arg(Args.sym_base), .offsets = offsets, .rank_idx = b.arg(Args.symRank(R)) };
        self.sm_locality_domains = b.ptrToInt(b.arg(Args.smLocalityDomains(R)), .i64);
        for (&self.descs, maps, 0..) |*d, *m, i| d.* = b.rawTmaDescAddr(b.arg(Args.map(R, i)), m.raw());
        return self;
    }

    fn c(self: *const Kernel) layout.Config {
        return self.k.cfg;
    }

    fn u(self: *const Kernel, v: u32) V {
        return ptx.constU32(self.b, v);
    }

    fn ci(self: *const Kernel, v: anytype) V {
        return self.b.cst(.i32, @as(i32, @intCast(v)));
    }

    fn desc(self: *const Kernel, comptime name: []const u8) V {
        inline for (layout.TensorMaps.names, 0..) |n, idx| {
            if (comptime std.mem.eql(u8, n, name)) return self.descs[idx];
        }
        @compileError("no tensor map " ++ name);
    }

    // ---------- shared memory (`SharedStorage` fields, `shared::cta` addresses) ----------

    fn at(self: *const Kernel, offset: u32) V {
        return self.smem.add(@as(i32, @intCast(offset)));
    }
    fn atIdx(self: *const Kernel, offset: u32, idx: V, stride: u32) V {
        return self.at(offset).add(idx.mul(@as(i32, @intCast(stride))));
    }
    fn expertTokenCount(self: *const Kernel, expert_idx: V) V {
        return self.atIdx(self.so.expert_token_count, expert_idx, 4);
    }
    fn dispatchSendBuffer(self: *const Kernel, warp: V) V {
        return self.atIdx(self.so.dispatch_send_buffer, warp, self.c().num_bytes_per_pull);
    }
    fn smemA(self: *const Kernel, stage: V) V {
        return self.atIdx(self.so.smem_a, stage, self.k.smemABytes());
    }
    fn smemB(self: *const Kernel, stage: V) V {
        return self.atIdx(self.so.smem_b, stage, self.k.smemBBytes());
    }
    fn smemSfa(self: *const Kernel, stage: V) V {
        return self.atIdx(self.so.smem_sfa, stage, self.k.smemSfaBytes());
    }
    fn smemSfb(self: *const Kernel, stage: V) V {
        return self.atIdx(self.so.smem_sfb, stage, self.k.smemSfbBytes());
    }
    fn dispatchBarrier(self: *const Kernel, idx: V) V {
        return self.atIdx(self.so.dispatch_barriers, idx, 8);
    }
    fn fullBarrier(self: *const Kernel, stage: V) V {
        return self.atIdx(self.so.full_barriers, stage, 8);
    }
    fn emptyBarrier(self: *const Kernel, stage: V) V {
        return self.atIdx(self.so.empty_barriers, stage, 8);
    }
    fn tmemFullBarrier(self: *const Kernel, stage: V) V {
        return self.atIdx(self.so.tmem_full_barriers, stage, 8);
    }
    fn tmemEmptyBarrier(self: *const Kernel, stage: V) V {
        return self.atIdx(self.so.tmem_empty_barriers, stage, 8);
    }
    fn tmemPtrInSmem(self: *const Kernel) V {
        return self.at(self.so.tmem_ptr_in_smem);
    }

    // ---------- `advance_pipeline` ----------

    const StagePhase = struct { stage: V, phase: V };

    /// `advance_pipeline(k_block_idx)` without the `++k_block_idx` (the `scf.for` step):
    /// `stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1; phase ^= stage_idx == 0`.
    fn advancePipeline(self: *const Kernel, sp: StagePhase) StagePhase {
        const b = self.b;
        const stage = b.select(sp.stage.eq(@as(i32, @intCast(self.c().num_stages - 1))), b.cst(.i32, 0), sp.stage.add(1));
        return .{ .stage = stage, .phase = sp.phase.bitXor(stage.eq(0).to(.i32)) };
    }

    // ==================== prologue ====================

    fn body(self: *Kernel) void {
        const b = self.b;
        const k = self.k;
        const cfg = self.cfg;

        // Thread indices
        self.is_leader_cta = ptx.blockRankInCluster(b).eq(0);
        self.sm_idx = b.blockIdx().x;
        self.thread_idx = b.threadIdx().x;
        self.warp_idx = ptx.canonicalWarpIdxSync(b);
        self.lane_idx = ptx.getLaneIdx(b);

        // Prefetch TMA descriptors at the very beginning
        {
            var warp0 = b.openIf(self.warp_idx.eq(0));
            for (self.descs) |d| ptx.prefetchTmaDescriptor(b, d);
            warp0.yieldThen(.{});
        }

        // Workspaces and Buffer (`layout::MegaMoEBuffer(sym_buffer.get_base_ptr(), ...)`)
        self.buffer = .init(b, self.sym.basePtr(), symmBuffer(cfg));
        self.workspace = self.buffer.workspace;

        // Shared memory: `extern __shared__ __align__(1024) uint8_t smem_buffer[]`, the
        // `SharedStorage` fields at `layout.SharedStorage` offsets.
        self.smem = ptx.sharedStorageBase(b, self.c().smem_size);
        self.sched = .{
            .b = b,
            .p = self.sp,
            .workspace = self.workspace,
            .task_info_full_barriers = self.at(self.so.task_info_full_barriers),
            .task_info_empty_barriers = self.at(self.so.task_info_empty_barriers),
            .task_infos = self.at(self.so.task_infos),
        };

        // A cluster sync is essential for 2CTA tensor memory allocation
        ptx.clusterSyncWithRelaxedArrive(b);

        // Initialization
        {
            var warp0 = b.openIf(self.warp_idx.eq(0));
            // Clean shared memory (the bytes must be 8 bytes aligned)
            var elected = b.openIf(ptx.electOneSync(b));
            ptx.stSharedBulk(b, self.at(self.so.expert_token_count), layout.alignUp(k.shape.num_experts * 4, 1024));
            elected.yieldThen(.{});
            warp0.yieldThen(.{});
        }
        {
            // Init m-barriers for dispatch
            var warp1 = b.openIf(self.warp_idx.eq(1));
            const num_dispatch_warps = numDispatchWarps(cfg);
            for (0..layout.ceilDiv(num_dispatch_warps, 32)) |j| {
                const idx = self.lane_idx.add(@as(i32, @intCast(j * 32)));
                var in_range = b.openIf(ptx.ltU(b, idx, @as(i32, @intCast(num_dispatch_warps))));
                ptx.mbarrierInit(b, self.dispatchBarrier(idx), 1);
                in_range.yieldThen(.{});
            }
            ptx.fenceBarrierInit(b);
            warp1.yieldThen(.{});
        }
        {
            // Init GEMM barriers
            var warp2 = b.openIf(self.warp_idx.eq(2));
            var elected = b.openIf(ptx.electOneSync(b));
            for (0..self.c().num_stages) |s| {
                // Arrive at 2 CTAs, A + B
                ptx.mbarrierInit(b, self.fullBarrier(self.ci(s)), 2 * 2);
                ptx.mbarrierInit(b, self.emptyBarrier(self.ci(s)), 1);
            }
            for (0..Kc.num_epilogue_stages) |s| {
                // Arrive at all CTAs
                ptx.mbarrierInit(b, self.tmemFullBarrier(self.ci(s)), 1);
                // Arrive only at the leader CTA
                ptx.mbarrierInit(b, self.tmemEmptyBarrier(self.ci(s)), 2 * self.c().num_epilogue_threads);
            }
            for (0..numEpilogueWarps(cfg) * 2) |s| ptx.mbarrierInit(b, self.at(self.so.combine_barriers + @as(u32, @intCast(8 * s))), 1);
            for (0..scheduler.num_schedule_stages) |s| {
                ptx.mbarrierInit(b, self.at(self.so.task_info_full_barriers + @as(u32, @intCast(8 * s))), 1);
                ptx.mbarrierInit(b, self.at(self.so.task_info_empty_barriers + @as(u32, @intCast(8 * s))), k.numScheduleConsumerThreads());
            }
            elected.yieldThen(.{});
            ptx.fenceBarrierInit(b);
            warp2.yieldThen(.{});
        }
        {
            // Allocate tensor memory
            var warp3 = b.openIf(self.warp_idx.eq(3));
            ptx.tmemAllocate2Sm(b, k.numTmemCols(), self.tmemPtrInSmem());
            warp3.yieldThen(.{});
        }
        ptx.clusterSync(b);

        // Wait for primary kernel completion
        ptx.gridDependencySynchronize(b);

        // Different warp roles (the C++ `if / else if` chain: disjoint warp ranges)
        const ndw = numDispatchWarps(cfg);
        {
            var role = b.openIf(ptx.ltU(b, self.warp_idx, @as(i32, @intCast(ndw))));
            self.dispatchRole();
            role.yieldThen(.{});
        }
        {
            var role = b.openIf(self.warp_idx.eq(@as(i32, @intCast(ndw))));
            self.tmaARole();
            role.yieldThen(.{});
        }
        {
            var role = b.openIf(self.warp_idx.eq(@as(i32, @intCast(ndw + 1))));
            self.tmaBRole();
            role.yieldThen(.{});
        }
        {
            var role = b.openIf(self.warp_idx.eq(@as(i32, @intCast(ndw + 2))));
            self.mmaRole();
            role.yieldThen(.{});
        }
        {
            var role = b.openIf(self.warp_idx.eq(@as(i32, @intCast(ndw + 3))));
            self.schedulerRole();
            role.yieldThen(.{});
        }
        {
            var role = b.openIf(ptx.geU(b, self.warp_idx, @as(i32, @intCast(ndw + numMmaNonEpilogueWarps(cfg)))));
            self.epilogueRole();
            role.yieldThen(.{});
        }
    }

    // ==================== dispatch warps ====================

    /// `read_topk_idx(process)`: each warp reads `kNumTokensPerWarp` tokens' top-k expert ids
    /// (one lane per token-topk) and calls `process(token_topk_idx, expert_idx)` for routed ones.
    fn readTopkIdx(self: *const Kernel, comptime push: bool) void {
        const b = self.b;
        const k = self.k;
        const topk = k.shape.num_topk;
        const ndw = numDispatchWarps(self.cfg);
        const tokens_per_warp = k.numTokensPerWarp();
        const start = self.sm_idx.mul(@as(i32, @intCast(ndw))).add(self.warp_idx).mul(@as(i32, @intCast(tokens_per_warp)));
        var loop = flow.forRange(b, start, self.num_tokens, @as(i32, @intCast(k.num_sms * ndw * tokens_per_warp)));
        {
            const t = loop.iv;
            // Allocate slots for each token-topk
            const has_token = ptx.ltU(b, t.add(ptx.divU(b, self.lane_idx, @as(i32, @intCast(topk)))), self.num_tokens)
                .bitAnd(ptx.ltU(b, self.lane_idx, @as(i32, @intCast(k.numActivateLanes()))));
            var active = b.openIf(has_token);
            {
                const token_topk_idx = t.mul(@as(i32, @intCast(topk))).add(self.lane_idx);
                const expert_idx = ptx.lo32(b, ptx.ldGlobalU64(b, device.index(b, self.buffer.input_topk_idx.basePtr(), token_topk_idx, 8)));
                var routed = b.openIf(expert_idx.ge(0));
                if (push) self.pushSourceIndex(token_topk_idx, expert_idx) else self.countToken(expert_idx);
                routed.yieldThen(.{});
                active.yieldThen(.{});
            }
            ptx.syncWarp(b);
            _ = loop.end({});
        }
    }

    /// Count experts' tokens: `atomicAdd_block(expert_token_count + expert_idx, 1)`.
    fn countToken(self: *const Kernel, expert_idx: V) void {
        _ = ptx.atomicAddBlockShared(self.b, self.expertTokenCount(expert_idx), @as(i32, 1));
    }

    /// Write source indices: claim a slot of the expert on the destination rank and store the
    /// token-topk index there (through NVLink for a peer rank).
    fn pushSourceIndex(self: *const Kernel, token_topk_idx: V, expert_idx: V) void {
        const b = self.b;
        const e_per_rank: i32 = @intCast(self.k.numExpertsPerRank());
        const dst_rank_idx = ptx.divU(b, expert_idx, e_per_rank);
        const dst_slot_idx = ptx.atomicAddBlockShared(b, self.expertTokenCount(expert_idx), @as(i32, 1));
        const dst_ptr = self.workspace.srcTokenTopkIdxPtr(ptx.remU(b, expert_idx, e_per_rank), self.sym.rank_idx, dst_slot_idx);
        self.sym.st(1, self.sym.map(dst_ptr, dst_rank_idx), .{token_topk_idx});
    }

    fn dispatchRole(self: *Kernel) void {
        const b = self.b;
        const k = self.k;
        const cfg = self.cfg;
        const c_ = self.c();
        const ndw = numDispatchWarps(cfg);
        const num_dispatch_threads = c_.num_dispatch_threads;
        const R = k.shape.num_ranks;
        const E = k.shape.num_experts;
        const scope: ptx.AlignedScope = .{ .num_threads = num_dispatch_threads, .barrier_idx = dispatch_barrier_idx };

        // Adjust registers
        ptx.warpgroupRegDealloc(b, k.numDispatchRegisters());

        // Count experts' tokens
        self.readTopkIdx(false);
        ptx.syncAligned(b, @as(i32, @intCast(num_dispatch_threads)), @as(i32, dispatch_barrier_idx));

        // Get SM offset
        {
            var loop = flow.forRange(b, self.thread_idx, @as(i32, @intCast(E)), @as(i32, @intCast(num_dispatch_threads)));
            const e = loop.iv;
            const count = b.loadWords(1, self.expertTokenCount(e), .shared, .{})[0];
            const send_value = ptx.constU64(b, 1 << 32).bitOr(ptx.zext(b, count, .i64));
            const old = ptx.atomicAddU64(b, self.workspace.expertSendCountPtr(e), send_value);
            b.storeWords(1, self.expertTokenCount(e), .shared, .{ptx.lo32(b, old)});
            _ = loop.end({});
        }
        ptx.syncAligned(b, @as(i32, @intCast(num_dispatch_threads)), @as(i32, dispatch_barrier_idx));

        // Write source indices
        self.readTopkIdx(true);

        // Grid sync
        ptx.gridSync(b, k.num_sms, self.workspace.gridSyncCountPtr(dispatch_grid_sync_index), self.sm_idx, self.thread_idx, scope, cfg.timeout_cycles);

        // Write expert count
        {
            var sm0 = b.openIf(self.sm_idx.eq(0));
            // Push this launch's grid index to every peer for tagging combine readiness; +1
            // differs from a zeroed workspace
            var rank_thread = b.openIf(ptx.ltU(b, self.thread_idx, @as(i32, @intCast(R))));
            self.sym.stU64(self.sym.map(self.workspace.peerGridIdxPtr(self.sym.rank_idx), self.thread_idx), ptx.getGridIdx(b).add(1));
            rank_thread.yieldThen(.{});
            ptx.syncWarp(b);

            const e_per_rank: i32 = @intCast(k.numExpertsPerRank());
            var loop = flow.forRange(b, self.thread_idx, @as(i32, @intCast(E)), @as(i32, @intCast(num_dispatch_threads)));
            {
                const e = loop.iv;
                const dst_rank_idx = ptx.divU(b, e, e_per_rank);
                const dst_local_expert_idx = ptx.remU(b, e, e_per_rank);
                const expert_status = ptx.ldGlobalU64(b, self.workspace.expertSendCountPtr(e));
                self.sym.stU64(self.sym.map(self.workspace.expertRecvCountPtr(self.sym.rank_idx, dst_local_expert_idx), dst_rank_idx), expert_status.bitAnd(ptx.constU64(b, 0xffffffff)));
                _ = ptx.atomicAddSys(b, self.sym.map(self.workspace.expertRecvCountSumPtr(dst_local_expert_idx), dst_rank_idx), expert_status);
                _ = loop.end({});
            }
            sm0.yieldThen(.{});
        }
        ptx.syncAligned(b, @as(i32, @intCast(num_dispatch_threads)), @as(i32, dispatch_barrier_idx));

        // Barrier before pulling (after the grid sync above, there is no more writes by other
        // SMs (except 0); after the NVLink barrier, there is a grid sync)
        device.nvlinkBarrier(b, self.workspace, self.sym, k.num_sms, num_dispatch_threads, dispatch_grid_sync_index, self.sm_idx, self.thread_idx, scope, false, true, cfg.timeout_cycles);

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx.syncUnaligned(b, @as(i32, @intCast(num_dispatch_threads + c_.num_epilogue_threads)), @as(i32, dispatch_with_epilogue_barrier_idx));

        // Pull token data and SF from remote ranks into local L1 buffer
        const counts = self.pullLoop();

        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx.syncUnaligned(b, @as(i32, @intCast(num_dispatch_threads + c_.num_epilogue_threads)), @as(i32, dispatch_with_epilogue_barrier_idx));
        self.cleanWorkspace(counts, ndw);

        // Wait for all ranks to finish cleaning (before the NVLink barrier, there is a grid
        // sync; at the end of kernel does not need to sync)
        device.nvlinkBarrier(b, self.workspace, self.sym, k.num_sms, num_dispatch_threads, dispatch_grid_sync_index, self.sm_idx, self.thread_idx, scope, true, false, cfg.timeout_cycles);
    }

    /// The dispatch pull loop. Each warp walks the pool tokens `sm_idx * kNumDispatchWarps +
    /// warp_idx + n * kNumGlobalWarps` of the local experts: finds the source rank by
    /// round-robin min-peeling over per-rank counts, waits for the ring slot, pulls the token
    /// (1-D TMA loads from the source rank, 1-D TMA stores into the L1 ring), copies its SF
    /// (UTCCP-transposed), weight and source metadata, and releases it to the TMA-A warp
    /// through `l1_full_count`. Returns the dispatch warps' scheduler counts (for the cleanup).
    fn pullLoop(self: *Kernel) scheduler.Counts {
        const b = self.b;
        const k = self.k;
        const c_ = self.c();
        const ndw = numDispatchWarps(self.cfg);
        const R = k.shape.num_ranks;
        const E: i32 = @intCast(k.numExpertsPerRank());
        const BM: i32 = @intCast(c_.block_m);
        const pull_buffer = self.dispatchSendBuffer(self.warp_idx);
        const pull_mbarrier = self.dispatchBarrier(self.warp_idx);
        const num_ranks_per_lane = layout.ceilDiv(R, 32);

        // Wait token data arrival
        const counts = self.sched.fetchExpertRecvCount();

        const zero_counts = b.arena.allocator().alloc(V, num_ranks_per_lane) catch @panic("OOM");
        for (zero_counts) |*z| z.* = b.cst(.i32, 0);
        const State = struct {
            token_idx: V,
            current_expert_idx: V,
            expert_start_idx: V,
            expert_end_idx: V,
            expert_pool_block_offset: V,
            stored_rank_count: []V,
            pull_mbarrier_phase: V,
        };
        const Fwd = struct { state: State, old_expert_idx: V };
        // for (token_idx = ...; ; token_idx += kNumGlobalWarps) { advance expert; if (done) break; ... }:
        // the before region advances the expert cursor and tests it; the after region is the rest.
        var loop = flow.While(State, Fwd).open(b, .{
            .token_idx = self.sm_idx.mul(@as(i32, @intCast(ndw))).add(self.warp_idx),
            .current_expert_idx = b.cst(.i32, -1),
            .expert_start_idx = b.cst(.i32, 0),
            .expert_end_idx = b.cst(.i32, 0),
            .expert_pool_block_offset = b.cst(.i32, 0),
            .stored_rank_count = zero_counts,
            .pull_mbarrier_phase = b.cst(.i32, 0),
        });
        var s = loop.state;
        {
            // Advance expert until within the range:
            //   while (token_idx >= expert_end_idx) {
            //       if (++ current_expert_idx >= kNumExpertsPerRank) break;
            //       expert_pool_block_offset += ceil_div(end - start, BLOCK_M); start = end; end += get_num_tokens(current);
            //   }
            // The before region does the test and the increment, the after region the rest.
            const Cursor = struct { cur: V, start: V, end: V, offset: V };
            var adv = flow.While(Cursor, Cursor).open(b, .{ .cur = s.current_expert_idx, .start = s.expert_start_idx, .end = s.expert_end_idx, .offset = s.expert_pool_block_offset });
            const a = adv.state;
            const go = ptx.geU(b, s.token_idx, a.end);
            const cur1 = b.select(go, a.cur.add(1), a.cur);
            const in = adv.cond(go.bitAnd(cur1.lt(E)), .{ .cur = cur1, .start = a.start, .end = a.end, .offset = a.offset });
            const cursor = adv.end(.{
                .cur = in.cur,
                .start = in.end,
                .end = in.end.add(self.sched.getNumTokens(counts, in.cur)),
                .offset = in.offset.add(ptx.ceilDivU(b, in.end.sub(in.start), BM)),
            });
            const old_expert_idx = s.current_expert_idx;
            s.current_expert_idx = cursor.cur;
            s.expert_start_idx = cursor.start;
            s.expert_end_idx = cursor.end;
            s.expert_pool_block_offset = cursor.offset;
            // Finish all tokens
            const fwd = loop.cond(cursor.cur.lt(E), .{ .state = s, .old_expert_idx = old_expert_idx });
            s = fwd.state;
            const current_expert_idx = s.current_expert_idx;

            // Load per-rank counts when expert changes
            {
                var reload = flow.IfElse([]V).open(b, fwd.old_expert_idx.ne(current_expert_idx));
                const fresh = b.arena.allocator().alloc(V, num_ranks_per_lane) catch @panic("OOM");
                for (fresh, 0..) |*f, idx| {
                    const j = self.lane_idx.add(@as(i32, @intCast(idx * 32)));
                    // TODO (DeepGEMM): this is not coalesced
                    var in_range = flow.IfElse(V).open(b, ptx.ltU(b, j, @as(i32, @intCast(R))));
                    in_range.orElse(ptx.lo32(b, ptx.ldGlobalU64(b, self.workspace.expertRecvCountPtr(j, current_expert_idx))));
                    f.* = in_range.end(b.cst(.i32, 0));
                }
                reload.orElse(fresh);
                s.stored_rank_count = reload.end(s.stored_rank_count);
            }

            // Round-robin rank selection via iterative min-peeling:
            //   while (true) { count actives, min; if (slot_idx < length * actives) { pick; break; } peel; }
            // The before region computes the round and tests the hit, the after region peels;
            // the pick runs after the loop on the forwarded values.
            const token_idx_in_expert = s.token_idx.sub(s.expert_start_idx);
            const Round = struct { remaining: []V, offset: V, slot_idx: V };
            const RoundFwd = struct { round: Round, num_active_ranks: V, length: V, num_round_tokens: V };
            var rr = flow.While(Round, RoundFwd).open(b, .{ .remaining = s.stored_rank_count, .offset = b.cst(.i32, 0), .slot_idx = token_idx_in_expert });
            const r = rr.state;
            var num_actives_in_lane = b.cst(.i32, 0);
            var min_in_lane = ptx.constU32(b, 0xffffffff);
            for (r.remaining) |rem| {
                const positive = rem.ne(0);
                num_actives_in_lane = num_actives_in_lane.add(positive.to(.i32));
                min_in_lane = b.select(positive, ptx.minU(b, min_in_lane, rem), min_in_lane);
            }
            const num_active_ranks = ptx.reduceAddSync(b, num_actives_in_lane);
            const length = ptx.reduceMinSync(b, min_in_lane);
            const num_round_tokens = length.mul(num_active_ranks);
            // Hit in the current round
            const hit = ptx.ltU(b, r.slot_idx, num_round_tokens);
            const peel = rr.cond(hit.bitXor(b.cst(.i1, true)), .{ .round = r, .num_active_ranks = num_active_ranks, .length = length, .num_round_tokens = num_round_tokens });
            // Move into the next round
            const peeled = b.arena.allocator().alloc(V, num_ranks_per_lane) catch @panic("OOM");
            for (peeled, peel.round.remaining) |*p, rem| p.* = rem.sub(ptx.minU(b, rem, peel.length));
            const picked = rr.end(.{ .remaining = peeled, .offset = peel.round.offset.add(peel.length), .slot_idx = peel.round.slot_idx.sub(peel.num_round_tokens) });

            const slot_idx_in_round = ptx.remU(b, picked.round.slot_idx, picked.num_active_ranks);
            var num_seen_ranks = b.cst(.i32, 0);
            var current_rank_in_expert_idx = b.cst(.i32, 0);
            for (picked.round.remaining, 0..) |rem, idx| {
                const mask = ptx.ballotSync(b, rem.ne(0));
                const num_active_lanes = ptx.popc(b, mask);
                const in_group = ptx.geU(b, slot_idx_in_round, num_seen_ranks).bitAnd(ptx.ltU(b, slot_idx_in_round, num_seen_ranks.add(num_active_lanes)));
                const lane_of_rank = ptx.fns(b, mask, @as(i32, 0), slot_idx_in_round.sub(num_seen_ranks).add(1));
                current_rank_in_expert_idx = b.select(in_group, lane_of_rank.add(@as(i32, @intCast(idx * 32))), current_rank_in_expert_idx);
                num_seen_ranks = num_seen_ranks.add(num_active_lanes);
            }
            const token_idx_in_rank = picked.round.offset.add(ptx.divU(b, picked.round.slot_idx, picked.num_active_ranks));

            s.pull_mbarrier_phase = self.pullToken(s, counts, current_rank_in_expert_idx, token_idx_in_rank, token_idx_in_expert, pull_buffer, pull_mbarrier);

            s.token_idx = s.token_idx.add(@as(i32, @intCast(k.num_sms * ndw)));
            _ = loop.end(s);
        }
        return counts;
    }

    /// The body of the pull loop once the source rank is known. Returns the elected lane's
    /// pull barrier phase.
    fn pullToken(self: *Kernel, s: anytype, counts: scheduler.Counts, current_rank_in_expert_idx: V, token_idx_in_rank: V, token_idx_in_expert: V, pull_buffer: V, pull_mbarrier: V) V {
        _ = counts;
        const b = self.b;
        const k = self.k;
        const c_ = self.c();
        const topk: i32 = @intCast(k.shape.num_topk);
        const BM: i32 = @intCast(c_.block_m);
        const num_ring_blocks: i32 = @intCast(k.numRingBlocks());
        const bytes_per_pull = c_.num_bytes_per_pull;
        const current_expert_idx = s.current_expert_idx;

        // Read source token-topk index (written by remote dispatch via NVLink)
        const src_token_topk_idx = ptx.ldGlobal(b, 1, self.workspace.srcTokenTopkIdxPtr(current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank))[0];
        const src_token_idx = ptx.divU(b, src_token_topk_idx, topk);
        const src_topk_idx = ptx.remU(b, src_token_topk_idx, topk);

        // Hidden bytes are divided into chunks
        const num_chunks = k.shape.hidden / bytes_per_pull;

        // TMA load token from remote rank and store into local
        const pool_token_idx = s.expert_pool_block_offset.mul(BM).add(token_idx_in_expert);
        const pool_block_idx = ptx.divU(b, pool_token_idx, BM);

        // Wait for ring buffer slot to be available (previous consumer must have finished all N blocks)
        const l1_empty_count_target = ptx.divU(b, pool_block_idx, num_ring_blocks).mul(@as(i32, @intCast(k.numL1BlockNs())));
        {
            var wait = b.openIf(l1_empty_count_target.ne(0));
            flow.spin(b, struct {
                ptr: V,
                target: V,
                pub fn pending(st: @This(), bb: *B) V {
                    return ptx.ltU(bb, ptx.ldAcq(bb, st.ptr), st.target);
                }
            }{ .ptr = self.workspace.l1EmptyCountPtr(ptx.remU(b, pool_block_idx, num_ring_blocks)), .target = l1_empty_count_target });
            wait.yieldThen(.{});
        }

        const src_base_ptr = self.sym.map(self.buffer.input_token.dataPtr(src_token_idx), current_rank_in_expert_idx);
        const dst_base_ptr = self.buffer.l1_token.dataPtr(ptx.remU(b, pool_token_idx, @as(i32, @intCast(c_.num_ring_tokens))));

        // `issue_and_wait_pull_store(i)`: wait chunk `i` in shared memory, store it to the ring
        const PullStore = struct {
            fn run(bb: *B, phase: V, bar: V, dst: V, src: V, chunk: u32, bytes: u32) V {
                const next = ptx.mbarrierWaitAndFlipPhase(bb, bar, phase);
                ptx.tmaStore1d(bb, dst.add(ptx.constU64(bb, @as(u64, chunk) * bytes)), src, @as(i32, @intCast(bytes)), ptx.CacheHint.evict_normal);
                ptx.tmaStoreArrive(bb);
                ptx.tmaStoreWait(bb, 0);
                return next;
            }
        };
        var phase = s.pull_mbarrier_phase;
        {
            var elected = flow.IfElse(V).open(b, ptx.electOneSync(b));
            var p = phase;
            for (0..num_chunks) |chunk_i| {
                const chunk: u32 = @intCast(chunk_i);
                ptx.tmaLoad1d(b, pull_buffer, src_base_ptr.add(ptx.constU64(b, @as(u64, chunk) * bytes_per_pull)), pull_mbarrier, @as(i32, @intCast(bytes_per_pull)), ptx.CacheHint.evict_first);
                ptx.mbarrierArriveAndSetTx(b, pull_mbarrier, @as(i32, @intCast(bytes_per_pull)));
                if (chunk != num_chunks - 1) p = PullStore.run(b, p, pull_mbarrier, dst_base_ptr, pull_buffer, chunk, bytes_per_pull);
            }
            elected.orElse(p);
            phase = elected.end(phase);
        }
        ptx.syncWarp(b);

        // Load the weight first, so that its remote latency overlaps with the SF copy below
        const weight = self.sym.ld(1, self.sym.map(device.index(b, self.buffer.input_topk_weights.basePtr(), src_token_topk_idx, 4), current_rank_in_expert_idx))[0];

        // Load and store SF (overlaps with last chunk's TMA load from remote)
        const num_sf_uint32 = k.shape.hidden / 128;
        const remote_sf_ptr = self.sym.map(self.buffer.input_sf.dataPtr(src_token_idx), current_rank_in_expert_idx);
        const local_sf_ptr = self.buffer.l1_sf.basePtr();
        const ring_block_idx = ptx.remU(b, pool_block_idx, num_ring_blocks);
        const token_idx_in_block = ptx.remU(b, token_idx_in_expert, BM);
        const sf_ring_token_idx = ring_block_idx.mul(@as(i32, @intCast(c_.sf_block_m))).add(self.transformSfTokenIdx(token_idx_in_block));
        for (0..layout.ceilDiv(num_sf_uint32, 32)) |si| {
            const j = self.lane_idx.add(@as(i32, @intCast(si * 32)));
            var in_range = b.openIf(ptx.ltU(b, j, @as(i32, @intCast(num_sf_uint32))));
            const value = self.sym.ld(1, device.index(b, remote_sf_ptr, j, 4))[0];
            ptx.stGlobal(b, 1, device.index(b, local_sf_ptr, j.mul(@as(i32, @intCast(c_.num_sf_ring_tokens))).add(sf_ring_token_idx), 4), .{value});
            in_range.yieldThen(.{});
        }
        ptx.syncWarp(b);

        // Store weights and metadata
        {
            var elected = flow.IfElse(V).open(b, ptx.electOneSync(b));
            ptx.stGlobal(b, 1, self.buffer.l1_topk_weights.dataPtr(ptx.remU(b, pool_token_idx, @as(i32, @intCast(c_.num_ring_tokens)))), .{weight});

            // Write source metadata for combine write-back (logical pool token)
            const meta = self.workspace.tokenSrcMetadataPtr(pool_token_idx);
            ptx.stGlobal(b, 1, meta, .{current_rank_in_expert_idx});
            ptx.stGlobal(b, 1, meta.add(4), .{src_token_idx});
            ptx.stGlobal(b, 1, meta.add(8), .{src_topk_idx});

            // Complete last chunk's store
            const next = PullStore.run(b, phase, pull_mbarrier, dst_base_ptr, pull_buffer, num_chunks - 1, bytes_per_pull);
            const is_last_token = s.token_idx.eq(s.expert_end_idx.sub(1));
            const count = b.select(is_last_token, b.cst(.i32, BM).sub(ptx.remU(b, token_idx_in_expert, BM)), b.cst(.i32, 1));
            ptx.redAddRel(b, self.workspace.l1FullCountPtr(ring_block_idx), count);
            elected.orElse(next);
            phase = elected.end(phase);
        }
        ptx.syncWarp(b);
        return phase;
    }

    /// `transform_sf_token_idx(token_idx_in_expert)`: UTCCP 4x32 transpose index mapping within
    /// each 128-element group.
    fn transformSfTokenIdx(self: *const Kernel, token_idx_in_expert: V) V {
        const b = self.b;
        const BM: i32 = @intCast(self.c().block_m);
        const idx = ptx.remU(b, token_idx_in_expert, BM);
        return ptx.divU(b, token_idx_in_expert, BM).mul(@as(i32, @intCast(self.c().sf_block_m)))
            .add(idx.bitAnd(ptx.constU32(b, ~@as(u32, 127))))
            .add(idx.bitAnd(31).mul(4))
            .add(idx.shrLogical(5).bitAnd(3));
    }

    /// Clean workspace for the next usage, and also do cumulative stats (SM 0: expert send
    /// counts and task counters; other SMs: per-expert counts and the ring signals of the
    /// expert's pool blocks).
    fn cleanWorkspace(self: *Kernel, counts: scheduler.Counts, ndw: u32) void {
        const b = self.b;
        const k = self.k;
        const num_dispatch_threads: i32 = @intCast(self.c().num_dispatch_threads);
        const ws = self.workspace;
        const num_ring_blocks: i32 = @intCast(k.numRingBlocks());
        _ = ndw;

        var sm0 = flow.IfElse(void).open(b, self.sm_idx.eq(0));
        {
            // SM 0: clear expert send count and schedule task counters
            {
                var loop = flow.forRange(b, self.thread_idx, @as(i32, @intCast(k.shape.num_experts)), num_dispatch_threads);
                ptx.stGlobalU64(b, ws.expertSendCountPtr(loop.iv), ptx.constU64(b, 0));
                _ = loop.end({});
            }
            {
                var warp0 = b.openIf(self.warp_idx.eq(0));
                var elected = b.openIf(ptx.electOneSync(b));
                for (0..layout.device_locality_domains) |d| {
                    const off: u64 = 4 * d;
                    const zero = b.cst(.i32, 0);
                    ptx.stGlobal(b, 1, device.offsetBy(b, ws.l2TaskCountPtr(), off), .{zero});
                    ptx.stGlobal(b, 1, device.offsetBy(b, ws.l1TaskCountPtr(), off), .{zero});
                    ptx.stGlobal(b, 1, device.offsetBy(b, ws.sharedL2TaskCountPtr(), off), .{zero});
                    ptx.stGlobal(b, 1, device.offsetBy(b, ws.sharedL1TaskCountPtr(), off), .{zero});
                }
                elected.yieldThen(.{});
                warp0.yieldThen(.{});
            }
            ptx.syncWarp(b);
            {
                var loop = flow.forRange(b, self.thread_idx, @as(i32, @intCast(ws.numSharedL2PoolBlocks())), num_dispatch_threads);
                ptx.stGlobal(b, 1, ws.sharedL2FullCountPtr(loop.iv), .{b.cst(.i32, 0)});
                _ = loop.end({});
            }
            ptx.syncWarp(b);
            sm0.orElse({});
        }
        {
            // Other SMs: clean blocks
            var loop = flow.forRange(b, self.sm_idx.sub(1), @as(i32, @intCast(k.numExpertsPerRank())), @as(i32, @intCast(k.num_sms - 1)));
            const e = loop.iv;
            // Read expert token count before clearing
            const num_recv_tokens = ptx.lo32(b, ptx.ldGlobalU64(b, ws.expertRecvCountSumPtr(e)));
            const num_recv_m_blocks = ptx.ceilDivU(b, num_recv_tokens, @as(i32, @intCast(self.c().block_m)));

            // Compute expert pool block offset
            const expert_pool_block_offset = self.sched.getPoolBlockOffset(counts.stored_num_tokens_per_expert, e);

            // Wait read count ready
            ptx.syncAligned(b, num_dispatch_threads, @as(i32, dispatch_barrier_idx));

            // Clean expert token count, and add cumulative results
            {
                var warp0 = b.openIf(self.warp_idx.eq(0));
                ptx.stGlobalU64(b, ws.expertRecvCountSumPtr(e), ptx.constU64(b, 0));
                warp0.yieldThen(.{});
            }
            {
                var warp1 = b.openIf(self.warp_idx.eq(1));
                const stats = self.cumulative_local_expert_recv_stats;
                var add = b.openIf(ptx.electOneSync(b).bitAnd(stats.ne(0)));
                ptx.redAddS32(b, device.index(b, stats, e, 4), num_recv_tokens);
                add.yieldThen(.{});
                ptx.syncWarp(b);
                warp1.yieldThen(.{});
            }

            // Clean per-rank token count
            {
                var ranks = flow.forRange(b, self.thread_idx, @as(i32, @intCast(k.shape.num_ranks)), num_dispatch_threads);
                ptx.stGlobalU64(b, ws.expertRecvCountPtr(ranks.iv, e), ptx.constU64(b, 0));
                _ = ranks.end({});
            }
            ptx.syncWarp(b);

            // Clean L1 and L2 full stuffs and ring buffer counts
            {
                var blocks = flow.forRange(b, self.thread_idx, num_recv_m_blocks, num_dispatch_threads);
                const rb = ptx.remU(b, expert_pool_block_offset.add(blocks.iv), num_ring_blocks);
                const zero = b.cst(.i32, 0);
                ptx.stGlobal(b, 1, ws.l1FullCountPtr(rb), .{zero});
                ptx.stGlobal(b, 1, ws.l1EmptyCountPtr(rb), .{zero});
                ptx.stGlobalU64(b, ws.l2FullMaskPtr(rb), ptx.constU64(b, 0));
                ptx.stGlobal(b, 1, ws.l2EmptyCountPtr(rb), .{zero});
                _ = blocks.end({});
            }
            ptx.syncWarp(b);
            _ = loop.end({});
        }
        sm0.end({});
    }

    // ==================== GEMM TMA load warp for tokens with SFA ====================

    /// The scheduler consumer loop shared by the TMA, MMA and epilogue roles:
    /// `while (scheduler.get_next_task(task_info)) { ... }` as an `scf.while` whose before
    /// region waits for and reads the next task.
    fn TaskLoop(comptime S: type) type {
        return struct {
            const State = struct { pipe: scheduler.Pipe, user: S };
            const Fwd = struct { state: State, task: scheduler.TaskInfo };
            loop: flow.While(State, Fwd),
            /// The task and the role's state inside the loop body.
            task: scheduler.TaskInfo,
            pipe: scheduler.Pipe,
            state: S,

            fn open(kern: *const Kernel, initial: S) @This() {
                var loop = flow.While(State, Fwd).open(kern.b, .{ .pipe = scheduler.Pipe.init(kern.b), .user = initial });
                const before = loop.state;
                const next = kern.sched.getNextTask(before.pipe);
                const fwd = loop.cond(next.valid, .{ .state = .{ .pipe = next.pipe, .user = before.user }, .task = next.task });
                return .{ .loop = loop, .task = fwd.task, .pipe = fwd.state.pipe, .state = fwd.state.user };
            }

            /// Ends the body; returns the role's state after the loop.
            fn end(self: *@This(), next: S) S {
                return self.loop.end(.{ .pipe = self.pipe, .user = next }).state.user;
            }
        };
    }

    /// `task_info.block_phase == Linear1 ? m.l1 : == Linear2 ? m.l2 : == SharedLinear1 ? m.sl1 : m.sl2`.
    fn selectByPhase(self: *const Kernel, task: scheduler.TaskInfo, l1: V, l2: V, sl1: V, sl2: V) V {
        const b = self.b;
        return b.select(task.isPhase(.linear1), l1, b.select(task.isPhase(.linear2), l2, b.select(task.isPhase(.shared_linear1), sl1, sl2)));
    }

    fn tmaARole(self: *Kernel) void {
        const b = self.b;
        const k = self.k;
        const c_ = self.c();
        const has_shared = k.hasShared();
        const num_ring_blocks: i32 = @intCast(k.numRingBlocks());

        // Adjust registers
        ptx.warpgroupRegDealloc(b, k.numNonEpilogueRegisters());

        var tasks = TaskLoop(StagePhase).open(self, .{ .stage = b.cst(.i32, 0), .phase = b.cst(.i32, 0) });
        {
            const t = tasks.task;
            const tensor_map_a = self.selectByPhase(t, self.desc("l1_acts"), self.desc("l2_acts"), self.desc("shared_l1_acts"), self.desc("shared_l2_acts"));
            const tensor_map_sfa = self.selectByPhase(t, self.desc("l1_acts_sf"), self.desc("l2_acts_sf"), self.desc("shared_l1_acts_sf"), self.desc("shared_l2_acts_sf"));
            const num_k_blocks = ptx.ceilDivU(b, t.shape_k, @as(i32, @intCast(c_.block_k)));

            // Compute pool block offset for this expert
            const pool_block_idx = t.pool_block_idx;
            const ring_block_idx = ptx.remU(b, pool_block_idx, num_ring_blocks);
            const block_idx = b.select(t.isShared(b, has_shared), pool_block_idx, ring_block_idx);

            // Wait the entire token arrival
            {
                var l1 = b.openIf(t.isPhase(.linear1));
                const num_expected_tokens = ptx.divU(b, pool_block_idx, num_ring_blocks).add(1).mul(@as(i32, @intCast(c_.block_m)));
                flow.spin(b, NotEqual{ .ptr = self.workspace.l1FullCountPtr(block_idx), .expected = num_expected_tokens });
                l1.yieldThen(.{});
            }
            {
                var sl2 = b.openIf(t.isPhase(.shared_linear2));
                const num_expected_blocks = self.ci(k.sharedL2ExpectedBlocks());
                flow.spin(b, NotEqual{ .ptr = self.workspace.sharedL2FullCountPtr(block_idx), .expected = num_expected_blocks });
                sl2.yieldThen(.{});
            }

            const dep = scheduler.L2KBlockDependency.init(b, self.sp, self.workspace.l2FullMaskPtr(ring_block_idx), ptx.divU(b, pool_block_idx, num_ring_blocks));
            const KState = struct { sp: StagePhase, pending_mask: V };
            var kloop = flow.For(KState).open(b, @as(i32, 0), num_k_blocks, @as(i32, 1), .{ .sp = tasks.state, .pending_mask = dep.pending_mask });
            {
                const k_block_idx = kloop.iv;
                const ks = kloop.state;
                var pending = ks.pending_mask;
                {
                    var l2 = flow.IfElse(V).open(b, t.isPhase(.linear2));
                    var d = dep;
                    d.pending_mask = pending;
                    l2.orElse(d.wait(b, self.sp, k_block_idx).pending_mask);
                    pending = l2.end(pending);
                }

                // Wait consumer release
                ptx.mbarrierWait(b, self.emptyBarrier(ks.sp.stage), ks.sp.phase.bitXor(1));

                // Compute token offsets from block index
                var m_idx = block_idx.mul(@as(i32, @intCast(c_.block_m)));
                const k_idx = k_block_idx.mul(@as(i32, @intCast(c_.block_k)));
                const sfa_m_idx = block_idx.mul(@as(i32, @intCast(c_.sf_block_m)));
                const sfa_k_idx = k_block_idx.mul(@as(i32, @intCast(c_.block_k / 128)));

                // Add 2 CTA offsets for non-leader CTA
                m_idx = b.select(self.is_leader_cta, m_idx, m_idx.add(ptx.divU(b, t.getUmmaAlignedValidM(b), @as(i32, 2))));

                // TMA copy tokens and SFA, then arrive at full barrier
                {
                    var elected = b.openIf(ptx.electOneSync(b));
                    const full = self.fullBarrier(ks.sp.stage);
                    ptx.tmaCopy2Sm(b, c_.block_k, k.loadBlockM(), Kc.swizzle_a_mode, 1, tensor_map_a, full, self.smemA(ks.sp.stage), k_idx, &.{m_idx});
                    ptx.tmaCopy2Sm(b, c_.sf_block_m, 1, 0, 4, tensor_map_sfa, full, self.smemSfa(ks.sp.stage), sfa_m_idx, &.{sfa_k_idx});
                    self.arriveFull(full, k.expectTxA());
                    elected.yieldThen(.{});
                }
                ptx.syncWarp(b);
                const after = kloop.end(.{ .sp = self.advancePipeline(ks.sp), .pending_mask = pending });
                _ = tasks.end(after.sp);
            }
        }
    }

    /// `if (is_leader_cta) full.arrive_and_expect_tx(bytes); else full.arrive(0u);`
    fn arriveFull(self: *const Kernel, full: V, bytes: u32) void {
        const b = self.b;
        var leader = b.openIf(self.is_leader_cta);
        ptx.mbarrierArriveAndExpectTx(b, full, @as(i32, @intCast(bytes)));
        leader.yieldThen(.{});
        var peer = b.openIf(self.is_leader_cta.bitXor(b.cst(.i1, true)));
        ptx.mbarrierArriveCluster(b, full, b.cst(.i32, 0));
        peer.yieldThen(.{});
    }

    // ==================== GEMM TMA load warp for weights with SF ====================

    fn tmaBRole(self: *Kernel) void {
        const b = self.b;
        const k = self.k;
        const c_ = self.c();
        const has_shared = k.hasShared();

        // Adjust registers
        ptx.warpgroupRegDealloc(b, k.numNonEpilogueRegisters());

        var tasks = TaskLoop(StagePhase).open(self, .{ .stage = b.cst(.i32, 0), .phase = b.cst(.i32, 0) });
        {
            const t = tasks.task;
            const tensor_map_b = self.selectByPhase(t, self.desc("l1_weights"), self.desc("l2_weights"), self.desc("shared_l1_weights"), self.desc("shared_l2_weights"));
            const tensor_map_sfb = self.selectByPhase(t, self.desc("l1_weights_sf"), self.desc("l2_weights_sf"), self.desc("shared_l1_weights_sf"), self.desc("shared_l2_weights_sf"));

            const shape_k = t.shape_k;
            const shape_n = t.shape_n;
            const shape_sfb_k = ptx.ceilDivU(b, shape_k, @as(i32, @intCast(Kc.gran_k * 4)));
            const n_block_idx = t.n_cluster_idx.mul(2).add(b.select(self.is_leader_cta, b.cst(.i32, 0), b.cst(.i32, 1)));
            const num_k_blocks = ptx.ceilDivU(b, shape_k, @as(i32, @intCast(c_.block_k)));

            const is_shared = t.isShared(b, has_shared);
            const half_n = ptx.divU(b, shape_n, @as(i32, 2));
            const n_idx = n_block_idx.mul(@as(i32, @intCast(c_.block_n)));
            const half_idx = ptx.geU(b, n_idx, half_n).to(.i32);
            const half_n_idx = n_idx.sub(half_idx.mul(half_n));
            const expert_idx = b.select(is_shared, b.cst(.i32, 0), t.local_expert_idx);

            var kloop = flow.For(StagePhase).open(b, @as(i32, 0), num_k_blocks, @as(i32, 1), tasks.state);
            {
                const k_block_idx = kloop.iv;
                const sp = kloop.state;
                // Wait consumer release
                ptx.mbarrierWait(b, self.emptyBarrier(sp.stage), sp.phase.bitXor(1));

                // Compute weight offset
                const k_idx = k_block_idx.mul(@as(i32, @intCast(c_.block_k)));
                const sfb_n_idx = n_block_idx.mul(@as(i32, @intCast(c_.block_n)));
                const sfb_k_step = k_block_idx.mul(@as(i32, @intCast(c_.block_k / 128)));
                const sfb_k_idx = b.select(is_shared, sfb_k_step, t.local_expert_idx.mul(shape_sfb_k).add(sfb_k_step));

                // TMA copy weights with SF
                {
                    var elected = b.openIf(ptx.electOneSync(b));
                    const full = self.fullBarrier(sp.stage);
                    const smem_b = self.smemB(sp.stage);
                    const smem_sfb = self.smemSfb(sp.stage);
                    {
                        var shared = b.openIf(is_shared);
                        ptx.tmaCopy2Sm(b, c_.block_k, k.loadBlockN(), Kc.swizzle_b_mode, 1, tensor_map_b, full, smem_b, k_idx, &.{ half_n_idx, half_idx, expert_idx });
                        ptx.tmaCopy2Sm(b, c_.block_n, 1, 0, 4, tensor_map_sfb, full, smem_sfb, sfb_n_idx, &.{sfb_k_idx});
                        self.arriveFull(full, k.expectTxSharedB());
                        shared.yieldThen(.{});
                    }
                    {
                        var routed = b.openIf(is_shared.bitXor(b.cst(.i1, true)));
                        ptx.tmaCopy2Sm(b, c_.block_k, k.loadBlockN(), Kc.swizzle_b_mode, 1, tensor_map_b, full, smem_b, k_idx, &.{ half_n_idx, half_idx, expert_idx });
                        ptx.tmaCopy2Sm(b, c_.block_n, 1, 0, 4, tensor_map_sfb, full, smem_sfb, sfb_n_idx, &.{sfb_k_idx});
                        self.arriveFull(full, k.expectTxRoutedB());
                        routed.yieldThen(.{});
                    }
                    elected.yieldThen(.{});
                }
                ptx.syncWarp(b);
                const after = kloop.end(self.advancePipeline(sp));
                _ = tasks.end(after);
            }
        }
    }

    // ==================== GEMM MMA issue warp ====================

    fn mmaRole(self: *Kernel) void {
        const b = self.b;
        const k = self.k;
        const c_ = self.c();
        const has_shared = k.hasShared();

        // Adjust registers
        ptx.warpgroupRegDealloc(b, k.numNonEpilogueRegisters());

        // GEMM MMA issue warp (only the leader CTA will run)
        var leader = b.openIf(self.is_leader_cta);
        {
            // Make instruction descriptor with block scaling (always swap A/B)
            const routed_instr_desc = ptx.constU32(b, k.routedInstrDesc().bits());
            const shared_instr_desc = ptx.constU32(b, k.sharedInstrDesc().bits());
            const sf_desc = ptx.makeSfDesc(b, b.cst(.i32, 0));

            const a_desc = ptx.makeUmmaDescKSw128(b, self.smemA(b.cst(.i32, 0)));
            const b_desc = ptx.makeUmmaDescKSw128(b, self.smemB(b.cst(.i32, 0)));
            const a_desc_lo = ptx.descLo(b, a_desc);
            const b_desc_lo = ptx.descLo(b, b_desc);

            // Persistently schedule over blocks
            const RoleState = struct { sp: StagePhase, current_iter_idx: V };
            var final: RoleState = undefined;
            var tasks = TaskLoop(RoleState).open(self, .{ .sp = .{ .stage = b.cst(.i32, 0), .phase = b.cst(.i32, 0) }, .current_iter_idx = b.cst(.i32, 0) });
            {
                const t = tasks.task;
                const num_k_blocks = ptx.divU(b, t.shape_k, @as(i32, @intCast(c_.block_k)));

                // Dynamic update of UMMA N based on effective M
                const instr_desc = ptx.updateInstrDescWithUmmaN(b, b.select(t.isShared(b, has_shared), shared_instr_desc, routed_instr_desc), t.getUmmaAlignedValidM(b));

                // Wait tensor memory empty barrier arrival
                const iter = tasks.state.current_iter_idx;
                const accum_stage_idx = ptx.remU(b, iter, @as(i32, Kc.num_epilogue_stages));
                const accum_phase = ptx.divU(b, iter, @as(i32, Kc.num_epilogue_stages)).bitAnd(1);
                ptx.mbarrierWait(b, self.tmemEmptyBarrier(accum_stage_idx), accum_phase.bitXor(1));
                ptx.tcgen05AfterThreadSync(b);

                // Launch MMAs
                var kloop = flow.For(StagePhase).open(b, @as(i32, 0), num_k_blocks, @as(i32, 1), tasks.state.sp);
                {
                    const k_block_idx = kloop.iv;
                    const sp = kloop.state;
                    // Wait TMA load completion
                    ptx.mbarrierWait(b, self.fullBarrier(sp.stage), sp.phase);
                    ptx.tcgen05AfterThreadSync(b);

                    const a_desc_base_lo = a_desc_lo.add(sp.stage.mul(@as(i32, @intCast(k.smemABytes() / 16))));
                    const b_desc_base_lo = b_desc_lo.add(sp.stage.mul(@as(i32, @intCast(k.smemBBytes() / 16))));
                    {
                        var elected = b.openIf(ptx.electOneSync(b));
                        for (0..c_.block_k / Kc.umma_block_k) |umma_k_block_idx| {
                            // UTCCP copy SFA and SFB to TMEM
                            for (0..c_.sf_block_m / Kc.num_utccp_aligned_elems) |si| {
                                const smem_ptr = self.smemSfa(sp.stage).add(@as(i32, @intCast(4 * (umma_k_block_idx * c_.sf_block_m + si * Kc.num_utccp_aligned_elems))));
                                ptx.utccp4x32dp128bit2cta(b, ptx.replaceSmemDescAddr(b, sf_desc, smem_ptr), self.ci(k.tmemStartColOfSfa() + si * 4));
                            }
                            for (0..c_.sf_block_n / Kc.num_utccp_aligned_elems) |si| {
                                const smem_ptr = self.smemSfb(sp.stage).add(@as(i32, @intCast(4 * (umma_k_block_idx * c_.sf_block_n + si * Kc.num_utccp_aligned_elems))));
                                ptx.utccp4x32dp128bit2cta(b, ptx.replaceSmemDescAddr(b, sf_desc, smem_ptr), self.ci(k.tmemStartColOfSfb() + si * 4));
                            }

                            // Issue UMMA
                            for (0..Kc.umma_block_k / Kc.umma_k) |kk| {
                                const kk32: u32 = @intCast(kk);
                                const runtime_instr_desc = ptx.makeRuntimeInstrDescWithSfId(b, instr_desc, kk32, kk32);
                                const a_lo = ptx.advanceUmmaDescLo(b, a_desc_base_lo, @as(i32, @intCast(umma_k_block_idx * Kc.umma_block_k * k.loadBlockM())), @as(i32, @intCast(kk * Kc.umma_k)));
                                const b_lo = ptx.advanceUmmaDescLo(b, b_desc_base_lo, @as(i32, @intCast(umma_k_block_idx * Kc.umma_block_k * k.loadBlockN())), @as(i32, @intCast(kk * Kc.umma_k)));
                                const accumulate = if (umma_k_block_idx > 0 or kk > 0) b.cst(.i32, 1) else k_block_idx.gt(0).to(.i32);
                                ptx.mmaMxf8f6f42x1SmSs(
                                    b,
                                    ptx.withDescLo(b, b_desc, b_lo),
                                    ptx.withDescLo(b, a_desc, a_lo),
                                    accum_stage_idx.mul(@as(i32, @intCast(k.ummaN()))),
                                    accumulate,
                                    runtime_instr_desc,
                                    self.ci(k.tmemStartColOfSfb()),
                                    self.ci(k.tmemStartColOfSfa()),
                                );
                            }
                        }
                        elected.yieldThen(.{});
                    }
                    ptx.syncWarp(b);

                    // Commit to the mbarrier object (`tcgen05.commit` implies
                    // `tcgen05.fence::before_thread_sync`)
                    self.emptyBarrierArrive(sp.stage, accum_stage_idx, k_block_idx.eq(num_k_blocks.sub(1)));
                    const after = kloop.end(self.advancePipeline(sp));
                    final = tasks.end(.{ .sp = after, .current_iter_idx = iter.add(1) });
                }
            }

            // To safely deconstruct barriers, we need another round of waits
            const final_iter = final.current_iter_idx;
            var any = b.openIf(final_iter.ne(0));
            const last = final_iter.sub(1);
            ptx.mbarrierWait(b, self.tmemEmptyBarrier(ptx.remU(b, last, @as(i32, Kc.num_epilogue_stages))), ptx.divU(b, last, @as(i32, Kc.num_epilogue_stages)).bitAnd(1));
            any.yieldThen(.{});
            leader.yieldThen(.{});
        }
    }

    /// `empty_barrier_arrive(do_tmem_full_arrive)`: commit the stage's MMAs to both CTAs'
    /// empty barriers, and to both CTAs' TMEM-full barriers after the last k block.
    fn emptyBarrierArrive(self: *const Kernel, stage: V, accum_stage_idx: V, do_tmem_full_arrive: V) void {
        const b = self.b;
        const cta_mask: u16 = (1 << 2) - 1;
        ptx.ummaArriveMulticast2x1Sm(b, self.emptyBarrier(stage), cta_mask);
        // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
        var full = b.openIf(do_tmem_full_arrive);
        ptx.ummaArriveMulticast2x1Sm(b, self.tmemFullBarrier(accum_stage_idx), cta_mask);
        full.yieldThen(.{});
        ptx.syncWarp(b);
    }

    // ==================== scheduler warp ====================

    fn schedulerRole(self: *Kernel) void {
        const b = self.b;
        // Adjust registers
        ptx.warpgroupRegDealloc(b, self.k.numNonEpilogueRegisters());

        // Do mainloop by the leader CTA
        var leader = b.openIf(self.is_leader_cta);
        self.sched.mainloop(self.num_tokens, self.sm_locality_domains);
        leader.yieldThen(.{});
    }

    // ==================== epilogue and combine warps ====================

    /// `shared_storage.smem_d.l1[wg][tma_stage_idx]`: `[STORE_BLOCK_M_L1][L1_OUT_BLOCK_N]` FP8.
    fn smemDL1(self: *const Kernel, wg: V, tma_stage_idx: u32) V {
        const tile = self.c().store_block_m_l1 * self.k.l1OutBlockN();
        return self.at(self.so.smem_d).add(wg.mul(@as(i32, @intCast(Kc.num_tma_store_stages * tile)))).add(@as(i32, @intCast(tma_stage_idx * tile)));
    }
    /// `shared_storage.smem_d.l2[wg]`: `[STORE_BLOCK_M_L2 * BLOCK_N]` BF16.
    fn smemDL2(self: *const Kernel, wg: V) V {
        return self.atIdx(self.so.smem_d, wg, self.c().store_block_m_l2 * self.c().block_n * 2);
    }
    /// `&shared_storage.amax_reduction[warp][idx]` (`float2`).
    fn amaxReduction(self: *const Kernel, warp: V, idx: V) V {
        return self.at(self.so.amax_reduction).add(warp.mul(@as(i32, @intCast(self.k.amaxReductionWarpBufferSize()))).add(idx).mul(8));
    }

    /// The epilogue warp's indices and the current task's offsets (`impl` 958-1007).
    const Epi = struct {
        epilogue_warp_idx: V,
        epilogue_wg_idx: V,
        warp_idx_in_wg: V,
        accum_stage_idx: V,
        is_shared: V,
        /// `ptx::exchange(task_info.valid_m, 0)`.
        valid_m: V,
        num_valid_wg_rows: V,
        pool_block_idx: V,
        ring_block_idx: V,
        /// `is_shared ? pool_block_idx : ring_block_idx`.
        block_idx: V,
        /// Ring-buffer offset for reusable data buffers.
        ring_m_idx: V,
        m_idx: V,
        /// Full-pool offset for non-ring metadata.
        pool_m_idx: V,
        n_block_idx: V,
        n_idx: V,
    };

    /// The store-block pipeline of one epilogue task (the lambdas `load_epi_block`,
    /// `load_epi_block_if_valid`, `prefetch_prologue`, `wait_and_prefetch_next`, `impl`
    /// 1012-1056): store blocks are loaded from TMEM into registers `kNumPrefetchStages` blocks
    /// ahead of use, and the TMEM stage is released right after the last valid block's loads
    /// complete. A block's values are `kNumAtomsPerStore * ATOM_M` words, atom `i` at
    /// `[8 i, 8 i + 8)` (`raw_values[.][i][0..8]`); the C++ ring of `kNumPrefetchStages + 1`
    /// buffers is the SSA values carried from the load to the use.
    const StorePipe = struct {
        kern: *const Kernel,
        accum_stage_idx: V,
        wg_idx: V,
        /// `kNumAtomsPerStore`.
        atoms: u32,
        /// `kNumStoreBlocks`.
        blocks: u32,
        /// `kNumPrefetchStages`.
        prefetch: u32,
        /// `num_valid_store_blocks`.
        num_valid: V,

        fn init(kern: *const Kernel, accum_stage_idx: V, wg_idx: V, store_block_m: u32, num_valid_wg_rows: V) StorePipe {
            const blocks = kern.k.wgBlockM() / store_block_m;
            return .{
                .kern = kern,
                .accum_stage_idx = accum_stage_idx,
                .wg_idx = wg_idx,
                .atoms = store_block_m / Kc.atom_m,
                .blocks = blocks,
                .prefetch = @min(kern.k.numEpiPrefetchStages(), blocks - 1),
                // Store blocks with at least one valid row (the last one may be partial);
                // `<= kNumStoreBlocks` by construction
                .num_valid = ptx.ceilDivU(kern.b, num_valid_wg_rows, @as(i32, @intCast(store_block_m))),
            };
        }

        /// `load_epi_block(s)`: per atom, `SM100_TMEM_LOAD_16dp256b1x` at lanes +0 and +16.
        fn load(self: StorePipe, s: u32) []V {
            const b = self.kern.b;
            const k = self.kern.k;
            const out = b.arena.allocator().alloc(V, self.atoms * Kc.atom_m) catch @panic("OOM");
            for (0..self.atoms) |i| {
                const tmem_addr = self.accum_stage_idx.mul(@as(i32, @intCast(k.ummaN())))
                    .add(self.wg_idx.mul(@as(i32, @intCast(k.wgBlockM()))))
                    .add(@as(i32, @intCast((s * self.atoms + @as(u32, @intCast(i))) * Kc.atom_m)));
                const lo = ptx.tmemLoad16dp256b1x(b, tmem_addr);
                const hi = ptx.tmemLoad16dp256b1x(b, tmem_addr.bitOr(@as(i32, 0x00100000)));
                @memcpy(out[i * 8 ..][0..4], &lo);
                @memcpy(out[i * 8 + 4 ..][0..4], &hi);
            }
            return out;
        }

        /// After block `s`'s loads: the last valid block is waited for right away and the TMEM
        /// stage is released, as nothing else will be read from it.
        fn releaseIfLast(self: StorePipe, s: u32) void {
            const b = self.kern.b;
            var last = b.openIf(self.num_valid.eq(@as(i32, @intCast(s + 1))));
            ptx.fenceViewAsyncTmemLoad(b);
            self.kern.releaseTmem(self.accum_stage_idx);
            last.yieldThen(.{});
        }

        /// `load_epi_block_if_valid(s)`: the block's values (zeros when `s` is not valid; they
        /// are only used under `s < num_valid_store_blocks`).
        fn loadIfValid(self: StorePipe, s: u32) []V {
            const b = self.kern.b;
            var valid = flow.IfElse([]V).open(b, ptx.ltU(b, self.kern.ci(s), self.num_valid));
            const values = self.load(s);
            self.releaseIfLast(s);
            valid.orElse(values);
            const zeros = b.arena.allocator().alloc(V, values.len) catch @panic("OOM");
            for (zeros) |*z| z.* = b.cst(.i32, 0);
            return valid.end(zeros);
        }

        /// `prefetch_prologue`: issue the first `kNumPrefetchStages` blocks (no wait); a
        /// warpgroup without valid tokens releases the stage right away. Returns block 0's
        /// values when it was prefetched.
        fn prologue(self: StorePipe) ?[]V {
            const b = self.kern.b;
            std.debug.assert(self.prefetch <= 1);
            const first: ?[]V = if (self.prefetch == 1) self.loadIfValid(0) else null;
            var none = b.openIf(self.num_valid.eq(0));
            self.kern.releaseTmem(self.accum_stage_idx);
            none.yieldThen(.{});
            return first;
        }

        const Ready = struct { values: []V, next: ?[]V };

        /// `wait_and_prefetch_next(s)` (inside the loop, where `s < num_valid_store_blocks`):
        /// wait for block `s`'s loads (issued `kNumPrefetchStages` iterations earlier, or right
        /// here without prefetch), then issue block `s + kNumPrefetchStages`. `pending` holds
        /// block `s`'s values when they were prefetched. Returns them threaded through the wait,
        /// and the next block's.
        fn waitAndPrefetchNext(self: StorePipe, s: u32, pending: ?[]V) Ready {
            const b = self.kern.b;
            if (self.prefetch == 0) {
                // `load_epi_block_if_valid(s)`: its `s < num_valid_store_blocks` holds here
                const values = self.load(s);
                self.releaseIfLast(s);
                return .{ .values = ptx.fenceViewAsyncTmemLoadSlice(b, values), .next = null };
            }
            const values = ptx.fenceViewAsyncTmemLoadSlice(b, pending.?);
            // `load_epi_block_if_valid(s + 1)` is a no-op for `s + 1 == kNumStoreBlocks`
            // (`num_valid_store_blocks <= kNumStoreBlocks`)
            const next: ?[]V = if (s + self.prefetch < self.blocks) self.loadIfValid(s + self.prefetch) else null;
            return .{ .values = values, .next = next };
        }
    };

    fn epilogueRole(self: *Kernel) void {
        const b = self.b;
        const k = self.k;
        const c_ = self.c();
        const cfg = self.cfg;
        const has_shared = k.hasShared();
        const num_epilogue_threads = c_.num_epilogue_threads;
        const num_ring_blocks: i32 = @intCast(k.numRingBlocks());
        const R = k.shape.num_ranks;
        const BM: i32 = @intCast(c_.block_m);
        const WG_BLOCK_M = k.wgBlockM();

        // Adjust registers
        ptx.warpgroupRegAlloc(b, k.numEpilogueRegisters());

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp
        // index bits; we also forbid two CTAs to share the same SM and its tensor memory
        {
            var nonzero = b.openIf(ptx.ldSharedU32(b, self.tmemPtrInSmem()).ne(0));
            ptx.trap(b);
            nonzero.yieldThen(.{});
        }

        // GEMM epilogue warps
        const epilogue_warp_idx = self.warp_idx.sub(@as(i32, @intCast(numDispatchWarps(cfg) + numMmaNonEpilogueWarps(cfg))));
        const epilogue_wg_idx = ptx.divU(b, epilogue_warp_idx, @as(i32, 4));
        const epilogue_thread_idx = epilogue_warp_idx.mul(32).add(self.lane_idx);
        const warp_idx_in_wg = ptx.remU(b, epilogue_warp_idx, @as(i32, 4));

        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks (`STORE_BLOCK_M_L1` / `STORE_BLOCK_M_L2`)
        //  - Store blocks are further divided into `ATOM_M`
        std.debug.assert(c_.block_m % Kc.num_epilogue_warpgroups == 0);
        std.debug.assert(WG_BLOCK_M % c_.store_block_m_l1 == 0 and WG_BLOCK_M % c_.store_block_m_l2 == 0);
        std.debug.assert(c_.store_block_m_l1 % Kc.atom_m == 0 and c_.store_block_m_l2 % Kc.atom_m == 0);
        std.debug.assert(c_.block_n == 128);

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx.syncUnaligned(b, @as(i32, @intCast(c_.num_dispatch_threads + num_epilogue_threads)), @as(i32, dispatch_with_epilogue_barrier_idx));

        // Persistently schedule over blocks
        var tasks = TaskLoop(V).open(self, b.cst(.i32, 0));
        {
            const t = tasks.task;
            // Wait UMMA arrival
            const iter = tasks.state;
            const accum_stage_idx = ptx.remU(b, iter, @as(i32, Kc.num_epilogue_stages));
            const accum_phase = ptx.divU(b, iter, @as(i32, Kc.num_epilogue_stages)).bitAnd(1);
            ptx.mbarrierWait(b, self.tmemFullBarrier(accum_stage_idx), accum_phase);
            ptx.tcgen05AfterThreadSync(b);

            // Now we can release the task
            self.sched.releaseTaskInfo(tasks.pipe);

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const valid_m = ptx.shflSync(b, t.valid_m, @as(i32, 0));
            const wg_start_m = epilogue_wg_idx.mul(@as(i32, @intCast(WG_BLOCK_M)));
            const num_valid_wg_rows = b.select(ptx.leU(b, valid_m, wg_start_m), b.cst(.i32, 0), ptx.minU(b, valid_m.sub(wg_start_m), @as(i32, @intCast(WG_BLOCK_M))));
            const is_shared = t.isShared(b, has_shared);
            const pool_block_idx = t.pool_block_idx;
            const ring_block_idx = ptx.remU(b, pool_block_idx, num_ring_blocks);
            const block_idx = b.select(is_shared, pool_block_idx, ring_block_idx);
            const n_block_idx = t.n_cluster_idx.mul(2).add(b.select(self.is_leader_cta, b.cst(.i32, 0), b.cst(.i32, 1)));
            const e: Epi = .{
                .epilogue_warp_idx = epilogue_warp_idx,
                .epilogue_wg_idx = epilogue_wg_idx,
                .warp_idx_in_wg = warp_idx_in_wg,
                .accum_stage_idx = accum_stage_idx,
                .is_shared = is_shared,
                .valid_m = valid_m,
                .num_valid_wg_rows = num_valid_wg_rows,
                .pool_block_idx = pool_block_idx,
                .ring_block_idx = ring_block_idx,
                .block_idx = block_idx,
                .ring_m_idx = ring_block_idx.mul(BM),
                .m_idx = block_idx.mul(BM),
                .pool_m_idx = pool_block_idx.mul(BM),
                .n_block_idx = n_block_idx,
                .n_idx = n_block_idx.mul(@as(i32, @intCast(c_.block_n))),
            };

            var l1 = flow.IfElse(void).open(b, t.isPhase(.linear1).bitOr(t.isPhase(.shared_linear1)));
            {
                {
                    // Wait L2 block empty
                    var routed = b.openIf(is_shared.bitXor(b.cst(.i1, true)));
                    const num_expected_blocks = ptx.divU(b, pool_block_idx, num_ring_blocks).mul(@as(i32, @intCast(k.numL2BlockNs())));
                    flow.spin(b, NotEqual{ .ptr = self.workspace.l2EmptyCountPtr(ring_block_idx), .expected = num_expected_blocks });
                    routed.yieldThen(.{});
                }

                self.l1Epilogue(&e);

                // Notify L2 and increment L1 empty count
                ptx.tmaStoreWait(b, 0);
                ptx.syncAligned(b, @as(i32, @intCast(num_epilogue_threads)), @as(i32, epilogue_full_barrier_idx));
                {
                    var notifier = b.openIf(epilogue_warp_idx.eq(0));
                    var elected = b.openIf(ptx.electOneSync(b));
                    {
                        var shared = b.openIf(is_shared);
                        ptx.redAddRel(b, self.workspace.sharedL2FullCountPtr(pool_block_idx), @as(i32, 1));
                        shared.yieldThen(.{});
                    }
                    {
                        var routed = b.openIf(is_shared.bitXor(b.cst(.i1, true)));
                        scheduler.L2KBlockDependency.arrive(b, self.workspace.l2FullMaskPtr(ring_block_idx), n_block_idx);
                        // Increment L1 empty count for this physical slot (one per N block)
                        ptx.redAddU32(b, self.workspace.l1EmptyCountPtr(ring_block_idx), @as(i32, 1));
                        routed.yieldThen(.{});
                    }
                    elected.yieldThen(.{});
                    notifier.yieldThen(.{});
                }
                ptx.syncWarp(b);
                l1.orElse({});
            }
            {
                // Increment L2 empty count for this physical slot (one per N block)
                {
                    var routed = b.openIf(is_shared.bitXor(b.cst(.i1, true)));
                    var notifier = b.openIf(epilogue_warp_idx.eq(0));
                    var elected = b.openIf(ptx.electOneSync(b));
                    ptx.redAddU32(b, self.workspace.l2EmptyCountPtr(ring_block_idx), @as(i32, 1));
                    elected.yieldThen(.{});
                    notifier.yieldThen(.{});
                    ptx.syncWarp(b);
                    routed.yieldThen(.{});
                }

                self.l2Epilogue(&e);

                // Ensure the next epilogue safe to use shared memory
                ptx.syncAligned(b, @as(i32, @intCast(num_epilogue_threads)), @as(i32, epilogue_full_barrier_idx));
                l1.end({});
            }
            _ = tasks.end(iter.add(1));
        }

        // Deallocate tensor memory (must be called by the same logical warp ID on both CTAs)
        {
            var warp0 = b.openIf(epilogue_warp_idx.eq(0));
            ptx.tmemFree2Sm(b, b.cst(.i32, 0), k.numTmemCols());
            warp0.yieldThen(.{});
        }

        // Peers' grid indices for tagging their combine readiness; the load overlaps with the grid sync
        std.debug.assert(R <= num_epilogue_threads);
        const tags = self.sm_idx.eq(0).bitAnd(ptx.ltU(b, epilogue_thread_idx, @as(i32, @intCast(R))));
        var peer_load = flow.IfElse(V).open(b, tags);
        peer_load.orElse(ptx.ldGlobalU64(b, self.workspace.peerGridIdxPtr(epilogue_thread_idx)));
        const peer_grid_idx = peer_load.end(ptx.constU64(b, 0));

        // All local L2 writes are done after this grid sync
        const scope: ptx.AlignedScope = .{ .num_threads = num_epilogue_threads, .barrier_idx = epilogue_full_barrier_idx };
        ptx.gridSync(b, k.num_sms, self.workspace.gridSyncCountPtr(epilogue_grid_sync_index), self.sm_idx, epilogue_thread_idx, scope, cfg.timeout_cycles);

        // Notify remote ranks; ordered before the cleanup barrier by the dispatch/epilogue sync below
        {
            var notify = b.openIf(tags);
            ptx.stRelSys(b, self.sym.map(self.workspace.combineReadyGridIdxPtr(self.sym.rank_idx), epilogue_thread_idx), peer_grid_idx);
            notify.yieldThen(.{});
        }

        // Barrier with dispatch warps, so that they can do clean workspace
        ptx.syncUnaligned(b, @as(i32, @intCast(c_.num_dispatch_threads + num_epilogue_threads)), @as(i32, dispatch_with_epilogue_barrier_idx));

        self.combine(epilogue_warp_idx);
    }

    /// `release_tmem()`: `tcgen05.fence::before_thread_sync`, then arrive on the leader CTA's
    /// TMEM-empty barrier of the stage.
    fn releaseTmem(self: *const Kernel, accum_stage_idx: V) void {
        const b = self.b;
        ptx.tcgen05BeforeThreadSync(b);
        ptx.mbarrierArriveCluster(b, self.tmemEmptyBarrier(accum_stage_idx), b.cst(.i32, 0));
    }

    // ---------- L1 epilogue ----------

    /// The unified L1 epilogue of one task (`impl` 1072-1235): SwiGLU in-place on the
    /// granularity-8 interleaved gate/up weights. With `SM100_TMEM_LOAD_16dp256b1x`, a warp's
    /// TMEM lanes 0-7 / 16-23 are gate rows and 8-15 / 24-31 up rows, so atom words
    /// `{0, 1}`/`{2, 3}` and `{4, 5}`/`{6, 7}` are (gate, up) pairs of two tokens.
    fn l1Epilogue(self: *const Kernel, e: *const Epi) void {
        const b = self.b;
        const stored_cached_weight = b.cst(.f32, 1.0);
        const pipe: StorePipe = .init(self, e.accum_stage_idx, e.epilogue_wg_idx, self.c().store_block_m_l1, e.num_valid_wg_rows);
        const first = pipe.prologue();
        self.l1StoreBlock(e, pipe, 0, first, stored_cached_weight);
    }

    /// Store block `s` of the L1 epilogue, then (nested) the next ones: the `#pragma unroll`
    /// loop `for (s ...) { if (s >= num_valid_store_blocks) break; ... }` as nested `if`s.
    /// `cached_weight` is `stored_cached_weight` carried from the previous block.
    fn l1StoreBlock(self: *const Kernel, e: *const Epi, pipe: StorePipe, s: u32, pending: ?[]V, cached_weight: V) void {
        if (s >= pipe.blocks) return;
        const b = self.b;
        const k = self.k;
        const c_ = self.c();
        const atoms = pipe.atoms;
        const WG: u32 = k.wgBlockM();
        const SBM1 = c_.store_block_m_l1;
        const L1_OUT_BLOCK_N: i32 = @intCast(k.l1OutBlockN());
        const lane = self.lane_idx;
        const a = b.arena.allocator();

        var valid = b.openIf(ptx.ltU(b, self.ci(s), pipe.num_valid));
        const ready = pipe.waitAndPrefetchNext(s, pending);

        // Iterate all atoms in the store block
        var stored_cached_weight = cached_weight;
        const activation_values = a.alloc([2][2]V, atoms) catch @panic("OOM");
        const amax_values = a.alloc([2]V, atoms) catch @panic("OOM");
        for (0..atoms) |i_| {
            const i: u32 = @intCast(i_);
            const j = s * atoms + i;

            // Load weights from global into register cache per 32 tokens
            std.debug.assert(32 % Kc.atom_m == 0);
            if ((j * Kc.atom_m) % 32 == 0) {
                var cond = e.is_shared.bitXor(b.cst(.i1, true));
                if (WG % 32 != 0) cond = cond.bitAnd(ptx.ltU(b, lane.add(@as(i32, @intCast(j * Kc.atom_m))), @as(i32, @intCast(WG))));
                var load = flow.IfElse(V).open(b, cond);
                const token = e.ring_m_idx.add(e.epilogue_wg_idx.mul(@as(i32, @intCast(WG)))).add(@as(i32, @intCast(j * Kc.atom_m))).add(lane);
                load.orElse(ptx.ldGlobal(b, 1, self.buffer.l1_topk_weights.dataPtr(token))[0].bitCast(.f32));
                stored_cached_weight = load.end(stored_cached_weight);
            }

            // Load weights from register cache
            const src_lane = ptx.remU(b, lane, @as(i32, 4)).mul(2).add(@as(i32, @intCast((j * Kc.atom_m) % 32)));
            const weights = [2]V{ ptx.shflSync(b, stored_cached_weight, src_lane), ptx.shflSync(b, stored_cached_weight, src_lane.add(1)) };

            // Apply SwiGLU: gate * sigmoid(alpha * gate) * (up + beta)
            const raw = ready.values[i * 8 ..][0..8];
            for (0..2) |kk| activation_values[i][kk] = self.swiglu(raw[4 * kk ..][0..4].*, weights);

            // Amax reduction (thread-level)
            var thread_local_amax = [2]V{ b.cst(.f32, 0.0), b.cst(.f32, 0.0) };
            for (0..2) |kk| {
                thread_local_amax[0] = ptx.fmaxf(b, thread_local_amax[0], ptx.fabsf(b, activation_values[i][kk][0]));
                thread_local_amax[1] = ptx.fmaxf(b, thread_local_amax[1], ptx.fabsf(b, activation_values[i][kk][1]));
            }

            // Amax reduction (warp-level)
            amax_values[i] = .{ ptx.warpReduce(b, 4, true, thread_local_amax[0], maxOp), ptx.warpReduce(b, 4, true, thread_local_amax[1], maxOp) };

            // Reduce amax (warp-pair-level)
            {
                var writer = b.openIf(ptx.ltU(b, lane, @as(i32, 4)));
                ptx.stSharedF32x2(b, self.amaxReduction(e.epilogue_warp_idx, lane.add(@as(i32, @intCast(i * (Kc.atom_m / 2))))), amax_values[i]);
                writer.yieldThen(.{});
            }
            ptx.syncWarp(b);
        }

        // Wait shared memory release from previous TMA store
        // And fence `shared_storage.amax_reduction`
        const tma_stage_idx = s % Kc.num_tma_store_stages;
        ptx.tmaStoreWait(b, Kc.num_tma_store_stages - 1);
        const wg_barrier = e.epilogue_wg_idx.add(@as(i32, epilogue_wg_barrier_start_idx));
        ptx.syncAligned(b, @as(i32, 128), wg_barrier);

        // Cast to FP8 E4M3 and store into shared memory
        for (0..atoms) |i_| {
            const i: u32 = @intCast(i_);
            // Reduce amax (warp-pair-level)
            const wp_amax = ptx.ldSharedF32x2(b, self.amaxReduction(e.epilogue_warp_idx.bitXor(1), ptx.remU(b, lane, @as(i32, 4)).add(@as(i32, @intCast(i * (Kc.atom_m / 2))))));
            amax_values[i][0] = ptx.fmaxf(b, amax_values[i][0], wp_amax[0]);
            amax_values[i][1] = ptx.fmaxf(b, amax_values[i][1], wp_amax[1]);

            // Calculate SF
            const sf_exp = [2]V{ ptx.getUe8m0SfExp(b, amax_values[i][0]), ptx.getUe8m0SfExp(b, amax_values[i][1]) };
            const sf_inv = [2]V{ ptx.getUe8m0SfInv(b, sf_exp[0]), ptx.getUe8m0SfInv(b, sf_exp[1]) };

            // Cast
            const upper = ptx.fmul2Rn(b, activation_values[i][0], sf_inv);
            const lower = ptx.fmul2Rn(b, activation_values[i][1], sf_inv);
            const fp8x4_values = ptx.fp8x4E4m3(b, upper[0], upper[1], lower[0], lower[1]);

            // STSM
            const row = lane;
            const col = e.warp_idx_in_wg;
            const smem_ptr = self.smemDL1(e.epilogue_wg_idx, tma_stage_idx)
                .add(@as(i32, @intCast(i * Kc.atom_m)) * L1_OUT_BLOCK_N)
                .add(row.mul(L1_OUT_BLOCK_N))
                // Use 64B swizzle for SwiGLU, so divided by 2
                .add(col.bitXor(ptx.divU(b, row, @as(i32, 2))).mul(@as(i32, Kc.num_bank_group_bytes)));
            ptx.stmatrixU8x4T(b, fp8x4_values, smem_ptr);

            // Store SF to `buffer.l2_sf_buffer` as UE8M0 (MN-major layout)
            // Only one warp per pair writes (both hold the same SF after cross-warp reduce)
            // Each lane < 4 holds SF for 2 rows (sf.x and sf.y)
            {
                var writer = b.openIf(ptx.remU(b, e.warp_idx_in_wg, @as(i32, 2)).eq(0).bitAnd(ptx.ltU(b, lane, @as(i32, 4))));
                const k_idx = e.n_block_idx.mul(2).add(ptx.divU(b, e.warp_idx_in_wg, @as(i32, 2)));
                const k_uint_idx = ptx.divU(b, k_idx, @as(i32, 4));
                const byte_idx = ptx.remU(b, k_idx, @as(i32, 4));
                const mn_stride = b.select(e.is_shared, self.ci(k.numSharedSfTokens() * 4), self.ci(c_.num_sf_ring_tokens * 4));
                const sf_base_ptr = b.select(e.is_shared, self.buffer.shared_l2_sf.basePtr(), self.buffer.l2_sf.basePtr());
                // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx`
                // differs by 4; `token_base_idx < BLOCK_M`, and `lane_idx * 2` (the lowest 3
                // bits) commutes with `transform_sf_token_idx`
                const token_base_idx = e.epilogue_wg_idx.mul(@as(i32, @intCast(WG))).add(@as(i32, @intCast(s * SBM1 + i * Kc.atom_m)));
                const sf_token_idx = e.block_idx.mul(@as(i32, @intCast(c_.sf_block_m))).add(self.transformSfTokenIdx(token_base_idx)).add(lane.mul(2).mul(4));
                const sf_addr = k_uint_idx.mul(mn_stride).add(sf_token_idx.mul(4)).add(byte_idx);
                ptx.stGlobalU8(b, device.index(b, sf_base_ptr, sf_addr, 1), sf_exp[0]);
                ptx.stGlobalU8(b, device.index(b, sf_base_ptr, sf_addr.add(4 * 4), 1), sf_exp[1]);
                writer.yieldThen(.{});
            }
            ptx.syncWarp(b);
        }
        ptx.syncAligned(b, @as(i32, 128), wg_barrier);

        // Issue TMA store after all atoms in this store block
        {
            var warp0 = b.openIf(e.warp_idx_in_wg.eq(0));
            var elected = b.openIf(ptx.electOneSync(b));
            const out_n_idx = e.n_block_idx.mul(L1_OUT_BLOCK_N);
            const tensor_map_l1_output = b.select(e.is_shared, self.desc("shared_l1_output"), self.desc("l1_output"));
            ptx.tmaStoreFence(b);
            ptx.tmaStore2d(b, tensor_map_l1_output, self.smemDL1(e.epilogue_wg_idx, tma_stage_idx), out_n_idx, e.m_idx.add(e.epilogue_wg_idx.mul(@as(i32, @intCast(WG)))).add(@as(i32, @intCast(s * SBM1))));
            ptx.tmaStoreArrive(b);
            elected.yieldThen(.{});
            warp0.yieldThen(.{});
        }
        ptx.syncWarp(b);

        self.l1StoreBlock(e, pipe, s + 1, ready.next, stored_cached_weight);
        valid.yieldThen(.{});
    }

    /// SwiGLU of one (gate, up) pair of two tokens with the routing weights (`impl`
    /// 1109-1135): `raw` = TMEM words `{gate.x, gate.y, up.x, up.y}` (f32 bits).
    fn swiglu(self: *const Kernel, raw: [4]V, weights: [2]V) [2]V {
        const b = self.b;
        const cfg = self.cfg;
        const clamp = cfg.activation_clamp;
        const clamped = clamp != std.math.inf(f32);
        var bf16_gate = ptx.float22bfloat162Rn(b, raw[0].bitCast(.f32), raw[1].bitCast(.f32));
        var bf16_up = ptx.float22bfloat162Rn(b, raw[2].bitCast(.f32), raw[3].bitCast(.f32));

        // Clamp
        if (clamped) {
            if (!cfg.clamp_after_silu) bf16_gate = ptx.hmin2Bf16(b, bf16_gate, ptx.bf16x2Const(b, clamp));
            bf16_up = ptx.hmax2Bf16(b, bf16_up, ptx.bf16x2Const(b, -clamp));
            bf16_up = ptx.hmin2Bf16(b, bf16_up, ptx.bf16x2Const(b, clamp));
        }

        // SwiGLU
        const gate = ptx.bfloat1622float2(b, bf16_gate);
        var sigmoid_input = gate;
        if (cfg.activation_alpha != 1.0)
            sigmoid_input = ptx.fmul2Rn(b, sigmoid_input, .{ b.cst(.f32, cfg.activation_alpha), b.cst(.f32, cfg.activation_alpha) });
        // `kFastMath` only: the IEEE `expf` and division of the other path are not ported
        // (`Config.check`)
        const neg_gate_exp = [2]V{ ptx.expfFastNeg(b, sigmoid_input[0]), ptx.expfFastNeg(b, sigmoid_input[1]) };
        const denom = ptx.fadd2Rn(b, .{ b.cst(.f32, 1.0), b.cst(.f32, 1.0) }, neg_gate_exp);
        var gated = ptx.fmul2Rn(b, gate, .{ ptx.fastRcp(b, denom[0]), ptx.fastRcp(b, denom[1]) });
        if (cfg.clamp_after_silu and clamped)
            gated = .{ ptx.fminf(b, gated[0], b.cst(.f32, clamp)), ptx.fminf(b, gated[1], b.cst(.f32, clamp)) };
        var up = ptx.bfloat1622float2(b, bf16_up);
        if (cfg.activation_beta != 0.0)
            up = ptx.fadd2Rn(b, up, .{ b.cst(.f32, cfg.activation_beta), b.cst(.f32, cfg.activation_beta) });
        return ptx.fmul2Rn(b, ptx.fmul2Rn(b, gated, up), weights);
    }

    // ---------- L2 epilogue ----------

    /// The L2 BF16 epilogue of one task (`impl` 1271-1366): write the GEMM output into the
    /// source ranks' combine buffers via NVLink.
    fn l2Epilogue(self: *const Kernel, e: *const Epi) void {
        const pipe: StorePipe = .init(self, e.accum_stage_idx, e.epilogue_wg_idx, self.c().store_block_m_l2, e.num_valid_wg_rows);
        const first = pipe.prologue();
        self.l2StoreBlock(e, pipe, 0, first);
    }

    /// `*workspace.get_token_src_metadata_ptr(pool_token_idx)`: `{rank, token, topk}`.
    fn loadTokenSrcMetadata(self: *const Kernel, pool_token_idx: V) [3]V {
        const b = self.b;
        const ptr = self.workspace.tokenSrcMetadataPtr(pool_token_idx);
        return .{ ptx.ldGlobal(b, 1, ptr)[0], ptx.ldGlobal(b, 1, ptr.add(4))[0], ptx.ldGlobal(b, 1, ptr.add(8))[0] };
    }

    /// Store block `s` of the L2 epilogue, then (nested) the next ones (see `l1StoreBlock`).
    fn l2StoreBlock(self: *const Kernel, e: *const Epi, pipe: StorePipe, s: u32, pending: ?[]V) void {
        if (s >= pipe.blocks) return;
        const b = self.b;
        const k = self.k;
        const c_ = self.c();
        const atoms = pipe.atoms;
        const WG: u32 = k.wgBlockM();
        const SBM2 = c_.store_block_m_l2;
        const sw: u32 = Kc.swizzle_cd_mode;
        const bank: i32 = Kc.num_bank_group_bytes;
        const lane = self.lane_idx;
        const a = b.arena.allocator();

        var valid = b.openIf(ptx.ltU(b, self.ci(s), pipe.num_valid));
        const ready = pipe.waitAndPrefetchNext(s, pending);

        // Read the source metadata of this warp's rows (2 per atom) before the shared memory
        // stores to overlap the latency
        const half = ptx.divU(b, lane, @as(i32, 16));
        const cached_src_metadata = a.alloc([3]V, atoms) catch @panic("OOM");
        for (0..atoms) |j_| {
            const j: u32 = @intCast(j_);
            const m_idx_in_block = e.epilogue_wg_idx.mul(@as(i32, @intCast(WG))).add(@as(i32, @intCast(s * SBM2 + j * Kc.atom_m))).add(e.warp_idx_in_wg.mul(2)).add(half);
            var in_range = flow.IfElse([3]V).open(b, ptx.ltU(b, m_idx_in_block, e.valid_m));
            const pool_token_idx = e.pool_m_idx.add(m_idx_in_block);
            var meta: [3]V = undefined;
            if (k.hasShared()) {
                var shared = flow.IfElse([3]V).open(b, e.is_shared);
                shared.orElse(.{ self.sym.rank_idx, pool_token_idx, self.ci(k.shape.num_topk) });
                meta = shared.end(self.loadTokenSrcMetadata(pool_token_idx));
            } else {
                meta = self.loadTokenSrcMetadata(pool_token_idx);
            }
            in_range.orElse(meta);
            const zero = b.cst(.i32, 0);
            cached_src_metadata[j] = in_range.end(.{ zero, zero, zero });
        }
        ptx.syncWarp(b);

        const wg_barrier = e.epilogue_wg_idx.add(@as(i32, epilogue_wg_barrier_start_idx));
        for (0..atoms) |i_| {
            const i: u32 = @intCast(i_);
            // Wait shared memory release from previous NVLink store
            // NOTES: skip for the first store block since the prior full barrier already ensures completion
            if (i == 0 and s > 0) ptx.syncAligned(b, @as(i32, 128), wg_barrier);

            // Store into shared memory
            // NOTES: each lane provides its own address for stmatrix; 2 warps share a BF16 swizzle atom
            const values = ready.values[i * 8 ..][0..8];
            const row = ptx.remU(b, lane, @as(i32, 8));
            const col = ptx.remU(b, e.epilogue_warp_idx, @as(i32, 2)).mul(4).add(ptx.divU(b, lane, @as(i32, 8)));
            const smem_ptr = self.smemDL2(e.epilogue_wg_idx)
                .add(ptx.divU(b, e.warp_idx_in_wg, @as(i32, 2)).mul(@as(i32, @intCast(SBM2 * sw))))
                .add(@as(i32, @intCast(i * Kc.atom_m * sw)))
                .add(row.mul(bank * 8))
                .add(col.bitXor(row).mul(bank));
            ptx.stmatrixU32x4T(b, .{
                ptx.castIntoBf16AndPack(b, values[0], values[1]),
                ptx.castIntoBf16AndPack(b, values[2], values[3]),
                ptx.castIntoBf16AndPack(b, values[4], values[5]),
                ptx.castIntoBf16AndPack(b, values[6], values[7]),
            }, smem_ptr);
        }

        // Wait shared memory ready
        ptx.syncAligned(b, @as(i32, 128), wg_barrier);

        // Write into remote buffers
        // Each warp writes 2 rows (lane_idx/16 splits the warp into two halves, one per row)
        const row_in_atom = ptx.remU(b, e.warp_idx_in_wg.mul(2).add(half), @as(i32, Kc.atom_m));
        const bank_group_idx = ptx.remU(b, lane, @as(i32, 8));
        self.l2WriteRow(e, s, 0, atoms, cached_src_metadata, half, row_in_atom, bank_group_idx);

        self.l2StoreBlock(e, pipe, s + 1, ready.next);
        valid.yieldThen(.{});
    }

    /// Row `j` of an L2 store block's remote writes, then (nested) the next ones: the
    /// `#pragma unroll` loop `for (j ...) { if (m_idx_in_block >= valid_m) break; ... }`.
    fn l2WriteRow(self: *const Kernel, e: *const Epi, s: u32, j: u32, atoms: u32, cached_src_metadata: []const [3]V, half: V, row_in_atom: V, bank_group_idx: V) void {
        if (j >= atoms) return;
        const b = self.b;
        const k = self.k;
        const SBM2 = self.c().store_block_m_l2;
        const sw: i32 = Kc.swizzle_cd_mode;
        const lane = self.lane_idx;
        const row_in_store = e.warp_idx_in_wg.mul(2).add(half).add(@as(i32, @intCast(j * Kc.atom_m)));
        const m_idx_in_block = e.epilogue_wg_idx.mul(@as(i32, @intCast(k.wgBlockM()))).add(@as(i32, @intCast(s * SBM2))).add(row_in_store);

        // Skip padding rows beyond the actual token count for this expert
        var in_range = b.openIf(ptx.ltU(b, m_idx_in_block, e.valid_m));
        const dst_rank_idx = cached_src_metadata[j][0];
        const dst_token_idx = cached_src_metadata[j][1];
        const dst_topk_idx = cached_src_metadata[j][2];

        // Read from shared memory
        const smem_ptr = self.smemDL2(e.epilogue_wg_idx)
            .add(ptx.divU(b, ptx.remU(b, lane, @as(i32, 16)), @as(i32, 8)).mul(@as(i32, @intCast(SBM2)) * sw))
            .add(row_in_store.mul(sw))
            .add(bank_group_idx.bitXor(row_in_atom).mul(@as(i32, Kc.num_bank_group_bytes)));
        const packed_ = ptx.ldSharedF32x4(b, smem_ptr);

        // Write into remote
        const dst_token = self.buffer.combine_token.slotDataPtr(dst_topk_idx, dst_token_idx);
        const dst_ptr = device.index(b, dst_token, e.n_idx.mul(2).add(ptx.remU(b, lane, @as(i32, 16)).mul(16)), 1);
        self.sym.st(4, self.sym.map(dst_ptr, dst_rank_idx), .{ packed_[0].bitCast(.i32), packed_[1].bitCast(.i32), packed_[2].bitCast(.i32), packed_[3].bitCast(.i32) });

        self.l2WriteRow(e, s, j + 1, atoms, cached_src_metadata, half, row_in_atom, bank_group_idx);
        in_range.yieldThen(.{});
    }

    // ---------- combine ----------

    /// The combine's per-warp shared-memory buffers and barriers (`impl` 1406-1416). The
    /// buffers reuse shared memory from the start up to the barriers.
    const Combine = struct {
        kern: *const Kernel,
        epilogue_warp_idx: V,
        /// `kNumChunkBytes`.
        chunk_bytes: u32,

        /// `combine_load_buffer[i]`.
        fn loadBuffer(self: Combine, i: V) V {
            const ne: i32 = @intCast(numEpilogueWarps(self.kern.cfg));
            return self.kern.smem.add(self.epilogue_warp_idx.add(i.mul(ne)).mul(@as(i32, @intCast(self.chunk_bytes))));
        }
        /// `combine_store_buffer`.
        fn storeBuffer(self: Combine) V {
            const ne: i32 = @intCast(numEpilogueWarps(self.kern.cfg));
            return self.kern.smem.add(self.epilogue_warp_idx.add(ne * 2).mul(@as(i32, @intCast(self.chunk_bytes))));
        }
        /// `combine_load_barriers[i]`.
        fn loadBarrier(self: Combine, i: V) V {
            return self.kern.at(self.kern.so.combine_barriers).add(i.add(self.epilogue_warp_idx.mul(2)).mul(8));
        }
    };

    const MaskLoad = struct { mask: V, loaded: V };

    /// `move_mask_and_load(i)`: pick the next slot of `mask` (the shared slot first, then the
    /// routed slots in top-k order) and 1-D TMA load its chunk into load stage `i`. Returns the
    /// remaining mask and whether a slot was loaded.
    fn moveMaskAndLoad(self: *const Kernel, cb: Combine, mask: V, i: V, token_idx: V, chunk_byte_offset: V) MaskLoad {
        const b = self.b;
        const topk: i32 = @intCast(self.k.shape.num_topk);
        var any = flow.IfElse(MaskLoad).open(b, mask.ne(0));
        // Shared slot goes first, then routed slots in top-k order
        const slot_idx = b.select(mask.shrLogical(topk).ne(0), b.cst(.i32, topk), ptx.ffs(b, mask).sub(1));
        const new_mask = mask.bitXor(b.cst(.i32, 1).shl(slot_idx));

        // Load
        {
            var elected = b.openIf(ptx.electOneSync(b));
            const src_ptr = device.index(b, self.buffer.combine_token.slotDataPtr(slot_idx, token_idx), chunk_byte_offset, 1);
            ptx.tmaLoad1d(b, cb.loadBuffer(i), src_ptr, cb.loadBarrier(i), @as(i32, @intCast(cb.chunk_bytes)), ptx.CacheHint.evict_first);
            ptx.mbarrierArriveAndSetTx(b, cb.loadBarrier(i), @as(i32, @intCast(cb.chunk_bytes)));
            elected.yieldThen(.{});
        }
        ptx.syncWarp(b);
        any.orElse(.{ .mask = new_mask, .loaded = b.cst(.i1, true) });
        return any.end(.{ .mask = mask, .loaded = b.cst(.i1, false) });
    }

    /// Combine: reduce top-k results and write back (`impl` 1380-1535, after the epilogue grid
    /// sync, the readiness push and the dispatch/epilogue barrier, which `epilogueRole` emits).
    fn combine(self: *const Kernel, epilogue_warp_idx: V) void {
        const b = self.b;
        const k = self.k;
        const cfg = self.cfg;
        const H = k.shape.hidden;
        const topk = k.shape.num_topk;
        const has_shared = k.hasShared();
        const ne = numEpilogueWarps(cfg);
        const lane = self.lane_idx;

        // NOTES: reuse shared memory from start up to the barriers
        const num_hidden_bytes: u32 = H * 2;
        const num_elems_per_uint4: u32 = 16 / 4;
        // 3 slots of chunk is needed: 2 load stages and 1 store
        const num_chunk_slots: u32 = 3;
        // NOTES: either 1 or 2 chunks for simplicity; restrict on both smem and register
        const num_chunks = k.numCombineChunks();
        const num_chunk_bytes = num_hidden_bytes / num_chunks;
        const num_chunk_uint4 = num_chunk_bytes / 16;
        const num_uint4_per_lane = num_chunk_uint4 / 32;
        std.debug.assert(H % num_chunks == 0);
        std.debug.assert(num_chunk_slots * ne * num_hidden_bytes / num_chunks <= self.so.reusableBytes());
        std.debug.assert(num_chunk_bytes % 16 == 0);
        std.debug.assert(num_chunk_uint4 % 32 == 0);
        std.debug.assert(topk + @intFromBool(has_shared) <= 32);
        // `mask >> kNumTopk` (C++ UB at 32)
        std.debug.assert(topk < 32);
        // Verify combined shared memory budget (DeepGEMM: device assert)
        std.debug.assert(num_chunk_slots * ne * num_chunk_bytes <= self.so.reusableBytes());

        const cb: Combine = .{ .kern = self, .epilogue_warp_idx = epilogue_warp_idx, .chunk_bytes = num_chunk_bytes };
        const e_per_rank: i32 = @intCast(k.numExpertsPerRank());

        // Iterate over all token chunks, 1 token 1 topk latency: ~3 us
        const grid_idx = ptx.getGridIdx(b).add(ptx.constU64(b, 1));
        const Outer = struct { load_stage_idx: V, combine_phase: V };
        var tokens = flow.For(Outer).open(
            b,
            epilogue_warp_idx.mul(@as(i32, @intCast(k.num_sms))).add(self.sm_idx),
            self.num_tokens.mul(@as(i32, @intCast(num_chunks))),
            @as(i32, @intCast(k.num_sms * ne)),
            .{ .load_stage_idx = b.cst(.i32, 0), .combine_phase = b.cst(.i32, 0) },
        );
        {
            const token_chunk_idx = tokens.iv;
            const token_idx = ptx.divU(b, token_chunk_idx, @as(i32, @intCast(num_chunks)));
            const chunk_idx = ptx.remU(b, token_chunk_idx, @as(i32, @intCast(num_chunks)));

            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            var read = flow.IfElse(V).open(b, ptx.ltU(b, lane, @as(i32, @intCast(topk))));
            read.orElse(ptx.lo32(b, ptx.ldGlobalU64(b, device.index(b, self.buffer.input_topk_idx.basePtr(), token_idx.mul(@as(i32, @intCast(topk))).add(lane), 8))));
            const shared_or_none = if (has_shared) b.select(lane.eq(@as(i32, @intCast(topk))), self.ci(topk), b.cst(.i32, -1)) else b.cst(.i32, -1);
            const stored_topk_slot_idx = read.end(shared_or_none);
            const total_mask = ptx.ballotSync(b, stored_topk_slot_idx.ge(0));

            // Wait for the ranks of the selected experts to finish their L2 writes
            const is_routed = ptx.ltU(b, lane, @as(i32, @intCast(topk))).bitAnd(stored_topk_slot_idx.ge(0));
            const peer_ready_ptr = self.workspace.combineReadyGridIdxPtr(b.select(is_routed, ptx.divU(b, stored_topk_slot_idx, e_per_rank), b.cst(.i32, 0)));
            ptx.waitUntil(b, cfg.timeout_cycles, CombineReady{ .is_routed = is_routed, .ptr = peer_ready_ptr, .grid_idx = grid_idx });

            const chunk_byte_offset = chunk_idx.mul(@as(i32, @intCast(num_chunk_bytes)));

            // Load the first selection
            const first = self.moveMaskAndLoad(cb, total_mask, tokens.state.load_stage_idx, token_idx, chunk_byte_offset);

            // Accumulate all top-k contributions for this chunk in float registers
            const zeros = b.arena.allocator().alloc([2]V, num_uint4_per_lane * num_elems_per_uint4) catch @panic("OOM");
            for (zeros) |*z| z.* = .{ b.cst(.f32, 0.0), b.cst(.f32, 0.0) };
            const Acc = struct { do_reduce: V, mask: V, load_stage_idx: V, combine_phase: V, reduced: [][2]V };
            var reduce = flow.While(Acc, Acc).open(b, .{ .do_reduce = first.loaded, .mask = first.mask, .load_stage_idx = tokens.state.load_stage_idx, .combine_phase = tokens.state.combine_phase, .reduced = zeros });
            const st = reduce.state;
            const it = reduce.cond(st.do_reduce, st);
            const reduced_out = blk: {
                // Prefetch next top-k into the buffer while current is being accumulated
                const next = self.moveMaskAndLoad(cb, it.mask, it.load_stage_idx.bitXor(1), token_idx, chunk_byte_offset);

                // Accumulate
                ptx.mbarrierWait(b, cb.loadBarrier(it.load_stage_idx), it.combine_phase);
                const reduced = b.arena.allocator().dupe([2]V, it.reduced) catch @panic("OOM");
                const buffer = cb.loadBuffer(it.load_stage_idx);
                for (0..num_uint4_per_lane) |j| {
                    const uint4_values = ptx.ldSharedU32x4(b, buffer.add(lane.add(@as(i32, @intCast(j * 32))).mul(16)));
                    for (0..num_elems_per_uint4) |l| reduced[j * num_elems_per_uint4 + l] = ptx.accumulate(b, reduced[j * num_elems_per_uint4 + l], uint4_values[l]);
                }
                ptx.fenceViewAsyncShared(b);
                const combine_phase = it.combine_phase.bitXor(it.load_stage_idx);
                const load_stage_idx = it.load_stage_idx.bitXor(1);
                break :blk reduce.end(.{ .do_reduce = next.loaded, .mask = next.mask, .load_stage_idx = load_stage_idx, .combine_phase = combine_phase, .reduced = reduced });
            };

            // Cast
            const store_buffer = cb.storeBuffer();
            for (0..num_uint4_per_lane) |j| {
                var casted: [4]V = undefined;
                for (0..num_elems_per_uint4) |l| {
                    const v = reduced_out.reduced[j * num_elems_per_uint4 + l];
                    casted[l] = ptx.float22bfloat162Rn(b, v[0], v[1]);
                }

                // Wait share memory release and write
                if (j == 0) {
                    ptx.tmaStoreWait(b, 0);
                    ptx.syncWarp(b);
                }
                ptx.stSharedU32x4(b, store_buffer.add(lane.add(@as(i32, @intCast(j * 32))).mul(16)), casted);
            }
            ptx.syncWarp(b);

            // TMA store the token chunk
            {
                var elected = b.openIf(ptx.electOneSync(b));
                ptx.tmaStoreFence(b);
                const dst = self.y.add(ptx.zext(b, token_idx, .i64).mul(ptx.constU64(b, num_hidden_bytes))).add(ptx.zext(b, chunk_byte_offset, .i64));
                ptx.tmaStore1d(b, dst, store_buffer, @as(i32, @intCast(num_chunk_bytes)), ptx.CacheHint.evict_normal);
                ptx.tmaStoreArrive(b);
                elected.yieldThen(.{});
            }
            ptx.syncWarp(b);
            _ = tokens.end(.{ .load_stage_idx = reduced_out.load_stage_idx, .combine_phase = reduced_out.combine_phase });
        }
    }
};

/// The combine's readiness wait: `__all_sync(0xffffffff, not is_routed or
/// ptx::ld_acq_sys(peer_ready_ptr) == grid_idx)` (the load only on routed lanes).
const CombineReady = struct {
    is_routed: V,
    ptr: V,
    grid_idx: V,
    pub fn pending(self: CombineReady, b: *B) V {
        var routed = flow.IfElse(V).open(b, self.is_routed);
        routed.orElse(ptx.ldAcqSysU64(b, self.ptr).eq(self.grid_idx));
        const ready = routed.end(b.cst(.i1, true));
        return ptx.allSync(b, ready).bitXor(b.cst(.i1, true));
    }
};

/// `math::ReduceMax<float>` (`fmaxf`).
fn maxOp(b: *B, x: V, y: V) V {
    return ptx.fmaxf(b, x, y);
}

/// `while (ptx::ld_acq(ptr) != expected);`
const NotEqual = struct {
    ptr: V,
    expected: V,
    pub fn pending(self: NotEqual, b: *B) V {
        return ptx.ldAcq(b, self.ptr).ne(self.expected);
    }
};

/// The symmetric buffer layout of an instantiation (`layout.MegaMoEBuffer` with its rings).
fn symmBuffer(cfg: Config) layout.MegaMoEBuffer {
    const k = cfg.kernel;
    const s = k.shape;
    return .init(s.hidden, s.intermediate_hidden, s.num_ranks, s.num_experts, s.num_max_tokens_per_rank, s.num_topk, k.cfg.num_ring_tokens, k.cfg.num_sf_ring_tokens, s.num_shared_experts);
}
