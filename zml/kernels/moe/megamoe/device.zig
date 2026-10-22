//! Device-side views of DeepGEMM MegaMoE's buffers and its cross-SM / cross-rank barriers
//! (vLLM's vendored DeepGEMM, `deep_gemm/include/deep_gemm/`):
//!
//! * `layout/sym_buffer.cuh`: `SymBuffer<kNumRanks>` (`Sym`): local base, per-rank offsets,
//!   rank, and `map(ptr, rank)`;
//! * `layout/mega_moe.cuh`: the `Workspace` getters (`Workspace`) and the `MegaMoEBuffer`
//!   regions (`Buffer`), as `i64` global addresses computed from `layout.zig`'s offsets with
//!   DeepGEMM's integer types (32-bit indices, zero-extended into 64-bit pointer arithmetic);
//! * `comm/barrier.cuh`: `nvlink_barrier` (`grid_sync` and `wait_until` are `ptx.gridSync` /
//!   `ptx.waitUntil`).
const std = @import("std");

const cute = @import("kernels/cute/builder");

const layout = @import("layout.zig");
const ptx = @import("ptx.zig");

const B = cute.Builder;
const V = cute.Value;

/// `comm::kNumTimeoutCycles`: 300 s at 2 GHz.
pub const default_timeout_cycles: i64 = 300 * 2_000_000_000;

/// `base + zext(idx) * elem_bytes` (C++ `T* + uint32_t`).
pub fn index(b: *B, base: V, idx: V, elem_bytes: u64) V {
    const i = ptx.zext(b, idx, .i64);
    return base.add(if (elem_bytes == 1) i else i.mul(ptx.constU64(b, elem_bytes)));
}

/// `base + offset` for a constant byte offset.
pub fn offsetBy(b: *B, base: V, offset: u64) V {
    return if (offset == 0) base else base.add(ptx.constU64(b, offset));
}

// ==================== layout/sym_buffer.cuh ====================

/// `layout::SymBuffer<kNumRanks>`: `base` (this rank's buffer, i64), `offsets[r]` (i64, peer
/// buffer minus `base`; `offsets[rank] == 0`) and `rank_idx` (i32), passed as kernel scalars.
pub const Sym = struct {
    b: *B,
    num_ranks: u32,
    base: V,
    offsets: []const V,
    rank_idx: V,

    /// `get_base_ptr()`.
    pub fn basePtr(self: Sym) V {
        return self.base;
    }

    /// `map(ptr, dst_rank_idx)`: identity when `kNumRanks == 1`, else `offsets[dst] + ptr`
    /// (DeepGEMM indexes the by-value param array; with scalar params the offset is selected).
    pub fn map(self: Sym, ptr: V, dst_rank_idx: V) V {
        if (self.num_ranks == 1) return ptr;
        const b = self.b;
        var offset = self.offsets[0];
        for (self.offsets[1..], 1..) |o, r| offset = b.select(dst_rank_idx.eq(@as(i32, @intCast(r))), o, offset);
        return offset.add(ptr);
    }

    /// Plain `*mapped = v` / `v = *mapped` through a pointer from `map`. DeepGEMM's mapped
    /// pointer is an integer cast, so nvcc emits a generic `st`/`ld` (`ST.E`/`LD.E`) when
    /// `kNumRanks > 1`; with one rank `map` returns the pointer itself and nvcc infers `.global`.
    pub fn st(self: Sym, comptime n: usize, mapped: V, words: [n]V) void {
        if (self.num_ranks == 1) ptx.stGlobal(self.b, n, mapped, words) else ptx.stGeneric(self.b, n, mapped, words);
    }
    pub fn stU64(self: Sym, mapped: V, value: V) void {
        if (self.num_ranks == 1) ptx.stGlobalU64(self.b, mapped, value) else ptx.stGenericU64(self.b, mapped, value);
    }
    pub fn ld(self: Sym, comptime n: usize, mapped: V) [n]V {
        return if (self.num_ranks == 1) ptx.ldGlobal(self.b, n, mapped) else ptx.ldGeneric(self.b, n, mapped);
    }
};

// ==================== layout/mega_moe.cuh: Workspace ====================

/// `layout::Workspace` getters over `signals` (the symmetric buffer base, i64).
pub const Workspace = struct {
    b: *B,
    signals: V,
    lw: layout.Workspace,

    fn field(self: Workspace, comptime name: []const u8) V {
        return offsetBy(self.b, self.signals, layout.Signals.offset(name));
    }

    /// `get_grid_sync_count_ptr<kIndex>()`.
    pub fn gridSyncCountPtr(self: Workspace, comptime grid_sync_index: u32) V {
        comptime std.debug.assert(grid_sync_index < layout.Signals.max_grid_sync_counters);
        return offsetBy(self.b, self.field("grid_sync_count"), 4 * grid_sync_index);
    }
    /// `get_nvl_barrier_counter_ptr()`.
    pub fn nvlBarrierCounterPtr(self: Workspace) V {
        return self.field("nvl_barrier_counter");
    }
    /// `get_nvl_barrier_signal_ptr(phase)`.
    pub fn nvlBarrierSignalPtr(self: Workspace, phase: V) V {
        return index(self.b, self.field("nvl_barrier_signals"), phase, 4);
    }
    /// `get_combine_ready_grid_idx_ptr(peer_rank_idx)`.
    pub fn combineReadyGridIdxPtr(self: Workspace, peer_rank_idx: V) V {
        return index(self.b, self.field("combine_ready_grid_idx"), peer_rank_idx, 8);
    }
    /// `get_peer_grid_idx_ptr(peer_rank_idx)`.
    pub fn peerGridIdxPtr(self: Workspace, peer_rank_idx: V) V {
        return index(self.b, self.field("peer_grid_idx"), peer_rank_idx, 8);
    }
    /// `get_l1_task_count_ptr()`.
    pub fn l1TaskCountPtr(self: Workspace) V {
        return self.field("l1_task_count");
    }
    /// `get_l2_task_count_ptr()`.
    pub fn l2TaskCountPtr(self: Workspace) V {
        return self.field("l2_task_count");
    }
    /// `get_shared_l1_task_count_ptr()`.
    pub fn sharedL1TaskCountPtr(self: Workspace) V {
        return self.field("shared_l1_task_count");
    }
    /// `get_shared_l2_task_count_ptr()`.
    pub fn sharedL2TaskCountPtr(self: Workspace) V {
        return self.field("shared_l2_task_count");
    }
    /// `get_expert_send_count_ptr(expert_idx)`.
    pub fn expertSendCountPtr(self: Workspace, expert_idx: V) V {
        return index(self.b, self.field("expert_send_count"), expert_idx, 8);
    }
    /// `get_expert_recv_count_ptr(rank_idx, expert_idx)`: `[rank][expert]` (u32 index).
    pub fn expertRecvCountPtr(self: Workspace, rank_idx: V, expert_idx: V) V {
        const i = rank_idx.mul(@as(i32, @intCast(self.lw.num_experts_per_rank))).add(expert_idx);
        return index(self.b, self.field("expert_recv_count"), i, 8);
    }
    /// `get_expert_recv_count_sum_ptr(expert_idx)`.
    pub fn expertRecvCountSumPtr(self: Workspace, expert_idx: V) V {
        return index(self.b, self.field("expert_recv_count_sum"), expert_idx, 8);
    }
    /// `get_l1_full_count_ptr(ring_block_idx)`.
    pub fn l1FullCountPtr(self: Workspace, ring_block_idx: V) V {
        return index(self.b, self.field("l1_full_count"), ring_block_idx, 4);
    }
    /// `get_l1_empty_count_ptr(ring_block_idx)`.
    pub fn l1EmptyCountPtr(self: Workspace, ring_block_idx: V) V {
        return index(self.b, self.field("l1_empty_count"), ring_block_idx, 4);
    }
    /// `get_l2_full_mask_ptr(ring_block_idx)`.
    pub fn l2FullMaskPtr(self: Workspace, ring_block_idx: V) V {
        return index(self.b, self.field("l2_full_mask"), ring_block_idx, 8);
    }
    /// `get_l2_empty_count_ptr(ring_block_idx)`.
    pub fn l2EmptyCountPtr(self: Workspace, ring_block_idx: V) V {
        return index(self.b, self.field("l2_empty_count"), ring_block_idx, 4);
    }
    /// `get_shared_l2_full_count_ptr(block_idx)`.
    pub fn sharedL2FullCountPtr(self: Workspace, block_idx: V) V {
        return index(self.b, self.field("shared_l2_full_count"), block_idx, 4);
    }
    /// `get_src_token_topk_idx_ptr(expert_idx, rank_idx, token_idx)`:
    /// `(expert * num_ranks + rank) * num_max_tokens_per_rank + token` u32s after the signals.
    pub fn srcTokenTopkIdxPtr(self: Workspace, expert_idx: V, rank_idx: V, token_idx: V) V {
        const i = expert_idx.mul(@as(i32, @intCast(self.lw.num_ranks))).add(rank_idx).mul(@as(i32, @intCast(self.lw.num_max_tokens_per_rank))).add(token_idx);
        return index(self.b, offsetBy(self.b, self.signals, @sizeOf(layout.Signals)), i, 4);
    }
    /// `get_token_src_metadata_ptr(pool_token_idx)` (12-byte `TokenSrcMetadata`).
    pub fn tokenSrcMetadataPtr(self: Workspace, pool_token_idx: V) V {
        return index(self.b, offsetBy(self.b, self.signals, self.lw.tokenSrcMetadataOffset(0)), pool_token_idx, @sizeOf(layout.TokenSrcMetadata));
    }
    /// `num_shared_l2_pool_blocks`.
    pub fn numSharedL2PoolBlocks(self: Workspace) u32 {
        return self.lw.num_shared_l2_pool_blocks;
    }
};

// ==================== layout/mega_moe.cuh: MegaMoEBuffer ====================

/// One `layout::Buffer` region of `MegaMoEBuffer` at `signals + region.offset`.
pub const Region = struct {
    b: *B,
    base: V,
    region: layout.Region,

    /// `get_base_ptr()`.
    pub fn basePtr(self: Region) V {
        return self.base;
    }
    /// `get_data_buffer(token_idx).get_base_ptr()` (`token_idx` u32, byte count u64).
    pub fn dataPtr(self: Region, token_idx: V) V {
        return index(self.b, self.base, token_idx, self.region.item_bytes);
    }
    /// `get_rank_buffer(slot).get_data_buffer(token_idx).get_base_ptr()`.
    pub fn slotDataPtr(self: Region, slot: V, token_idx: V) V {
        const slot_base = index(self.b, self.base, slot, self.region.bytesPerSlot());
        return index(self.b, slot_base, token_idx, self.region.item_bytes);
    }
};

/// `layout::MegaMoEBuffer` over the symmetric buffer base.
pub const Buffer = struct {
    workspace: Workspace,
    input_token: Region,
    input_sf: Region,
    input_topk_idx: Region,
    input_topk_weights: Region,
    shared_l1_sf: Region,
    shared_l2_token: Region,
    shared_l2_sf: Region,
    l1_token: Region,
    l1_sf: Region,
    l1_topk_weights: Region,
    l2_token: Region,
    l2_sf: Region,
    combine_token: Region,

    pub fn init(b: *B, base: V, buf: layout.MegaMoEBuffer) Buffer {
        var out: Buffer = undefined;
        out.workspace = .{ .b = b, .signals = base, .lw = buf.workspace };
        inline for (@typeInfo(Buffer).@"struct".field_names) |f| {
            if (comptime std.mem.eql(u8, f, "workspace")) continue;
            const r: layout.Region = @field(buf, f);
            @field(out, f) = .{ .b = b, .base = offsetBy(b, base, r.offset), .region = r };
        }
        return out;
    }
};

// ==================== comm/barrier.cuh ====================

/// `comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, kGridSyncIndex, kTag>(workspace,
/// sym_buffer, sm_idx, thread_idx, sync_scope, sync_prologue, sync_epilogue)`: grid syncs around
/// a cross-rank barrier on SM 0 (signed `±1` release adds to every rank's signal of the
/// current phase, then an acquire spin on the local one until it reaches `kNumRanks` or 0).
pub fn nvlinkBarrier(
    b: *B,
    ws: Workspace,
    sym: Sym,
    num_sms: u32,
    num_threads: u32,
    comptime grid_sync_index: u32,
    sm_idx: V,
    thread_idx: V,
    sync_scope: anytype,
    sync_prologue: bool,
    sync_epilogue: bool,
    timeout_cycles: i64,
) void {
    std.debug.assert(sym.num_ranks <= num_threads);
    if (sync_prologue) ptx.gridSync(b, num_sms, ws.gridSyncCountPtr(grid_sync_index), sm_idx, thread_idx, sync_scope, timeout_cycles);

    var sm0 = b.openIf(sm_idx.eq(0));
    {
        const counter_ptr = ws.nvlBarrierCounterPtr();
        const status = ptx.ldGlobal(b, 1, counter_ptr)[0].bitAnd(3);
        const signal_phase = status.bitAnd(1);
        const signal_sign = status.shrLogical(1);
        const signal_ptr = ws.nvlBarrierSignalPtr(signal_phase);

        // Send signals to remote ranks
        var send = b.openIf(ptx.ltU(b, thread_idx, @as(i32, @intCast(sym.num_ranks))));
        ptx.redAddRelSys(b, sym.map(signal_ptr, thread_idx), b.select(signal_sign.ne(0), b.cst(.i32, -1), b.cst(.i32, 1)));
        send.yieldThen(.{});
        sync_scope.run(b);

        // Update status and wait arrival
        var leader = b.openIf(thread_idx.eq(0));
        {
            ptx.redAddU32(b, counter_ptr, 1);
            const target = b.select(signal_sign.ne(0), b.cst(.i32, 0), b.cst(.i32, @as(i32, @intCast(sym.num_ranks))));
            ptx.waitUntil(b, timeout_cycles, struct {
                ptr: V,
                target: V,
                pub fn pending(self: @This(), bb: *B) V {
                    return ptx.ldAcqSysS32(bb, self.ptr).ne(self.target);
                }
            }{ .ptr = signal_ptr, .target = target });
            leader.yieldThen(.{});
        }
        sm0.yieldThen(.{});
    }

    if (sync_epilogue) ptx.gridSync(b, num_sms, ws.gridSyncCountPtr(grid_sync_index), sm_idx, thread_idx, sync_scope, timeout_cycles);
}
