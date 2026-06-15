const std = @import("std");

const mlir = @import("mlir");
const zml = @import("zml");
const mtt = @import("kernels/mosaic_tpu/builder");
const Pipeline = mtt.pipeline;
const vector = @import("mlir/dialects").vector;
const tpu = @import("mlir/dialects/mosaic_tpu");

const dynamic = std.math.minInt(i64);

fn allocaVmem(b: *mtt.Builder, shape: []const i64, dtype: mtt.DType) mtt.Value {
    return Pipeline.memRefAlloca(b, mlir.Type.memRef(dtype.toMlir(b.ctx), shape, null, mtt.MemorySpace.vmem.attribute(b.ctx)));
}

fn cdiv(x: i64, y: i64) i64 {
    return @divFloor(x + y - 1, y);
}

fn alignTo(x: i64, y: i64) i64 {
    return cdiv(x, y) * y;
}

fn dtypePacking(dtype: mtt.DType) i64 {
    return switch (dtype) {
        .bf16, .f16 => 2,
        .f32, .i32 => 1,
        .i8 => 4,
        else => 1,
    };
}

fn dmaSemAt2(b: *mtt.Builder, sems: mtt.Value, i: mtt.Value, j: mtt.Value) mtt.Value {
    const sl = b.memRefSlice(sems, &.{ i, j }, &.{ 1, 1 }, &.{});
    return b.memRefSqueeze(sl, &.{});
}

fn dmaSemAt1(b: *mtt.Builder, sems: mtt.Value, i: mtt.Value) mtt.Value {
    const sl = b.memRefSlice(sems, &.{i}, &.{1}, &.{});
    return b.memRefSqueeze(sl, &.{});
}

fn syncBarrier(b: *mtt.Builder, cfg: Cfg) void {
    const sem = b.semBarrier();
    const c1 = b.lift(@as(i32, 1));
    inline for (0..16) |i| {
        if (i < cfg.ep_size) {
            b.semSignalOpts(sem, c1, .{ .device_id = b.lift(@as(i32, @intCast(i))) });
        }
    }
    b.semWait(sem, b.lift(@as(i32, @intCast(cfg.ep_size))));
}

fn fetchGating(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    bt_id: mtt.Value,
    sem_slot: mtt.Value,
    sz: mtt.Value,
    priority: i32,
) void {
    const c0 = b.lift(@as(i32, 0));
    const bt = b.lift(@as(i32, @intCast(cfg.bt)));
    const start = b.muli(bt_id, bt);
    const sz_assumed = b.assumeMultiple(sz, @intCast(cfg.bt));
    const src = b.memRefSlice(a.gating_hbm, &.{ start, c0 }, &.{ dynamic, cfg.paddedNumExperts() }, &.{sz_assumed});
    const dst_sl = b.memRefSlice(a.b_gating_x2_vmem, &.{ sem_slot, c0, c0 }, &.{ 1, dynamic, cfg.paddedNumExperts() }, &.{sz_assumed});
    const dst = b.memRefSqueeze(dst_sl, &.{ dynamic, cfg.paddedNumExperts() });
    b.enqueueDma(src, dst, dmaSemAt2(b, a.local_sems, sem_slot, c0), .{ .priority = priority });
}

fn waitGating(b: *mtt.Builder, cfg: Cfg, a: anytype, sem_slot: mtt.Value) void {
    const c0 = b.lift(@as(i32, 0));
    const src_sl = b.memRefSlice(a.b_gating_x2_vmem, &.{ sem_slot, c0, c0 }, &.{ 1, cfg.bt, cfg.paddedNumExperts() }, &.{});
    const src = b.memRefSqueeze(src_sl, &.{ cfg.bt, cfg.paddedNumExperts() });
    b.waitDma2(dmaSemAt2(b, a.local_sems, sem_slot, c0), src, src, .{});
}

const TopKInfo = struct {
    logits: mtt.Value,
    logits_list: [32]mtt.Value,
    routing: mtt.Value,
    sizes: mtt.Value,
    starts: mtt.Value,
};

fn extToF32(b: *mtt.Builder, x: mtt.Value, src_dtype: mtt.DType) mtt.Value {
    return if (src_dtype == .f32) x else b.arithExtf(x, .f32);
}

fn truncFromF32(b: *mtt.Builder, x: mtt.Value, dst_dtype: mtt.DType) mtt.Value {
    return if (dst_dtype == .f32) x else b.arithTruncf(x, dst_dtype);
}

fn vectorBitcastShape(b: *mtt.Builder, x: mtt.Value, shape: []const i64, dtype: mtt.DType) mtt.Value {
    return b.emit(vector.bitcast(b.ctx, x.inner, b.vectorTy(shape, dtype), b.loc()));
}

fn tpuBitcastShape(b: *mtt.Builder, x: mtt.Value, shape: []const i64, dtype: mtt.DType) mtt.Value {
    return b.emit(tpu.bitcast(b.ctx, x.inner, b.vectorTy(shape, dtype), b.loc()));
}

fn broadcastMinorTiled(b: *mtt.Builder, value: mtt.Value, rows: i64, cols: i64) mtt.Value {
    if (cols <= 128) return b.broadcastTo(value, &.{ rows, cols });
    const tile = b.broadcastTo(value, &.{ rows, 128 });
    const reps = @divExact(cols, 128);
    var parts: [32]mtt.Value = undefined;
    inline for (0..32) |i| {
        if (i < reps) parts[i] = tile;
    }
    return b.concatenate(parts[0..@intCast(reps)], 1);
}

fn scoreAndTopK1(b: *mtt.Builder, cfg: Cfg, gating: mtt.Value) TopKInfo {
    const bt = cfg.bt;
    const padded_num_experts = cfg.paddedNumExperts();
    const padded_top_k = cfg.paddedTopK();

    const gating_2d = b.shapeCast(gating, &.{ bt, padded_num_experts });
    const scored = switch (cfg.scoring_fn) {
        .softmax => blk: {
            const neg_inf_bf16 = b.full(&.{bt}, -std.math.inf(f32), cfg.token_dtype);
            const max_bf16 = b.multiReduction(.maximumf, gating_2d, neg_inf_bf16, &.{1});
            const max_bc = b.broadcastTo(b.shapeCast(max_bf16, &.{ bt, 1 }), &.{ bt, padded_num_experts });
            const shifted = b.subf(gating_2d, max_bc);
            const exp_bf16 = b.exp(shifted);
            const exp_f32 = extToF32(b, exp_bf16, cfg.token_dtype);
            const sum = b.multiReduction(.add, exp_f32, b.zeros(&.{bt}, .f32), &.{1});
            const sum_bf16 = truncFromF32(b, b.shapeCast(sum, &.{ bt, 1 }), cfg.token_dtype);
            const denom = b.broadcastTo(sum_bf16, &.{ bt, padded_num_experts });
            break :blk extToF32(b, b.divf(exp_bf16, denom), cfg.token_dtype);
        },
        .sigmoid => blk: {
            const one = b.splat(@as(f32, 1.0), &.{ bt, padded_num_experts }, cfg.token_dtype);
            const zero = b.zeros(&.{ bt, padded_num_experts }, cfg.token_dtype);
            const exp_neg = b.exp(b.subf(zero, gating_2d));
            break :blk extToF32(b, b.divf(one, b.addf(exp_neg, one)), cfg.token_dtype);
        },
    };

    // The Pallas path for top_k == 1 slices the real expert columns for
    // argmax, then broadcasts the result into padded_top_k-shaped routing and
    // logit buffers.
    const active_scores = b.vectorExtractStridedSlice(scored, &.{ 0, 0 }, &.{ bt, cfg.num_experts }, &.{ 1, 1 });
    const max_logits = b.multiReduction(.maximumf, active_scores, b.full(&.{bt}, -std.math.inf(f32), .f32), &.{1});
    const indices = b.reduceIndex(active_scores, .max, 1);
    const logits = b.broadcastTo(b.shapeCast(max_logits, &.{ bt, 1 }), &.{ bt, padded_top_k });
    const idx_bc = b.broadcastTo(b.shapeCast(indices, &.{ bt, 1 }), &.{ bt, padded_top_k });
    const k_iota = b.iota(&.{ bt, padded_top_k }, .i32, &.{1});
    const valid_k = b.cmpi(.eq, k_iota, b.zeros(&.{ bt, padded_top_k }, .i32));
    const routing = b.select(valid_k, idx_bc, b.zeros(&.{ bt, padded_top_k }, .i32));
    const iota = b.iota(&.{ bt, padded_num_experts }, .i32, &.{1});
    const mask = b.cmpi(.eq, iota, broadcastMinorTiled(b, b.shapeCast(indices, &.{ bt, 1 }), bt, padded_num_experts));
    const t2e = b.extui(mask, .i32);
    const sizes = b.shapeCast(b.multiReduction(.add, t2e, b.zeros(&.{padded_num_experts}, .i32), &.{0}), &.{ 1, padded_num_experts });
    var logits_list: [32]mtt.Value = undefined;
    logits_list[0] = logits;
    return .{
        .logits = logits,
        .logits_list = logits_list,
        .routing = routing,
        .sizes = sizes,
        .starts = b.zeros(&.{ 1, padded_num_experts }, .i32),
    };
}

fn scoreAndTopK(b: *mtt.Builder, cfg: Cfg, gating: mtt.Value) TopKInfo {
    if (cfg.top_k == 1) return scoreAndTopK1(b, cfg, gating);

    const bt = cfg.bt;
    const padded_num_experts = cfg.paddedNumExperts();
    const padded_top_k = cfg.paddedTopK();

    const gating_2d = b.shapeCast(gating, &.{ bt, padded_num_experts });
    var input = switch (cfg.scoring_fn) {
        .softmax => blk: {
            const neg_inf_bf16 = b.full(&.{bt}, -std.math.inf(f32), cfg.token_dtype);
            const max_bf16 = b.multiReduction(.maximumf, gating_2d, neg_inf_bf16, &.{1});
            const max_bc = b.broadcastTo(b.shapeCast(max_bf16, &.{ bt, 1 }), &.{ bt, padded_num_experts });
            const shifted = b.subf(gating_2d, max_bc);
            const exp_bf16 = b.exp(shifted);
            const exp_f32 = extToF32(b, exp_bf16, cfg.token_dtype);
            const sum = b.multiReduction(.add, exp_f32, b.zeros(&.{bt}, .f32), &.{1});
            const sum_bf16 = truncFromF32(b, b.shapeCast(sum, &.{ bt, 1 }), cfg.token_dtype);
            const denom = b.broadcastTo(sum_bf16, &.{ bt, padded_num_experts });
            break :blk extToF32(b, b.divf(exp_bf16, denom), cfg.token_dtype);
        },
        .sigmoid => blk: {
            const one = b.splat(@as(f32, 1.0), &.{ bt, padded_num_experts }, cfg.token_dtype);
            const zero = b.zeros(&.{ bt, padded_num_experts }, cfg.token_dtype);
            const exp_neg = b.exp(b.subf(zero, gating_2d));
            break :blk extToF32(b, b.divf(one, b.addf(exp_neg, one)), cfg.token_dtype);
        },
    };

    const iota = b.iota(&.{ bt, padded_num_experts }, .i32, &.{1});
    const padded_k_iota = b.iota(&.{ bt, padded_top_k }, .i32, &.{1});
    var routing = b.zeros(&.{ bt, padded_top_k }, .i32);
    var t2e = b.zeros(&.{ bt, padded_num_experts }, .i32);
    var logits_planes: [32]mtt.Value = undefined;
    var logits_sum = b.zeros(&.{ bt, padded_top_k }, .f32);

    inline for (0..32) |k_id_usize| {
        if (k_id_usize < cfg.top_k) {
            const max_scores = b.vectorExtractStridedSlice(input, &.{ 0, 0 }, &.{ bt, cfg.num_experts }, &.{ 1, 1 });
            const max_logits = b.multiReduction(.maximumf, max_scores, b.full(&.{bt}, -std.math.inf(f32), .f32), &.{1});
            const max_bc = b.broadcastTo(b.shapeCast(max_logits, &.{ bt, 1 }), &.{ bt, padded_top_k });
            logits_planes[k_id_usize] = max_bc;
            if (cfg.renormalize_topk_logits) logits_sum = b.addf(logits_sum, max_bc);

            const index_scores = b.vectorExtractStridedSlice(input, &.{ 0, 0 }, &.{ bt, cfg.num_experts }, &.{ 1, 1 });
            const indices = b.reduceIndex(index_scores, .max, 1);
            const indices_bc = b.broadcastTo(b.shapeCast(indices, &.{ bt, 1 }), &.{ bt, padded_top_k });
            const k_match = b.cmpi(.eq, padded_k_iota, b.splat(@as(i32, @intCast(k_id_usize)), &.{ bt, padded_top_k }, .i32));
            routing = b.select(k_match, indices_bc, routing);
            const mask = b.cmpi(.eq, iota, broadcastMinorTiled(b, b.shapeCast(indices, &.{ bt, 1 }), bt, padded_num_experts));
            t2e = b.addi(t2e, b.extui(mask, .i32));
            if (k_id_usize != cfg.top_k - 1) {
                input = b.select(mask, b.full(&.{ bt, padded_num_experts }, -std.math.inf(f32), .f32), input);
            }
        }
    }

    if (cfg.renormalize_topk_logits) {
        inline for (0..32) |k_id_usize| {
            if (k_id_usize < cfg.top_k) logits_planes[k_id_usize] = b.divf(logits_planes[k_id_usize], logits_sum);
        }
    }
    const sizes = b.shapeCast(b.multiReduction(.add, t2e, b.zeros(&.{padded_num_experts}, .i32), &.{0}), &.{ 1, padded_num_experts });
    return .{
        .logits = logits_planes[0],
        .logits_list = logits_planes,
        .routing = routing,
        .sizes = sizes,
        .starts = b.zeros(&.{ 1, padded_num_experts }, .i32),
    };
}

fn copyMetadataSingleDevice(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    bt_sem_id: mtt.Value,
    my_id: mtt.Value,
    right_id: mtt.Value,
    info: TopKInfo,
) void {
    const c0 = b.lift(@as(i32, 0));

    const region = b.openRegion();
    {
        const t2e_vmem = allocaVmem(b, &.{ cfg.bt, cfg.paddedTopK() }, .i32);
        const d2e_count_vmem = allocaVmem(b, &.{ cfg.ep_size, 1, cfg.paddedNumExperts() }, .i32);
        const offsets_vmem = allocaVmem(b, &.{ 2, cfg.paddedNumExperts() }, .i32);
        const starts_vmem = allocaVmem(b, &.{ 1, cfg.paddedNumExperts() }, .i32);
        const sizes_vmem = allocaVmem(b, &.{ 1, cfg.paddedNumExperts() }, .i32);
        const send_sem = b.memRefSqueeze(b.memRefSlice(a.send_sems, &.{c0}, &.{1}, &.{}), &.{});

        const offsets_dst_sl = b.memRefSlice(a.expert_offsets_x2_smem, &.{ bt_sem_id, c0, c0 }, &.{ 1, 2, cfg.paddedNumExperts() }, &.{});
        const offsets_dst = b.memRefSqueeze(offsets_dst_sl, &.{ 2, cfg.paddedNumExperts() });
        b.vectorStoreAt(offsets_vmem, b.zeros(&.{ 2, cfg.paddedNumExperts() }, .i32), &.{ b.cIndex(0), b.cIndex(0) });
        b.enqueueDma(offsets_vmem, offsets_dst, send_sem, .{});

        const routing_dst_sl = b.memRefSlice(a.t2e_routing_x2_smem, &.{ bt_sem_id, c0, c0 }, &.{ 1, cfg.bt, cfg.paddedTopK() }, &.{});
        const routing_dst = b.memRefSqueeze(routing_dst_sl, &.{ cfg.bt, cfg.paddedTopK() });
        b.vectorStoreAt(t2e_vmem, info.routing, &.{ b.cIndex(0), b.cIndex(0) });
        b.enqueueDma(t2e_vmem, routing_dst, send_sem, .{});

        var reduced_sizes = info.sizes;
        var reduced_starts = info.starts;
        if (cfg.ep_size == 1) {
            b.vectorStoreAt(d2e_count_vmem, b.shapeCast(info.sizes, &.{ cfg.ep_size, 1, cfg.paddedNumExperts() }), &.{ b.cIndex(0), b.cIndex(0), b.cIndex(0) });
        } else {
            const recv_sem = b.memRefSqueeze(b.memRefSlice(a.recv_sems, &.{c0}, &.{1}, &.{}), &.{});
            b.vectorStoreAt(d2e_count_vmem, b.shapeCast(info.sizes, &.{ 1, 1, cfg.paddedNumExperts() }), &.{ b.toIndex(my_id), b.cIndex(0), b.cIndex(0) });
            var row_id = my_id;
            inline for (0..16) |i_usize| {
                if (i_usize < cfg.ep_size - 1) {
                    syncBarrier(b, cfg);
                    const row_sl = b.memRefSlice(d2e_count_vmem, &.{ row_id, c0, c0 }, &.{ 1, 1, cfg.paddedNumExperts() }, &.{});
                    const row_ref = b.memRefSqueeze(row_sl, &.{ 1, cfg.paddedNumExperts() });
                    b.enqueueDma(row_ref, row_ref, recv_sem, .{ .source_semaphore = send_sem, .device_id = right_id });
                    b.waitDma2(send_sem, row_ref, row_ref, .{ .device_id = right_id });
                    b.waitDma2(recv_sem, row_ref, row_ref, .{ .device_id = right_id });
                    row_id = floorModPos(b, b.addi(row_id, b.lift(@as(i32, @intCast(cfg.ep_size - 1)))), b.lift(@as(i32, @intCast(cfg.ep_size))));
                    const new_sizes_3d = b.vectorLoadShape(d2e_count_vmem, &.{ b.toIndex(row_id), b.cIndex(0), b.cIndex(0) }, &.{ 1, 1, cfg.paddedNumExperts() });
                    const new_sizes = b.shapeCast(new_sizes_3d, &.{ 1, cfg.paddedNumExperts() });
                    reduced_sizes = b.addi(reduced_sizes, new_sizes);
                    const add_to_starts = b.cmpi(.sgt, my_id, b.lift(@as(i32, @intCast(i_usize))));
                    reduced_starts = b.addi(reduced_starts, b.select(add_to_starts, new_sizes, b.zeros(&.{ 1, cfg.paddedNumExperts() }, .i32)));
                }
            }
        }

        const starts_dst_sl = b.memRefSlice(a.expert_starts_x2_smem, &.{ bt_sem_id, c0, c0 }, &.{ 1, 1, cfg.paddedNumExperts() }, &.{});
        const starts_dst = b.memRefSqueeze(starts_dst_sl, &.{ 1, cfg.paddedNumExperts() });
        b.vectorStoreAt(starts_vmem, reduced_starts, &.{ b.cIndex(0), b.cIndex(0) });

        const sizes_dst_sl = b.memRefSlice(a.expert_sizes_x2_smem, &.{ bt_sem_id, c0, c0 }, &.{ 1, 1, cfg.paddedNumExperts() }, &.{});
        const sizes_dst = b.memRefSqueeze(sizes_dst_sl, &.{ 1, cfg.paddedNumExperts() });
        b.vectorStoreAt(sizes_vmem, reduced_sizes, &.{ b.cIndex(0), b.cIndex(0) });

        b.enqueueDma(starts_vmem, starts_dst, send_sem, .{});
        b.enqueueDma(sizes_vmem, sizes_dst, send_sem, .{});

        const counts_dst_sl = b.memRefSlice(a.d2e_count_x2_smem, &.{ bt_sem_id, c0, c0, c0 }, &.{ 1, cfg.ep_size, 1, cfg.paddedNumExperts() }, &.{});
        const counts_dst = b.memRefSqueeze(counts_dst_sl, &.{ cfg.ep_size, 1, cfg.paddedNumExperts() });
        b.enqueueDma(d2e_count_vmem, counts_dst, send_sem, .{});

        b.waitDma2(send_sem, t2e_vmem, routing_dst, .{});
        b.waitDma2(send_sem, d2e_count_vmem, counts_dst, .{});
        b.waitDma2(send_sem, offsets_vmem, offsets_dst, .{});
        b.waitDma2(send_sem, starts_vmem, starts_dst, .{});
        b.waitDma2(send_sem, sizes_vmem, sizes_dst, .{});
    }
    b.closeRegion(region);
}

fn floorModPos(b: *mtt.Builder, x: mtt.Value, y: mtt.Value) mtt.Value {
    const c0 = b.lift(@as(i32, 0));
    const r = b.remsi(x, y);
    const add = b.addi(r, y);
    const r_neg = b.cmpi(.slt, r, c0);
    const r_nz = b.cmpi(.ne, r, c0);
    return b.select(b.andi(r_neg, r_nz), add, r);
}

fn floorDivScalar(b: *mtt.Builder, x: mtt.Value, y: mtt.Value) mtt.Value {
    const c0 = b.lift(@as(i32, 0));
    const c1 = b.lift(@as(i32, 1));
    const q = b.divsi(x, y);
    const sign_x = b.subi(b.extui(b.cmpi(.sgt, x, c0), .i32), b.extui(b.cmpi(.slt, x, c0), .i32));
    const sign_y = b.subi(b.extui(b.cmpi(.sgt, y, c0), .i32), b.extui(b.cmpi(.slt, y, c0), .i32));
    const sign_diff = b.cmpi(.ne, sign_x, sign_y);
    const rem_nz = b.cmpi(.ne, b.remsi(x, y), c0);
    return b.select(b.andi(sign_diff, rem_nz), b.subi(q, c1), q);
}

fn myDeviceId(b: *mtt.Builder, cfg: Cfg) mtt.Value {
    return if (cfg.ep_size == 1) b.lift(@as(i32, 0)) else b.remsi(b.deviceId(), b.lift(@as(i32, @intCast(cfg.ep_size))));
}

fn startA2aScatter(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    bt_id: mtt.Value,
    e_sem_id: mtt.Value,
    local_e_id: mtt.Value,
) void {
    @setEvalBranchQuota(100_000);
    const c0 = b.lift(@as(i32, 0));
    const c1 = b.lift(@as(i32, 1));
    const local_num_experts = b.lift(@as(i32, @intCast(cfg.localNumExperts())));
    const bt_sem_id = floorModPos(b, bt_id, b.lift(@as(i32, 2)));
    const my_id = myDeviceId(b, cfg);
    const send_sem = b.memRefSqueeze(b.memRefSlice(a.send_sems, &.{e_sem_id}, &.{1}, &.{}), &.{});
    const recv_sem = b.memRefSqueeze(b.memRefSlice(a.recv_sems, &.{e_sem_id}, &.{1}, &.{}), &.{});
    var send_sz = c0;

    inline for (0..64) |bt_t_id_usize| {
        if (bt_t_id_usize < cfg.bt) {
            const bt_t_id = b.lift(@as(i32, @intCast(bt_t_id_usize)));
            inline for (0..32) |k_id_usize| {
                if (k_id_usize < cfg.top_k) {
                    const k_id = b.lift(@as(i32, @intCast(k_id_usize)));
                    const e_id = b.scalarLoad(a.t2e_routing_x2_smem, &.{ b.toIndex(bt_sem_id), b.toIndex(bt_t_id), b.toIndex(k_id) });
                    const recv_id = floorDivScalar(b, e_id, local_num_experts);
                    const e_rem = floorModPos(b, e_id, local_num_experts);
                    const active = b.cmpi(.eq, e_rem, local_e_id);
                    const sz = b.select(active, c1, c0);
                    const is_local = b.cmpi(.eq, recv_id, my_id);
                    const local_sz = b.select(is_local, sz, c0);
                    const remote_sz = b.select(is_local, c0, sz);
                    send_sz = b.addi(send_sz, remote_sz);

                    const e_idx = b.toIndex(e_id);
                    const offset = b.scalarLoad(a.expert_offsets_x2_smem, &.{ b.toIndex(bt_sem_id), b.cIndex(0), e_idx });
                    const next_offset = b.addi(b.addi(offset, local_sz), remote_sz);
                    b.scalarStore(a.expert_offsets_x2_smem, next_offset, &.{ b.toIndex(bt_sem_id), b.cIndex(0), e_idx });
                    const start = b.addi(b.scalarLoad(a.expert_starts_x2_smem, &.{ b.toIndex(bt_sem_id), b.cIndex(0), e_idx }), offset);
                    const t_id = b.addi(b.muli(bt_id, b.lift(@as(i32, @intCast(cfg.bt)))), bt_t_id);

                    const local_src = b.memRefSlice(a.tokens_hbm, &.{ t_id, c0, c0 }, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{local_sz});
                    const local_dst_sl = b.memRefSlice(a.a2a_s_x2_vmem, &.{ e_sem_id, start, c0, c0 }, &.{ 1, dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{local_sz});
                    const local_dst = b.memRefSqueeze(local_dst_sl, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
                    b.enqueueDma(local_src, local_dst, recv_sem, .{});

                    const remote_src = b.memRefSlice(a.tokens_hbm, &.{ t_id, c0, c0 }, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{remote_sz});
                    const remote_dst_sl = b.memRefSlice(a.a2a_s_x2_vmem, &.{ e_sem_id, start, c0, c0 }, &.{ 1, dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{remote_sz});
                    const remote_dst = b.memRefSqueeze(remote_dst_sl, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
                    b.enqueueDma(remote_src, remote_dst, recv_sem, .{ .source_semaphore = send_sem, .device_id = recv_id });
                }
            }
        }
    }

    b.scalarStore(a.a2a_s_sends_x2_smem, send_sz, &.{b.toIndex(e_sem_id)});
}

fn waitA2aScatterRecv(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    bt_id: mtt.Value,
    e_sem_id: mtt.Value,
    local_e_id: mtt.Value,
) void {
    const c0 = b.lift(@as(i32, 0));
    const local_num_experts = b.lift(@as(i32, @intCast(cfg.localNumExperts())));
    const bt_sem_id = floorModPos(b, bt_id, b.lift(@as(i32, 2)));
    const my_e_id = b.addi(b.muli(myDeviceId(b, cfg), local_num_experts), local_e_id);
    const sz = b.scalarLoad(a.expert_sizes_x2_smem, &.{ b.toIndex(bt_sem_id), b.cIndex(0), b.toIndex(my_e_id) });
    const ref_sl = b.memRefSlice(a.a2a_s_x2_vmem, &.{ e_sem_id, c0, c0, c0 }, &.{ 1, dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{sz});
    const ref = b.memRefSqueeze(ref_sl, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
    const recv_sem = b.memRefSqueeze(b.memRefSlice(a.recv_sems, &.{e_sem_id}, &.{1}, &.{}), &.{});
    b.waitDma2(recv_sem, ref, ref, .{});
}

fn waitA2aScatterSend(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    e_sem_id: mtt.Value,
) void {
    const c0 = b.lift(@as(i32, 0));
    const sz = b.scalarLoad(a.a2a_s_sends_x2_smem, &.{b.toIndex(e_sem_id)});
    const ref_sl = b.memRefSlice(a.a2a_s_x2_vmem, &.{ e_sem_id, c0, c0, c0 }, &.{ 1, dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{sz});
    const ref = b.memRefSqueeze(ref_sl, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
    const send_sem = b.memRefSqueeze(b.memRefSlice(a.send_sems, &.{e_sem_id}, &.{1}, &.{}), &.{});
    b.waitDma2(send_sem, ref, ref, .{});
}

fn startFetchBw1(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    local_e_id: mtt.Value,
    sem_id: mtt.Value,
    bf_id: mtt.Value,
    bd1_id: mtt.Value,
) void {
    const bd1_per_pack = @divExact(cfg.bd1, cfg.tokenPacking());
    const sem = dmaSemAt2(b, a.local_sems, sem_id, b.lift(@as(i32, 1)));
    inline for (0..4) |p_usize| {
        if (p_usize < cfg.tokenPacking()) {
            const p = b.lift(@as(i32, @intCast(p_usize)));
            const offset = b.addi(b.muli(p, b.lift(@as(i32, @intCast(@divExact(cfg.hidden_size, cfg.tokenPacking()))))), b.muli(bd1_id, b.lift(@as(i32, @intCast(bd1_per_pack)))));
            const col = b.muli(bf_id, b.lift(@as(i32, @intCast(cfg.bf))));
            const src = b.memRefSlice(a.w1_hbm, &.{ local_e_id, b.lift(@as(i32, 0)), offset, col }, &.{ 1, 1, bd1_per_pack, cfg.bf }, &.{});
            const src_sq = b.memRefSqueeze(src, &.{ bd1_per_pack, cfg.bf });
            const dst_sl = b.memRefSlice(a.b_w1_x2_vmem, &.{ sem_id, p, b.lift(@as(i32, 0)), b.lift(@as(i32, 0)) }, &.{ 1, 1, bd1_per_pack, cfg.bf }, &.{});
            const dst = b.memRefSqueeze(dst_sl, &.{ bd1_per_pack, cfg.bf });
            b.enqueueDma(src_sq, dst, sem, .{});
        }
    }
}

fn startFetchBw3(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    local_e_id: mtt.Value,
    sem_id: mtt.Value,
    bf_id: mtt.Value,
    bd1_id: mtt.Value,
) void {
    const bd1_per_pack = @divExact(cfg.bd1, cfg.tokenPacking());
    const sem = dmaSemAt2(b, a.local_sems, sem_id, b.lift(@as(i32, 3)));
    inline for (0..4) |p_usize| {
        if (p_usize < cfg.tokenPacking()) {
            const p = b.lift(@as(i32, @intCast(p_usize)));
            const offset = b.addi(b.muli(p, b.lift(@as(i32, @intCast(@divExact(cfg.hidden_size, cfg.tokenPacking()))))), b.muli(bd1_id, b.lift(@as(i32, @intCast(bd1_per_pack)))));
            const col = b.muli(bf_id, b.lift(@as(i32, @intCast(cfg.bf))));
            const src = b.memRefSlice(a.w1_hbm, &.{ local_e_id, b.lift(@as(i32, 1)), offset, col }, &.{ 1, 1, bd1_per_pack, cfg.bf }, &.{});
            const src_sq = b.memRefSqueeze(src, &.{ bd1_per_pack, cfg.bf });
            const dst_sl = b.memRefSlice(a.b_w3_x2_vmem, &.{ sem_id, p, b.lift(@as(i32, 0)), b.lift(@as(i32, 0)) }, &.{ 1, 1, bd1_per_pack, cfg.bf }, &.{});
            const dst = b.memRefSqueeze(dst_sl, &.{ bd1_per_pack, cfg.bf });
            b.enqueueDma(src_sq, dst, sem, .{});
        }
    }
}

fn startFetchBw2(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    local_e_id: mtt.Value,
    sem_id: mtt.Value,
    bf_id: mtt.Value,
    bd2_id: mtt.Value,
) void {
    const bd2_per_pack = @divExact(cfg.bd2, cfg.tokenPacking());
    const sem = dmaSemAt2(b, a.local_sems, sem_id, b.lift(@as(i32, 2)));
    inline for (0..4) |p_usize| {
        if (p_usize < cfg.tokenPacking()) {
            const p = b.lift(@as(i32, @intCast(p_usize)));
            const row = b.muli(bf_id, b.lift(@as(i32, @intCast(cfg.bf))));
            const offset = b.addi(b.muli(p, b.lift(@as(i32, @intCast(@divExact(cfg.hidden_size, cfg.tokenPacking()))))), b.muli(bd2_id, b.lift(@as(i32, @intCast(bd2_per_pack)))));
            const src = b.memRefSlice(a.w2_hbm, &.{ local_e_id, row, offset }, &.{ 1, cfg.bf, bd2_per_pack }, &.{});
            const src_sq = b.memRefSqueeze(src, &.{ cfg.bf, bd2_per_pack });
            const dst_sl = b.memRefSlice(a.b_w2_x2_vmem, &.{ sem_id, p, b.lift(@as(i32, 0)), b.lift(@as(i32, 0)) }, &.{ 1, 1, cfg.bf, bd2_per_pack }, &.{});
            const dst = b.memRefSqueeze(dst_sl, &.{ cfg.bf, bd2_per_pack });
            b.enqueueDma(src_sq, dst, sem, .{});
        }
    }
}

fn waitFetchBw1(b: *mtt.Builder, cfg: Cfg, a: anytype, sem_id: mtt.Value) void {
    const ref_sl = b.memRefSlice(a.b_w1_x2_vmem, &.{ sem_id, b.lift(@as(i32, 0)), b.lift(@as(i32, 0)), b.lift(@as(i32, 0)) }, &.{ 1, cfg.tokenPacking(), @divExact(cfg.bd1, cfg.tokenPacking()), cfg.bf }, &.{});
    const ref = b.memRefSqueeze(ref_sl, &.{ cfg.tokenPacking(), @divExact(cfg.bd1, cfg.tokenPacking()), cfg.bf });
    const sem = dmaSemAt2(b, a.local_sems, sem_id, b.lift(@as(i32, 1)));
    b.waitDma2(sem, ref, ref, .{});
}

fn waitFetchBw2(b: *mtt.Builder, cfg: Cfg, a: anytype, sem_id: mtt.Value) void {
    const ref_sl = b.memRefSlice(a.b_w2_x2_vmem, &.{ sem_id, b.lift(@as(i32, 0)), b.lift(@as(i32, 0)), b.lift(@as(i32, 0)) }, &.{ 1, cfg.tokenPacking(), cfg.bf, @divExact(cfg.bd2, cfg.tokenPacking()) }, &.{});
    const ref = b.memRefSqueeze(ref_sl, &.{ cfg.tokenPacking(), cfg.bf, @divExact(cfg.bd2, cfg.tokenPacking()) });
    const sem = dmaSemAt2(b, a.local_sems, sem_id, b.lift(@as(i32, 2)));
    b.waitDma2(sem, ref, ref, .{});
}

fn waitFetchBw3(b: *mtt.Builder, cfg: Cfg, a: anytype, sem_id: mtt.Value) void {
    const ref_sl = b.memRefSlice(a.b_w3_x2_vmem, &.{ sem_id, b.lift(@as(i32, 0)), b.lift(@as(i32, 0)), b.lift(@as(i32, 0)) }, &.{ 1, cfg.tokenPacking(), @divExact(cfg.bd1, cfg.tokenPacking()), cfg.bf }, &.{});
    const ref = b.memRefSqueeze(ref_sl, &.{ cfg.tokenPacking(), @divExact(cfg.bd1, cfg.tokenPacking()), cfg.bf });
    const sem = dmaSemAt2(b, a.local_sems, sem_id, b.lift(@as(i32, 3)));
    b.waitDma2(sem, ref, ref, .{});
}

fn silu(b: *mtt.Builder, x: mtt.Value) mtt.Value {
    const shape = x.shape().constSlice();
    const one = b.splat(@as(f32, 1.0), shape, .f32);
    const sigmoid = b.divf(one, b.addf(one, b.exp(b.negf(x))));
    return b.mulf(x, sigmoid);
}

fn gelu(b: *mtt.Builder, x: mtt.Value) mtt.Value {
    const shape = x.shape().constSlice();
    const half = b.splat(@as(f32, 0.5), shape, .f32);
    const one = b.splat(@as(f32, 1.0), shape, .f32);
    const cubic_coeff = b.splat(@as(f32, 0.044715), shape, .f32);
    const sqrt_two_over_pi = b.splat(@as(f32, 0.7978845608028654), shape, .f32);
    const x2 = b.mulf(x, x);
    const x3 = b.mulf(x, x2);
    const cubic = b.mulf(cubic_coeff, x3);
    const inner = b.addf(x, cubic);
    const scaled = b.mulf(sqrt_two_over_pi, inner);
    const t = b.tanh(scaled);
    const prob = b.mulf(half, b.addf(one, t));
    return b.mulf(x, prob);
}

fn swigluoai(b: *mtt.Builder, gate: mtt.Value, up: mtt.Value) mtt.Value {
    const shape = gate.shape().constSlice();
    const neg_limit = b.splat(@as(f32, -7.0), shape, .f32);
    const limit = b.splat(@as(f32, 7.0), shape, .f32);
    const alpha = b.splat(@as(f32, 1.702), shape, .f32);
    const one = b.splat(@as(f32, 1.0), shape, .f32);
    const clipped_gate = b.minimumf(gate, limit);
    const sigmoid = b.divf(one, b.addf(one, b.exp(b.negf(b.mulf(alpha, clipped_gate)))));
    const glu = b.mulf(clipped_gate, sigmoid);
    const clipped_up = b.minimumf(b.maximumf(up, neg_limit), limit);
    return b.mulf(b.addf(clipped_up, one), glu);
}

fn applyActFn(b: *mtt.Builder, cfg: Cfg, gate: mtt.Value, up: mtt.Value) mtt.Value {
    return switch (cfg.act_fn) {
        .silu => b.mulf(silu(b, gate), up),
        .gelu => b.mulf(gelu(b, gate), up),
        .swigluoai => swigluoai(b, gate, up),
    };
}

fn matmul2d(b: *mtt.Builder, lhs: mtt.Value, rhs: mtt.Value, acc: mtt.Value) mtt.Value {
    return b.matmulOpts(lhs, rhs, acc, .{
        .dimension_numbers = b.dotDimensionNumbers(&.{1}, &.{0}, &.{0}, &.{1}, &.{ 0, 0, 1, 1 }, &.{}, &.{}),
    });
}

fn weightBuffer(
    b: *mtt.Builder,
    mem: mtt.Value,
    sem_id: mtt.Value,
    packs: i64,
    rows: i64,
    cols: i64,
) mtt.Value {
    const c0 = b.lift(@as(i32, 0));
    const sl = b.memRefSlice(mem, &.{ sem_id, c0, c0, c0 }, &.{ 1, packs, rows, cols }, &.{});
    return b.memRefSqueeze(sl, &.{ packs, rows, cols });
}

fn loadWeightPack(
    b: *mtt.Builder,
    mem: mtt.Value,
    pack_id: comptime_int,
    rows: i64,
    cols: i64,
) mtt.Value {
    return b.shapeCast(
        b.vectorLoadShape(mem, &.{ b.cIndex(@intCast(pack_id)), b.cIndex(0), b.cIndex(0) }, &.{ 1, rows, cols }),
        &.{ rows, cols },
    );
}

fn expertFfnBaseline(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    bt_id: mtt.Value,
    e_sem_id: mtt.Value,
    local_e_id: mtt.Value,
) void {
    @setEvalBranchQuota(100_000);
    const c0i = b.cIndex(0);
    const c0 = b.lift(@as(i32, 0));
    const c1 = b.lift(@as(i32, 1));
    const t_pack = cfg.tokenPacking();
    const hidden_per_pack = @divExact(cfg.hidden_size, t_pack);
    const bd1_per_pack = @divExact(cfg.bd1, t_pack);
    const bd2_per_pack = @divExact(cfg.bd2, t_pack);
    const bd1_per_pack_usize: usize = @intCast(bd1_per_pack);
    const bd2_per_pack_usize: usize = @intCast(bd2_per_pack);
    const num_bf = @divExact(cfg.intermediate_size, cfg.bf);
    const num_bd1 = @divExact(cfg.hidden_size, cfg.bd1);
    const num_bd2 = @divExact(cfg.hidden_size, cfg.bd2);
    const num_bf_usize: usize = @intCast(num_bf);
    const num_bd1_usize: usize = @intCast(num_bd1);
    const num_bd2_usize: usize = @intCast(num_bd2);
    const local_num_experts = b.lift(@as(i32, @intCast(cfg.localNumExperts())));
    const my_id = myDeviceId(b, cfg);
    const my_e_id = b.addi(b.muli(my_id, local_num_experts), local_e_id);
    const bt_sem_id = floorModPos(b, bt_id, b.lift(@as(i32, 2)));
    const dyn_sz = b.scalarLoad(a.expert_sizes_x2_smem, &.{ b.toIndex(bt_sem_id), b.cIndex(0), b.toIndex(my_e_id) });
    const num_loops = b.divsi(b.addi(dyn_sz, b.lift(@as(i32, @intCast(cfg.btc - 1)))), b.lift(@as(i32, @intCast(cfg.btc))));

    inline for (0..8) |bf_id_usize| {
        if (bf_id_usize < num_bf_usize) {
            inline for (0..16) |bd1_id_usize| {
                if (bd1_id_usize < num_bd1_usize) {
                    const cur_sem = b.lift(@as(i32, @intCast((bf_id_usize * (num_bd1_usize + num_bd2_usize) + bd1_id_usize) % 2)));
                    const next_sem = b.lift(@as(i32, @intCast((bf_id_usize * (num_bd1_usize + num_bd2_usize) + bd1_id_usize + 1) % 2)));
                    if (bd1_id_usize + 1 < num_bd1_usize) {
                        startFetchBw1(b, cfg, a, local_e_id, next_sem, b.lift(@as(i32, @intCast(bf_id_usize))), b.lift(@as(i32, @intCast(bd1_id_usize + 1))));
                        startFetchBw3(b, cfg, a, local_e_id, next_sem, b.lift(@as(i32, @intCast(bf_id_usize))), b.lift(@as(i32, @intCast(bd1_id_usize + 1))));
                    } else {
                        startFetchBw2(b, cfg, a, local_e_id, next_sem, b.lift(@as(i32, @intCast(bf_id_usize))), c0);
                    }

                    waitFetchBw1(b, cfg, a, cur_sem);
                    waitFetchBw3(b, cfg, a, cur_sem);

                    var loop = b.openFor(c0, num_loops, c1, .{});
                    const b_acc_2d = b.memRefReshape(a.b_acc_vmem, &.{ 2, cfg.bt * cfg.ep_size, cfg.bf });
                    const a2a_s_b32_rank4 = b.memRefBitcastShape(a.a2a_s_x2_vmem, &.{ 2, cfg.bt * cfg.ep_size, 1, hidden_per_pack }, .i32);
                    const row = b.muli(b.lift(@as(i32, @intCast(cfg.btc))), loop.iv);
                    const row_idx = b.toIndex(row);
                    const a2a_s_b32 = b.memRefReshape(a2a_s_b32_rank4, &.{ 2, cfg.bt * cfg.ep_size, hidden_per_pack });
                    const a2a_s_b32_sl = b.memRefSlice(a2a_s_b32, &.{ e_sem_id, c0, c0 }, &.{ 1, cfg.bt * cfg.ep_size, hidden_per_pack }, &.{});
                    const a2a_s_b32_e = b.memRefSqueeze(a2a_s_b32_sl, &.{ cfg.bt * cfg.ep_size, hidden_per_pack });
                    const w1_buf = weightBuffer(b, a.b_w1_x2_vmem, cur_sem, t_pack, bd1_per_pack, cfg.bf);
                    const w3_buf = weightBuffer(b, a.b_w3_x2_vmem, cur_sem, t_pack, bd1_per_pack, cfg.bf);
                    const b_acc1_sl = b.memRefSlice(b_acc_2d, &.{ c0, c0, c0 }, &.{ 1, cfg.bt * cfg.ep_size, cfg.bf }, &.{});
                    const b_acc3_sl = b.memRefSlice(b_acc_2d, &.{ c1, c0, c0 }, &.{ 1, cfg.bt * cfg.ep_size, cfg.bf }, &.{});
                    const b_acc1 = b.memRefSqueeze(b_acc1_sl, &.{ cfg.bt * cfg.ep_size, cfg.bf });
                    const b_acc3 = b.memRefSqueeze(b_acc3_sl, &.{ cfg.bt * cfg.ep_size, cfg.bf });
                    const col_start = b.lift(@as(i32, @intCast(bd1_id_usize * bd1_per_pack_usize)));
                    const a2a_s_b32_tile = if (num_bd1_usize == 1)
                        a2a_s_b32_e
                    else
                        b.memRefSlice(a2a_s_b32_e, &.{ c0, col_start }, &.{ cfg.bt * cfg.ep_size, bd1_per_pack }, &.{});
                    const t_b32 = b.vectorLoadShape(a2a_s_b32_tile, &.{ row_idx, c0i }, &.{ cfg.btc, bd1_per_pack });

                    inline for (0..4) |p_usize| {
                        if (p_usize < cfg.tokenPacking()) {
                            const shifted = if (p_usize == 0) t_b32 else b.shrui(t_b32, b.splat(@as(i32, 16), &.{ cfg.btc, bd1_per_pack }, .i32));
                            const token = if (cfg.token_dtype == .f32)
                                b.bitcastTo(shifted, .f32)
                            else
                                b.bitcastTo(b.trunci(shifted, .i16), cfg.token_dtype);
                            const w1 = loadWeightPack(b, w1_buf, p_usize, bd1_per_pack, cfg.bf);
                            const w3 = loadWeightPack(b, w3_buf, p_usize, bd1_per_pack, cfg.bf);
                            const acc1_part = matmul2d(b, token, w1, b.zeros(&.{ cfg.btc, cfg.bf }, .f32));
                            const acc3_part = matmul2d(b, token, w3, b.zeros(&.{ cfg.btc, cfg.bf }, .f32));
                            if (bd1_id_usize == 0 and p_usize == 0) {
                                b.vectorStoreAt(b_acc1, acc1_part, &.{ row_idx, c0i });
                                b.vectorStoreAt(b_acc3, acc3_part, &.{ row_idx, c0i });
                            } else {
                                const old1 = b.vectorLoadShape(b_acc1, &.{ row_idx, c0i }, &.{ cfg.btc, cfg.bf });
                                b.vectorStoreAt(b_acc1, b.addf(old1, acc1_part), &.{ row_idx, c0i });
                                const old3 = b.vectorLoadShape(b_acc3, &.{ row_idx, c0i }, &.{ cfg.btc, cfg.bf });
                                b.vectorStoreAt(b_acc3, b.addf(old3, acc3_part), &.{ row_idx, c0i });
                            }
                        }
                    }
                    loop.yield(.{});
                }
            }

            inline for (0..16) |bd2_id_usize| {
                if (bd2_id_usize < num_bd2_usize) {
                    const cur_idx = bf_id_usize * (num_bd1_usize + num_bd2_usize) + num_bd1_usize + bd2_id_usize;
                    const cur_sem = b.lift(@as(i32, @intCast(cur_idx % 2)));
                    const next_sem = b.lift(@as(i32, @intCast((cur_idx + 1) % 2)));
                    if (bd2_id_usize + 1 < num_bd2) {
                        startFetchBw2(b, cfg, a, local_e_id, next_sem, b.lift(@as(i32, @intCast(bf_id_usize))), b.lift(@as(i32, @intCast(bd2_id_usize + 1))));
                    } else if (bf_id_usize + 1 < num_bf_usize) {
                        startFetchBw1(b, cfg, a, local_e_id, next_sem, b.lift(@as(i32, @intCast(bf_id_usize + 1))), c0);
                        startFetchBw3(b, cfg, a, local_e_id, next_sem, b.lift(@as(i32, @intCast(bf_id_usize + 1))), c0);
                    }

                    waitFetchBw2(b, cfg, a, cur_sem);
                    if (bf_id_usize == 0 and bd2_id_usize == 0) {
                        waitA2aGatherSend(b, cfg, a, bt_id, e_sem_id, b.subi(local_e_id, b.lift(@as(i32, 2))));
                    }

                    var loop = b.openFor(c0, num_loops, c1, .{});
                    const b_acc_2d = b.memRefReshape(a.b_acc_vmem, &.{ 2, cfg.bt * cfg.ep_size, cfg.bf });
                    const a2a_s_acc_b32_rank4 = b.memRefBitcastShape(a.a2a_s_acc_x2_vmem, &.{ 2, cfg.bt * cfg.ep_size, 1, hidden_per_pack }, .i32);
                    const row = b.muli(b.lift(@as(i32, @intCast(cfg.btc))), loop.iv);
                    const row_idx = b.toIndex(row);
                    const a2a_s_acc_b32 = b.memRefReshape(a2a_s_acc_b32_rank4, &.{ 2, cfg.bt * cfg.ep_size, hidden_per_pack });
                    const a2a_s_acc_b32_sl = b.memRefSlice(a2a_s_acc_b32, &.{ e_sem_id, c0, c0 }, &.{ 1, cfg.bt * cfg.ep_size, hidden_per_pack }, &.{});
                    const a2a_s_acc_b32_e = b.memRefSqueeze(a2a_s_acc_b32_sl, &.{ cfg.bt * cfg.ep_size, hidden_per_pack });
                    const w2_buf = weightBuffer(b, a.b_w2_x2_vmem, cur_sem, t_pack, cfg.bf, bd2_per_pack);
                    const b_acc1_sl = b.memRefSlice(b_acc_2d, &.{ c0, c0, c0 }, &.{ 1, cfg.bt * cfg.ep_size, cfg.bf }, &.{});
                    const b_acc3_sl = b.memRefSlice(b_acc_2d, &.{ c1, c0, c0 }, &.{ 1, cfg.bt * cfg.ep_size, cfg.bf }, &.{});
                    const b_acc1 = b.memRefSqueeze(b_acc1_sl, &.{ cfg.bt * cfg.ep_size, cfg.bf });
                    const b_acc3 = b.memRefSqueeze(b_acc3_sl, &.{ cfg.bt * cfg.ep_size, cfg.bf });
                    const acc1 = b.vectorLoadShape(b_acc1, &.{ row_idx, c0i }, &.{ cfg.btc, cfg.bf });
                    const acc3 = b.vectorLoadShape(b_acc3, &.{ row_idx, c0i }, &.{ cfg.btc, cfg.bf });
                    const act = applyActFn(b, cfg, acc1, acc3);
                    var res_b32 = b.zeros(&.{ cfg.btc, bd2_per_pack }, .i32);
                    inline for (0..4) |p_usize| {
                        if (p_usize < cfg.tokenPacking()) {
                            const w2 = loadWeightPack(b, w2_buf, p_usize, cfg.bf, bd2_per_pack);
                            const res = matmul2d(b, act, w2, b.zeros(&.{ cfg.btc, bd2_per_pack }, .f32));
                            const shifted = if (cfg.token_dtype == .f32)
                                b.bitcastTo(res, .i32)
                            else blk: {
                                const res_u32 = b.shrui(b.bitcastTo(res, .i32), b.splat(@as(i32, 16), &.{ cfg.btc, bd2_per_pack }, .i32));
                                break :blk if (p_usize == 0) res_u32 else b.shli(res_u32, b.splat(@as(i32, 16), &.{ cfg.btc, bd2_per_pack }, .i32));
                            };
                            res_b32 = b.ori(res_b32, shifted);
                        }
                    }
                    const col_start = b.lift(@as(i32, @intCast(bd2_id_usize * bd2_per_pack_usize)));
                    const a2a_s_acc_b32_e_tile = if (num_bd2_usize == 1)
                        a2a_s_acc_b32_e
                    else
                        b.memRefSlice(a2a_s_acc_b32_e, &.{ c0, col_start }, &.{ cfg.bt * cfg.ep_size, bd2_per_pack }, &.{});
                    if (bf_id_usize == 0) {
                        b.vectorStoreAt(a2a_s_acc_b32_e_tile, res_b32, &.{ row_idx, c0i });
                    } else {
                        const packed_res = if (cfg.token_dtype == .f32) blk: {
                            const old_b32 = b.vectorLoadShape(a2a_s_acc_b32_e_tile, &.{ row_idx, c0i }, &.{ cfg.btc, bd2_per_pack });
                            const old_token = b.bitcastTo(old_b32, .f32);
                            const res_token = b.bitcastTo(res_b32, .f32);
                            break :blk b.bitcastTo(b.addf(old_token, res_token), .i32);
                        } else blk: {
                            const row_tile = b.memRefSlice(a2a_s_acc_b32_e_tile, &.{ row, c0 }, &.{ cfg.btc, bd2_per_pack }, &.{});
                            const old_ref = b.memRefBitcastShape(row_tile, &.{ cfg.btc * t_pack, bd2_per_pack }, cfg.token_dtype);
                            const old_token = b.vectorLoadShape(old_ref, &.{ b.cIndex(0), b.cIndex(0) }, &.{ cfg.btc * t_pack, bd2_per_pack });
                            const res_token = tpuBitcastShape(b, res_b32, &.{ cfg.btc * t_pack, bd2_per_pack }, cfg.token_dtype);
                            break :blk tpuBitcastShape(b, b.addf(old_token, res_token), &.{ cfg.btc, bd2_per_pack }, .i32);
                        };
                        b.vectorStoreAt(a2a_s_acc_b32_e_tile, packed_res, &.{ row_idx, c0i });
                    }
                    loop.yield(.{});
                }
            }
        }
    }
}

fn startA2aGather(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    bt_id: mtt.Value,
    e_sem_id: mtt.Value,
    local_e_id: mtt.Value,
) void {
    const c0 = b.lift(@as(i32, 0));
    const local_num_experts = b.lift(@as(i32, @intCast(cfg.localNumExperts())));
    const my_id = myDeviceId(b, cfg);
    const my_e_id = b.addi(b.muli(my_id, local_num_experts), local_e_id);
    const bt_sem_id = floorModPos(b, bt_id, b.lift(@as(i32, 2)));
    var start = c0;
    inline for (0..16) |recv_id_usize| {
        if (recv_id_usize < cfg.ep_size) {
            const recv_id = b.lift(@as(i32, @intCast(recv_id_usize)));
            const sz = b.scalarLoad(a.d2e_count_x2_smem, &.{ b.toIndex(bt_sem_id), b.toIndex(recv_id), b.cIndex(0), b.toIndex(my_e_id) });
            const is_local = b.cmpi(.eq, recv_id, my_id);
            const local_sz = b.select(is_local, sz, c0);
            const remote_sz = b.select(is_local, c0, sz);

            const local_src_sl = b.memRefSlice(a.a2a_s_acc_x2_vmem, &.{ e_sem_id, start, c0, c0 }, &.{ 1, dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{local_sz});
            const local_src = b.memRefSqueeze(local_src_sl, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
            const local_dst_sl = b.memRefSlice(a.a2a_g_hbm, &.{ my_e_id, c0, c0, c0 }, &.{ 1, dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{local_sz});
            const local_dst = b.memRefSqueeze(local_dst_sl, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
            b.enqueueDma(local_src, local_dst, a.a2a_gather_sem, .{});

            const remote_src_sl = b.memRefSlice(a.a2a_s_acc_x2_vmem, &.{ e_sem_id, start, c0, c0 }, &.{ 1, dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{remote_sz});
            const remote_src = b.memRefSqueeze(remote_src_sl, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
            const remote_dst_sl = b.memRefSlice(a.a2a_g_hbm, &.{ my_e_id, c0, c0, c0 }, &.{ 1, dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{remote_sz});
            const remote_dst = b.memRefSqueeze(remote_dst_sl, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
            b.enqueueDma(remote_src, remote_dst, a.a2a_gather_sem, .{ .source_semaphore = dmaSemAt1(b, a.send_sems, e_sem_id), .device_id = recv_id });
            start = b.addi(start, sz);
        }
    }
}

fn waitA2aGatherRecvAll(b: *mtt.Builder, cfg: Cfg, a: anytype) void {
    const ref = b.memRefReshape(a.a2a_g_hbm, &.{ cfg.num_experts * cfg.bt, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
    const sl = b.memRefSlice(ref, &.{ b.lift(@as(i32, 0)), b.lift(@as(i32, 0)), b.lift(@as(i32, 0)) }, &.{ cfg.top_k * cfg.bt, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{});
    b.waitDma2(a.a2a_gather_sem, sl, sl, .{});
}

fn waitA2aGatherSend(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    bt_id: mtt.Value,
    e_sem_id: mtt.Value,
    local_e_id: mtt.Value,
) void {
    const c0 = b.lift(@as(i32, 0));
    const local_num_experts = b.lift(@as(i32, @intCast(cfg.localNumExperts())));
    const my_id = myDeviceId(b, cfg);
    const my_e_id = b.addi(b.muli(my_id, local_num_experts), local_e_id);
    const bt_sem_id = floorModPos(b, bt_id, b.lift(@as(i32, 2)));
    const sz = b.scalarLoad(a.expert_sizes_x2_smem, &.{ b.toIndex(bt_sem_id), b.cIndex(0), b.toIndex(my_e_id) });
    const local_sz = b.scalarLoad(a.d2e_count_x2_smem, &.{ b.toIndex(bt_sem_id), b.toIndex(my_id), b.cIndex(0), b.toIndex(my_e_id) });
    const remote_sz_raw = b.subi(sz, local_sz);
    const is_valid = b.andi(b.cmpi(.sge, local_e_id, c0), b.cmpi(.slt, local_e_id, local_num_experts));
    const remote_sz = b.select(is_valid, remote_sz_raw, c0);
    const ref = b.memRefReshape(a.a2a_g_hbm, &.{ cfg.num_experts * cfg.bt, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
    const sl = b.memRefSlice(ref, &.{ c0, c0, c0 }, &.{ dynamic, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{remote_sz});
    b.waitDma2(dmaSemAt1(b, a.send_sems, e_sem_id), sl, sl, .{});
}

fn repeatLogitsToHidden(b: *mtt.Builder, cfg: Cfg, logits: mtt.Value) mtt.Value {
    const reps = @divExact(cfg.hidden_size, cfg.paddedTopK());
    if (reps == 1) return logits;
    if (reps > 16) {
        const tail_reps = reps - 16;
        var tail_parts: [16]mtt.Value = undefined;
        inline for (0..16) |i| {
            if (i < tail_reps) tail_parts[i] = logits;
        }
        const tail = b.concatenate(tail_parts[0..@intCast(tail_reps)], 1);

        var head_parts: [16]mtt.Value = undefined;
        inline for (0..16) |i| {
            head_parts[i] = logits;
        }
        const head = b.concatenate(head_parts[0..16], 1);
        return b.concatenate(&.{ head, tail }, 1);
    }
    var parts: [32]mtt.Value = undefined;
    inline for (0..32) |i| {
        if (i < reps) parts[i] = logits;
    }
    return b.concatenate(parts[0..@intCast(reps)], 1);
}

fn btAcc(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    bt_id: mtt.Value,
    info: TopKInfo,
) mtt.Value {
    @setEvalBranchQuota(100_000);
    const c0 = b.lift(@as(i32, 0));
    const c1 = b.lift(@as(i32, 1));
    const bt_sem_id = floorModPos(b, bt_id, b.lift(@as(i32, 2)));
    inline for (0..64) |bt_t_id_usize| {
        if (bt_t_id_usize < cfg.bt) {
            const bt_t_id = b.lift(@as(i32, @intCast(bt_t_id_usize)));
            inline for (0..32) |k_id_usize| {
                if (k_id_usize < cfg.top_k) {
                    const k_id = b.lift(@as(i32, @intCast(k_id_usize)));
                    const e_id = b.scalarLoad(a.t2e_routing_x2_smem, &.{ b.toIndex(bt_sem_id), b.toIndex(bt_t_id), b.toIndex(k_id) });
                    const e_idx = b.toIndex(e_id);
                    const offset = b.scalarLoad(a.expert_offsets_x2_smem, &.{ b.toIndex(bt_sem_id), b.cIndex(1), e_idx });
                    b.scalarStore(a.expert_offsets_x2_smem, b.addi(offset, c1), &.{ b.toIndex(bt_sem_id), b.cIndex(1), e_idx });
                    const src_sl = b.memRefSlice(a.a2a_g_hbm, &.{ e_id, offset, c0, c0 }, &.{ 1, 1, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{});
                    const src = b.memRefSqueeze(src_sl, &.{ 1, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
                    const dst_sl = b.memRefSlice(a.a2a_g_acc_vmem, &.{ k_id, bt_t_id, c0, c0 }, &.{ 1, 1, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }, &.{});
                    const dst = b.memRefSqueeze(dst_sl, &.{ 1, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) });
                    b.enqueueDma(src, dst, a.a2a_acc_sem, .{});
                }
            }
        }
    }
    b.waitDma2(a.a2a_acc_sem, a.a2a_g_acc_vmem, a.a2a_g_acc_vmem, .{});
    if (cfg.top_k == 1) {
        const acc = b.shapeCast(b.vectorLoadShape(a.a2a_g_acc_vmem, &.{ b.cIndex(0), b.cIndex(0), b.cIndex(0), b.cIndex(0) }, &.{ cfg.top_k, cfg.bt, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) }), &.{ cfg.bt, cfg.hidden_size });
        const logits = b.concatenate(&.{ info.logits, info.logits }, 1);
        return truncFromF32(b, b.mulf(extToF32(b, acc, cfg.token_dtype), logits), cfg.token_dtype);
    }

    var output: mtt.Value = undefined;
    inline for (0..32) |k_id_usize| {
        if (k_id_usize < cfg.top_k) {
            const acc_4d = b.vectorLoadShape(
                a.a2a_g_acc_vmem,
                &.{ b.cIndex(@intCast(k_id_usize)), b.cIndex(0), b.cIndex(0), b.cIndex(0) },
                &.{ 1, cfg.bt, cfg.tokenPacking(), @divExact(cfg.hidden_size, cfg.tokenPacking()) },
            );
            const acc = b.shapeCast(acc_4d, &.{ cfg.bt, cfg.hidden_size });
            const logits = repeatLogitsToHidden(b, cfg, info.logits_list[k_id_usize]);
            const term = b.mulf(extToF32(b, acc, cfg.token_dtype), logits);
            output = if (k_id_usize == 0) term else b.addf(output, term);
        }
    }
    return truncFromF32(b, output, cfg.token_dtype);
}

fn storeOutput(b: *mtt.Builder, cfg: Cfg, a: anytype, bt_id: mtt.Value, output: mtt.Value) void {
    const bt_sem_id = floorModPos(b, bt_id, b.lift(@as(i32, 2)));
    b.vectorStoreAt(a.b_output_x2_vmem, b.shapeCast(output, &.{ 1, cfg.bt, cfg.hidden_size }), &.{ b.toIndex(bt_sem_id), b.cIndex(0), b.cIndex(0) });
}

fn startSendBo(b: *mtt.Builder, cfg: Cfg, a: anytype, bt_id: mtt.Value) void {
    const c0 = b.lift(@as(i32, 0));
    const bt_sem_id = floorModPos(b, bt_id, b.lift(@as(i32, 2)));
    const out_ref = b.memRefSqueeze(b.memRefSlice(a.b_output_x2_vmem, &.{ bt_sem_id, c0, c0 }, &.{ 1, cfg.bt, cfg.hidden_size }, &.{}), &.{ cfg.bt, cfg.hidden_size });
    const dst = b.memRefSlice(a.output_hbm, &.{ b.muli(bt_id, b.lift(@as(i32, @intCast(cfg.bt)))), c0 }, &.{ cfg.bt, cfg.hidden_size }, &.{});
    b.enqueueDma(out_ref, dst, dmaSemAt2(b, a.local_sems, bt_sem_id, b.lift(@as(i32, 4))), .{});
}

fn waitSendBo(b: *mtt.Builder, cfg: Cfg, a: anytype, bt_id: mtt.Value) void {
    const c0 = b.lift(@as(i32, 0));
    const c2 = b.lift(@as(i32, 2));
    const num_bt = b.lift(@as(i32, @intCast(@divExact(cfg.localNumTokens(), cfg.bt))));
    const valid = b.andi(b.cmpi(.sge, bt_id, c0), b.cmpi(.slt, bt_id, num_bt));
    const sz = b.assumeMultiple(b.select(valid, b.lift(@as(i32, @intCast(cfg.bt))), c0), @intCast(cfg.bt));
    const bt_sem_id = floorModPos(b, b.addi(bt_id, c2), c2);
    const ref = b.memRefSlice(a.output_hbm, &.{ c0, c0 }, &.{ dynamic, cfg.hidden_size }, &.{sz});
    b.waitDma2(dmaSemAt2(b, a.local_sems, bt_sem_id, b.lift(@as(i32, 4))), ref, ref, .{});
}

fn nextSemId(b: *mtt.Builder, sem_id: mtt.Value) mtt.Value {
    const c0 = b.lift(@as(i32, 0));
    const c1 = b.lift(@as(i32, 1));
    return b.select(b.cmpi(.eq, sem_id, c0), c1, c0);
}

fn runExpertLoop(
    b: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    bt_id: mtt.Value,
    initial_e_sem_id: mtt.Value,
) mtt.Value {
    const c0 = b.lift(@as(i32, 0));
    const c1 = b.lift(@as(i32, 1));
    const local_num_experts = b.lift(@as(i32, @intCast(cfg.localNumExperts())));

    var loop = b.openFor(c0, local_num_experts, c1, .{initial_e_sem_id});
    const local_e_id = loop.iv;
    const e_sem_id = loop.carried[0];

    startFetchBw1(b, cfg, a, local_e_id, c0, c0, c0);
    startFetchBw3(b, cfg, a, local_e_id, c0, c0, c0);

    const next_e_sem_id = nextSemId(b, e_sem_id);
    const next_local_e_id = b.addi(local_e_id, c1);
    {
        var when_next = b.openIf(b.cmpi(.slt, next_local_e_id, local_num_experts));
        startA2aScatter(b, cfg, a, bt_id, next_e_sem_id, next_local_e_id);
        when_next.yieldThen(.{});
    }

    waitA2aScatterRecv(b, cfg, a, bt_id, e_sem_id, local_e_id);
    expertFfnBaseline(b, cfg, a, bt_id, e_sem_id, local_e_id);
    startA2aGather(b, cfg, a, bt_id, e_sem_id, local_e_id);
    waitA2aScatterSend(b, cfg, a, e_sem_id);
    syncBarrier(b, cfg);

    loop.yield(.{next_e_sem_id});
    return loop.results[0];
}

pub const Cfg = struct {
    pub const ActFn = enum {
        silu,
        gelu,
        swigluoai,
    };

    pub const ScoringFn = enum {
        softmax,
        sigmoid,
    };

    num_tokens: i64,
    hidden_size: i64,
    intermediate_size: i64,
    num_experts: i64,
    top_k: i64,
    ep_size: i64,
    token_dtype: mtt.DType = .bf16,
    weight_dtype: mtt.DType = .bf16,
    renormalize_topk_logits: bool = false,
    act_fn: ActFn = .silu,
    scoring_fn: ScoringFn = .softmax,
    bt: i64,
    bf: i64,
    bd1: i64,
    bd2: i64,
    btc: i64,
    bfc: i64,
    bd1c: i64,
    bd2c: i64,

    pub fn localNumTokens(self: Cfg) i64 {
        return @divExact(self.num_tokens, self.ep_size);
    }

    pub fn localNumExperts(self: Cfg) i64 {
        return @divExact(self.num_experts, self.ep_size);
    }

    pub fn paddedNumExperts(self: Cfg) i64 {
        return alignTo(self.num_experts, 128);
    }

    pub fn paddedTopK(self: Cfg) i64 {
        return alignTo(self.top_k, 128);
    }

    pub fn tokenPacking(self: Cfg) i64 {
        return dtypePacking(self.token_dtype);
    }
};

pub fn run(b: *mtt.Builder, cfg: Cfg) !void {
    const t_packing = cfg.tokenPacking();
    const local_tokens = cfg.localNumTokens();
    const hidden_per_pack = @divExact(cfg.hidden_size, t_packing);
    const bt_devices = cfg.bt * cfg.ep_size;

    const a = try b.declareArgsOpts(.{
        .tokens_hbm = .{ .ref = .{
            .shape = &.{ local_tokens, t_packing, hidden_per_pack },
            .dtype = cfg.token_dtype,
            .memory_space = .hbm,
        } },
        .w1_hbm = .{ .ref = .{
            .shape = &.{ cfg.localNumExperts(), 2, cfg.hidden_size, cfg.intermediate_size },
            .dtype = cfg.weight_dtype,
            .memory_space = .hbm,
        } },
        .w2_hbm = .{ .ref = .{
            .shape = &.{ cfg.localNumExperts(), cfg.intermediate_size, cfg.hidden_size },
            .dtype = cfg.weight_dtype,
            .memory_space = .hbm,
        } },
        .gating_hbm = .{ .ref = .{
            .shape = &.{ local_tokens, cfg.paddedNumExperts() },
            .dtype = cfg.token_dtype,
            .memory_space = .hbm,
        } },
        .a2a_g_hbm = .{ .ref = .{
            .shape = &.{ cfg.num_experts, cfg.bt, t_packing, hidden_per_pack },
            .dtype = cfg.token_dtype,
            .memory_space = .hbm,
        } },
        .output_hbm = .{ .ref = .{
            .shape = &.{ local_tokens, cfg.hidden_size },
            .dtype = cfg.token_dtype,
            .memory_space = .hbm,
            .role = .output,
        } },
        .t2e_routing_x2_smem = .{ .ref = .{
            .shape = &.{ 2, cfg.bt, cfg.paddedTopK() },
            .dtype = .i32,
            .memory_space = .smem,
            .role = .scratch,
        } },
        .d2e_count_x2_smem = .{ .ref = .{
            .shape = &.{ 2, cfg.ep_size, 1, cfg.paddedNumExperts() },
            .dtype = .i32,
            .memory_space = .smem,
            .role = .scratch,
        } },
        .expert_offsets_x2_smem = .{ .ref = .{
            .shape = &.{ 2, 2, cfg.paddedNumExperts() },
            .dtype = .i32,
            .memory_space = .smem,
            .role = .scratch,
        } },
        .expert_starts_x2_smem = .{ .ref = .{
            .shape = &.{ 2, 1, cfg.paddedNumExperts() },
            .dtype = .i32,
            .memory_space = .smem,
            .role = .scratch,
        } },
        .expert_sizes_x2_smem = .{ .ref = .{
            .shape = &.{ 2, 1, cfg.paddedNumExperts() },
            .dtype = .i32,
            .memory_space = .smem,
            .role = .scratch,
        } },
        .a2a_s_sends_x2_smem = .{ .ref = .{
            .shape = &.{2},
            .dtype = .i32,
            .memory_space = .smem,
            .role = .scratch,
        } },
        .a2a_s_x2_vmem = .{ .ref = .{
            .shape = &.{ 2, bt_devices, t_packing, hidden_per_pack },
            .dtype = cfg.token_dtype,
            .memory_space = .vmem,
            .role = .scratch,
        } },
        .a2a_s_acc_x2_vmem = .{ .ref = .{
            .shape = &.{ 2, bt_devices, t_packing, hidden_per_pack },
            .dtype = cfg.token_dtype,
            .memory_space = .vmem,
            .role = .scratch,
        } },
        .a2a_g_acc_vmem = .{ .ref = .{
            .shape = &.{ cfg.top_k, cfg.bt, t_packing, hidden_per_pack },
            .dtype = cfg.token_dtype,
            .memory_space = .vmem,
            .role = .scratch,
        } },
        .b_gating_x2_vmem = .{ .ref = .{
            .shape = &.{ 2, cfg.bt, cfg.paddedNumExperts() },
            .dtype = cfg.token_dtype,
            .memory_space = .vmem,
            .role = .scratch,
        } },
        .b_output_x2_vmem = .{ .ref = .{
            .shape = &.{ 2, cfg.bt, cfg.hidden_size },
            .dtype = cfg.token_dtype,
            .memory_space = .vmem,
            .role = .scratch,
        } },
        .b_w1_x2_vmem = .{ .ref = .{
            .shape = &.{ 2, t_packing, @divExact(cfg.bd1, t_packing), cfg.bf },
            .dtype = cfg.weight_dtype,
            .memory_space = .vmem,
            .role = .scratch,
        } },
        .b_w3_x2_vmem = .{ .ref = .{
            .shape = &.{ 2, t_packing, @divExact(cfg.bd1, t_packing), cfg.bf },
            .dtype = cfg.weight_dtype,
            .memory_space = .vmem,
            .role = .scratch,
        } },
        .b_w2_x2_vmem = .{ .ref = .{
            .shape = &.{ 2, t_packing, cfg.bf, @divExact(cfg.bd2, t_packing) },
            .dtype = cfg.weight_dtype,
            .memory_space = .vmem,
            .role = .scratch,
        } },
        .b_acc_vmem = .{ .ref = .{
            .shape = &.{ 2, bt_devices, 1, cfg.bf },
            .dtype = .f32,
            .memory_space = .vmem,
            .role = .scratch,
        } },
        .local_sems = .{ .sem_array = .{ .shape = &.{ 2, 5 }, .kind = .dma } },
        .send_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
        .recv_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
        .a2a_gather_sem = .{ .sem_array = .{ .shape = &.{}, .kind = .dma } },
        .a2a_acc_sem = .{ .sem_array = .{ .shape = &.{}, .kind = .dma } },
    }, &.{}, .{
        .dimension_semantics = &.{},
        .scalar_prefetch = 0,
        .scratch_operands = 20,
        .pallas_window_params = false,
    });

    const c0 = b.lift(@as(i32, 0));
    const c1 = b.lift(@as(i32, 1));
    const c2 = b.lift(@as(i32, 2));
    const c_bt = b.lift(@as(i32, @intCast(cfg.bt)));
    const c_zero_sz = b.lift(@as(i32, 0));
    const num_bt = b.lift(@as(i32, @intCast(@divExact(cfg.localNumTokens(), cfg.bt))));

    syncBarrier(b, cfg);
    fetchGating(b, cfg, a, c0, c0, c_bt, 0);
    const my_id = myDeviceId(b, cfg);
    const right_id = if (cfg.ep_size == 1) c0 else floorModPos(b, b.addi(my_id, c1), b.lift(@as(i32, @intCast(cfg.ep_size))));
    const local_num_experts = b.lift(@as(i32, @intCast(cfg.localNumExperts())));

    if (@divExact(cfg.localNumTokens(), cfg.bt) == 1) {
        fetchGating(b, cfg, a, c1, c1, c_zero_sz, 0);
        waitGating(b, cfg, a, c0);

        const gating_raw = b.vectorLoadShape(a.b_gating_x2_vmem, &.{ b.cIndex(0), b.cIndex(0), b.cIndex(0) }, &.{ 1, cfg.bt, cfg.paddedNumExperts() });
        const topk = scoreAndTopK(b, cfg, gating_raw);
        copyMetadataSingleDevice(b, cfg, a, c0, my_id, right_id, topk);
        syncBarrier(b, cfg);
        startA2aScatter(b, cfg, a, c0, c0, c0);
        const e_sem_id = runExpertLoop(b, cfg, a, c0, c0);
        waitA2aGatherRecvAll(b, cfg, a);
        syncBarrier(b, cfg);
        const output = btAcc(b, cfg, a, c0, topk);
        waitSendBo(b, cfg, a, b.subi(c0, c2));
        storeOutput(b, cfg, a, c0, output);
        startSendBo(b, cfg, a, c0);
        waitA2aGatherSend(b, cfg, a, c0, e_sem_id, b.subi(local_num_experts, c2));
        waitA2aGatherSend(b, cfg, a, c0, nextSemId(b, e_sem_id), b.subi(local_num_experts, c1));
        syncBarrier(b, cfg);
    } else {
        var loop = b.openFor(c0, num_bt, c1, .{c0});
        const bt_id = loop.iv;
        const bt_sem_id = floorModPos(b, bt_id, c2);
        const initial_e_sem_id = loop.carried[0];
        const next_bt_id = b.addi(bt_id, c1);
        const next_fetch_sem = floorModPos(b, b.addi(next_bt_id, c2), c2);
        const next_valid = b.andi(b.cmpi(.sge, next_bt_id, c0), b.cmpi(.slt, next_bt_id, num_bt));
        const next_sz = b.select(next_valid, c_bt, c_zero_sz);
        fetchGating(b, cfg, a, next_bt_id, next_fetch_sem, next_sz, 0);
        waitGating(b, cfg, a, bt_sem_id);

        const gating_raw = b.vectorLoadShape(a.b_gating_x2_vmem, &.{ b.toIndex(bt_sem_id), b.cIndex(0), b.cIndex(0) }, &.{ 1, cfg.bt, cfg.paddedNumExperts() });
        const topk = scoreAndTopK(b, cfg, gating_raw);
        copyMetadataSingleDevice(b, cfg, a, bt_sem_id, my_id, right_id, topk);
        syncBarrier(b, cfg);
        startA2aScatter(b, cfg, a, bt_id, initial_e_sem_id, c0);
        const e_sem_id = runExpertLoop(b, cfg, a, bt_id, initial_e_sem_id);
        waitA2aGatherRecvAll(b, cfg, a);
        syncBarrier(b, cfg);
        const output = btAcc(b, cfg, a, bt_id, topk);
        waitSendBo(b, cfg, a, b.subi(bt_id, c2));
        storeOutput(b, cfg, a, bt_id, output);
        startSendBo(b, cfg, a, bt_id);
        waitA2aGatherSend(b, cfg, a, bt_id, e_sem_id, b.subi(local_num_experts, c2));
        waitA2aGatherSend(b, cfg, a, bt_id, nextSemId(b, e_sem_id), b.subi(local_num_experts, c1));
        syncBarrier(b, cfg);
        loop.yield(.{e_sem_id});
    }
    waitSendBo(b, cfg, a, b.subi(num_bt, c2));
    waitSendBo(b, cfg, a, b.subi(num_bt, c1));
}

pub const Kernel = struct {
    pub const name: [:0]const u8 = "fused_moe";
    pub const Config = Cfg;

    pub fn emit(allocator: std.mem.Allocator, cfg: Config) ![:0]const u8 {
        const ctx = try zml.kernel.mosaic_tpu.newContext();
        defer ctx.deinit();

        const renorm_str = if (cfg.renormalize_topk_logits) "-renorm_k" else "";
        const fn_name = try std.fmt.allocPrint(
            allocator,
            "fused-moe-k_{d}{s}-bt_{d}_{d}-bf_{d}_{d}-bd1_{d}_{d}-bd2_{d}_{d}",
            .{ cfg.top_k, renorm_str, cfg.bt, cfg.btc, cfg.bf, cfg.bfc, cfg.bd1, cfg.bd1c, cfg.bd2, cfg.bd2c },
        );
        defer allocator.free(fn_name);

        var b = try mtt.Builder.open(allocator, ctx, fn_name);
        defer b.deinit();
        try run(&b, cfg);
        return b.finishOpts(&.{}, .{ .canonicalize = false });
    }
};
