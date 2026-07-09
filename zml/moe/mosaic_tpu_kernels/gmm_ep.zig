const std = @import("std");

const zml = @import("../../zml.zig");
const mtt = @import("kernels/mosaic_tpu/builder");
const pipeline = mtt.pipeline;
const mtt_kernel = zml.kernel.mosaic_tpu;

fn cdiv(x: i64, y: i64) i64 {
    return @divFloor(x + y - 1, y);
}

fn alignTo(x: i64, y: i64) i64 {
    return cdiv(x, y) * y;
}

fn floorModI32(b: *mtt.Builder, x: mtt.Value, y: mtt.Value) mtt.Value {
    const c0 = b.lift(@as(i32, 0));
    const r = b.remsi(x, y);
    const r_nz = b.cmpi(.ne, r, c0);
    const r_neg = b.cmpi(.slt, r, c0);
    return b.select(b.andi(r_neg, r_nz), b.addi(r, y), r);
}

fn floorDivPosI32(b: *mtt.Builder, x: mtt.Value, y: mtt.Value) mtt.Value {
    const c0 = b.lift(@as(i32, 0));
    const c1 = b.lift(@as(i32, 1));
    const q = b.divsi(x, y);
    const sign = b.subi(
        b.extui(b.cmpi(.sgt, x, c0), .i32),
        b.extui(b.cmpi(.slt, x, c0), .i32),
    );
    const sign_not_pos = b.cmpi(.ne, sign, c1);
    const rem_nz = b.cmpi(.ne, b.remsi(x, y), c0);
    return b.select(b.andi(sign_not_pos, rem_nz), b.subi(q, c1), q);
}

fn ceilDivPosI32(b: *mtt.Builder, x: mtt.Value, y: mtt.Value) mtt.Value {
    return b.divsi(b.addi(x, b.subi(y, b.lift(@as(i32, 1)))), y);
}

pub const Cfg = struct {
    pub const FuseAct = enum {
        none,
        silu,
        gelu,
    };

    size_m: i64,
    size_k: i64,
    size_n: i64,
    size_group: i64,
    size_lhs_group: i64,
    tile_m: i64,
    tile_k: i64,
    tile_n: i64,
    lhs_dtype: mtt.DType = .bf16,
    rhs_dtype: mtt.DType = .bf16,
    out_dtype: mtt.DType = .bf16,
    acc_dtype: mtt.DType = .f32,
    size_lhs_sublane: i64 = 8,
    fuse_act: FuseAct = .none,

    pub fn outSizeN(self: Cfg) i64 {
        return switch (self.fuse_act) {
            .none => self.size_n,
            .silu, .gelu => @divExact(self.size_n, 2),
        };
    }

    pub fn alignedN(self: Cfg) i64 {
        return switch (self.fuse_act) {
            .none => alignTo(self.outSizeN(), 128),
            .silu, .gelu => self.outSizeN(),
        };
    }

    pub fn rhsTileN(self: Cfg) i64 {
        return switch (self.fuse_act) {
            .none => self.tile_n,
            .silu, .gelu => self.tile_n * 2,
        };
    }

    pub fn maxNumGm(self: Cfg) i64 {
        return self.size_group + cdiv(self.size_m, self.tile_m) - 1;
    }
};

fn validateConfig(cfg: Cfg) void {
    std.debug.assert(cfg.size_group <= cfg.size_lhs_group);
    std.debug.assert(cfg.size_m > 0);
    std.debug.assert(cfg.size_k > 0);
    std.debug.assert(cfg.size_n > 0);
    std.debug.assert(cfg.tile_m > 0);
    std.debug.assert(cfg.tile_k > 0);
    std.debug.assert(cfg.tile_n > 0);
    std.debug.assert(cfg.size_lhs_sublane > 0);
    std.debug.assert(@mod(cfg.size_m, cfg.size_lhs_sublane) == 0);
    std.debug.assert(cdiv(cfg.size_k, cfg.tile_k) == 1);
    std.debug.assert(@mod(cfg.tile_k, 128) == 0);
    std.debug.assert(@mod(cfg.rhsTileN(), 128) == 0);
    if (cfg.fuse_act != .none) {
        std.debug.assert(@mod(cfg.size_n, 2) == 0);
        std.debug.assert(cdiv(cfg.alignedN(), cfg.tile_n) == 1);
    } else {
        std.debug.assert(@mod(cfg.alignedN(), cfg.tile_n) == 0);
    }
    std.debug.assert(cfg.acc_dtype == .f32);
}

const Meta = struct {
    group_id: mtt.Value,
    m_offset: mtt.Value,
};

const PipeCtx = struct {
    cfg: Cfg,
    meta: Meta,
    num_gm: mtt.Value,
};

fn winRef(b: *mtt.Builder, rw: pipeline.RefWindow, shape: []const i64) mtt.Value {
    const buf = rw.buf.?;
    const slot = rw.slot orelse return buf;
    const a = b.arena.allocator();
    const rank = shape.len + 1;
    const base = a.alloc(mtt.Value, rank) catch @panic("gmm_ep winRef OOM");
    const result_shape = a.alloc(i64, rank) catch @panic("gmm_ep winRef OOM");
    base[0] = slot;
    result_shape[0] = 1;
    for (shape, 0..) |d, i| {
        base[i + 1] = b.lift(@as(i32, 0));
        result_shape[i + 1] = d;
    }
    const sliced = b.memRefSlice(buf, base, result_shape, &.{});
    return b.memRefSqueeze(sliced, shape);
}

fn metaLoad(b: *mtt.Builder, ref: mtt.Value, idx: mtt.Value) mtt.Value {
    return b.scalarLoad(ref, &.{b.toIndex(idx)});
}

fn lhsIndexMap(b: *mtt.Builder, indices: []const mtt.Value, ctx_opaque: ?*anyopaque) []const pipeline.BlockIndex {
    const ctx: *PipeCtx = @ptrCast(@alignCast(ctx_opaque.?));
    const cfg = ctx.cfg;
    const gm_id = indices[1];
    const k_id = indices[2];

    const m_start = metaLoad(b, ctx.meta.m_offset, gm_id);
    const m_end = metaLoad(b, ctx.meta.m_offset, b.addi(gm_id, b.lift(@as(i32, 1))));
    const sublane = b.lift(@as(i32, @intCast(cfg.size_lhs_sublane)));
    const row_start = floorDivPosI32(b, m_start, sublane);
    const row_end = ceilDivPosI32(b, m_end, sublane);
    const row_size = b.subi(row_end, row_start);

    const out = b.arena.allocator().alloc(pipeline.BlockIndex, 3) catch @panic("gmm_ep lhsIndexMap OOM");
    out[0] = .{ .slice = .{ .start = row_start, .size = row_size } };
    out[1] = .{ .scalar = b.lift(@as(i32, 0)) };
    out[2] = .{ .scalar = k_id };
    return out;
}

fn rhsIndexMap(b: *mtt.Builder, indices: []const mtt.Value, ctx_opaque: ?*anyopaque) []const pipeline.BlockIndex {
    const ctx: *PipeCtx = @ptrCast(@alignCast(ctx_opaque.?));
    const group_id = metaLoad(b, ctx.meta.group_id, indices[1]);
    const out = b.arena.allocator().alloc(pipeline.BlockIndex, 3) catch @panic("gmm_ep rhsIndexMap OOM");
    out[0] = .{ .scalar = group_id };
    out[1] = .{ .scalar = indices[2] };
    out[2] = .{ .scalar = indices[0] };
    return out;
}

fn outIndexMap(b: *mtt.Builder, indices: []const mtt.Value, ctx_opaque: ?*anyopaque) []const pipeline.BlockIndex {
    const ctx: *PipeCtx = @ptrCast(@alignCast(ctx_opaque.?));
    const cfg = ctx.cfg;
    const gm_id = indices[1];
    const n_id = indices[0];

    const m_start = metaLoad(b, ctx.meta.m_offset, gm_id);
    const m_end = metaLoad(b, ctx.meta.m_offset, b.addi(gm_id, b.lift(@as(i32, 1))));
    const sublane = b.lift(@as(i32, @intCast(cfg.size_lhs_sublane)));
    const row_start = floorDivPosI32(b, m_start, sublane);
    const capped_row_end = floorDivPosI32(b, m_end, sublane);
    const last_row_end = ceilDivPosI32(b, m_end, sublane);
    const row_end = b.select(
        b.cmpi(.eq, gm_id, b.subi(ctx.num_gm, b.lift(@as(i32, 1)))),
        last_row_end,
        capped_row_end,
    );
    const row_size = b.subi(row_end, row_start);

    const out = b.arena.allocator().alloc(pipeline.BlockIndex, 3) catch @panic("gmm_ep outIndexMap OOM");
    out[0] = .{ .slice = .{ .start = row_start, .size = row_size } };
    out[1] = .{ .scalar = b.lift(@as(i32, 0)) };
    out[2] = .{ .scalar = n_id };
    return out;
}

fn indices1DTo3D(b: *mtt.Builder, indices: []const mtt.Value) []const mtt.Value {
    const out = b.arena.allocator().alloc(mtt.Value, 3) catch @panic("gmm_ep indices1DTo3D OOM");
    out[0] = b.lift(@as(i32, 0));
    out[1] = indices[0];
    out[2] = b.lift(@as(i32, 0));
    return out;
}

fn lhsIndexMap1D(b: *mtt.Builder, indices: []const mtt.Value, ctx_opaque: ?*anyopaque) []const pipeline.BlockIndex {
    return lhsIndexMap(b, indices1DTo3D(b, indices), ctx_opaque);
}

fn rhsIndexMap1D(b: *mtt.Builder, indices: []const mtt.Value, ctx_opaque: ?*anyopaque) []const pipeline.BlockIndex {
    return rhsIndexMap(b, indices1DTo3D(b, indices), ctx_opaque);
}

fn outIndexMap1D(b: *mtt.Builder, indices: []const mtt.Value, ctx_opaque: ?*anyopaque) []const pipeline.BlockIndex {
    return outIndexMap(b, indices1DTo3D(b, indices), ctx_opaque);
}

fn fillMetadata(b: *mtt.Builder, cfg: Cfg, a: anytype) mtt.Value {
    const c0 = b.lift(@as(i32, 0));
    const c1 = b.lift(@as(i32, 1));
    const group_offset = b.scalarLoad(a.group_offset, &.{b.cIndex(0)});
    const max_num_group = b.addi(group_offset, b.lift(@as(i32, @intCast(cfg.size_group))));
    b.scalarStore(a.metadata_m_offset, c0, &.{b.cIndex(0)});

    var outer = b.openFor(c0, max_num_group, c1, .{ c0, c0 });
    {
        b.traceStart(10, "outer_group_loop");
        const lhs_group_id = outer.iv;
        const num_gm = outer.carried[0];
        const start_m_offset = outer.carried[1];
        const group_id = b.subi(lhs_group_id, group_offset);
        const group_size = b.scalarLoad(a.group_sizes, &.{b.toIndex(lhs_group_id)});
        const end_m_offset = b.addi(start_m_offset, group_size);
        const sublane = b.lift(@as(i32, @intCast(cfg.size_lhs_sublane)));
        const local_offset = floorModI32(b, start_m_offset, sublane);
        const aligned_group_size = b.addi(group_size, local_offset);
        const tile_m = b.lift(@as(i32, @intCast(cfg.tile_m)));
        var curr_num_gm = b.divsi(b.addi(aligned_group_size, b.subi(tile_m, c1)), tile_m);
        const should_process = b.andi(b.cmpi(.sgt, group_size, c0), b.cmpi(.sge, group_id, c0));
        curr_num_gm = b.select(should_process, curr_num_gm, c0);
        const next_num_gm = b.addi(num_gm, curr_num_gm);

        var inner = b.openFor(num_gm, next_num_gm, c1, .{start_m_offset});
        {
            b.traceStart(10, "inner_tm_loop");
            const tm_id = inner.iv;
            const curr_m_offset = inner.carried[0];
            const local = floorModI32(b, curr_m_offset, sublane);
            const tm_room = b.subi(tile_m, local);
            const remaining = b.subi(end_m_offset, curr_m_offset);
            const tm_size = b.minsi(tm_room, remaining);
            const next_m_offset = b.addi(curr_m_offset, tm_size);
            b.scalarStore(a.metadata_group_id, group_id, &.{b.toIndex(tm_id)});
            b.scalarStore(a.metadata_m_offset, curr_m_offset, &.{b.toIndex(tm_id)});
            b.scalarStore(a.metadata_m_offset, next_m_offset, &.{b.toIndex(b.addi(tm_id, c1))});
            b.traceStop();
            inner.yield(.{next_m_offset});
        }

        b.traceStop();
        outer.yield(.{ next_num_gm, end_m_offset });
    }
    return outer.results[0];
}

fn matmulByMxuColumns(b: *mtt.Builder, cfg: Cfg, lhs: mtt.Value, rhs: mtt.Value, cols: i64) mtt.Value {
    if (cfg.tile_k > cfg.size_k) {
        std.debug.assert(cdiv(cfg.size_k, cfg.tile_k) == 1);
        std.debug.assert(@mod(cols, 128) == 0);

        const chunks = @divExact(cols, 128);
        const parts = b.arena.allocator().alloc(mtt.Value, @intCast(chunks)) catch @panic("gmm_ep matmul parts OOM");
        const tail_k = cfg.tile_k - cfg.size_k;
        const lhs_valid = b.vectorExtractStridedSlice(lhs, &.{ 0, 0 }, &.{ cfg.tile_m, cfg.size_k }, &.{ 1, 1 });
        const lhs_tail = b.vectorExtractStridedSlice(lhs, &.{ 0, cfg.size_k }, &.{ cfg.tile_m, tail_k }, &.{ 1, 1 });

        var rev: i64 = 0;
        while (rev < chunks) : (rev += 1) {
            const chunk = chunks - 1 - rev;
            const start_n = chunk * 128;
            const rhs_valid = b.vectorExtractStridedSlice(rhs, &.{ 0, start_n }, &.{ cfg.size_k, 128 }, &.{ 1, 1 });
            const rhs_tail = b.vectorExtractStridedSlice(rhs, &.{ cfg.size_k, start_n }, &.{ tail_k, 128 }, &.{ 1, 1 });
            const zero = b.zeros(&.{ cfg.tile_m, 128 }, cfg.acc_dtype);
            const tail_acc = b.matmulOpts(lhs_tail, rhs_tail, zero, .{
                .dimension_numbers = b.dotDimensionNumbers(&.{1}, &.{0}, &.{0}, &.{1}, &.{ 0, 0, 1, 1 }, &.{}, &.{}),
            });
            parts[@intCast(chunk)] = b.matmulOpts(lhs_valid, rhs_valid, tail_acc, .{
                .dimension_numbers = b.dotDimensionNumbers(&.{1}, &.{0}, &.{0}, &.{1}, &.{ 0, 0, 1, 1 }, &.{}, &.{}),
            });
        }

        if (chunks == 1) return parts[0];

        const max_concat_inputs = 16;
        if (chunks <= max_concat_inputs) return b.concatenate(parts, 1);

        const groups = cdiv(chunks, max_concat_inputs);
        const group_parts = b.arena.allocator().alloc(mtt.Value, @intCast(groups)) catch @panic("gmm_ep matmul group parts OOM");
        var group: i64 = 0;
        while (group < groups) : (group += 1) {
            const start = group * max_concat_inputs;
            const end = @min(start + max_concat_inputs, chunks);
            const start_usize: usize = @intCast(start);
            const end_usize: usize = @intCast(end);
            const group_slice = parts[start_usize..end_usize];
            group_parts[@intCast(group)] = if (group_slice.len == 1) group_slice[0] else b.concatenate(group_slice, 1);
        }
        return b.concatenate(group_parts, 1);
    }

    if (cols <= 128) {
        return b.matmulOpts(lhs, rhs, b.zeros(&.{ cfg.tile_m, cols }, cfg.acc_dtype), .{
            .dimension_numbers = b.dotDimensionNumbers(&.{1}, &.{0}, &.{0}, &.{1}, &.{ 0, 0, 1, 1 }, &.{}, &.{}),
        });
    }

    std.debug.assert(@mod(cols, 128) == 0);
    const chunks = @divExact(cols, 128);
    const parts = b.arena.allocator().alloc(mtt.Value, @intCast(chunks)) catch @panic("gmm_ep matmul parts OOM");
    var rev: i64 = 0;
    while (rev < chunks) : (rev += 1) {
        const chunk = chunks - 1 - rev;
        const start_n = chunk * 128;
        const rhs_chunk = b.vectorExtractStridedSlice(rhs, &.{ 0, start_n }, &.{ cfg.tile_k, 128 }, &.{ 1, 1 });
        const zero = b.zeros(&.{ cfg.tile_m, 128 }, cfg.acc_dtype);
        parts[@intCast(chunk)] = b.matmulOpts(lhs, rhs_chunk, zero, .{
            .dimension_numbers = b.dotDimensionNumbers(&.{1}, &.{0}, &.{0}, &.{1}, &.{ 0, 0, 1, 1 }, &.{}, &.{}),
        });
    }

    if (chunks == 1) return parts[0];

    const max_concat_inputs = 16;
    if (chunks <= max_concat_inputs) return b.concatenate(parts, 1);

    const groups = cdiv(chunks, max_concat_inputs);
    const group_parts = b.arena.allocator().alloc(mtt.Value, @intCast(groups)) catch @panic("gmm_ep matmul group parts OOM");
    var group: i64 = 0;
    while (group < groups) : (group += 1) {
        const start = group * max_concat_inputs;
        const end = @min(start + max_concat_inputs, chunks);
        const start_usize: usize = @intCast(start);
        const end_usize: usize = @intCast(end);
        const group_slice = parts[start_usize..end_usize];
        group_parts[@intCast(group)] = if (group_slice.len == 1) group_slice[0] else b.concatenate(group_slice, 1);
    }
    return b.concatenate(group_parts, 1);
}

fn geluApprox(b: *mtt.Builder, x: mtt.Value, shape: []const i64) mtt.Value {
    const x2 = b.mulf(x, x);
    const x3 = b.mulf(x2, x);
    const cubic = b.mulf(x3, b.splat(@as(f32, 0.044715), shape, cfgAccDtype()));
    const inner = b.mulf(b.addf(x, cubic), b.splat(@as(f32, 0.7978845608028654), shape, cfgAccDtype()));
    return b.mulf(
        b.mulf(x, b.splat(@as(f32, 0.5), shape, cfgAccDtype())),
        b.addf(b.tanh(inner), b.splat(@as(f32, 1.0), shape, cfgAccDtype())),
    );
}

fn cfgAccDtype() mtt.DType {
    return .f32;
}

fn silu(b: *mtt.Builder, x: mtt.Value, shape: []const i64) mtt.Value {
    const one = b.splat(@as(f32, 1.0), shape, cfgAccDtype());
    const denom = b.addf(one, b.exp(b.negf(x)));
    return b.divf(x, denom);
}

fn applyFuseAct(b: *mtt.Builder, cfg: Cfg, acc: mtt.Value) mtt.Value {
    return switch (cfg.fuse_act) {
        .none => acc,
        .silu, .gelu => {
            const shape = &.{ cfg.tile_m, cfg.tile_n };
            const gate = b.vectorExtractStridedSlice(acc, &.{ 0, 0 }, shape, &.{ 1, 1 });
            const up = b.vectorExtractStridedSlice(acc, &.{ 0, cfg.outSizeN() }, shape, &.{ 1, 1 });
            const activated = switch (cfg.fuse_act) {
                .silu => silu(b, gate, shape),
                .gelu => geluApprox(b, gate, shape),
                .none => unreachable,
            };
            return b.mulf(activated, up);
        },
    };
}

fn innerBody(
    b: *mtt.Builder,
    indices: []const mtt.Value,
    refs: []const pipeline.RefWindow,
    scratches: []const mtt.Value,
    ctx_opaque: ?*anyopaque,
) mtt.FinishError!void {
    const ctx: *PipeCtx = @ptrCast(@alignCast(ctx_opaque.?));
    const cfg = ctx.cfg;
    const gm_id = indices[1];

    const lhs_shape = &.{ cdiv(cfg.tile_m, cfg.size_lhs_sublane), cfg.size_lhs_sublane, cfg.tile_k };
    const out_shape = &.{ cdiv(cfg.tile_m, cfg.size_lhs_sublane), cfg.size_lhs_sublane, cfg.tile_n };
    const rhs_tile_n = cfg.rhsTileN();

    b.traceStart(10, "matmul_first_last");
    const lhs_ref = winRef(b, refs[0], lhs_shape);
    const lhs_ref_2d = b.memRefReshape(lhs_ref, &.{ cfg.tile_m, cfg.tile_k });
    const lhs = b.refLoad(lhs_ref_2d);
    const rhs_slot = b.toIndex(refs[1].slot.?);
    var rhs = b.shapeCast(
        b.vectorLoadShape(refs[1].buf.?, &.{ rhs_slot, b.cIndex(0), b.cIndex(0) }, &.{ 1, cfg.tile_k, rhs_tile_n }),
        &.{ cfg.tile_k, rhs_tile_n },
    );
    const valid_k = @mod(cfg.size_k, cfg.tile_k);
    if (valid_k != 0) {
        const k_iota = b.iota(&.{ cfg.tile_k, rhs_tile_n }, .i32, &.{0});
        const valid = b.splat(@as(i32, @intCast(valid_k)), &.{ cfg.tile_k, rhs_tile_n }, .i32);
        rhs = b.select(b.cmpi(.slt, k_iota, valid), rhs, b.zeros(&.{ cfg.tile_k, rhs_tile_n }, cfg.rhs_dtype));
    }
    var acc = matmulByMxuColumns(b, cfg, lhs, rhs, rhs_tile_n);
    acc = applyFuseAct(b, cfg, acc);

    const m_start = metaLoad(b, ctx.meta.m_offset, gm_id);
    const m_end = metaLoad(b, ctx.meta.m_offset, b.addi(gm_id, b.lift(@as(i32, 1))));
    const sublane = b.lift(@as(i32, @intCast(cfg.size_lhs_sublane)));
    const m_offset = b.subi(m_start, floorModI32(b, m_start, sublane));
    const m_start_local = b.subi(m_start, m_offset);
    const m_end_local = b.subi(m_end, m_offset);
    const row_iota = b.iota(&.{ cfg.tile_m, cfg.tile_n }, .i32, &.{0});
    const row_mask = b.andi(
        b.cmpi(.sle, b.broadcastTo(m_start_local, &.{ cfg.tile_m, cfg.tile_n }), row_iota),
        b.cmpi(.slt, row_iota, b.broadcastTo(m_end_local, &.{ cfg.tile_m, cfg.tile_n })),
    );
    acc = b.select(row_mask, acc, b.zeros(&.{ cfg.tile_m, cfg.tile_n }, cfg.acc_dtype));

    const out_slot_i32 = refs[2].slot.?;
    const out_slot = b.toIndex(out_slot_i32);
    b.vectorStoreAt(
        refs[2].buf.?,
        b.shapeCast(b.shapeCast(acc, out_shape).to(cfg.out_dtype), &.{ 1, cdiv(cfg.tile_m, cfg.size_lhs_sublane), cfg.size_lhs_sublane, cfg.tile_n }),
        &.{ out_slot, b.cIndex(0), b.cIndex(0), b.cIndex(0) },
    );

    const out_ref = winRef(b, refs[2], out_shape);
    if (cfg.tile_m == cfg.size_lhs_sublane) {
        const out_value = b.shapeCast(b.refLoad(out_ref), &.{ cfg.tile_m, cfg.tile_n });
        const partial_zero = b.zeros(&.{ cfg.tile_m, cfg.tile_n }, cfg.out_dtype);
        const partial_prev = b.select(
            b.cmpi(.eq, gm_id, b.lift(@as(i32, 0))),
            partial_zero,
            b.refLoad(scratches[0]),
        );
        const out_accum = b.addf(out_value, partial_prev);
        const out_ref_for_store = winRef(b, refs[2], out_shape);
        b.refStore(out_ref_for_store, b.shapeCast(out_accum, out_shape));
    } else {
        const first_row = b.shapeCast(
            b.vectorLoadShape(out_ref, &.{ b.cIndex(0), b.cIndex(0), b.cIndex(0) }, &.{ 1, cfg.size_lhs_sublane, cfg.tile_n }),
            &.{ cfg.size_lhs_sublane, cfg.tile_n },
        );
        const partial_zero = b.zeros(&.{ cfg.size_lhs_sublane, cfg.tile_n }, cfg.out_dtype);
        const partial_prev = b.select(
            b.cmpi(.eq, gm_id, b.lift(@as(i32, 0))),
            partial_zero,
            b.refLoad(scratches[0]),
        );
        const out_accum = b.addf(first_row, partial_prev);
        const out_ref_for_store = winRef(b, refs[2], out_shape);
        b.vectorStoreAt(
            out_ref_for_store,
            b.shapeCast(out_accum, &.{ 1, cfg.size_lhs_sublane, cfg.tile_n }),
            &.{ b.cIndex(0), b.cIndex(0), b.cIndex(0) },
        );
    }

    const last_row = floorDivPosI32(b, m_end_local, sublane);
    const m_end_mod = floorModI32(b, m_end_local, sublane);
    const partial_complete = b.cmpi(.eq, m_end_mod, b.lift(@as(i32, 0)));
    const out_ref_for_partial = winRef(b, refs[2], out_shape);
    const partial_shape = if (cfg.tile_m == cfg.size_lhs_sublane)
        &.{ cfg.tile_m, cfg.tile_n }
    else
        &.{ cfg.size_lhs_sublane, cfg.tile_n };
    const partial_zero = b.zeros(partial_shape, cfg.out_dtype);
    const partial_row = b.shapeCast(
        b.vectorLoadShape(out_ref_for_partial, &.{ b.toIndex(last_row), b.cIndex(0), b.cIndex(0) }, &.{ 1, cfg.size_lhs_sublane, cfg.tile_n }),
        partial_shape,
    );
    b.refStore(scratches[0], b.select(partial_complete, partial_zero, partial_row));
    b.traceStop();
}

fn innerBody1D(
    b: *mtt.Builder,
    indices: []const mtt.Value,
    refs: []const pipeline.RefWindow,
    scratches: []const mtt.Value,
    ctx_opaque: ?*anyopaque,
) mtt.FinishError!void {
    try innerBody(b, indices1DTo3D(b, indices), refs, scratches, ctx_opaque);
}

pub fn run(b: *mtt.Builder, cfg: Cfg) mtt.FinishError!void {
    validateConfig(cfg);

    const a = try b.declareArgsOpts(.{
        .group_sizes = .{ .ref = .{ .shape = &.{cfg.size_lhs_group}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },
        .group_offset = .{ .ref = .{ .shape = &.{1}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },
        .lhs = .{ .ref = .{ .shape = &.{ cfg.size_m, cfg.size_k }, .dtype = cfg.lhs_dtype, .memory_space = .hbm } },
        .rhs = .{ .ref = .{ .shape = &.{ cfg.size_group, cfg.size_k, cfg.size_n }, .dtype = cfg.rhs_dtype, .memory_space = .hbm } },
        .out = .{ .ref = .{ .shape = &.{ cfg.size_m, cfg.alignedN() }, .dtype = cfg.out_dtype, .memory_space = .hbm, .role = .output } },
        .partial_out = .{ .ref = .{ .shape = &.{ cfg.size_lhs_sublane, cfg.tile_n }, .dtype = cfg.out_dtype, .role = .scratch } },
        .acc = .{ .ref = .{ .shape = &.{ cfg.tile_m, cfg.tile_n }, .dtype = cfg.acc_dtype, .role = .scratch } },
        .metadata_group_id = .{ .ref = .{ .shape = &.{cfg.maxNumGm()}, .dtype = .i32, .memory_space = .smem, .role = .scratch } },
        .metadata_m_offset = .{ .ref = .{ .shape = &.{cfg.maxNumGm() + 1}, .dtype = .i32, .memory_space = .smem, .role = .scratch } },
    }, &.{}, .{
        .dimension_semantics = &.{},
        .scalar_prefetch = 2,
        .scratch_operands = 4,
    });

    const num_k = b.lift(@as(i32, @intCast(cdiv(cfg.size_k, cfg.tile_k))));
    const num_n = b.lift(@as(i32, @intCast(cdiv(cfg.alignedN(), cfg.tile_n))));
    const num_gm = fillMetadata(b, cfg, a);

    const ctx = b.arena.allocator().create(PipeCtx) catch @panic("gmm_ep PipeCtx OOM");
    ctx.* = .{
        .cfg = cfg,
        .meta = .{ .group_id = a.metadata_group_id, .m_offset = a.metadata_m_offset },
        .num_gm = num_gm,
    };

    const lhs_spec = pipeline.PipeSpec{
        .full_shape = &.{ @divExact(cfg.size_m, cfg.size_lhs_sublane), cfg.size_lhs_sublane, cfg.size_k },
        .dtype = cfg.lhs_dtype,
        .source_reshape_shape = &.{ @divExact(cfg.size_m, cfg.size_lhs_sublane), cfg.size_lhs_sublane, cfg.size_k },
        .block_shape = &.{ .{ .bounded_slice = cdiv(cfg.tile_m, cfg.size_lhs_sublane) }, .{ .blocked = cfg.size_lhs_sublane }, .{ .blocked = cfg.tile_k } },
        .index_map = if (cdiv(cfg.alignedN(), cfg.tile_n) == 1 and cdiv(cfg.size_k, cfg.tile_k) == 1 and cfg.size_m > cfg.tile_m) lhsIndexMap1D else lhsIndexMap,
        .index_map_ctx = ctx,
        .bounded_slice_tiling = 1,
    };
    const rhs_spec = pipeline.PipeSpec{
        .full_shape = &.{ cfg.size_group, cfg.size_k, cfg.size_n },
        .dtype = cfg.rhs_dtype,
        .block_shape = &.{ .squeezed, .{ .blocked = cfg.tile_k }, .{ .blocked = cfg.rhsTileN() } },
        .index_map = if (cdiv(cfg.alignedN(), cfg.tile_n) == 1 and cdiv(cfg.size_k, cfg.tile_k) == 1 and cfg.size_m > cfg.tile_m) rhsIndexMap1D else rhsIndexMap,
        .index_map_ctx = ctx,
        .buffer_count = 3,
    };
    const out_spec = pipeline.PipeSpec{
        .full_shape = &.{ @divExact(cfg.size_m, cfg.size_lhs_sublane), cfg.size_lhs_sublane, cfg.alignedN() },
        .dtype = cfg.out_dtype,
        .source_reshape_shape = &.{ @divExact(cfg.size_m, cfg.size_lhs_sublane), cfg.size_lhs_sublane, cfg.alignedN() },
        .block_shape = &.{ .{ .bounded_slice = cdiv(cfg.tile_m, cfg.size_lhs_sublane) }, .{ .blocked = cfg.size_lhs_sublane }, .{ .blocked = cfg.tile_n } },
        .index_map = if (cdiv(cfg.alignedN(), cfg.tile_n) == 1 and cdiv(cfg.size_k, cfg.tile_k) == 1 and cfg.size_m > cfg.tile_m) outIndexMap1D else outIndexMap,
        .index_map_ctx = ctx,
        .bounded_slice_tiling = 1,
    };
    const use_1d_pipeline = cdiv(cfg.alignedN(), cfg.tile_n) == 1 and cdiv(cfg.size_k, cfg.tile_k) == 1 and cfg.size_m > cfg.tile_m;
    try pipeline.emitPipeline(b, &.{ a.lhs, a.rhs, a.out }, &.{ a.partial_out, a.acc, a.metadata_group_id, a.metadata_m_offset }, .{
        .body = if (use_1d_pipeline) innerBody1D else innerBody,
        .body_ctx = ctx,
        .grid = if (use_1d_pipeline) &.{num_gm} else &.{ num_n, num_gm, num_k },
        .in_specs = &.{ lhs_spec, rhs_spec },
        .out_specs = &.{out_spec},
    });
}

pub const TpuKernel = mtt_kernel.Kernel(Cfg, .{
    .name = "gmm_ep_kernel",
    .inputs = &.{ "group_sizes", "group_offset", "lhs", "rhs" },
    .outputs = &.{"out"},
    .run = run,
});
