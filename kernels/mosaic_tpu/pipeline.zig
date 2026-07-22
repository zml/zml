//! Zig port of `pltpu.emit_pipeline` (jax/_src/pallas/mosaic/pipeline.py) —
//! emits lowered `tpu`-dialect IR byte-equivalent to `pallas_call`.
//!
//! `.blocked`/`.squeezed`/one `.bounded_slice` block dim; trivial-windowing
//! inputs peeled as sync copy; `trace_scopes`. Unsupported features panic.

const std = @import("std");
const mlir = @import("mlir");
const dialects = @import("mlir/dialects");
const builder = @import("builder.zig");

const tpu = @import("mlir/dialects/mosaic_tpu");
const arith = dialects.arith;
const scf = dialects.scf;
const memref = dialects.memref;

const Builder = builder.Builder;
const Value = builder.Value;
const DType = builder.DType;
const MemorySpace = builder.MemorySpace;
const FinishError = builder.FinishError;

/// `ShapedType::kDynamic` — sentinel for a `?` dim in a memref shape.
const kDyn: i64 = std.math.minInt(i64);

/// One entry of `pl.BlockSpec.block_shape`.
pub const BlockDim = union(enum) {
    /// `pl.Blocked(n)` — windowed dim of size `n`.
    blocked: i64,
    /// `pl.Squeezed()` / `None` — dim dropped in the window.
    squeezed,
    /// `pl.BoundedSlice(n)` — index_map returns `.slice`; `n` is the max.
    bounded_slice: i64,
};

/// One entry returned by `pl.BlockSpec.index_map`.
pub const BlockIndex = union(enum) {
    /// `pl.ds(idx*block, block)` — for `.blocked`/`.squeezed` dims.
    scalar: Value,
    /// `pl.ds(start, size)` — for `.bounded_slice` dims.
    slice: struct { start: Value, size: Value },
};

/// `pl.BlockSpec.index_map`. `ctx` is opaque user data (Zig has no closures).
pub const IndexMapFn = *const fn (b: *Builder, indices: []const Value, ctx: ?*anyopaque) []const BlockIndex;

/// Working window for one pipeline operand. Body must re-emit
/// `memref_slice`/`squeeze` at each use site. `slot == null` ⇒ trivial-windowing;
/// `buf == null` ⇒ no operand.
pub const RefWindow = struct { buf: ?Value = null, slot: ?Value = null };

/// The pipeline body — `pallas_call`'s inner kernel.
pub const BodyFn = *const fn (b: *Builder, grid_indices: []const Value, refs: []const RefWindow, scratches: []const Value, ctx: ?*anyopaque) FinishError!void;

// Optional overrides for a buffered reference's DMA operations.
// Unspecified operations retain the standard pipeline behavior.
pub const BufferedRefStrategy = struct { ctx: ?*anyopaque = null, copy_in: ?*const fn (br: *BufferedRef, b: *Builder, indices: []const Value, slot_cumulative: Value, static_zero_start: bool, ctx: ?*anyopaque) void = null, wait_in: ?*const fn (br: *BufferedRef, b: *Builder, indices: []const Value, ctx: ?*anyopaque) void = null, copy_out: ?*const fn (br: *BufferedRef, b: *Builder, indices: []const Value, ctx: ?*anyopaque) void = null, wait_out: ?*const fn (br: *BufferedRef, b: *Builder, indices: []const Value, ctx: ?*anyopaque) void = null };

/// `pl.BlockSpec` for one pipeline operand. `null` slot ↔ `None` spec.
pub const PipeSpec = struct {
    full_shape: []const i64,
    dtype: DType,
    /// Optional HBM source/result view to materialize at each DMA use site.
    source_reshape_shape: ?[]const i64 = null,
    /// `null` ⇒ trivial windowing (the whole array).
    block_shape: ?[]const BlockDim = null,
    /// `null` only allowed with trivial windowing.
    index_map: ?IndexMapFn = null,
    index_map_ctx: ?*anyopaque = null,
    /// Only `.vmem` supported.
    memory_space: MemorySpace = .vmem,

    buffer_count: ?usize = null,
    /// Tile that `_create_bounded_slice` rounds the DMA size up to.
    bounded_slice_tiling: i32 = 1,
    /// Pallas's `disable_bounds_checks`: skip OOB clamp, emit only `tpu.assume_multiple`.
    bounded_slice_skip_clamp: bool = false,

    /// Optional custom DMA behavior for this operand.
    transfer_strategy: ?BufferedRefStrategy = null,
    /// Override the role infered from whether this spec appears
    /// in `in_specs` or `out_specs`
    buffer_type: ?BufferType = null,
};

pub const PipelineOpts = struct {
    body: BodyFn,
    body_ctx: ?*anyopaque = null,
    /// Dynamic (`i32`) grid bounds. Comptime: pass `b.lift(@as(i32, N))`.
    grid: []const Value,
    /// One per HBM input ref (slot `null` <-> `None` spec).
    in_specs: []const ?PipeSpec,
    out_specs: []const ?PipeSpec,
    trace_scopes: bool = true,
};

/// Emits `pltpu.emit_pipeline(body, grid=..., in_specs=..., out_specs=...)(*refs, scratches=...)`.
/// `refs` = `in_refs ++ out_refs`. Emits into `b.currentBlock()`.
pub fn emitPipeline(
    b: *Builder,
    refs: []const ?Value,
    scratches: []const Value,
    opts: PipelineOpts,
) FinishError!void {
    std.debug.assert(opts.grid.len >= 1);
    if (opts.grid.len > 4) @panic("emitPipeline: grids with rank > 4 unsupported");
    const n_in = opts.in_specs.len;
    std.debug.assert(refs.len == n_in + opts.out_specs.len);

    const a = b.arena.allocator();

    var brefs: std.ArrayList(BufferedRef) = try .initCapacity(a, opts.in_specs.len + opts.out_specs.len);
    for (opts.in_specs, 0..) |maybe_spec, i| {
        if (maybe_spec) |spec| {
            std.debug.assert(refs[i] != null);
            brefs.appendAssumeCapacity(BufferedRef.create(b, spec, .input, refs[i].?, i));
        }
    }
    for (opts.out_specs, 0..) |maybe_spec, j| {
        if (maybe_spec) |spec| {
            std.debug.assert(refs[n_in + j] != null);
            brefs.appendAssumeCapacity(BufferedRef.create(b, spec, .output, refs[n_in + j].?, n_in + j));
        }
    }

    // Whole pipeline lives in a `tpu.region`.
    const region_block = mlir.Block.init(&.{}, &.{});
    b.pushBlock(region_block);

    for (brefs.items) |*br| br.allocate(b);

    const num_steps = gridSize(b, opts.grid);
    const c0_i32 = b.lift(@as(i32, 0));
    const c1_i32 = b.lift(@as(i32, 1));
    {
        var when = b.openIf(b.cmpi(.sgt, num_steps, c0_i32));
        {
            for (brefs.items) |*br| if (br.is_trivial and (br.buffer_type == .input or br.buffer_type == .input_output)) br.syncCopyIn(b);

            const num_stages = blk: {
                var m: usize = 2;
                for (brefs.items) |br| if (br.buffer_count > m) {
                    m = br.buffer_count;
                };
                break :blk m;
            };
            var sched0 = Scheduler.init(b, c0_i32, opts.grid, num_steps, num_stages, opts.trace_scopes);
            for (0..num_stages - 1) |step| {
                if (opts.trace_scopes) {
                    var buf: [32]u8 = undefined;
                    b.traceStart(10, std.fmt.bufPrint(&buf, "ep_initialize_{d}", .{step}) catch unreachable);
                }
                for (brefs.items) |*br| sched0.initializeStep(b, br, step);
                if (opts.trace_scopes) b.traceStop();
            }

            // iter_args: each bref's slots in spec order (see appendSlotInits), then grid indices.
            var inits: std.ArrayList(*const mlir.Value) = try .initCapacity(a, brefs.items.len * 4 + opts.grid.len);
            for (brefs.items) |*br| br.appendSlotInits(b, &inits);
            for (0..opts.grid.len) |_| inits.appendAssumeCapacity(c0_i32.inner);
            const n_slots = inits.items.len - opts.grid.len;

            const i32_ty = b.scalarTy(.i32);
            const block_types = a.alloc(*const mlir.Type, inits.items.len + 1) catch unreachable;
            const block_locs = a.alloc(*const mlir.Location, inits.items.len + 1) catch unreachable;
            for (block_types) |*t| t.* = i32_ty;
            for (block_locs) |*l| l.* = b.loc();
            const for_body = mlir.Block.init(block_types, block_locs);
            b.pushBlock(for_body);

            const iv: Value = .{ .inner = for_body.argument(0), .kernel = b };
            {
                var cur: usize = 1;
                for (brefs.items) |*br| cur = br.bindSlotsAt(b, for_body, cur);
                std.debug.assert(cur == n_slots + 1);
            }
            var grid_indices_buf: [4]Value = undefined;
            for (0..opts.grid.len) |i| {
                grid_indices_buf[i] = .{ .inner = for_body.argument(@intCast(n_slots + 1 + i)), .kernel = b };
            }

            var sched = Scheduler.init(b, iv, opts.grid, num_steps, num_stages, opts.trace_scopes);
            for (0..opts.grid.len) |i| sched.indices_storage[i] = grid_indices_buf[i];
            sched.recomputeDerivedIndices(b);

            for (brefs.items) |*br| {
                if (br.buffer_type == .input or br.buffer_type == .input_output)
                    sched.copyIn(b, br);
            }
            for (brefs.items) |*br| {
                if (br.buffer_type == .input or br.buffer_type == .input_output)
                    sched.waitIn(b, br);
            }
            // Emit slot `rem` ops BEFORE the body; body re-emits slice/squeeze per use.
            var current_refs = a.alloc(RefWindow, refs.len) catch unreachable;
            for (current_refs) |*r| r.* = .{};
            for (brefs.items) |*br| {
                if (br.is_trivial or !br.isBuffered()) {
                    current_refs[br.spec_index] = .{ .buf = br.window_ref };
                } else {
                    const active = switch (br.buffer_type) {
                        .input => br.wait_in_slot.?,
                        .output, .input_output => br.copy_out_slot.?,
                    };
                    current_refs[br.spec_index] = .{ .buf = br.window_ref, .slot = b.remui(active, b.lift(@as(i32, @intCast(br.buffer_count)))) };
                }
            }

            if (opts.trace_scopes) b.traceStart(10, "ep_run_kernel");
            try opts.body(b, sched.indices_storage[0..opts.grid.len], current_refs, scratches, opts.body_ctx);
            if (opts.trace_scopes) b.traceStop();

            for (brefs.items) |*br| {
                if (br.buffer_type == .output or br.buffer_type == .input_output)
                    sched.copyOut(b, br);
            }
            for (brefs.items) |*br| {
                if (br.buffer_type == .output or br.buffer_type == .input_output)
                    sched.waitOut(b, br);
            }

            for (brefs.items) |*br| {
                if (br.buffer_type == .input or br.buffer_type == .input_output)
                    sched.advanceSlots(b, br);
            }
            var yields: std.ArrayList(*const mlir.Value) = try .initCapacity(a, brefs.items.len * 4 + opts.grid.len);
            for (brefs.items) |*br| br.appendSlotYields(b, &yields);
            const next_indices = nextIndexND(b, sched.indices(), opts.grid);
            for (next_indices) |idx| yields.appendAssumeCapacity(idx.inner);
            _ = scf.yield(b.ctx, yields.items, b.loc()).appendTo(for_body);
            b.popBlock();

            const for_op = scf.for_(b.ctx, c0_i32.inner, num_steps.inner, c1_i32.inner, inits.items, for_body, .{}, b.loc());
            _ = for_op.appendTo(b.currentBlock());
            var k: usize = 0;
            for (brefs.items) |*br| k = br.readbackSlots(b, for_op, k);
            var final_grid_indices_buf: [4]Value = undefined;
            for (0..opts.grid.len) |i| {
                final_grid_indices_buf[i] = .{ .inner = for_op.result(@intCast(n_slots + i)), .kernel = b };
            }

            // prev_index BEFORE Scheduler.init so common arithmetic CSEs with `step`.
            const final_idx = prevIndexND(b, final_grid_indices_buf[0..opts.grid.len], opts.grid);
            var sched_fin = Scheduler.init(b, b.subi(num_steps, c1_i32), opts.grid, num_steps, num_stages, opts.trace_scopes);
            for (0..opts.grid.len) |i| sched_fin.indices_storage[i] = final_idx[i];
            sched_fin.recomputeDerivedIndices(b);
            for (brefs.items) |*br| sched_fin.finalize(b, br);
            for (brefs.items) |*br| if (br.is_trivial and (br.buffer_type == .output or br.buffer_type == .input_output)) br.syncCopyOut(b);

            when.yieldThen(.{});
        }
    }

    _ = tpu.yield(b.ctx, &.{}, b.loc()).appendTo(region_block);
    b.popBlock();
    _ = tpu.region(b.ctx, region_block, &.{}, b.loc()).appendTo(b.currentBlock());
}

pub const BufferType = enum { input, output, input_output };

/// VMEM double-buffer + DMA semaphores for one pipeline operand.
pub const BufferedRef = struct {
    spec: PipeSpec,
    buffer_type: BufferType,
    buffer_count: usize,
    is_trivial: bool,
    /// Index into `refs` / `current_refs`.
    spec_index: usize,
    src: Value,
    /// Block shape with `.squeezed` dims removed.
    block_compact: []const i64,
    /// Indices into `spec.block_shape` parallel to `block_compact`.
    block_keep: []const usize,

    /// `memref<buffer_count x block_compact…, vmem>` (or full shape if trivial).
    window_ref: Value = undefined,
    sem_recvs: ?Value = null,
    sem_sends: ?Value = null,

    // Slot iter_arg state — see appendSlotInits for init values.
    copy_in_slot: ?Value = null,
    wait_in_slot: ?Value = null,
    copy_out_slot: ?Value = null,
    wait_out_slot: ?Value = null,
    slot: ?Value = null,

    fn create(b: *Builder, spec: PipeSpec, bt: BufferType, src: Value, spec_index: usize) BufferedRef {
        const a = b.arena.allocator();
        const buffer_type = spec.buffer_type orelse bt;
        if (spec.memory_space != .vmem) @panic("emitPipeline: only VMEM operands supported");
        const trivial = specHasTrivialWindowing(spec);
        var bc: usize = if (spec.buffer_count) |c| c else 2;
        if (spec.buffer_count == null and trivial) bc = 1;
        if (buffer_type != .input and bc > 2) @panic("emitPipeline: output and input_output buffer_count > 2 unsupported");

        const block_len = if (spec.block_shape) |bs| bs.len else 0;
        var keep = std.ArrayList(usize).initCapacity(a, block_len) catch unreachable;
        var compact = std.ArrayList(i64).initCapacity(a, block_len) catch unreachable;
        if (spec.block_shape) |bs| {
            for (bs, 0..) |bd, di| switch (bd) {
                .blocked => |n| {
                    keep.appendAssumeCapacity(di);
                    compact.appendAssumeCapacity(n);
                },
                .bounded_slice => |n| {
                    keep.appendAssumeCapacity(di);
                    compact.appendAssumeCapacity(n);
                },
                .squeezed => {},
            };
        }
        return .{
            .spec = spec,
            .buffer_type = buffer_type,
            .buffer_count = bc,
            .is_trivial = trivial,
            .spec_index = spec_index,
            .src = src,
            .block_compact = compact.items,
            .block_keep = keep.items,
        };
    }

    fn isBuffered(self: *const BufferedRef) bool {
        return self.buffer_count > 0;
    }

    fn allocate(self: *BufferedRef, b: *Builder) void {
        const a = b.arena.allocator();
        const vmem = MemorySpace.vmem.attribute(b.ctx);
        const elem = self.spec.dtype.toMlir(b.ctx);
        if (self.is_trivial) {
            const ty = mlir.Type.memRef(elem, self.spec.full_shape, null, vmem);
            self.window_ref = memRefAlloca(b, ty);
        } else {
            if (self.block_compact.len == 1) @panic("emitPipeline: 1-D flat block buffers unsupported (TODO)");
            var shape = a.alloc(i64, self.block_compact.len + 1) catch unreachable;
            shape[0] = @intCast(self.buffer_count);
            @memcpy(shape[1..], self.block_compact);
            const ty = mlir.Type.memRef(elem, shape, null, vmem);
            self.window_ref = memRefAlloca(b, ty);
        }
        if (!self.is_trivial) {
            const sem_arr_ty = dmaSemRefTy(b, self.buffer_count);
            if (self.buffer_type == .input or self.buffer_type == .input_output) self.sem_recvs = semAllocOp(b, sem_arr_ty);
            if (self.buffer_type == .output or self.buffer_type == .input_output) self.sem_sends = semAllocOp(b, sem_arr_ty);
        }
    }

    fn appendSlotInits(self: *BufferedRef, b: *Builder, out: *std.ArrayList(*const mlir.Value)) void {
        const c0 = b.lift(@as(i32, 0)).inner;
        switch (self.buffer_type) {
            .input => {
                if (self.is_trivial) {
                    out.appendAssumeCapacity(c0);
                } else {
                    out.appendAssumeCapacity((self.copy_in_slot orelse b.lift(@as(i32, 0))).inner);
                    out.appendAssumeCapacity(c0);
                }
            },
            .output => {
                out.appendAssumeCapacity(c0);
                out.appendAssumeCapacity(c0);
            },
            .input_output => {
                out.appendAssumeCapacity(
                    (self.copy_in_slot orelse b.lift(@as(i32, 0))).inner,
                );
                out.appendAssumeCapacity(c0);
                out.appendAssumeCapacity(c0);
                out.appendAssumeCapacity(c0);
            },
        }
    }

    fn nSlots(self: *const BufferedRef) usize {
        return switch (self.buffer_type) {
            .input => if (self.is_trivial) 1 else 2,
            .output => 2,
            .input_output => 4,
        };
    }

    fn bindSlotsAt(self: *BufferedRef, b: *Builder, body: *mlir.Block, start: usize) usize {
        var k = start;
        const arg = struct {
            fn f(b_: *Builder, body_: *mlir.Block, i: *usize) Value {
                const v: Value = .{ .inner = body_.argument(@intCast(i.*)), .kernel = b_ };
                i.* += 1;
                return v;
            }
        }.f;
        switch (self.buffer_type) {
            .input => {
                if (self.is_trivial) {
                    self.slot = arg(b, body, &k);
                } else {
                    self.copy_in_slot = arg(b, body, &k);
                    self.wait_in_slot = arg(b, body, &k);
                }
            },
            .output => {
                self.copy_out_slot = arg(b, body, &k);
                self.wait_out_slot = arg(b, body, &k);
            },
            .input_output => {
                self.copy_in_slot = arg(b, body, &k);
                self.wait_in_slot = arg(b, body, &k);
                self.copy_out_slot = arg(b, body, &k);
                self.wait_out_slot = arg(b, body, &k);
            },
        }
        return k;
    }

    fn appendSlotYields(self: *BufferedRef, b: *Builder, out: *std.ArrayList(*const mlir.Value)) void {
        _ = b;
        switch (self.buffer_type) {
            .input => {
                if (self.is_trivial) {
                    out.appendAssumeCapacity(self.slot.?.inner);
                } else {
                    out.appendAssumeCapacity(self.copy_in_slot.?.inner);
                    out.appendAssumeCapacity(self.wait_in_slot.?.inner);
                }
            },
            .output => {
                out.appendAssumeCapacity(self.copy_out_slot.?.inner);
                out.appendAssumeCapacity(self.wait_out_slot.?.inner);
            },
            .input_output => {
                out.appendAssumeCapacity(self.copy_in_slot.?.inner);
                out.appendAssumeCapacity(self.wait_in_slot.?.inner);
                out.appendAssumeCapacity(self.copy_out_slot.?.inner);
                out.appendAssumeCapacity(self.wait_out_slot.?.inner);
            },
        }
    }

    fn readbackSlots(self: *BufferedRef, b: *Builder, for_op: *mlir.Operation, start: usize) usize {
        var k = start;
        const take = struct {
            fn f(b_: *Builder, op: *mlir.Operation, i: *usize) Value {
                const v: Value = .{ .inner = op.result(@intCast(i.*)), .kernel = b_ };
                i.* += 1;
                return v;
            }
        }.f;
        switch (self.buffer_type) {
            .input => {
                if (self.is_trivial) {
                    self.slot = take(b, for_op, &k);
                } else {
                    self.copy_in_slot = take(b, for_op, &k);
                    self.wait_in_slot = take(b, for_op, &k);
                }
            },
            .output => {
                self.copy_out_slot = take(b, for_op, &k);
                self.wait_out_slot = take(b, for_op, &k);
            },
            .input_output => {
                self.copy_in_slot = take(b, for_op, &k);
                self.wait_in_slot = take(b, for_op, &k);
                self.copy_out_slot = take(b, for_op, &k);
                self.wait_out_slot = take(b, for_op, &k);
            },
        }
        return k;
    }

    fn computeIndex(self: *const BufferedRef, b: *Builder, indices: []const Value) []const BlockIndex {
        if (self.spec.index_map) |im| return im(b, indices, self.spec.index_map_ctx);
        @panic("emitPipeline: index_map required for non-trivial windowing");
    }

    /// Per `block_shape` dim: `{base, size?}`. `size == null` ⇒ static-full.
    /// `static_zero_start` enables prologue shortcut (Python-int 0 start folds).
    const DmaSlice = struct { base: []Value, size: []?Value };
    fn getDmaSlice(self: *const BufferedRef, b: *Builder, indices: []const Value, static_zero_start: bool) DmaSlice {
        const a = b.arena.allocator();
        const bs = self.spec.block_shape.?;
        const bidx = self.computeIndex(b, indices);
        std.debug.assert(bidx.len == bs.len);
        var base = a.alloc(Value, bs.len) catch unreachable;
        var size = a.alloc(?Value, bs.len) catch unreachable;
        for (bs, 0..) |bd, d| {
            switch (bd) {
                .squeezed => {
                    base[d] = switch (bidx[d]) {
                        .scalar => |v| v,
                        .slice => @panic("emitPipeline: squeezed dim must get a scalar index"),
                    };
                    size[d] = null;
                },
                .blocked => |n| {
                    const idx = switch (bidx[d]) {
                        .scalar => |v| v,
                        .slice => @panic("emitPipeline: blocked dim must get a scalar index"),
                    };
                    base[d] = if (n == 1) idx else idx.mul(@as(i32, @intCast(n)));
                    const dim_size = self.spec.full_shape[d];
                    if (@mod(dim_size, n) == 0) {
                        size[d] = null;
                    } else {
                        const tiling: i64 = if (d == bs.len - 1) 128 else if (bs.len >= 2 and d == bs.len - 2) 8 else 1;
                        const clamped = @min(n, dim_size);
                        const sz_const = @divFloor(clamped + tiling - 1, tiling) * tiling;
                        size[d] = b.emit(tpu.assume_multiple(b.ctx, b.lift(@as(i32, @intCast(sz_const))).inner, @as(i32, @intCast(tiling)), b.loc()));
                    }
                },
                .bounded_slice => {
                    const sl = switch (bidx[d]) {
                        .scalar => @panic("emitPipeline: bounded_slice dim must get a `.slice` index from index_map"),
                        .slice => |s| s,
                    };
                    const tiling: i32 = self.spec.bounded_slice_tiling;
                    const dim_size = self.spec.full_shape[d];
                    base[d] = sl.start;
                    if (self.spec.bounded_slice_skip_clamp) {
                        size[d] = b.emit(tpu.assume_multiple(b.ctx, sl.size.inner, tiling, b.loc()));
                        continue;
                    }
                    // Prologue shortcut: Python-int 0 start ⇒ `_round_up_to_nearest_multiple` folds.
                    if (static_zero_start and sl.start.asConstantInt() == @as(?i64, 0)) {
                        const is_oob = b.cmpi(.sgt, sl.size, b.lift(@as(i32, @intCast(dim_size))));
                        const sz = b.select(is_oob, b.lift(@as(i32, @intCast(dim_size))), sl.size);
                        size[d] = b.emit(tpu.assume_multiple(b.ctx, sz.inner, tiling, b.loc()));
                        continue;
                    }
                    const is_oob = b.cmpi(.sgt, b.addi(sl.start, sl.size), b.lift(@as(i32, @intCast(dim_size))));
                    const rounded = blk: {
                        if (tiling == 1) break :blk b.subi(b.lift(@as(i32, @intCast(dim_size + 1))), sl.start);
                        // lax sign-fixup for `jnp.remainder`.
                        const m = b.lift(tiling);
                        const c0 = b.lift(@as(i32, 0));
                        const rem_arg = b.subi(b.lift(@as(i32, @intCast(dim_size))), sl.start);
                        const r = b.remsi(rem_arg, m);
                        const r_nz = b.cmpi(.ne, r, c0);
                        const r_neg = b.cmpi(.slt, r, c0);
                        const fixup = b.andi(r_neg, r_nz);
                        const r_plus = b.addi(r, m);
                        const floormod = b.select(fixup, r_plus, r);
                        const round_down = b.subi(rem_arg, floormod);
                        break :blk b.addi(round_down, m);
                    };
                    const sz = b.select(is_oob, rounded, sl.size);
                    size[d] = b.emit(tpu.assume_multiple(b.ctx, sz.inner, tiling, b.loc()));
                },
            }
        }
        return .{ .base = base, .size = size };
    }

    fn sliceSrc(self: *const BufferedRef, b: *Builder, dma: DmaSlice) Value {
        const a = b.arena.allocator();
        const bs = self.spec.block_shape.?;
        const src = if (self.spec.source_reshape_shape) |shape| b.memRefReshape(self.src, shape) else self.src;
        var base = a.alloc(Value, bs.len) catch unreachable;
        var dyn = std.ArrayList(Value).initCapacity(a, bs.len) catch unreachable;
        var result_shape = a.alloc(i64, bs.len) catch unreachable;
        for (bs, 0..) |bd, d| {
            base[d] = dma.base[d];
            if (dma.size[d]) |sz| {
                result_shape[d] = kDyn;
                dyn.appendAssumeCapacity(sz);
            } else {
                result_shape[d] = switch (bd) {
                    .squeezed => 1,
                    .blocked => |n| n,
                    .bounded_slice => unreachable,
                };
            }
        }
        const sliced = b.memRefSlice(src, base, result_shape, dyn.items);
        if (self.block_keep.len == bs.len) return sliced;

        const squeezed_shape = a.alloc(i64, self.block_compact.len) catch unreachable;
        for (self.block_keep, 0..) |sd, ci| squeezed_shape[ci] = result_shape[sd];
        return b.memRefSqueeze(sliced, squeezed_shape);
    }

    /// Slices `alloca[slot, 0…]` (non-squeezed dims) then squeezes the slot dim.
    fn sliceWindowAt(self: *const BufferedRef, b: *Builder, slot: Value, dma: DmaSlice) Value {
        const a = b.arena.allocator();
        const bs = self.spec.block_shape.?;
        const c0 = b.lift(@as(i32, 0));
        const nd = 1 + self.block_compact.len;
        var base = a.alloc(Value, nd) catch unreachable;
        var result_shape = a.alloc(i64, nd) catch unreachable;
        var dyn = std.ArrayList(Value).initCapacity(a, self.block_keep.len) catch unreachable;
        base[0] = slot;
        result_shape[0] = 1;
        for (self.block_keep, 0..) |sd, ci| {
            base[ci + 1] = c0;
            if (dma.size[sd]) |sz| {
                result_shape[ci + 1] = kDyn;
                dyn.appendAssumeCapacity(sz);
            } else {
                result_shape[ci + 1] = switch (bs[sd]) {
                    .blocked => |n| n,
                    else => unreachable,
                };
            }
        }
        const sliced = b.memRefSlice(self.window_ref, base, result_shape, dyn.items);
        const squeezed_shape = a.alloc(i64, self.block_compact.len) catch unreachable;
        @memcpy(squeezed_shape, result_shape[1..]);
        return b.memRefSqueeze(sliced, squeezed_shape);
    }

    fn semAt(b: *Builder, sem_array: Value, i: Value) Value {
        const sliced = b.memRefSlice(sem_array, &.{i}, &.{1}, &.{});
        return b.memRefSqueeze(sliced, &.{});
    }

    fn currentRef(self: *BufferedRef, b: *Builder) Value {
        if (self.is_trivial or !self.isBuffered()) return self.window_ref;
        const slot = switch (self.buffer_type) {
            .output => b.remui(self.copy_out_slot.?, b.lift(@as(i32, @intCast(self.buffer_count)))),
            .input => b.remui(self.wait_in_slot.?, b.lift(@as(i32, @intCast(self.buffer_count)))),
        };
        const a = b.arena.allocator();
        const nd = 1 + self.block_compact.len;
        var base = a.alloc(Value, nd) catch unreachable;
        var result_shape = a.alloc(i64, nd) catch unreachable;
        const c0 = b.lift(@as(i32, 0));
        base[0] = slot;
        result_shape[0] = 1;
        for (self.block_compact, 0..) |n, ci| {
            base[ci + 1] = c0;
            result_shape[ci + 1] = n;
        }
        const sliced = b.memRefSlice(self.window_ref, base, result_shape, &.{});
        return b.memRefSqueeze(sliced, self.block_compact);
    }

    fn syncCopyIn(self: *BufferedRef, b: *Builder) void {
        const blk = mlir.Block.init(&.{}, &.{});
        b.pushBlock(blk);
        const sem = semAllocOp(b, dmaSemRefTy(b, null));
        b.enqueueDma(self.src, self.window_ref, sem, .{});
        b.waitDma2(sem, self.src, self.window_ref, .{});
        _ = tpu.yield(b.ctx, &.{}, b.loc()).appendTo(blk);
        b.popBlock();
        _ = tpu.region(b.ctx, blk, &.{}, b.loc()).appendTo(b.currentBlock());
    }

    fn syncCopyOut(self: *BufferedRef, b: *Builder) void {
        const blk = mlir.Block.init(&.{}, &.{});
        b.pushBlock(blk);
        const sem = semAllocOp(b, dmaSemRefTy(b, null));
        b.enqueueDma(self.window_ref, self.src, sem, .{});
        b.waitDma2(sem, self.window_ref, self.src, .{});
        _ = tpu.yield(b.ctx, &.{}, b.loc()).appendTo(blk);
        b.popBlock();
        _ = tpu.region(b.ctx, blk, &.{}, b.loc()).appendTo(b.currentBlock());
    }

    // Op order mirrors pipeline.py: copy_in/copy_out/wait_out take slot rem
    // before get_dma_slice; wait_in does it after.

    fn defaultCopyIn(self: *BufferedRef, b: *Builder, indices: []const Value, slot_cumulative: Value, static_zero_start: bool) void {
        const slot = b.remui(slot_cumulative, b.lift(@as(i32, @intCast(self.buffer_count))));
        const dma = self.getDmaSlice(b, indices, static_zero_start);
        const src_sl = self.sliceSrc(b, dma);
        const dst_sl = self.sliceWindowAt(b, slot, dma);
        const sem = semAt(b, self.sem_recvs.?, slot);
        b.enqueueDma(src_sl, dst_sl, sem, .{});
    }

    fn defaultWaitIn(self: *BufferedRef, b: *Builder, indices: []const Value) void {
        const dma = self.getDmaSlice(b, indices, false);
        const slot = b.remui(self.wait_in_slot.?, b.lift(@as(i32, @intCast(self.buffer_count))));
        const src_sl = self.sliceSrc(b, dma);
        const dst_sl = self.sliceWindowAt(b, slot, dma);
        const sem = semAt(b, self.sem_recvs.?, slot);
        b.waitDma2(sem, src_sl, dst_sl, .{});
    }

    fn defaultCopyOut(self: *BufferedRef, b: *Builder, indices: []const Value) void {
        if (self.buffer_count == 1) @panic("emitPipeline: single-buffered output sync_copy_out unsupported (TODO)");
        const slot = b.remui(self.copy_out_slot.?, b.lift(@as(i32, @intCast(self.buffer_count))));
        const dma = self.getDmaSlice(b, indices, false);
        // Pallas evaluates `src` (VMEM window) before `dst` (HBM).
        const src_sl = self.sliceWindowAt(b, slot, dma);
        const dst_sl = self.sliceSrc(b, dma);
        const sem = semAt(b, self.sem_sends.?, slot);
        b.enqueueDma(src_sl, dst_sl, sem, .{});
    }

    fn defaultWaitOut(self: *BufferedRef, b: *Builder, indices: []const Value) void {
        if (self.buffer_count <= 1) return;
        const slot = b.remui(self.wait_out_slot.?, b.lift(@as(i32, @intCast(self.buffer_count))));
        const dma = self.getDmaSlice(b, indices, false);
        const src_sl = self.sliceWindowAt(b, slot, dma);
        const dst_sl = self.sliceSrc(b, dma);
        const sem = semAt(b, self.sem_sends.?, slot);
        b.waitDma2(sem, src_sl, dst_sl, .{});
    }

    fn copyIn(self: *BufferedRef, b: *Builder, indices: []const Value, slot_cumulative: Value, static_zero_start: bool) void {
        if (self.is_trivial) return;

        if (self.spec.transfer_strategy) |strategy| {
            if (strategy.copy_in) |customCopyIn| {
                customCopyIn(self, b, indices, slot_cumulative, static_zero_start, strategy.ctx);
                return;
            }
        }

        self.defaultCopyIn(b, indices, slot_cumulative, static_zero_start);
    }

    fn waitIn(self: *BufferedRef, b: *Builder, indices: []const Value) void {
        if (self.spec.transfer_strategy) |strategy| {
            if (strategy.wait_in) |customWaitIn| {
                customWaitIn(self, b, indices, strategy.ctx);
                return;
            }
        }
        self.defaultWaitIn(b, indices);
    }

    fn copyOut(self: *BufferedRef, b: *Builder, indices: []const Value) void {
        if (self.spec.transfer_strategy) |strategy| {
            if (strategy.copy_out) |customCopyOut| {
                customCopyOut(self, b, indices, strategy.ctx);
                return;
            }
        }
        self.defaultCopyOut(b, indices);
    }

    fn waitOut(self: *BufferedRef, b: *Builder, indices: []const Value) void {
        if (self.spec.transfer_strategy) |strategy| {
            if (strategy.wait_out) |customWaitOut| {
                customWaitOut(self, b, indices, strategy.ctx);
                return;
            }
        }
        self.defaultWaitOut(b, indices);
    }

    pub fn sourceRef(self: *const BufferedRef) Value {
        return self.src;
    }

    pub fn windowRef(self: *const BufferedRef) Value {
        return self.window_ref;
    }

    pub fn copyInSlot(self: *const BufferedRef, b: *Builder, slot_cumulative: Value) Value {
        return b.remui(slot_cumulative, b.lift(@as(i32, @intCast(self.buffer_count))));
    }

    pub fn waitInSlot(self: *const BufferedRef, b: *Builder) Value {
        return b.remui(self.wait_in_slot.?, b.lift(@as(i32, @intCast(self.buffer_count))));
    }

    pub fn receiveSemaphoreAt(self: *const BufferedRef, b: *Builder, slot: Value) Value {
        return semAt(b, self.sem_recvs.?, slot);
    }

    pub fn copyOutSlot(self: *const BufferedRef, b: *Builder) Value {
        const buffer_count: i32 = @intCast(self.buffer_count);
        return b.remui(self.copy_out_slot.?, b.lift(buffer_count));
    }

    pub fn waitOutSlot(self: *const BufferedRef, b: *Builder) Value {
        const buffer_count: i32 = @intCast(self.buffer_count);
        return b.remui(self.wait_out_slot.?, b.lift(buffer_count));
    }

    pub fn sendSemaphoreAt(
        self: *const BufferedRef,
        b: *Builder,
        slot: Value,
    ) Value {
        return semAt(b, self.sem_sends.?, slot);
    }
};

/// Prologue/loop/epilogue copy+wait schedule.
const Scheduler = struct {
    step: Value,
    grid: []const Value,
    num_steps: Value,
    num_stages: usize,
    trace_scopes: bool,
    first_step: Value,
    last_step: Value,
    indices_storage: [4]Value = undefined,
    prev_storage: [4]Value = undefined,
    next_storage: [4]Value = undefined,
    fetch_storage: [8][4]Value = undefined,
    fetch_len: usize,
    indices_len: usize,

    fn init(b: *Builder, step: Value, grid: []const Value, num_steps: Value, num_stages: usize, trace_scopes: bool) Scheduler {
        if (num_stages + 1 > 8) @panic("emitPipeline: buffer_count > 7 unsupported");
        const c0 = b.lift(@as(i32, 0));
        const c1 = b.lift(@as(i32, 1));
        var s: Scheduler = .{
            .step = step,
            .grid = grid,
            .num_steps = num_steps,
            .num_stages = num_stages,
            .trace_scopes = trace_scopes,
            .first_step = b.cmpi(.eq, step, c0),
            .last_step = b.cmpi(.eq, step, b.subi(num_steps, c1)),
            .fetch_len = num_stages + 1,
            .indices_len = grid.len,
        };
        for (0..grid.len) |i| s.indices_storage[i] = c0;
        s.recomputeDerivedIndices(b);
        return s;
    }

    fn indices(self: *Scheduler) []const Value {
        return self.indices_storage[0..self.indices_len];
    }
    fn prevIndices(self: *Scheduler) []const Value {
        return self.prev_storage[0..self.indices_len];
    }
    fn nextIndices(self: *Scheduler) []const Value {
        return self.next_storage[0..self.indices_len];
    }
    fn fetchIndices(self: *Scheduler, lookahead: usize) []const Value {
        std.debug.assert(lookahead < self.fetch_len);
        return self.fetch_storage[lookahead][0..self.indices_len];
    }

    fn recomputeDerivedIndices(self: *Scheduler, b: *Builder) void {
        const prev = prevIndexND(b, self.indices(), self.grid);
        const next = nextIndexND(b, self.indices(), self.grid);
        for (0..self.indices_len) |i| {
            self.prev_storage[i] = prev[i];
            self.next_storage[i] = next[i];
            self.fetch_storage[0][i] = self.indices_storage[i];
            self.fetch_storage[1][i] = next[i];
        }
        var fetch = next;
        var lookahead: usize = 2;
        while (lookahead < self.fetch_len) : (lookahead += 1) {
            fetch = nextIndexND(b, fetch, self.grid);
            for (0..self.indices_len) |i| self.fetch_storage[lookahead][i] = fetch[i];
        }
    }

    fn namedScope(self: *Scheduler, b: *Builder, name: []const u8) void {
        if (self.trace_scopes) b.traceStart(10, name);
    }
    fn endScope(self: *Scheduler, b: *Builder) void {
        if (self.trace_scopes) b.traceStop();
    }

    /// `~out_of_fetch` as `step < num_steps + (1 - buffer_count)`.
    fn notOutOfFetch(self: *Scheduler, b: *Builder, br: *const BufferedRef) Value {
        std.debug.assert(br.isBuffered());
        const off: i32 = 1 - @as(i32, @intCast(br.buffer_count));
        return b.cmpi(.slt, self.step, b.addi(self.num_steps, b.lift(off)));
    }

    fn notFirstStep(self: *Scheduler, b: *Builder) Value {
        return b.cmpi(.ne, self.step, b.lift(@as(i32, 0)));
    }

    fn indicesDiffer(b: *Builder, br: *const BufferedRef, ai: []const Value, ci: []const Value) Value {
        const xa = br.computeIndex(b, ai);
        const xc = br.computeIndex(b, ci);
        std.debug.assert(xa.len == xc.len and xa.len > 0);
        var acc: ?Value = null;
        for (xa, xc) |x, y| {
            const d: Value = switch (x) {
                .scalar => |xv| b.cmpi(.ne, xv, y.scalar),
                .slice => |xs| b.ori(b.cmpi(.ne, xs.start, y.slice.start), b.cmpi(.ne, xs.size, y.slice.size)),
            };
            acc = if (acc) |p| b.ori(p, d) else d;
        }
        return acc.?;
    }

    fn hasChanged(self: *Scheduler, b: *Builder, br: *const BufferedRef) Value {
        std.debug.assert(br.isBuffered() and !br.is_trivial);
        return indicesDiffer(b, br, self.indices(), self.prevIndices());
    }
    fn willChangeCurrent(self: *Scheduler, b: *Builder, br: *const BufferedRef) Value {
        std.debug.assert(br.isBuffered() and !br.is_trivial);
        return indicesDiffer(b, br, self.indices(), self.nextIndices());
    }
    fn willChangeFetch(self: *Scheduler, b: *Builder, br: *const BufferedRef) Value {
        std.debug.assert(br.isBuffered() and !br.is_trivial);
        if (br.buffer_count < 2) return self.hasChanged(b, br);
        return indicesDiffer(b, br, self.fetchIndices(br.buffer_count - 2), self.fetchIndices(br.buffer_count - 1));
    }

    fn advance(b: *Builder, slot: Value, pred: Value) Value {
        return b.select(pred, b.addi(slot, b.lift(@as(i32, 1))), slot);
    }

    fn initializeStep(self: *Scheduler, b: *Builder, br: *BufferedRef, step: usize) void {
        if ((br.buffer_type != .input and br.buffer_type != .input_output) or !br.isBuffered() or br.is_trivial) return;
        if (step + 1 >= br.buffer_count) return;
        if (br.copy_in_slot == null) br.copy_in_slot = b.lift(@as(i32, 0));

        const pred = if (step == 0) self.first_step else blk: {
            const block_changed = indicesDiffer(b, br, self.fetchIndices(step), self.fetchIndices(step - 1));
            break :blk b.andi(self.first_step, block_changed);
        };
        const fetch_indices = self.fetchIndices(step);
        if (step == 0) {
            // static_zero_start=true: Python-int 0 grid indices enable folds in getDmaSlice.
            br.copyIn(b, fetch_indices, br.copy_in_slot.?, true);
        } else {
            var when = b.openIf(pred);
            {
                br.copyIn(b, fetch_indices, br.copy_in_slot.?, false);
                when.yieldThen(.{});
            }
        }
        br.copy_in_slot = advance(b, br.copy_in_slot.?, pred);
    }

    fn copyIn(self: *Scheduler, b: *Builder, br: *BufferedRef) void {
        if (br.is_trivial) return;
        var pred = b.andi(self.willChangeFetch(b, br), self.notOutOfFetch(b, br));
        if (br.isBuffered() and br.buffer_count < 2) pred = b.ori(pred, self.first_step);
        if (br.buffer_type != .input and br.buffer_type != .input_output) return;
        {
            var when = b.openIf(pred);
            {
                self.namedScope(b, "ep_copy_in");
                br.copyIn(b, self.fetchIndices(br.buffer_count - 1), br.copy_in_slot.?, false);
                self.endScope(b);
                when.yieldThen(.{});
            }
        }
        br.copy_in_slot = advance(b, br.copy_in_slot.?, pred);
    }

    fn waitIn(self: *Scheduler, b: *Builder, br: *BufferedRef) void {
        if (br.is_trivial) return;
        const pred = b.ori(self.hasChanged(b, br), self.first_step);
        {
            var when = b.openIf(pred);
            {
                self.namedScope(b, "ep_wait_in");
                if (br.buffer_type == .input or br.buffer_type == .input_output) br.waitIn(b, self.indices());
                self.endScope(b);
                when.yieldThen(.{});
            }
        }
    }

    fn copyOut(self: *Scheduler, b: *Builder, br: *BufferedRef) void {
        if (br.is_trivial) return;
        const pred = b.ori(self.willChangeCurrent(b, br), self.last_step);
        {
            var when = b.openIf(pred);
            {
                self.namedScope(b, "ep_copy_out");
                if (br.buffer_type == .output or br.buffer_type == .input_output) br.copyOut(b, self.indices());
                self.endScope(b);
                when.yieldThen(.{});
            }
        }
        br.copy_out_slot = advance(b, br.copy_out_slot.?, pred);
    }

    fn waitOut(self: *Scheduler, b: *Builder, br: *BufferedRef) void {
        if (br.is_trivial) return;
        const pred = b.andi(self.hasChanged(b, br), self.notFirstStep(b));
        {
            var when = b.openIf(pred);
            {
                self.namedScope(b, "ep_wait_out");
                // copy_out was issued the previous iteration.
                if (br.buffer_type == .output or br.buffer_type == .input_output) br.waitOut(b, self.prevIndices());
                self.endScope(b);
                when.yieldThen(.{});
            }
        }
        br.wait_out_slot = advance(b, br.wait_out_slot.?, pred);
    }

    fn finalize(self: *Scheduler, b: *Builder, br: *BufferedRef) void {
        if (br.is_trivial or (br.buffer_type != .output and br.buffer_type != .input_output)) return;
        var when = b.openIf(self.last_step);
        {
            self.namedScope(b, "ep_finalize");
            br.waitOut(b, self.indices());
            self.endScope(b);
            when.yieldThen(.{});
        }
    }

    fn advanceSlots(self: *Scheduler, b: *Builder, br: *BufferedRef) void {
        if (br.buffer_type != .input and br.buffer_type != .input_output) return;
        if (br.is_trivial) {
            switch (br.buffer_type) {
                .input => br.slot = advance(b, br.slot.?, self.last_step),
                .input_output => br.wait_in_slot = advance(b, br.wait_in_slot.?, self.last_step),
                .output => unreachable,
            }
        } else {
            const pred = b.ori(self.willChangeCurrent(b, br), self.last_step);
            br.wait_in_slot = advance(b, br.wait_in_slot.?, pred);
        }
    }
};

/// `verify = false`: `AutomaticAllocationScope` requirement is only satisfied
/// once it's inside `func.func`; full verify at `Builder.finish`.
pub fn memRefAlloca(b: *Builder, ty: *const mlir.Type) Value {
    return b.emit(mlir.Operation.make(b.ctx, "memref.alloca", .{
        .results = .{ .flat = &.{ty} },
        .attributes = &.{.named(b.ctx, "operandSegmentSizes", .denseArray(b.ctx, .i32, &.{ 0, 0 }))},
        .verify = false,
        .location = b.loc(),
    }));
}

pub fn semAllocOp(b: *Builder, ty: *const mlir.Type) Value {
    return b.emit(mlir.Operation.make(b.ctx, "tpu.sem_alloc", .{
        .results = .{ .flat = &.{ty} },
        .verify = false,
        .location = b.loc(),
    }));
}

/// `pl.SemaphoreType.DMA((n,))`, or rank-0 for `n == null`.
pub fn dmaSemRefTy(b: *Builder, n: ?usize) *const mlir.Type {
    const shape: []const i64 = if (n) |m| &.{@intCast(m)} else &.{};
    return mlir.Type.memRef(tpu.dmaSemaphoreType(b.ctx), shape, null, MemorySpace.semaphore_mem.attribute(b.ctx));
}

fn gridSize(b: *Builder, grid: []const Value) Value {
    var acc = grid[0];
    for (grid[1..]) |g| acc = b.muli(acc, g);
    return acc;
}

fn filterIndices(b: *Builder, indices: []const Value, grid: []const Value) [4]Value {
    var out: [4]Value = undefined;
    const c0 = b.lift(@as(i32, 0));
    for (indices, grid, 0..) |i, g, pos| {
        out[pos] = if (g.asConstantInt() == @as(?i64, 1)) c0 else i;
    }
    return out;
}

fn nextIndexND(b: *Builder, indices: []const Value, grid: []const Value) []const Value {
    const a = b.arena.allocator();
    const out = a.alloc(Value, indices.len) catch @panic("nextIndexND OOM");
    const c0 = b.lift(@as(i32, 0));
    var carry: Value = b.cmpi(.eq, c0, c0);
    var rev: usize = 0;
    while (rev < indices.len) : (rev += 1) {
        const pos = indices.len - 1 - rev;
        const inc = b.select(carry, b.addi(indices[pos], b.lift(@as(i32, 1))), indices[pos]);
        carry = b.cmpi(.eq, inc, grid[pos]);
        out[pos] = b.select(carry, c0, inc);
    }
    const filtered = filterIndices(b, out, grid);
    for (0..indices.len) |i| out[i] = filtered[i];
    return out;
}

fn prevIndexND(b: *Builder, indices: []const Value, grid: []const Value) []const Value {
    const a = b.arena.allocator();
    const out = a.alloc(Value, indices.len) catch @panic("prevIndexND OOM");
    const c0 = b.lift(@as(i32, 0));
    var borrow: Value = b.cmpi(.eq, c0, c0);
    const cm1 = b.lift(@as(i32, -1));
    var rev: usize = 0;
    while (rev < indices.len) : (rev += 1) {
        const pos = indices.len - 1 - rev;
        const dec = b.select(borrow, b.subi(indices[pos], b.lift(@as(i32, 1))), indices[pos]);
        borrow = b.cmpi(.eq, dec, cm1);
        out[pos] = b.select(borrow, b.subi(grid[pos], b.lift(@as(i32, 1))), dec);
    }
    const filtered = filterIndices(b, out, grid);
    for (0..indices.len) |i| out[i] = filtered[i];
    return out;
}

/// `null` block_shape, or every dim is `.blocked(full_dim)`.
fn specHasTrivialWindowing(spec: PipeSpec) bool {
    const bs = spec.block_shape orelse return true;
    if (bs.len != spec.full_shape.len) return false;
    for (bs, spec.full_shape) |bd, fs| switch (bd) {
        .blocked => |n| if (n != fs) return false,
        else => return false,
    };
    return true;
}

test {
    std.testing.refAllDecls(@This());
}

test "trivial input-output ref is copied in and out" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (builder.dialects_needed) |dialect| registry.registerDialect(dialect);
    mlir.registerFuncExtensions(registry);

    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    var b = try Builder.init(std.testing.allocator, ctx, "trivial_input_output", &.{
        .{
            .name = "state",
            .kind = .{ .ref = .{
                .shape = &.{ 8, 128 },
                .dtype = .bf16,
                .memory_space = .hbm,
            } },
        },
    }, &.{});
    defer b.deinit();

    const noOpBody = struct {
        fn call(
            _: *Builder,
            _: []const Value,
            _: []const RefWindow,
            _: []const Value,
            _: ?*anyopaque,
        ) FinishError!void {}
    }.call;

    try emitPipeline(&b, &.{b.arg(0)}, &.{}, .{
        .body = noOpBody,
        .grid = &.{b.lift(@as(i32, 1))},
        .in_specs = &.{.{
            .full_shape = &.{ 8, 128 },
            .dtype = .bf16,
            .buffer_type = .input_output,
        }},
        .out_specs = &.{},
        .trace_scopes = false,
    });

    const ir = try b.finish(&.{});
    defer std.testing.allocator.free(ir);

    // One enqueue/wait pair for copy_in, one for copy_out.
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "tpu.enqueue_dma"));
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "tpu.wait_dma"));
}
