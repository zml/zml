//! Exact top-k over a long axis, eg the vocabulary axis of LM logits.
//!
//! `topK` returns the same values and indices as
//! `x.topK(axis, k, .{ .descending = true })`: values sorted in descending
//! order, ties broken by ascending index, -0.0 equal to +0.0.
//!
//! XLA lowers a stable top-k with k > 16 to a full segmented radix sort of the
//! axis. On CUDA, Triton kernels select the k best elements of each row instead:
//!
//! 1. `GroupMax` reduces each row to the maxima of small groups.
//! 2. `RowBound` finds the k-th largest group maximum. At least k elements are
//!    above it, so it bounds the k-th largest element from below.
//! 3. `Candidates` collects the few elements above the bound, per chunk of the
//!    row, as packed integers (see `packValues`): larger is better and every
//!    candidate is unique.
//! 4. `FixRows` replaces the candidates of rows with a truncated chunk (eg
//!    many ties) by an exact radix select of the whole row.
//!
//! A final sort of the packed candidates of each row gives the top k in order.
//! When the axis is explicitly sharded, each shard selects its own top k before
//! they are gathered and merged.
const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../zml.zig");
const Shape = zml.Shape;
const Tensor = zml.Tensor;
const tri = zml.kernel.triton;

/// Smallest k handled by the kernels: XLA's own top-k kernel (k <= 16) takes
/// about 12us per k on a [256, 64640] input, the kernels about 35us.
pub const min_k = 3;
/// Largest k handled by the kernels.
pub const max_k = 256;
/// Shortest axis worth the kernels; XLA sorts shorter axes quickly.
pub const min_axis_len = 4096;

pub const Opts = struct {
    /// Rows to compute, with the shape of the input without the top-k axis,
    /// which must be the last one. Inactive rows are skipped by the kernels and
    /// return -inf values at index 0.
    row_mask: ?Tensor = null,
};

/// Same as `x.topK(named_axis, k, .{ .descending = true })`, see module doc.
pub fn topK(x: Tensor, named_axis: anytype, k: u32, opts: Opts) Tensor.SortRes {
    const err_msg = "topK named axis should be an integer or a named axis, eg `x.topK(.{{ .best_token = .token }}, 16)` or `x.topK(-1, 16)`";
    const new_tag: ?[:0]const u8, const a = switch (@typeInfo(@TypeOf(named_axis))) {
        .int, .comptime_int => .{ null, x.axis(@as(i64, @intCast(named_axis))) },
        .@"struct" => |info| blk: {
            stdx.debug.assertComptime(info.field_names.len == 1, err_msg, .{});
            break :blk .{ info.field_names[0], x.axis(@field(named_axis, info.field_names[0])) };
        },
        else => stdx.debug.compileError(err_msg, .{}),
    };

    var result = if (useKernels(x, a, k)) kernelTopK(x, a, k, opts.row_mask) else x.topK(a, k, .{ .descending = true });
    if (new_tag) |tag| {
        result.values._shape._tags.set(a, tag.ptr);
        result.indices._shape._tags.set(a, tag.ptr);
    }
    return result;
}

/// Static launch parameters for one row length.
const Plan = struct {
    n: i64,
    k: i64,
    group_chunks: i64,
    group_chunk_len: i64,
    chunks: i64,
    chunk_len: i64,
    /// Candidate slots per chunk, about twice the expected count.
    capacity: i64,

    const group_block = 8192;
    const groups = 256;
    const block = 4096;
    const block_groups = 128;
    const max_candidates = 2048;

    fn init(n: i64, k: u32) ?Plan {
        if (n < min_axis_len or n < k or n > std.math.maxInt(i32)) return null;
        const group_chunks = std.math.divCeil(i64, n, group_block) catch unreachable;
        const chunks = std.math.divCeil(i64, n, block) catch unreachable;
        // A row has a bit more than k candidates, spread over its chunks.
        const per_chunk = std.math.divCeil(i64, k, chunks) catch unreachable;
        const capacity = pow2(@max(8, 2 * per_chunk + 4));
        if (chunks * capacity > max_candidates) return null;
        return .{
            .n = n,
            .k = k,
            .group_chunks = group_chunks,
            .group_chunk_len = std.math.divCeil(i64, n, group_chunks) catch unreachable,
            .chunks = chunks,
            .chunk_len = std.math.divCeil(i64, n, chunks) catch unreachable,
            .capacity = capacity,
        };
    }

    fn candidates(plan: Plan) i64 {
        return plan.chunks * plan.capacity;
    }
};

fn useKernels(x: Tensor, a: u3, k: u32) bool {
    const compiler = zml.Compiler.current();
    if (compiler.platform.target != .cuda) return false;
    if (x.dtype() != .f32 and x.dtype() != .bf16) return false;
    if (k < min_k or k > max_k) return false;
    if (x.count() > std.math.maxInt(i32)) return false;

    const local_shape = zml.Sharding.shardedShape(x.shape()) catch return false;
    if (Plan.init(local_shape.dim(a), k) == null) return false;

    // Without an explicit sharding of the axis, a multi-device program could
    // still shard it through propagation: keep XLA's partitioned TopK there.
    const num_partitions = if (compiler.meshes.len > 0) compiler.meshes[0].numPartitions() else 1;
    const explicit = x.shape()._sharding.mesh != null and x.shape().partition(a).isClosed();
    return num_partitions == 1 or explicit;
}

fn kernelTopK(x: Tensor, a: u3, k: u32, row_mask: ?Tensor) Tensor.SortRes {
    const last: u3 = @intCast(x.rank() - 1);
    const xt = if (a == last) x else x.swapAxes(a, last);
    const n = xt.dim(last);
    const rows = @divExact(@as(i64, @intCast(xt.count())), n);
    const flat = xt.reshape(.{ rows, n });
    const active = if (row_mask) |m| blk: {
        stdx.debug.assert(a == last and m.count() == rows, "topK expects a row mask shaped like {f} without its last axis, got {f}", .{ x, m });
        break :blk m.convert(.i32).reshape(.{rows});
    } else Tensor.scalar(1, .i32).broad(.init(.{rows}, .i32));

    const sorted = if (xt.shape().partition(last).isSharded())
        shardedTopK(flat, active, xt.shape(), k)
    else
        localTopK(flat, active, k);

    const keys = sorted.shiftRightArithmetic(.scalar(32, .i64)).convert(.i32);
    const low = sorted.logical(.AND, .scalar(0xffffffff, .i64));
    const indices = Tensor.scalar(std.math.maxInt(i32), .i64).sub(low.shiftRightLogical(.scalar(1, .i64))).convert(.i32);
    const negative_zero = low.logical(.AND, .scalar(1, .i64)).cmp(.EQ, .scalar(1, .i64));

    const out_shape = xt.shape().setDim(last, k);
    var values = negative_zero.select(Tensor.scalar(-0.0, .f32), keysToValues(keys)).convert(x.dtype()).reshape(out_shape);
    var idx = indices.reshape(out_shape.withDtype(.i32));
    if (a != last) {
        values = values.swapAxes(a, last);
        idx = idx.swapAxes(a, last);
    }
    return .{ .values = values, .indices = idx };
}

/// Top k packed values of each row of `[rows, n]`, in descending order.
fn localTopK(flat: Tensor, active: Tensor, k: u32) Tensor {
    return sortPacked(candidatesTopK(flat, active, k), k);
}

/// Packed values are unique, larger is better.
fn sortPacked(x: Tensor, k: u32) Tensor {
    const sorted = zml.ops.sort(.{x}, 1, struct {
        fn call(lhs: zml.ops.SortArgs) Tensor {
            return lhs.left.cmp(.GT, lhs.right);
        }
    }.call, .{}, false);
    return sorted[0].slice(1, .{ .end = k });
}

/// Packed candidates of each row of `[rows, n]`, in no particular order,
/// padded with `pad`, holding the top k of the row.
fn candidatesTopK(flat: Tensor, active: Tensor, k: u32) Tensor {
    const rows = flat.dim(0);
    const plan = Plan.init(flat.dim(1), k).?;

    const group_max = GroupMax.Kernel.call(
        .{ .x = flat, .active = active },
        .{ .group_max = .init(.{ rows, plan.group_chunks * Plan.groups }, .i32) },
        .{
            .cfg = .{ .dtype = tri.from(flat.dtype()), .plan = plan },
            .grid = .{ @intCast(rows), @intCast(plan.group_chunks), 1 },
            .num_stages = 1,
            .num_warps = 8,
        },
    ).group_max;

    const bound = RowBound.Kernel.call(
        .{ .group_max = group_max, .active = active },
        .{ .bound = .init(.{rows}, .i32) },
        .{
            .cfg = .{ .plan = plan },
            .grid = .{ @intCast(rows), 1, 1 },
            .num_stages = 1,
            .num_warps = 8,
        },
    ).bound;

    const candidates = Candidates.Kernel.call(
        .{ .x = flat, .bound = bound, .active = active },
        .{ .candidates = .init(.{ rows, plan.candidates() }, .i64), .counts = .init(.{ rows, plan.chunks }, .i32) },
        .{
            .cfg = .{ .dtype = tri.from(flat.dtype()), .plan = plan },
            .grid = .{ @intCast(rows), @intCast(plan.chunks), 1 },
            .num_stages = 1,
            .num_warps = 4,
        },
    );

    return FixRows.Kernel.call(
        .{ .x = flat, .candidates = candidates.candidates, .counts = candidates.counts, .active = active },
        .{ .fixed = candidates.candidates.shape() },
        .{
            .cfg = .{ .dtype = tri.from(flat.dtype()), .plan = plan },
            .grid = .{ @intCast(rows), 1, 1 },
            .num_stages = 1,
            .num_warps = 8,
            .output_operand_aliases = .{ .fixed = .candidates },
        },
    ).fixed;
}

/// Top k packed values of rows whose last axis is sharded: top k of each
/// shard, gathered, with global indices.
fn shardedTopK(flat: Tensor, active: Tensor, original: Shape, k: u32) Tensor {
    const last = original.rank() - 1;
    const mesh = original._sharding.mesh.?;
    const spec = original.partition(last);
    const local_n = (zml.Sharding.shardedShape(original) catch unreachable).dim(last);
    const shard_count = @divExact(original.dim(last), local_n);

    const partitioning: zml.Sharding.Partitioning = .init(&.{ .replicated, spec });
    const local_tops = zml.ops.manualComputation(
        (struct {
            fn body(inputs: struct { x: Tensor, active: Tensor, k: u32 }, _: Shape) Tensor {
                return localTopK(inputs.x, inputs.active, inputs.k);
            }
        }).body,
        .{
            .x = flat.withPartitioning(mesh, partitioning),
            .active = active.withPartitioning(mesh, zml.Sharding.Partitioning.init(&.{.replicated})),
            .k = k,
        },
        Shape.init(.{ flat.dim(0), shard_count * k }, .i64).withPartitioning(mesh, partitioning),
    );

    return mergeShardTops(local_tops, local_n, k);
}

/// Top k of `[rows, shards * k]` packed shard top k, shard s holding indices
/// local to elements `[s * local_n, (s + 1) * local_n)`.
fn mergeShardTops(local_tops: Tensor, local_n: i64, k: u32) Tensor {
    // Add s * local_n to the indices, ie subtract it from the packed low bits,
    // see `packValues`.
    const shard = Tensor.iota(local_tops.shape(), 1).convert(.i64).divByConst(k);
    return sortPacked(local_tops.sub(shard.scale(2 * local_n)), k);
}

/// Inverse of `signedKey`.
fn keysToValues(keys: Tensor) Tensor {
    const flipped = keys.logical(.XOR, Tensor.scalar(std.math.maxInt(i32), .i32).broad(keys.shape()));
    const bits = keys.cmp(.LT, Tensor.scalar(0, .i32).broad(keys.shape())).select(flipped, keys);
    return bits.bitCast(.f32);
}

const KernelConfig = struct {
    dtype: tri.DType = .f32,
    plan: Plan,
};

/// Maxima of groups of `group_block / groups` elements, per chunk of
/// `group_chunk_len` elements of each active row.
pub const GroupMax = struct {
    pub const Kernel = tri.Kernel(KernelConfig, .{
        .name = "zml_topk_group_max",
        .inputs = &.{ "x", "active" },
        .outputs = &.{"group_max"},
        .run = run,
    });

    fn run(b: *tri.Builder, cfg: KernelConfig) tri.FinishError!void {
        const plan = cfg.plan;
        const args = try b.declareArgs(.{
            .x = .{ .ptr = cfg.dtype },
            .active = .{ .ptr = .i32 },
            .group_max = .{ .ptr = .i32 },
        });
        const row = b.programId(.x);
        const chunk = b.programId(.y);
        b.returnIf(b.load(args.active.addPtr(row)).eq(0), .{});

        const shape = [_]i64{Plan.group_block};
        const chunk_len: i32 = @intCast(plan.group_chunk_len);
        const start = chunk.mul(chunk_len);
        const len = b.minimum(b.lift(@as(i32, @intCast(plan.n))).sub(start), chunk_len);
        const lanes = b.arange(0, Plan.group_block, .i32);
        const valid = lanes.lt(len);
        const x = b.loadOpts(rowPtr(args.x, row, plan).addPtr(start.add(lanes)), .{
            .mask = valid,
            .other = b.full(&shape, 0.0, cfg.dtype),
        }).to(.f32);
        const skey = b.select(valid, signedKey(b, x), b.full(&shape, min_key, .i32));
        const group_max = b.reshapeOpts(skey, &.{ Plan.groups, Plan.group_block / Plan.groups }, .{ .can_reorder = true })
            .maxOpts(.{ .axis = 1 });
        const offset = row.mul(@as(i32, @intCast(plan.group_chunks * Plan.groups))).add(chunk.mul(Plan.groups));
        b.store(args.group_max.addPtr(offset.add(b.arange(0, Plan.groups, .i32))), group_max);
    }
};

/// Signed key lower bound of the k-th largest element of each active row: the
/// k-th largest group maximum, truncated to 16 bits.
pub const RowBound = struct {
    pub const Kernel = tri.Kernel(KernelConfig, .{
        .name = "zml_topk_row_bound",
        .inputs = &.{ "group_max", "active" },
        .outputs = &.{"bound"},
        .run = run,
    });

    fn run(b: *tri.Builder, cfg: KernelConfig) tri.FinishError!void {
        const plan = cfg.plan;
        const args = try b.declareArgs(.{
            .group_max = .{ .ptr = .i32 },
            .active = .{ .ptr = .i32 },
            .bound = .{ .ptr = .i32 },
        });
        const row = b.programId(.x);
        b.returnIf(b.load(args.active.addPtr(row)).eq(0), .{});

        const count: i32 = @intCast(plan.group_chunks * Plan.groups);
        const block = pow2(count);
        const lanes = b.arange(0, block, .i32);
        const group_max = b.loadOpts(args.group_max.addPtr(row.mul(count).add(lanes)), .{
            .mask = lanes.lt(count),
            .other = b.full(&.{block}, min_key, .i32),
        });
        const selection = radixSelect(b, KeyTensor{ .ukey = unsignedKey(b, group_max), .mask = null }, b.lift(@as(i32, @intCast(plan.k))), 2);
        b.store(args.bound.addPtr(row), b.xori(selection.prefix, b.lift(min_key)));
    }
};

/// Up to `capacity` packed elements >= bound per chunk of `chunk_len`
/// elements, in no particular order and padded with `pad`, and the candidate
/// count of each chunk. A count above `capacity` means the candidates were
/// truncated.
pub const Candidates = struct {
    pub const Kernel = tri.Kernel(KernelConfig, .{
        .name = "zml_topk_candidates",
        .inputs = &.{ "x", "bound", "active" },
        .outputs = &.{ "candidates", "counts" },
        .run = run,
    });

    fn run(b: *tri.Builder, cfg: KernelConfig) tri.FinishError!void {
        const plan = cfg.plan;
        const args = try b.declareArgs(.{
            .x = .{ .ptr = cfg.dtype },
            .bound = .{ .ptr = .i32 },
            .active = .{ .ptr = .i32 },
            .candidates = .{ .ptr = .i64 },
            .counts = .{ .ptr = .i32 },
        });
        const row = b.programId(.x);
        const chunk = b.programId(.y);
        const capacity: i32 = @intCast(plan.capacity);
        const chunk_id = row.mul(@as(i32, @intCast(plan.chunks))).add(chunk);
        const out = args.candidates.addPtr(chunk_id.mul(capacity));
        const slots = b.arange(0, capacity, .i32);

        const inactive = b.load(args.active.addPtr(row)).eq(0);
        var skip = b.openIf(inactive);
        {
            b.store(args.counts.addPtr(chunk_id), b.lift(@as(i32, 0)));
            skip.yieldThen(.{});
        }
        b.returnIf(inactive, .{});

        const shape = [_]i64{Plan.block};
        const groups_shape = [_]i64{ Plan.block_groups, Plan.block / Plan.block_groups };
        const bound = b.load(args.bound.addPtr(row));
        const chunk_len: i32 = @intCast(plan.chunk_len);
        const start = chunk.mul(chunk_len);
        const len = b.minimum(b.lift(@as(i32, @intCast(plan.n))).sub(start), chunk_len);
        const lanes = b.arange(0, Plan.block, .i32);
        const valid = lanes.lt(len);
        const x = b.loadOpts(rowPtr(args.x, row, plan).addPtr(start.add(lanes)), .{
            .mask = valid,
            .other = b.full(&shape, 0.0, cfg.dtype),
        }).to(.f32);
        const skey = signedKey(b, x);
        const candidate = valid.bitAnd(skey.ge(bound));
        const all_packed = b.select(candidate, packValues(b, x, skey, start.add(lanes)), b.full(&shape, pad, .i64));
        // Groups map to threads: extraction only takes in-thread maxima.
        const candidates = b.reshapeOpts(all_packed, &groups_shape, .{ .can_reorder = true });
        const group_counts = candidates.ne(pad).to(.i32).sumOpts(.{ .axis = 1 });
        const count = group_counts.sum();
        const offsets = b.cumsum(group_counts).sub(group_counts);
        // Skip the extraction of truncated candidates.
        const max_count = b.select(count.gt(capacity), b.lift(@as(i32, 0)), group_counts.max());

        // Emit the best remaining candidate of every group, max_count times.
        const first = candidates.maxOpts(.{ .axis = 1 });
        b.storeOpts(out.addPtr(offsets), first, .{ .mask = group_counts.gt(0).bitAnd(offsets.lt(capacity)) });
        var extract = b.openFor(@as(i32, 1), max_count, @as(i32, 1), .{ candidates, first });
        {
            const remaining = b.select(extract.carried[0].lt(b.expandDims(extract.carried[1], 1)), extract.carried[0], b.full(&groups_shape, pad, .i64));
            const next = remaining.maxOpts(.{ .axis = 1 });
            b.storeOpts(out.addPtr(offsets.add(extract.iv)), next, .{ .mask = group_counts.gt(extract.iv) });
            extract.yield(.{ remaining, next });
        }

        b.storeOpts(out.addPtr(slots), b.full(&.{capacity}, pad, .i64), .{ .mask = slots.ge(count) });
        b.store(args.counts.addPtr(chunk_id), count);
    }
};

/// Rewrites the candidates of rows the other kernels could not handle: the
/// exact top k of rows with a truncated chunk, and -inf at index 0 for
/// inactive rows.
pub const FixRows = struct {
    pub const Kernel = tri.Kernel(KernelConfig, .{
        .name = "zml_topk_fix_rows",
        .inputs = &.{ "x", "candidates", "counts", "active" },
        .outputs = &.{"fixed"},
        .run = run,
    });

    fn run(b: *tri.Builder, cfg: KernelConfig) tri.FinishError!void {
        const plan = cfg.plan;
        const args = try b.declareArgs(.{
            .x = .{ .ptr = cfg.dtype },
            .candidates = .{ .ptr = .i64 },
            .counts = .{ .ptr = .i32 },
            .active = .{ .ptr = .i32 },
            .fixed = .{ .ptr = .i64 },
        });
        const row = b.programId(.x);
        const candidate_count: i32 = @intCast(plan.candidates());
        const out_base = row.mul(candidate_count);
        const slots = b.arange(0, pow2(candidate_count), .i32);
        const slots_shape = [_]i64{slots.dim(0)};

        const inactive = b.load(args.active.addPtr(row)).eq(0);
        var skip = b.openIf(inactive);
        {
            const negative_infinity = b.full(&slots_shape, (@as(i64, inf_key) << 32) | (std.math.maxInt(i32) << 1), .i64);
            b.storeOpts(args.fixed.addPtr(out_base.add(slots)), negative_infinity, .{ .mask = slots.lt(candidate_count) });
            skip.yieldThen(.{});
        }
        b.returnIf(inactive, .{});

        const chunks: i32 = @intCast(plan.chunks);
        const chunk_ids = b.arange(0, pow2(chunks), .i32);
        const counts = b.loadOpts(args.counts.addPtr(row.mul(chunks).add(chunk_ids)), .{
            .mask = chunk_ids.lt(chunks),
            .other = b.full(&.{pow2(chunks)}, 0, .i32),
        });
        b.returnIf(counts.max().le(@as(i32, @intCast(plan.capacity))), .{});

        b.storeOpts(args.fixed.addPtr(out_base.add(slots)), b.full(&slots_shape, pad, .i64), .{
            .mask = slots.ge(@as(i32, @intCast(plan.k))).bitAnd(slots.lt(candidate_count)),
        });
        selectRow(b, cfg, rowPtr(args.x, row, plan), args.fixed, out_base);
    }

    /// Exact top k of a whole row in index order, streamed in blocks: four
    /// histogram passes, then one emission pass.
    fn selectRow(b: *tri.Builder, cfg: KernelConfig, row_ptr: tri.Value, top: tri.Value, out_base: tri.Value) void {
        const row: KeyRow = .{ .cfg = cfg, .row_ptr = row_ptr };
        const selection = radixSelect(b, row, b.lift(@as(i32, @intCast(cfg.plan.k))), 4);
        const threshold = b.xori(selection.prefix, b.lift(min_key));

        var blocks = b.openFor(@as(i32, 0), row.blockCount(), @as(i32, 1), .{ b.lift(@as(i32, 0)), b.lift(@as(i32, 0)) });
        {
            const block = row.load(b, blocks.iv);
            const skey = signedKey(b, block.x);
            const above = block.valid.bitAnd(skey.gt(threshold));
            const tied = block.valid.bitAnd(skey.eq(threshold));
            const tie_rank = b.cumsum(tied.to(.i32)).add(blocks.carried[1]);
            const mask = above.bitOr(tied.bitAnd(tie_rank.le(selection.rank)));
            const position = out_base.add(blocks.carried[0]).add(b.cumsum(mask.to(.i32)).sub(1));
            b.storeOpts(top.addPtr(position), packValues(b, block.x, skey, block.offsets), .{ .mask = mask });
            blocks.yield(.{ blocks.carried[0].add(mask.to(.i32).sum()), blocks.carried[1].add(tied.to(.i32).sum()) });
        }
    }
};

fn pow2(x: i64) i32 {
    return @intCast(std.math.ceilPowerOfTwoAssert(u64, @intCast(x)));
}

const min_key: i32 = std.math.minInt(i32);
/// `signedKey(-inf)`.
const inf_key: i32 = @bitCast(@as(u32, 0x807fffff));
/// Packed candidate padding, below every real candidate.
const pad: i64 = std.math.minInt(i64);

fn byteMask(shift: i32) i32 {
    return @bitCast(@as(u32, 0xff) << @intCast(shift));
}

fn rowPtr(x: tri.Value, row: tri.Value, plan: Plan) tri.Value {
    return x.addPtr(row.to(.i64).mul(plan.n));
}

/// Signed integer with the order of the float, -0.0 equal to +0.0.
fn signedKey(b: *tri.Builder, x: tri.Value) tri.Value {
    const dims = x.shape().constSlice();
    const raw = b.bitcast(x, .i32);
    const bits = b.select(raw.eq(min_key), b.full(dims, 0, .i32), raw);
    const flip = b.andi(b.shrsi(bits, b.full(dims, 31, .i32)), b.full(dims, std.math.maxInt(i32), .i32));
    return b.xori(bits, flip);
}

/// Key compared with unsigned semantics, used for radix digits.
fn unsignedKey(b: *tri.Builder, skey: tri.Value) tri.Value {
    return b.xori(skey, b.full(skey.shape().constSlice(), min_key, .i32));
}

/// `(key << 32) | ((maxInt(i32) - index) << 1) | is_negative_zero`: larger key
/// first, then smaller index. The last bit recovers -0.0 from its key.
fn packValues(b: *tri.Builder, x: tri.Value, skey: tri.Value, index: tri.Value) tri.Value {
    const dims = skey.shape().constSlice();
    const high = b.shli(skey.to(.i64), b.full(dims, 32, .i64));
    const reversed = b.full(dims, std.math.maxInt(i32), .i32).sub(index).to(.i64);
    const negative_zero = b.bitcast(x, .i32).eq(min_key).to(.i64);
    return b.ori(high, b.ori(b.shli(reversed, b.full(dims, 1, .i64)), negative_zero));
}

const BinStep = struct {
    /// Selected digit, shifted in place.
    digit: tri.Value,
    /// Keys in better bins.
    count_before: tri.Value,
};

/// Finds the bin of a descending histogram that holds the rank-th largest key.
fn selectBin(b: *tri.Builder, histogram: tri.Value, bins: tri.Value, rank: tri.Value, shift: i32) BinStep {
    const cumulative = b.cumsum(histogram);
    const selected_bin = b.select(cumulative.ge(rank), bins, b.full(&.{256}, 256, .i32)).min();
    const count_before = b.select(bins.lt(selected_bin), histogram, b.full(&.{256}, 0, .i32)).sum();
    return .{
        .digit = b.shli(b.lift(@as(i32, 255)).sub(selected_bin), b.lift(shift)),
        .count_before = count_before,
    };
}

const Selection = struct {
    /// Unsigned key prefix of the k-th largest key.
    prefix: tri.Value,
    /// Number of keys with that prefix in the top k.
    rank: tri.Value,
};

/// Finds the top `passes` bytes of the k-th largest unsigned key of `keys`,
/// one 8-bit histogram per pass.
fn radixSelect(b: *tri.Builder, keys: anytype, k: tri.Value, comptime passes: u3) Selection {
    const bins = b.arange(0, 256, .i32);
    var prefix = b.lift(@as(i32, 0));
    var prefix_mask = b.lift(@as(i32, 0));
    var rank = k;
    inline for (0..passes) |pass| {
        const shift: i32 = 24 - 8 * @as(i32, pass);
        const histogram = keys.histogram(b, prefix, prefix_mask, shift);
        const step = selectBin(b, histogram, bins, rank, shift);
        prefix = prefix.bitOr(step.digit);
        prefix_mask = prefix_mask.bitOr(b.lift(byteMask(shift)));
        rank = rank.sub(step.count_before);
    }
    return .{ .prefix = prefix, .rank = rank };
}

/// Descending histogram of the digit at `shift` of the keys under `prefix`.
fn digitHistogram(b: *tri.Builder, ukey: tri.Value, mask: ?tri.Value, prefix: tri.Value, prefix_mask: tri.Value, shift: i32) tri.Value {
    const shape = ukey.shape().constSlice();
    const in_prefix = b.andi(ukey, b.splat(prefix_mask, shape)).eq(b.splat(prefix, shape));
    const matches = if (mask) |m| m.bitAnd(in_prefix) else in_prefix;
    const digit = b.andi(b.shrui(ukey, b.full(shape, shift, .i32)), b.full(shape, 0xff, .i32));
    return b.histogramOpts(b.full(shape, 255, .i32).sub(digit), 256, .{ .mask = matches });
}

/// Unsigned keys held by the program, outside of `mask` ignored.
const KeyTensor = struct {
    ukey: tri.Value,
    mask: ?tri.Value,

    fn histogram(self: KeyTensor, b: *tri.Builder, prefix: tri.Value, prefix_mask: tri.Value, shift: i32) tri.Value {
        return digitHistogram(b, self.ukey, self.mask, prefix, prefix_mask, shift);
    }
};

/// Keys of a whole row, loaded in blocks.
const KeyRow = struct {
    cfg: KernelConfig,
    row_ptr: tri.Value,

    const block = Plan.block;

    fn blockCount(self: KeyRow) i32 {
        return std.math.divCeil(i32, @intCast(self.cfg.plan.n), block) catch unreachable;
    }

    fn load(self: KeyRow, b: *tri.Builder, index: tri.Value) struct { x: tri.Value, valid: tri.Value, offsets: tri.Value } {
        const offsets = index.mul(block).add(b.arange(0, block, .i32));
        const valid = offsets.lt(@as(i32, @intCast(self.cfg.plan.n)));
        const x = b.loadOpts(self.row_ptr.addPtr(offsets), .{
            .mask = valid,
            .other = b.full(&.{block}, 0.0, self.cfg.dtype),
        }).to(.f32);
        return .{ .x = x, .valid = valid, .offsets = offsets };
    }

    fn histogram(self: KeyRow, b: *tri.Builder, prefix: tri.Value, prefix_mask: tri.Value, shift: i32) tri.Value {
        var blocks = b.openFor(@as(i32, 0), self.blockCount(), @as(i32, 1), .{b.full(&.{256}, 0, .i32)});
        {
            const loaded = self.load(b, blocks.iv);
            const ukey = unsignedKey(b, signedKey(b, loaded.x));
            blocks.yield(.{blocks.carried[0].add(digitHistogram(b, ukey, loaded.valid, prefix, prefix_mask, shift))});
        }
        return blocks.results[0];
    }
};

const TestFill = enum { normal, ties, masked, zeros };

fn testValues(allocator: std.mem.Allocator, fill: TestFill, rows: usize, n: usize, dtype: zml.DataType) !zml.Slice {
    var prng: std.Random.DefaultPrng = .init(@as(u64, @backingInt(fill)) * 7919 + n);
    const random = prng.random();
    const slice = try zml.Slice.alloc(allocator, .init(.{ rows, n }, dtype));
    for (0..rows * n) |i| {
        const row = i / n;
        const v: f32 = switch (fill) {
            .normal => random.floatNorm(f32) * 3,
            // Few distinct values: ties straddle the top-k boundary.
            .ties => @round(random.floatNorm(f32) * 2) / 2,
            // Mostly -inf, a row-dependent number of finite values.
            .masked => if (random.uintLessThan(usize, n) < row * 13) random.floatNorm(f32) else -std.math.inf(f32),
            // A few positives, then signed zeros at the boundary.
            .zeros => switch (random.uintLessThan(u32, 4096)) {
                0 => 0.5,
                1...2047 => -0.0,
                2048...4000 => 0.0,
                else => -1.0,
            },
        };
        switch (dtype) {
            .f32 => slice.items(f32)[i] = v,
            .bf16 => slice.items(zml.floats.BFloat16)[i] = .fromF32(v),
            else => unreachable,
        }
    }
    return slice;
}

fn expectSameTopK(io: std.Io, actual: zml.Bufferized(Tensor.SortRes), expected: zml.Bufferized(Tensor.SortRes), row_mask: ?[]const bool) !void {
    const allocator = std.testing.allocator;
    const actual_values = try actual.values.toSliceAlloc(allocator, io);
    defer actual_values.free(allocator);
    const expected_values = try expected.values.toSliceAlloc(allocator, io);
    defer expected_values.free(allocator);
    const actual_indices = try actual.indices.toSliceAlloc(allocator, io);
    defer actual_indices.free(allocator);
    const expected_indices = try expected.indices.toSliceAlloc(allocator, io);
    defer expected_indices.free(allocator);

    const k: usize = @intCast(actual.values.shape().dim(-1));
    const value_bytes = actual.values.shape().dtype().sizeOf();
    for (0..actual.values.shape().count() / k) |row| {
        if (row_mask) |mask| if (!mask[row]) continue;
        try std.testing.expectEqualSlices(i32, expected_indices.items(i32)[row * k ..][0..k], actual_indices.items(i32)[row * k ..][0..k]);
        // Bitwise, -0.0 included.
        try std.testing.expectEqualSlices(u8, expected_values.data()[row * k * value_bytes ..][0 .. k * value_bytes], actual_values.data()[row * k * value_bytes ..][0 .. k * value_bytes]);
    }
}

test "topK kernels match the XLA top-k" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Fns = struct {
        fn both(x: Tensor, k: u32) [2]Tensor.SortRes {
            return .{ topK(x, .{ .top = .n }, k, .{}), x.topK(.{ .top = .n }, k, .{ .descending = true }) };
        }
    };
    const Case = struct { rows: usize, n: usize, k: u32, dtype: zml.DataType, fill: TestFill };
    const cases = [_]Case{
        .{ .rows = 9, .n = 64640, .k = 50, .dtype = .f32, .fill = .normal },
        .{ .rows = 9, .n = 64640, .k = 50, .dtype = .f32, .fill = .ties },
        .{ .rows = 9, .n = 64640, .k = 50, .dtype = .f32, .fill = .masked },
        .{ .rows = 9, .n = 64640, .k = 50, .dtype = .f32, .fill = .zeros },
        .{ .rows = 9, .n = 64640, .k = 50, .dtype = .bf16, .fill = .ties },
        .{ .rows = 5, .n = 10007, .k = 17, .dtype = .f32, .fill = .ties },
        .{ .rows = 5, .n = 4096, .k = 256, .dtype = .f32, .fill = .normal },
        .{ .rows = 2, .n = 129280, .k = 128, .dtype = .bf16, .fill = .normal },
    };
    for (cases) |case| {
        const x: Tensor = .init(.{ .b = case.rows, .n = case.n }, case.dtype);
        var exe = try platform.compileFn(allocator, io, Fns.both, .{ x, case.k }, .{});
        defer exe.deinit();
        const values = try testValues(allocator, case.fill, case.rows, case.n, case.dtype);
        defer values.free(allocator);
        var x_buffer: zml.Buffer = try .fromSlice(io, platform, values);
        defer x_buffer.deinit();
        var results: [2]zml.Bufferized(Tensor.SortRes) = undefined;
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        runner.run(io, .{x_buffer}, &results, .{ .wait = true });
        defer for (&results) |*r| zml.Buffer.deinitAll(Tensor.SortRes, r);
        expectSameTopK(io, results[0], results[1], null) catch |err| {
            std.log.err("topK mismatch for {any}", .{case});
            return err;
        };
    }
}

test "topK kernels skip inactive rows" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Fns = struct {
        fn both(x: Tensor, mask: Tensor) [2]Tensor.SortRes {
            return .{ topK(x, .{ .top = .n }, 50, .{ .row_mask = mask }), x.topK(.{ .top = .n }, 50, .{ .descending = true }) };
        }
    };
    const rows = 6;
    const n = 64640;
    const x: Tensor = .init(.{ .b = rows, .n = n }, .f32);
    const mask: Tensor = .init(.{ .b = rows }, .bool);
    var exe = try platform.compileFn(allocator, io, Fns.both, .{ x, mask }, .{});
    defer exe.deinit();
    const values = try testValues(allocator, .ties, rows, n, .f32);
    defer values.free(allocator);
    var x_buffer: zml.Buffer = try .fromSlice(io, platform, values);
    defer x_buffer.deinit();
    const mask_values = [rows]bool{ true, false, false, true, false, true };
    var mask_buffer: zml.Buffer = try .fromBytes(io, platform, mask.shape(), std.mem.sliceAsBytes(&mask_values));
    defer mask_buffer.deinit();
    var results: [2]zml.Bufferized(Tensor.SortRes) = undefined;
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    runner.run(io, .{ x_buffer, mask_buffer }, &results, .{ .wait = true });
    defer for (&results) |*r| zml.Buffer.deinitAll(Tensor.SortRes, r);
    try expectSameTopK(io, results[0], results[1], &mask_values);

    const inactive_values = try results[0].values.getValue([rows][50]f32, io);
    const inactive_indices = try results[0].indices.getValue([rows][50]i32, io);
    for (mask_values, inactive_values, inactive_indices) |active, row_values, row_indices| {
        if (active) continue;
        try std.testing.expectEqual(@as([50]f32, @splat(-std.math.inf(f32))), row_values);
        try std.testing.expectEqual(@as([50]i32, @splat(0)), row_indices);
    }
}

test "topK kernels merge shard top k" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const shards = 3;
    const local_n = 21000;
    const k = 50;
    const Fns = struct {
        // What the manual computation of `shardedTopK` does on each shard.
        fn merged(x: Tensor) Tensor {
            const rows = x.dim(0);
            const active = Tensor.scalar(1, .i32).broad(.init(.{rows}, .i32));
            var tops: [shards]Tensor = undefined;
            for (&tops, 0..) |*top, s| {
                top.* = localTopK(x.slice(1, .{ .start = @intCast(s * local_n), .end = @intCast((s + 1) * local_n) }), active, k);
            }
            return mergeShardTops(Tensor.concatenate(&tops, 1), local_n, k);
        }

        fn both(x: Tensor) [2]Tensor {
            const packed_top = merged(x);
            const expected = x.topK(1, k, .{ .descending = true });
            const indices = Tensor.scalar(std.math.maxInt(i32), .i64).sub(packed_top.logical(.AND, .scalar(0xffffffff, .i64)).shiftRightLogical(.scalar(1, .i64))).convert(.i32);
            return .{ indices, expected.indices };
        }
    };
    const rows = 4;
    const x: Tensor = .init(.{ rows, shards * local_n }, .f32);
    var exe = try platform.compileFn(allocator, io, Fns.both, .{x}, .{});
    defer exe.deinit();
    const values = try testValues(allocator, .ties, rows, shards * local_n, .f32);
    defer values.free(allocator);
    var x_buffer: zml.Buffer = try .fromSlice(io, platform, values);
    defer x_buffer.deinit();
    var results: [2]zml.Buffer = undefined;
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    runner.run(io, .{x_buffer}, &results, .{ .wait = true });
    defer for (&results) |*r| r.deinit();
    try std.testing.expectEqual(try results[1].getValue([rows][k]i32, io), try results[0].getValue([rows][k]i32, io));
}

test "topK kernels on an explicitly sharded axis" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const mesh = platform.meshes.get("model").?;
    const Fns = struct {
        fn both(x: Tensor, m: *const zml.Mesh) [2]Tensor.SortRes {
            const sharded = x.withPartitioning(m, .{ .b = .replicated, .n = .model });
            return .{ topK(sharded, .{ .top = .n }, 50, .{}), x.topK(.{ .top = .n }, 50, .{ .descending = true }) };
        }
    };
    const rows = 4;
    const n = 64640;
    const x: Tensor = .init(.{ .b = rows, .n = n }, .f32);
    var exe = try platform.compileFn(allocator, io, Fns.both, .{ x, mesh }, .{ .meshes = &.{mesh} });
    defer exe.deinit();
    const values = try testValues(allocator, .ties, rows, n, .f32);
    defer values.free(allocator);
    var x_buffer: zml.Buffer = try .fromSlice(io, platform, values);
    defer x_buffer.deinit();
    var results: [2]zml.Bufferized(Tensor.SortRes) = undefined;
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    runner.run(io, .{x_buffer}, &results, .{ .wait = true });
    defer for (&results) |*r| zml.Buffer.deinitAll(Tensor.SortRes, r);
    try expectSameTopK(io, results[0], results[1], null);
}
