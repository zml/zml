//! Pieces of the CuTe paged attention kernels that every architecture computes the same way.
const std = @import("std");
const stdx = @import("stdx");

const zml = @import("zml/core");
const host = @import("cute.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;

const PointerArgument = struct { ptr: cute.DType };

pub const KernelArguments = struct {
    q: PointerArgument,
    k: PointerArgument,
    v: PointerArgument,
    table: PointerArgument,
    lengths: PointerArgument,
    starts: PointerArgument,
    sink: PointerArgument,
    out: PointerArgument,

    // Split kernels write fp32 partials for the merge kernel instead of the final output.
    pub fn init(dtype: cute.DType, splits: i64) KernelArguments {
        return .{
            .q = .{ .ptr = dtype },
            .k = .{ .ptr = dtype },
            .v = .{ .ptr = dtype },
            .table = .{ .ptr = cute.DType.i32 },
            .lengths = .{ .ptr = cute.DType.i32 },
            .starts = .{ .ptr = cute.DType.i32 },
            .sink = .{ .ptr = cute.DType.f32 },
            .out = .{ .ptr = outputDtype(dtype, splits) },
        };
    }
};

fn outputDtype(dtype: cute.DType, splits: i64) cute.DType {
    if (splits > 1) {
        return .f32;
    }

    return dtype;
}

pub fn allowMaximumSharedMemory(b: *B) void {
    b.setFunctionAttribute(.named(b.ctx, "cu_attrs", .dict(b.ctx, &.{.named(b.ctx, "max_dynamic_shared_size_bytes", b.devMaxSharedMemoryOptin())})));
}

pub fn loadI32(b: *B, ptr: V, index: V) V {
    return b.loadWords(1, ptr.add(index.to(.i64).mul(4)), .global, .{})[0];
}

pub const KeyTiles = struct { first: V, count: V };

// The tiles of keys that the query rows [first_row, first_row + rows) can see, so every kernel and its merge agree on the partition.
pub fn keyTiles(b: *B, c: anytype, first_row: V, rows: i64, qlen: V, klen: V, tile_keys: i64) KeyTiles {
    const first_position = klen.sub(qlen).add(first_row.div(c.group));
    const last_position = klen.sub(qlen).add(first_row.add(rows - 1).div(c.group).minimum(qlen.sub(1)));
    const begin = firstVisibleKey(b, c.window, first_position);
    const end = endVisibleKey(c.causal, last_position, klen);
    const first = begin.div(tile_keys);
    return .{ .first = first, .count = end.add(tile_keys - 1).div(tile_keys).sub(first).maximum(0) };
}

fn firstVisibleKey(b: *B, window: i64, first_position: V) V {
    if (window > 0) {
        return first_position.sub(window - 1).maximum(0);
    }

    return b.constant(.i32, 0);
}

fn endVisibleKey(causal: bool, last_position: V, klen: V) V {
    if (causal) {
        return last_position.add(1).minimum(klen);
    }

    return klen;
}

pub const TileSpan = struct { lo: V, hi: V };

pub fn splitSpan(tiles: KeyTiles, split: V, splits: i64) TileSpan {
    return .{
        .lo = tiles.first.add(tiles.count.mul(split).div(splits)),
        .hi = tiles.first.add(tiles.count.mul(split.add(1)).div(splits)),
    };
}

// Element offset of one key row of one KV head. Rows past the end read page entry 0 so the page table is never read out of range.
pub fn pagedElement(b: *B, c: anytype, table: V, seq: V, head: V, key: V, key_valid: V, column: V) V {
    const safe_key = b.select(key_valid, key, b.constant(.i32, 0));
    const page = loadI32(b, table, seq.mul(c.pages_per_seq).add(safe_key.div(c.page_size)));
    return page.to(.i64).mul(c.page_size).add(safe_key.rem(c.page_size).to(.i64)).mul(c.kv_heads).add(head.to(.i64)).mul(c.head_dim).add(column.to(.i64));
}

// Index of a packed (token, group member) query row among the rows of one KV head's output, in units of head_dim.
pub fn queryRowIndex(c: anytype, token: V, head: V, packed_row: V) V {
    return token.to(.i64).mul(c.kv_heads).add(head.to(.i64)).mul(c.group).add(packed_row.rem(c.group).to(.i64));
}

// Element offset of the start of an output row: a final row, or an fp32 partial row with the maximum and sum after the features.
pub fn outputRowStart(c: anytype, split: V, row_index: V) V {
    if (c.splits > 1) {
        return split.to(.i64).mul(c.tokens * c.kv_heads * c.group).add(row_index).mul(c.head_dim + 2);
    }

    return row_index.mul(c.head_dim);
}

pub fn storeOutput(b: *B, c: anytype, out_pointer: V, split: V, start: V, head: V, row: V, dim: V, value: V) void {
    const token = start.add(row.div(c.group));
    const row_start = outputRowStart(c, split, queryRowIndex(c, token, head, row)).add(dim.to(.i64));
    if (c.splits > 1) {
        b.storeWords(1, out_pointer.add(row_start.mul(4)), .global, .{value.bitCast(.i32)});
    } else {
        b.storeHalf(out_pointer.add(row_start.mul(2)), .global, value.to(c.dtype).bitCast(.i16).to(.i32));
    }
}

// Reciprocal of the softmax denominator, folding the attention sink in as an extra zero-valued entry when the kernel is not split.
pub fn rowNormalizer(b: *B, c: anytype, sink_pointer: V, head: V, row: V, maximum: V, denominator: V, zero: V) V {
    const norm = b.select(denominator.gt(0), b.rcpApprox(denominator), zero);
    if (c.splits == 1 and c.sink) {
        const sink = loadI32(b, sink_pointer, head.mul(c.group).add(row.rem(c.group))).bitCast(.f32).mul(std.math.log2e);
        const joint_maximum = maximum.maximum(sink);
        const weight = b.exp2Approx(maximum.sub(joint_maximum));
        return weight.div(denominator.mul(weight).add(b.exp2Approx(sink.sub(joint_maximum))));
    }

    return norm;
}

pub fn multiprocessorCount() i64 {
    return @intCast(host.multiprocessorCount(zml.Compiler.current().platform));
}

pub fn pagedOperands(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
    const q_ = q.transpose(.{ .b, .hkv, .hg, .hd });
    const k_ = k.transpose(.{ .page, .k_chunk, .hkv, .hd });
    const v_ = v.transpose(.{ .page, .k_chunk, .hkv, .hd });
    stdx.debug.assert(q_.dim(.hkv) == k_.dim(.hkv) and k_.shape().eql(v_.shape()) and k_.dim(.hd) == q_.dim(.hd) and q_.dim(.hg) > 0, "CuTe paged attention requires aligned local Q/KV heads and matching caches", .{});
    return .{ q_, k_, v_ };
}

// Kernels that take no sink still receive a buffer, so it is zeros.
pub fn sinkOperand(opts: host.AttentionOptions, entries: i64) zml.Tensor {
    if (opts.sink) |sink| {
        return sink.convert(.f32).reshape(.{entries});
    }

    return zml.Tensor.zeroes(.init(.{entries}, .f32));
}

// Keys a query can reach at most, which bounds how many splits are worth launching.
pub fn keyBound(c: anytype) i64 {
    const cached_keys = c.pages_per_seq * c.page_size;
    if (c.window > 0) {
        return @min(c.window, cached_keys);
    }

    return cached_keys;
}

pub fn prefillQueryTiles(c: anytype, rows: i64) i64 {
    return c.batch + @divTrunc(c.tokens * c.group + rows - 1, rows);
}

// Enough splits to fill twice the multiprocessors, but none smaller than `keys_per_split` keys.
pub fn splitCount(c: anytype, multiprocessors: i64, query_tiles: i64, bound: i64, keys_per_split: i64) i64 {
    const work = query_tiles * c.kv_heads;
    return @max(1, @min(32, @min(@divTrunc(multiprocessors * 2 + work - 1, work), @divTrunc(bound + keys_per_split - 1, keys_per_split))));
}

pub fn splitOutputShape(c: anytype, final_shape: zml.Shape) zml.Shape {
    if (c.splits > 1) {
        return zml.Shape.init(.{ c.splits, c.tokens, c.kv_heads * c.group, c.head_dim + 2 }, .f32);
    }

    return final_shape;
}
