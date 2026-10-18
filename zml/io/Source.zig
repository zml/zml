//! A tensor the loader assembles from checkpoint bytes by moving them, never
//! by computing: a checkpoint tensor, or slices, reshapes, transposes,
//! broadcasts, paddings, concatenations and stacks of checkpoint tensors.
//! The ops are `zml.Tensor`'s shape ops, with the same names, tags and
//! argument conventions. A source touches neither the registry nor the
//! store's bindings: `bind` makes the model tensor the loader fills from it,
//! placed by the caller's partitioning, in the memory the caller names.
//! Sources are built during model init and live in the store's arena.
const std = @import("std");

const stdx = @import("stdx");

const constants = @import("../constants.zig");
const DataType = @import("../dtype.zig").DataType;
const Memory = @import("../platform.zig").Memory;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;
const TensorStore = @import("TensorStore.zig");

const Source = @This();

store: *TensorStore,
_shape: Shape,
/// Disjoint lattices of elements read from checkpoint tensors; elements no
/// piece covers are zero.
pieces: []const Piece,

/// Checkpoint bits read through an affine lattice: for every `j` with
/// `j[k] < dims[k].count`, the element at row-major index
/// `origin + sum(j[k] * dims[k].step)` of the source's shape is the element
/// at bit `offset + sum(j[k] * dims[k].stride)` of `tensor`. Reshapes keep
/// the row-major order, so they never touch a piece.
pub const Piece = struct {
    tensor: *const safetensors.Tensor,
    offset: u64,
    origin: u64,
    /// Outermost first, none of count 1.
    dims: []const Dim,
};

pub const Dim = struct {
    count: u64,
    /// Row-major elements of the source's shape between two steps.
    step: u64,
    /// Bits of the checkpoint tensor between two steps; zero repeats them.
    stride: u64,
};

/// `pad`'s per-axis options, as for `Tensor.pad`.
pub const Pad = Tensor.Pad;
/// `slice`'s range, as for `Tensor.slice`.
pub const Slice = Tensor.Slice;

pub const BindOptions = struct {
    /// See `TensorStore.Binding.memory`.
    memory: Memory.Kind = .default,
};

/// The checkpoint tensor itself, with `tags` (null keeps the checkpoint's).
pub fn init(store: *TensorStore, tensor: *const safetensors.Tensor, tags: ?[]const Shape.Tag) Source {
    var shape_ = tensor.shape;
    if (tags) |user_tags| {
        stdx.debug.assert(user_tags.len == shape_.rank(), "checkpoint tensor {s} has shape {f}, but got {d} tags: {f}", .{ tensor.name, shape_, user_tags.len, stdx.fmt.stringsZ(user_tags) });
        @memcpy(shape_._tags.slice(), user_tags);
    }
    shape_._sharding = .replicated;
    const arena = store.arena.allocator();
    const elements: u64 = shape_.count();
    const pieces: []const Piece = if (elements == 0) &.{} else blk: {
        const dims_: []const Dim = if (elements == 1)
            &.{}
        else
            arena.dupe(Dim, &.{.{ .count = elements, .step = 1, .stride = shape_.dtype().bitSizeOf() }}) catch oom();
        break :blk arena.dupe(Piece, &.{.{ .tensor = tensor, .offset = 0, .origin = 0, .dims = dims_ }}) catch oom();
    };
    return .{ .store = store, ._shape = shape_, .pieces = pieces };
}

pub fn shape(self: Source) Shape {
    return self._shape;
}

pub fn dim(self: Source, axis_: anytype) i64 {
    return self._shape.dim(axis_);
}

pub fn dims(self: *const Source) []const i64 {
    return self._shape.dims();
}

pub fn dtype(self: Source) DataType {
    return self._shape.dtype();
}

pub fn rank(self: Source) u4 {
    return self._shape.rank();
}

pub fn axis(self: Source, axis_: anytype) u3 {
    return self._shape.axis(axis_);
}

pub fn count(self: Source) usize {
    return self._shape.count();
}

pub fn format(self: Source, writer: *std.Io.Writer) !void {
    try writer.print("Source({f}, {d} pieces)", .{ self._shape, self.pieces.len });
}

// Reshapes: the row-major order is unchanged, so are the pieces.

pub fn withTags(self: Source, tags: anytype) Source {
    return self.reshaped(self._shape.withTags(tags));
}

pub fn rename(self: Source, renames: anytype) Source {
    return self.reshaped(self._shape.rename(renames));
}

pub fn renameTag(self: Source, old: anytype, new: anytype) Source {
    return self.reshaped(self._shape.setTag(old, new));
}

pub fn splitAxis(self: Source, ax: anytype, split_shape: anytype) Source {
    return self.reshaped(self._shape.splitAxis(ax, split_shape));
}

pub fn merge(self: Source, merges: anytype) Source {
    return self.reshaped(self._shape.mergeAxes(merges));
}

pub fn reshape(self: Source, output_shape: anytype) Source {
    return self.reshaped(self._shape.reshape(output_shape));
}

pub fn flatten(self: Source) Source {
    return self.reshape(.{self.count()});
}

pub fn insertAxes(self: Source, axis_: anytype, tags: anytype) Source {
    const tags_ = Shape.parseTags(tags);
    const ax = if (@TypeOf(axis_) == @EnumLiteral() and axis_ == .last) self.rank() else self.axis(axis_);
    var res = self._shape;
    const ones: [constants.MAX_RANK]i64 = @splat(1);
    res._dims.insertSlice(ax, ones[0..tags_.len]) catch unreachable;
    res._tags.insertSlice(ax, tags_.constSlice()) catch unreachable;
    return self.reshaped(res);
}

pub fn appendAxes(self: Source, tags: anytype) Source {
    return self.insertAxes(.last, tags);
}

pub fn squeeze(self: Source, axis_: anytype) Source {
    const a = self.axis(axis_);
    stdx.debug.assert(self.dim(a) == 1, "squeeze expects axis to be squeezed to have a dimension of 1, got {}", .{self.dim(a)});
    return self.reshaped(self._shape.remove(a));
}

fn reshaped(self: Source, new_shape: Shape) Source {
    stdx.debug.assert(new_shape.count() == self.count(), "Source reshape from {f} to {f} changes the element count", .{ self._shape, new_shape });
    var res = self;
    res._shape = new_shape.withDtype(self.dtype());
    res._shape._sharding = .replicated;
    return res;
}

/// Reinterprets the bits as `dt`, like `Tensor.bitCast`: to a narrower type,
/// each element becomes the elements of a new last axis `.bitcast`; to a
/// wider one, the last axis must hold one element of `dt`.
pub fn bitCast(self: Source, dt: DataType) Source {
    const src_bits: u64 = self.dtype().bitSizeOf();
    const tgt_bits: u64 = dt.bitSizeOf();
    if (src_bits == tgt_bits) return .{ .store = self.store, ._shape = self._shape.withDtype(dt), .pieces = self.pieces };
    var ctx: Rewrite = .init(self.store);
    defer ctx.deinit();
    if (src_bits > tgt_bits) {
        const ratio = std.math.divExact(u64, src_bits, tgt_bits) catch std.debug.panic("bitCast from {t} to {t}: bit sizes are not multiples", .{ self.dtype(), dt });
        var res_shape = self._shape.append(.{ .bitcast = @as(i64, @intCast(ratio)) }).withDtype(dt);
        res_shape._sharding = .replicated;
        for (self.pieces) |piece| {
            var flat: Flat = .fromPiece(piece);
            flat.origin *= ratio;
            for (flat.dims.slice()) |*d| d.step *= ratio;
            flat.dims.append(.{ .count = ratio, .step = 1, .stride = tgt_bits }) catch tooManyDims();
            ctx.emitFlat(flat);
        }
        return ctx.finish(res_shape);
    }
    const ratio = std.math.divExact(u64, tgt_bits, src_bits) catch std.debug.panic("bitCast from {t} to {t}: bit sizes are not multiples", .{ self.dtype(), dt });
    const last: u3 = @intCast(self.rank() - 1);
    stdx.debug.assert(self.dim(last) == @as(i64, @intCast(ratio)), "bitCast from {f} to {t} expects a last axis of {d} elements", .{ self._shape, dt, ratio });
    var res_shape = self._shape.remove(last).withDtype(dt);
    res_shape._sharding = .replicated;
    for (self.pieces) |piece| {
        ctx.decompose(self._shape, piece);
        for (ctx.placed.items) |*placed| {
            // The elements merged into one must be one contiguous run of bits.
            const inner: ?usize = for (placed.dims.constSlice(), 0..) |d, k| {
                if (d.axis == last) break k;
            } else null;
            const whole = placed.origin[last] == 0 and inner != null and blk: {
                const d = placed.dims.get(inner.?);
                break :blk d.count == ratio and d.step == 1 and d.stride == src_bits;
            };
            if (!whole) std.debug.panic("bitCast of {f} to {t}: an element of {t} would gather bits that are not contiguous in the checkpoint", .{ self._shape, dt, dt });
            _ = placed.dims.orderedRemove(inner.?);
            ctx.emit(res_shape, placed.*);
        }
    }
    return ctx.finish(res_shape);
}

// Ops that move elements.

/// Like `Tensor.slice`, for a static, contiguous slice (`step` 1).
pub fn slice(self: Source, axis_: anytype, s: Tensor.Slice) Source {
    stdx.debug.assert(s.dyn_start == null, "Source.slice takes static slices, got {f}", .{s});
    stdx.debug.assert(s.step == 1, "Source.slice takes contiguous slices, got {f}", .{s});
    const a = self.axis(axis_);
    const range = s.absolute(self.dim(a));
    const lo: u64 = @intCast(range.start);
    const hi: u64 = @intCast(range.end);
    const res_shape = self._shape.setDim(a, range.end - range.start);
    var ctx: Rewrite = .init(self.store);
    defer ctx.deinit();
    for (self.pieces) |piece| {
        ctx.decompose(self._shape, piece);
        for (ctx.placed.items) |placed| ctx.clip(placed, a, lo, hi);
        for (ctx.clipped.items) |*placed| {
            placed.origin[a] -= lo;
            ctx.emit(res_shape, placed.*);
        }
    }
    const sliced = ctx.finish(res_shape);
    return if (s.singleton) sliced.reshaped(res_shape.remove(a)) else sliced;
}

/// Like `Tensor.split`; the result lives in the store's arena.
pub fn split(self: Source, axis_: anytype, split_sizes: []const i64) []Source {
    const a = self.axis(axis_);
    var total: i64 = 0;
    for (split_sizes) |n| total += n;
    stdx.debug.assert(total == self.dim(a), "split expects sum of 'split_sizes' values and axis dimension to be equal, got {} and {}", .{ total, self.dim(a) });
    const res = self.store.arena.allocator().alloc(Source, split_sizes.len) catch oom();
    var start: i64 = 0;
    for (split_sizes, res) |n, *part| {
        part.* = self.slice(a, .{ .start = start, .end = start + n });
        start += n;
    }
    return res;
}

pub fn chunkExact(self: Source, axis_: anytype, comptime n_chunks: comptime_int) [n_chunks]Source {
    const a = self.axis(axis_);
    const size = @divExact(self.dim(a), n_chunks);
    var chunks: [n_chunks]Source = undefined;
    for (&chunks, 0..) |*chunk, i| {
        const start = @as(i64, @intCast(i)) * size;
        chunk.* = self.slice(a, .{ .start = start, .end = start + size });
    }
    return chunks;
}

/// Like `Tensor.transpose`: no axes reverses them.
pub fn transpose(self: Source, axes_: anytype) Source {
    const given = self._shape.axes(axes_);
    const r = self.rank();
    var permutation: [constants.MAX_RANK]u3 = undefined;
    if (given.len == 0) {
        for (0..r) |i| permutation[i] = @intCast(r - 1 - i);
    } else {
        stdx.debug.assert(given.len == r, "transpose expects {d} axes for {f}, got {d}", .{ r, self._shape, given.len });
        @memcpy(permutation[0..r], given.constSlice());
    }
    const identity = for (permutation[0..r], 0..) |p, i| {
        if (p != i) break false;
    } else true;
    if (identity) return self;

    var as_i64: [constants.MAX_RANK]i64 = undefined;
    for (permutation[0..r], as_i64[0..r]) |p, *q| q.* = p;
    const res_shape = self._shape.transpose(as_i64[0..r]);
    var inverse: [constants.MAX_RANK]u3 = undefined;
    for (permutation[0..r], 0..) |p, i| inverse[p] = @intCast(i);
    var ctx: Rewrite = .init(self.store);
    defer ctx.deinit();
    for (self.pieces) |piece| {
        ctx.decompose(self._shape, piece);
        for (ctx.placed.items) |placed| {
            var moved = placed;
            for (0..r) |i| moved.origin[i] = placed.origin[permutation[i]];
            for (moved.dims.slice()) |*d| d.axis = inverse[d.axis];
            ctx.emit(res_shape, moved);
        }
    }
    return ctx.finish(res_shape);
}

/// Like `Tensor.pad`, with zero as the only padding value.
pub fn pad(self: Source, padding_value: anytype, paddings: anytype) Source {
    stdx.debug.assert(isZero(padding_value), "Source.pad pads with zeros, got {any}", .{padding_value});
    const options = self._shape.parseAxesOptions(Tensor.Pad, paddings, .{});
    var res_shape = self._shape;
    var moves = false;
    for (options.constSlice(), 0..) |p, a| {
        stdx.debug.assert(p.low >= 0 and p.high >= 0 and p.interior >= 0, "Source.pad cannot crop, got {any} on axis {d}", .{ p, a });
        const d = self.dim(a);
        res_shape._dims.set(a, p.low + d + @max(d - 1, 0) * p.interior + p.high);
        if (p.low != 0 or p.interior != 0 or (a != 0 and p.high != 0)) moves = true;
    }
    // Rows appended after the last one move no element.
    if (!moves) return .{ .store = self.store, ._shape = res_shape, .pieces = self.pieces };
    var ctx: Rewrite = .init(self.store);
    defer ctx.deinit();
    for (self.pieces) |piece| {
        ctx.decompose(self._shape, piece);
        for (ctx.placed.items) |*placed| {
            for (options.constSlice(), 0..) |p, a| {
                placed.origin[a] = @as(u64, @intCast(p.low)) + placed.origin[a] * @as(u64, @intCast(p.interior + 1));
            }
            for (placed.dims.slice()) |*d| d.step *= @intCast(options.get(d.axis).interior + 1);
            ctx.emit(res_shape, placed.*);
        }
    }
    return ctx.finish(res_shape);
}

/// Like `Tensor.broad`: repeats the elements along the axes `other` adds or
/// grows from 1. The loader reads a few large repetitions once each, and
/// copies many small ones from one read of the source per job.
pub fn broad(self: Source, other: Shape) Source {
    stdx.debug.assert(self._shape.canBroadcastTo(other), "Can't broadcast {f} to {f}", .{ self._shape, other });
    if (std.mem.eql(i64, self.dims(), other.dims())) return self;
    const r = self.rank();
    var axes_map: [constants.MAX_RANK]u3 = undefined;
    if (r == 0 or r == other.rank()) {
        for (0..r) |i| axes_map[i] = @intCast(i);
    } else {
        for (self._shape.tags(), 0..) |t, i| axes_map[i] = other.axis(t);
    }
    var res_shape = other.withDtype(self.dtype());
    res_shape._sharding = .replicated;
    // The axes the elements repeat along: new ones and those grown from 1.
    var repeated: [constants.MAX_RANK]bool = @splat(true);
    for (0..r) |i| {
        if (self.dim(i) != 1) repeated[axes_map[i]] = false;
    }
    var ctx: Rewrite = .init(self.store);
    defer ctx.deinit();
    for (self.pieces) |piece| {
        ctx.decompose(self._shape, piece);
        for (ctx.placed.items) |placed| {
            var moved: Placed = .{ .tensor = placed.tensor, .offset = placed.offset };
            for (0..r) |i| moved.origin[axes_map[i]] = placed.origin[i];
            for (placed.dims.constSlice()) |d| {
                var mapped = d;
                mapped.axis = axes_map[d.axis];
                moved.dims.append(mapped) catch tooManyDims();
            }
            for (0..other.rank()) |b| {
                if (!repeated[b] or other.dim(b) == 1) continue;
                moved.dims.append(.{ .count = @intCast(other.dim(b)), .axis = @intCast(b), .step = 1, .stride = 0 }) catch tooManyDims();
            }
            ctx.emit(res_shape, moved);
        }
    }
    return ctx.finish(res_shape);
}

/// Like `Tensor.concatenate`; the sources must share every dimension but
/// `axis_`, and the dtype.
pub fn concatenate(sources: []const Source, axis_: anytype) Source {
    stdx.debug.assert(sources.len > 0, "concatenate expects at least one source", .{});
    if (sources.len == 1) return sources[0];
    const first = sources[0];
    const a = first.axis(axis_);
    var total: i64 = 0;
    for (sources) |s| {
        stdx.debug.assert(s.store == first.store, "concatenate expects sources of one store", .{});
        const same = s.dtype() == first.dtype() and s.rank() == first.rank() and for (0..first.rank()) |i| {
            if (i != a and s.dim(i) != first.dim(i)) break false;
        } else true;
        stdx.debug.assert(same, "concatenate along axis {d} expects matching shapes, got {f} and {f}", .{ a, first._shape, s._shape });
        total += s.dim(a);
    }
    const res_shape = first._shape.setDim(a, total);
    const outer_stride = elementStrides(res_shape)[a];
    var ctx: Rewrite = .init(first.store);
    defer ctx.deinit();
    var offset: u64 = 0;
    for (sources) |s| {
        for (s.pieces) |piece| {
            if (a == 0) {
                // Along the outermost axis, a source's elements follow the previous one's.
                var flat: Flat = .fromPiece(piece);
                flat.origin += offset * outer_stride;
                ctx.emitFlat(flat);
                continue;
            }
            ctx.decompose(s._shape, piece);
            for (ctx.placed.items) |*placed| {
                placed.origin[a] += offset;
                ctx.emit(res_shape, placed.*);
            }
        }
        offset += @intCast(s.dim(a));
    }
    return ctx.finish(res_shape);
}

/// Like `Tensor.stack`; the sources must have the same shape and tags.
pub fn stack(sources: []const Source, axis_: anytype, tag: anytype) Source {
    stdx.debug.assert(sources.len > 0, "stack expects at least one source", .{});
    const first = sources[0];
    for (sources[1..]) |s| {
        stdx.debug.assert(s.store == first.store, "stack expects sources of one store", .{});
        stdx.debug.assert(first._shape.eqlWithTags(s._shape), "stack expects source shapes to match, got {f} and {f}", .{ first._shape, s._shape });
    }
    const a: u3 = if (@TypeOf(axis_) == @EnumLiteral() and axis_ == .last) @intCast(first.rank()) else first.axis(axis_);
    var res_shape = first._shape.insertTag(axis_, @intCast(sources.len), tag);
    res_shape._sharding = .replicated;
    const r = first.rank();
    var ctx: Rewrite = .init(first.store);
    defer ctx.deinit();
    for (sources, 0..) |s, i| {
        for (s.pieces) |piece| {
            if (a == 0) {
                var flat: Flat = .fromPiece(piece);
                flat.origin += i * first.count();
                ctx.emitFlat(flat);
                continue;
            }
            ctx.decompose(s._shape, piece);
            for (ctx.placed.items) |placed| {
                var moved = placed;
                std.mem.copyBackwards(u64, moved.origin[a + 1 .. r + 1], placed.origin[a..r]);
                moved.origin[a] = i;
                for (moved.dims.slice()) |*d| {
                    if (d.axis >= a) d.axis += 1;
                }
                ctx.emit(res_shape, moved);
            }
        }
    }
    return ctx.finish(res_shape);
}

/// Makes the model tensor the loader fills from this source, placed on
/// `mesh_` by `partitioning`: a struct keyed by this source's tags (axes it
/// does not name are replicated), or a `Sharding.Partitioning`. `.replicated`
/// places it on every device without a mesh. A source that is one whole
/// checkpoint tensor binds it like `View.createTensor`; any other source
/// binds a composite (`TensorStore.Layout`).
pub fn bind(self: Source, mesh_: @EnumLiteral(), partitioning: anytype, opts: BindOptions) Tensor {
    const sharding: Sharding = if (mesh_ == .replicated) .replicated else blk: {
        const mesh = self.store.mesh(mesh_);
        const partition: Sharding.Partitioning = if (@TypeOf(partitioning) == Sharding.Partitioning)
            partitioning
        else
            .parse(self._shape.tags(), mesh, .replicated, partitioning);
        break :blk .{ .mesh = mesh, .partition = partition };
    };
    var placed = self._shape;
    placed._sharding = sharding;
    if (sharding.mesh) |mesh| placed = placed.withPartitioning(mesh, sharding.partition);

    const tensor: Tensor = .fromShape(placed);
    self.store.putSourcesNoClobber(tensor.id, self.lower(opts.memory)) catch oom();
    return tensor;
}

/// Like `bind`, with the tags and sharding of `placed`, a shape built
/// elsewhere; its dimensions and dtype must be the source's.
pub fn bindAs(self: Source, placed: Shape, opts: BindOptions) Tensor {
    stdx.debug.assert(placed.eql(self._shape), "Source {f} cannot bind as {f}", .{ self._shape, placed });
    const tensor: Tensor = .fromShape(placed);
    self.store.putSourcesNoClobber(tensor.id, self.lower(opts.memory)) catch oom();
    return tensor;
}

/// The binding: the checkpoint tensors the pieces read, deduplicated and
/// copied for the binding to own, and the pieces as byte parts.
fn lower(self: Source, memory: Memory.Kind) TensorStore.Binding {
    const arena = self.store.arena.allocator();
    var tensors: std.ArrayList(*safetensors.Tensor) = .empty;
    // Each `View.source` copies its registry entry, and the copies share the entry's name:
    // the key identifies the entry, not the copy.
    const Entry = struct { name: [*]const u8, len: usize, offset: u64 };
    var indices: std.AutoHashMapUnmanaged(Entry, u32) = .empty;
    defer indices.deinit(self.store.allocator);
    const parts = arena.alloc(TensorStore.Part, self.pieces.len) catch oom();
    for (self.pieces, parts) |piece, *part| {
        const entry: Entry = .{ .name = piece.tensor.name.ptr, .len = piece.tensor.name.len, .offset = piece.tensor.offset };
        const found = indices.getOrPut(self.store.allocator, entry) catch oom();
        if (!found.found_existing) {
            const copy = arena.create(safetensors.Tensor) catch oom();
            copy.* = piece.tensor.*;
            tensors.append(arena, copy) catch oom();
            found.value_ptr.* = @intCast(tensors.items.len - 1);
        }
        part.* = lowerPiece(arena, self._shape, piece);
        part.source = found.value_ptr.*;
    }
    const owned = tensors.toOwnedSlice(arena) catch oom();

    // One whole checkpoint tensor is a plain binding.
    if (parts.len == 1 and parts[0].dims.len == 0 and parts[0].src_offset == 0 and parts[0].dst_offset == 0 and
        parts[0].width == self._shape.byteSize() and owned[0].byteSize() == parts[0].width)
    {
        return .{ .tensors = owned, .transformed = false, .memory = memory };
    }
    const layout = arena.create(TensorStore.Layout) catch oom();
    layout.* = .{ .shape = self._shape, .parts = parts };
    return .{ .tensors = owned, .transformed = false, .memory = memory, .layout = layout };
}

/// A piece in bytes: its dimensions outermost first, the innermost run that
/// is contiguous on both sides as the width.
fn lowerPiece(arena: std.mem.Allocator, res_shape: Shape, piece: Piece) TensorStore.Part {
    const ebits: u64 = res_shape.dtype().bitSizeOf();
    var lowered: stdx.BoundedArray(TensorStore.Part.Dim, max_dims) = .empty;
    for (piece.dims) |d| lowered.appendAssumeCapacity(.{ .count = d.count, .src_stride = d.stride, .dst_stride = d.step * ebits });
    var width: u64 = ebits;
    if (lowered.len != 0) {
        const inner = lowered.get(lowered.len - 1);
        if (inner.dst_stride == ebits and inner.src_stride == ebits) {
            width = inner.count * ebits;
            _ = lowered.pop();
        }
    }
    const dst_bits = piece.origin * ebits;
    var aligned = piece.offset % 8 == 0 and dst_bits % 8 == 0 and width % 8 == 0;
    for (lowered.slice()) |*d| {
        aligned = aligned and d.src_stride % 8 == 0 and d.dst_stride % 8 == 0;
        d.src_stride /= 8;
        d.dst_stride /= 8;
    }
    if (!aligned) std.debug.panic("Source {f} places {s} at sub-byte positions, which the loader cannot do", .{ res_shape, piece.tensor.name });
    return .{
        .source = 0,
        .src_offset = piece.offset / 8,
        .dst_offset = dst_bits / 8,
        .width = width / 8,
        .dims = if (lowered.len == 0) &.{} else arena.dupe(TensorStore.Part.Dim, lowered.constSlice()) catch oom(),
    };
}

// The machinery. A piece is kept flat; an op that addresses axes rewrites it
// in coordinates of the input shape (`Placed`, every dimension moving one
// axis), moves the coordinates, then flattens it in the result's shape.

const max_dims = 2 * constants.MAX_RANK;

/// A piece while an op rewrites it, its dimensions inline.
const Flat = struct {
    tensor: *const safetensors.Tensor,
    offset: u64,
    origin: u64,
    dims: stdx.BoundedArray(Dim, max_dims) = .empty,

    fn fromPiece(piece: Piece) Flat {
        var flat: Flat = .{ .tensor = piece.tensor, .offset = piece.offset, .origin = piece.origin };
        flat.dims.appendSlice(piece.dims) catch tooManyDims();
        return flat;
    }

    /// The same lattice with dimension `k` cut to `n` steps from step `first`.
    fn sub(self: Flat, k: usize, first: u64, n: u64) Flat {
        var res = self;
        const d = self.dims.get(k);
        res.origin += first * d.step;
        res.offset += first * d.stride;
        res.dims.buffer[k].count = n;
        return res;
    }
};

/// A piece in coordinates of a shape: the element at
/// `origin + sum(j[k] * dims[k].step along dims[k].axis)`.
const Placed = struct {
    tensor: *const safetensors.Tensor,
    offset: u64,
    origin: [constants.MAX_RANK]u64 = @splat(0),
    dims: stdx.BoundedArray(AxisDim, max_dims) = .empty,
};

const AxisDim = struct {
    count: u64,
    axis: u3,
    step: u64,
    stride: u64,
};

/// The scratch of one op and the pieces it makes.
const Rewrite = struct {
    store: *TensorStore,
    scratch: std.heap.ArenaAllocator,
    work: std.ArrayList(Flat) = .empty,
    /// The current piece in coordinates of its shape (`decompose`).
    placed: std.ArrayList(Placed) = .empty,
    /// What `clip` kept of them.
    clipped: std.ArrayList(Placed) = .empty,
    out: std.ArrayList(Piece) = .empty,

    fn init(store: *TensorStore) Rewrite {
        return .{ .store = store, .scratch = .init(store.allocator) };
    }

    fn deinit(self: *Rewrite) void {
        self.scratch.deinit();
    }

    fn finish(self: *Rewrite, res_shape: Shape) Source {
        var s = res_shape;
        s._sharding = .replicated;
        return .{
            .store = self.store,
            ._shape = s,
            .pieces = self.store.arena.allocator().dupe(Piece, self.out.items) catch oom(),
        };
    }

    /// Rewrites `piece` in coordinates of `shape` into `placed`, cutting it
    /// where a dimension would carry from one axis into the next.
    fn decompose(self: *Rewrite, shape_: Shape, piece: Piece) void {
        const scratch = self.scratch.allocator();
        self.placed.clearRetainingCapacity();
        self.clipped.clearRetainingCapacity();
        self.work.clearRetainingCapacity();
        self.work.append(scratch, .fromPiece(piece)) catch oom();
        const strides = elementStrides(shape_);
        while (self.work.pop()) |flat| {
            switch (place(shape_, strides, flat)) {
                .placed => |placed| self.placed.append(scratch, placed) catch oom(),
                .overflow => |overflowing| self.cut(shape_, strides, flat, overflowing),
            }
        }
    }

    /// Splits `flat`, which carries past the end of axis `a`, into pieces
    /// that do not: at the row boundaries of `a` when its longest dimension
    /// there steps regularly through the rows, else one piece per step.
    fn cut(self: *Rewrite, shape_: Shape, strides: [constants.MAX_RANK]u64, flat: Flat, a: u3) void {
        const scratch = self.scratch.allocator();
        const n: u64 = @intCast(shape_.dim(a));
        const c = (flat.origin / strides[a]) % n;
        var longest: ?usize = null;
        var longest_span: u64 = 0;
        var total: u64 = 0;
        for (flat.dims.constSlice(), 0..) |d, k| {
            if (d.count <= 1 or axisOfStep(shape_.rank(), strides, d.step) != a) continue;
            const span = (d.count - 1) * (d.step / strides[a]);
            total += span;
            if (longest == null or span > longest_span) {
                longest = k;
                longest_span = span;
            }
        }
        const k = longest.?;
        const d = flat.dims.get(k);
        const s = d.step / strides[a];
        const rest = total - longest_span;
        if (n % s == 0 and c % s + rest < s) {
            const per_row = n / s;
            var j: u64 = 0;
            if (c >= s) {
                // A partial first row.
                const head = @min((n - c + s - 1) / s, d.count);
                self.work.append(scratch, flat.sub(k, 0, head)) catch oom();
                j = head;
            }
            const rows = (d.count - j) / per_row;
            if (rows > 0) {
                var body = flat.sub(k, j, per_row);
                body.dims.append(.{ .count = rows, .step = d.step * per_row, .stride = d.stride * per_row }) catch tooManyDims();
                self.work.append(scratch, body) catch oom();
                j += rows * per_row;
            }
            if (j < d.count) self.work.append(scratch, flat.sub(k, j, d.count - j)) catch oom();
            return;
        }
        // An irregular lattice: one piece per step of that dimension.
        for (0..d.count) |step_index| {
            var single = flat.sub(k, step_index, 1);
            _ = single.dims.orderedRemove(k);
            self.work.append(scratch, single) catch oom();
        }
    }

    /// Appends the elements of `placed` whose coordinate on `a` is in
    /// `[lo, hi)` to `clipped`.
    fn clip(self: *Rewrite, placed: Placed, a: u3, lo: u64, hi: u64) void {
        const scratch = self.scratch.allocator();
        var span: u64 = 0;
        var outer: ?usize = null;
        for (placed.dims.constSlice(), 0..) |d, k| {
            if (d.axis != a or d.count <= 1) continue;
            span += (d.count - 1) * d.step;
            if (outer == null or d.step > placed.dims.get(outer.?).step) outer = k;
        }
        const o = placed.origin[a];
        if (o >= lo and o + span < hi) {
            self.clipped.append(scratch, placed) catch oom();
            return;
        }
        if (o + span < lo or o >= hi) return;
        // Cut along the outermost dimension on `a`: the steps whose whole
        // sub-lattice is inside stay one piece, the partial ones recurse.
        const k = outer.?;
        const d = placed.dims.get(k);
        const inner = span - (d.count - 1) * d.step;
        const full_lo: u64 = if (o >= lo) 0 else std.math.divCeil(u64, lo - o, d.step) catch unreachable;
        const full_hi: u64 = if (o + inner >= hi) 0 else @min(d.count, (hi - 1 - inner - o) / d.step + 1);
        if (full_lo < full_hi) {
            var kept = placed;
            kept.dims.buffer[k].count = full_hi - full_lo;
            kept.origin[a] = o + full_lo * d.step;
            kept.offset += full_lo * d.stride;
            self.clipped.append(scratch, kept) catch oom();
        }
        const touch_lo: u64 = if (o + inner >= lo) 0 else std.math.divCeil(u64, lo - o - inner, d.step) catch unreachable;
        const touch_hi: u64 = @min(d.count, (hi - 1 - o) / d.step + 1);
        const partial = [_][2]u64{
            .{ touch_lo, @min(full_lo, touch_hi) },
            .{ @max(full_hi, touch_lo, full_lo), touch_hi },
        };
        for (partial) |range| {
            var j = range[0];
            while (j < range[1]) : (j += 1) {
                var part = placed;
                part.origin[a] = o + j * d.step;
                part.offset += j * d.stride;
                _ = part.dims.orderedRemove(k);
                self.clip(part, a, lo, hi);
            }
        }
    }

    /// Flattens `placed` in `res_shape` into the result.
    fn emit(self: *Rewrite, res_shape: Shape, placed: Placed) void {
        const strides = elementStrides(res_shape);
        var flat: Flat = .{ .tensor = placed.tensor, .offset = placed.offset, .origin = 0 };
        for (0..res_shape.rank()) |b| flat.origin += placed.origin[b] * strides[b];
        for (placed.dims.constSlice()) |d| {
            if (d.count <= 1) continue;
            flat.dims.append(.{ .count = d.count, .step = d.step * strides[d.axis], .stride = d.stride }) catch tooManyDims();
        }
        self.emitFlat(flat);
    }

    /// Adds `flat` to the result, its dimensions outermost first and merged
    /// where one continues the other.
    fn emitFlat(self: *Rewrite, flat: Flat) void {
        var sorted: stdx.BoundedArray(Dim, max_dims) = .empty;
        for (flat.dims.constSlice()) |d| {
            if (d.count > 1) sorted.appendAssumeCapacity(d);
        }
        std.mem.sort(Dim, sorted.slice(), {}, struct {
            fn moreOuter(_: void, lhs: Dim, rhs: Dim) bool {
                return lhs.step > rhs.step;
            }
        }.moreOuter);
        var merged: stdx.BoundedArray(Dim, max_dims) = .empty;
        for (sorted.constSlice()) |d| {
            if (merged.len != 0) {
                const last = &merged.slice()[merged.len - 1];
                if (last.step == d.count * d.step and last.stride == d.count * d.stride) {
                    last.* = .{ .count = last.count * d.count, .step = d.step, .stride = d.stride };
                    continue;
                }
            }
            merged.appendAssumeCapacity(d);
        }
        const arena = self.store.arena.allocator();
        self.out.append(self.scratch.allocator(), .{
            .tensor = flat.tensor,
            .offset = flat.offset,
            .origin = flat.origin,
            .dims = if (merged.len == 0) &.{} else arena.dupe(Dim, merged.constSlice()) catch oom(),
        }) catch oom();
    }
};

const Placement = union(enum) {
    placed: Placed,
    /// The axis some coordinate would run past.
    overflow: u3,
};

/// `flat` in coordinates of `shape_`, if no dimension carries into the next
/// axis: each dimension moves the outermost axis its step is a multiple of.
fn place(shape_: Shape, strides: [constants.MAX_RANK]u64, flat: Flat) Placement {
    const r = shape_.rank();
    var placed: Placed = .{ .tensor = flat.tensor, .offset = flat.offset };
    for (0..r) |b| placed.origin[b] = (flat.origin / strides[b]) % @as(u64, @intCast(shape_.dim(b)));
    var extent = placed.origin;
    for (flat.dims.constSlice()) |d| {
        if (d.count <= 1) continue;
        const b = axisOfStep(r, strides, d.step);
        const step = d.step / strides[b];
        placed.dims.appendAssumeCapacity(.{ .count = d.count, .axis = b, .step = step, .stride = d.stride });
        extent[b] += (d.count - 1) * step;
    }
    for (0..r) |b| {
        if (extent[b] >= @as(u64, @intCast(shape_.dim(b)))) return .{ .overflow = @intCast(b) };
    }
    return .{ .placed = placed };
}

fn axisOfStep(r: u4, strides: [constants.MAX_RANK]u64, step: u64) u3 {
    for (0..r) |b| {
        if (step % strides[b] == 0) return @intCast(b);
    }
    std.debug.panic("a lattice step of {d} elements in a shape of rank {d}", .{ step, r });
}

/// Row-major element strides.
fn elementStrides(shape_: Shape) [constants.MAX_RANK]u64 {
    var strides: [constants.MAX_RANK]u64 = @splat(0);
    var acc: u64 = 1;
    var b: usize = shape_.rank();
    while (b > 0) {
        b -= 1;
        strides[b] = acc;
        acc *= @intCast(shape_.dim(b));
    }
    return strides;
}

fn isZero(value: anytype) bool {
    return switch (@typeInfo(@TypeOf(value))) {
        .comptime_int, .int, .comptime_float, .float => value == 0,
        else => false,
    };
}

fn oom() noreturn {
    @panic("OOM");
}

fn tooManyDims() noreturn {
    std.debug.panic("a Source piece needs more than {d} dimensions", .{max_dims});
}

// Tests: every op is checked against the same function run as a program
// over `zml.Tensor`, the loader's byte placement replayed on the host.

const testing = @import("../testing.zig");
const Buffer = @import("../buffer.zig").Buffer;

/// A checkpoint of in-memory tensors (the registry points nowhere) and the
/// bytes each holds.
const TestCheckpoint = struct {
    registry: safetensors.TensorRegistry,
    store: TensorStore,
    bytes: std.StringHashMapUnmanaged([]u8) = .empty,
    allocator: std.mem.Allocator,
    next_offset: u64 = 0,

    fn init(self: *TestCheckpoint, allocator: std.mem.Allocator) void {
        self.* = .{ .registry = .init(allocator), .store = undefined, .allocator = allocator };
        self.store = .fromRegistry(allocator, &self.registry, &.{});
    }

    fn deinit(self: *TestCheckpoint) void {
        var it = self.bytes.valueIterator();
        while (it.next()) |bytes| self.allocator.free(bytes.*);
        self.bytes.deinit(self.allocator);
        self.store.deinit();
        self.registry.deinit();
    }

    /// Registers `name` of `shape` holding distinct bytes.
    fn add(self: *TestCheckpoint, name: []const u8, shape_: Shape) !void {
        try self.registry.registerTensor(.{ .file_uri = "memory", .name = name, .shape = shape_, .offset = self.next_offset });
        self.next_offset += shape_.byteSize();
        const bytes = try self.allocator.alloc(u8, shape_.byteSize());
        for (bytes, 0..) |*byte, i| byte.* = @truncate(i *% 7 +% self.bytes.count() *% 61 +% 1);
        try self.bytes.put(self.allocator, self.registry.tensors.getKey(name).?, bytes);
    }

    /// The bytes the loader would assemble for `tensor`, placed on the host.
    fn assemble(self: *TestCheckpoint, tensor: Tensor) ![]u8 {
        const binding = self.store.getSourcesById(tensor.id).?;
        const out = try self.allocator.alloc(u8, tensor.shape().byteSize());
        @memset(out, 0);
        const layout = binding.layout orelse {
            @memcpy(out, self.bytes.get(binding.tensors[0].name).?);
            return out;
        };
        for (layout.parts) |part| {
            const src = self.bytes.get(binding.tensors[part.source].name).?;
            var chunk: u64 = 0;
            while (chunk < part.chunks()) : (chunk += 1) {
                var rest = chunk;
                var src_offset = part.src_offset;
                var dst_offset = part.dst_offset;
                var d = part.dims.len;
                while (d > 0) {
                    d -= 1;
                    src_offset += (rest % part.dims[d].count) * part.dims[d].src_stride;
                    dst_offset += (rest % part.dims[d].count) * part.dims[d].dst_stride;
                    rest /= part.dims[d].count;
                }
                @memcpy(out[dst_offset..][0..part.width], src[src_offset..][0..part.width]);
            }
        }
        return out;
    }
};

/// Each case builds a tensor from its inputs with shape ops only, the same
/// code for `Tensor` and `Source`.
const cases = struct {
    fn interleave(comptime T: type, members: []const T, groups: i64) T {
        var parts: [4]T = undefined;
        for (members, parts[0..members.len]) |member, *part| part.* = member.splitAxis(.row, .{ .group = groups, .r = -1 });
        return T.concatenate(parts[0..members.len], .r).merge(.{ .row = .{ .group, .r } });
    }

    const ConcatenateRows = struct {
        const inputs = .{ .{ .row = 3, .col = 4 }, .{ .row = 2, .col = 4 }, .{ .row = 4, .col = 4 } };
        fn build(comptime T: type, x: []const T) T {
            return T.concatenate(x, .row);
        }
    };
    const ConcatenateColumns = struct {
        const inputs = .{ .{ .row = 3, .col = 4 }, .{ .row = 3, .col = 2 } };
        fn build(comptime T: type, x: []const T) T {
            return T.concatenate(x, .col);
        }
    };
    const InterleaveByHeads = struct {
        const inputs = .{ .{ .row = 4, .col = 3 }, .{ .row = 2, .col = 3 }, .{ .row = 2, .col = 3 } };
        fn build(comptime T: type, x: []const T) T {
            return interleave(T, x, 2);
        }
    };
    const InterleaveRows = struct {
        const inputs = .{ .{ .row = 4, .col = 3 }, .{ .row = 4, .col = 3 } };
        fn build(comptime T: type, x: []const T) T {
            return interleave(T, x, 4);
        }
    };
    const StackExperts = struct {
        const inputs = .{ .{ .row = 2, .col = 3 }, .{ .row = 2, .col = 3 }, .{ .row = 2, .col = 3 }, .{ .row = 2, .col = 3 } };
        fn build(comptime T: type, x: []const T) T {
            const members = [2]T{ T.stack(x[0..2], 0, .expert), T.stack(x[2..4], 0, .expert) };
            return T.concatenate(&members, .row);
        }
    };
    const InterleaveExperts = struct {
        const inputs = StackExperts.inputs;
        fn build(comptime T: type, x: []const T) T {
            const members = [2]T{ T.stack(x[0..2], 0, .expert), T.stack(x[2..4], 0, .expert) };
            return T.stack(&members, 2, .proj).merge(.{ .rows = .{ .row, .proj } });
        }
    };
    const InterleaveExpertVectors = struct {
        // Per-expert vectors (biases): row r of projection p lands at r * 2 + p.
        const inputs = .{ .{ .row = 3 }, .{ .row = 3 }, .{ .row = 3 }, .{ .row = 3 } };
        fn build(comptime T: type, x: []const T) T {
            const members = [2]T{ T.stack(x[0..2], 0, .expert), T.stack(x[2..4], 0, .expert) };
            return T.stack(&members, .last, .proj).merge(.{ .rows = .{ .row, .proj } });
        }
    };
    const Transpose = struct {
        const inputs = .{.{ .row = 3, .col = 4 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].transpose(.{ .col, .row });
        }
    };
    const RowsOfThree = struct {
        // lfm2's in_proj: rows [B; C; x] become b0 c0 x0 b1 c1 x1.
        const inputs = .{.{ .row = 6, .col = 4 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].splitAxis(.row, .{ .m = 3, .c = 2 }).transpose(.{ .c, .m, .col }).merge(.{ .row = .{ .c, .m } });
        }
    };
    const SliceRows = struct {
        const inputs = .{.{ .row = 5, .col = 4 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].slice(.row, .{ .start = 1, .end = 4 });
        }
    };
    const SliceColumns = struct {
        const inputs = .{.{ .row = 3, .col = 4 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].slice(.col, .{ .start = 1, .end = 3 });
        }
    };
    const SliceByLength = struct {
        const inputs = .{.{ .row = 6, .col = 4 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].slice(.row, .{ .start = 2, .len = 3 });
        }
    };
    const SliceInterleaved = struct {
        const inputs = InterleaveByHeads.inputs;
        fn build(comptime T: type, x: []const T) T {
            return interleave(T, x, 2).slice(.row, .{ .start = 3, .end = 7 });
        }
    };
    const SingleRow = struct {
        const inputs = .{.{ .row = 5, .col = 4 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].slice(.row, .single(2));
        }
    };
    const Pad = struct {
        const inputs = .{.{ .row = 4, .col = 4 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].pad(0, .{ .row = Tensor.Pad{ .high = 4 }, .col = Tensor.Pad{ .low = 1, .interior = 1 } });
        }
    };
    const PadRows = struct {
        const inputs = .{.{ .row = 5, .col = 4 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].pad(0, .{ .row = Tensor.Pad{ .high = 3 } });
        }
    };
    const BroadcastScalar = struct {
        const inputs = .{.{}};
        fn build(comptime T: type, x: []const T) T {
            return x[0].broad(Shape.init(.{ .row = 3, .col = 2 }, .u16));
        }
    };
    const BroadcastColumn = struct {
        // A per-tensor scale repeated over each member's rows.
        const inputs = .{ .{ .row = 1, .col = 1 }, .{ .row = 1, .col = 1 } };
        fn build(comptime T: type, x: []const T) T {
            const members = [2]T{ x[0].broad(Shape.init(.{ .row = 4, .col = 1 }, .u16)), x[1].broad(Shape.init(.{ .row = 2, .col = 1 }, .u16)) };
            return interleave(T, &members, 2);
        }
    };
    const NarrowBits = struct {
        const inputs = .{.{ .row = 3, .col = 4 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].bitCast(.u8).merge(.{ .col = .{ .col, .bitcast } });
        }
    };
    const WidenBits = struct {
        const inputs = .{.{ .row = 2, .col = 6 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].bitCast(.u8).splitAxis(.col, .{ .col = 3, .pair = 4 }).merge(.{ .pair = .{.pair} }).bitCast(.u32);
        }
    };
    const Swizzle = struct {
        // The 128x4 block-scale layout, at 8x4: rows padded to 8 and tiled
        // by 2x4, columns padded to 4.
        const inputs = .{.{ .e = 2, .m = 7, .k = 6 }};
        fn build(comptime T: type, x: []const T) T {
            return x[0].pad(0, .{ .m = Tensor.Pad{ .high = 1 }, .k = Tensor.Pad{ .high = 2 } })
                .splitAxis(.m, .{ .mo = -1, .m4 = 2, .mi = 4 })
                .splitAxis(.k, .{ .ko = -1, .ki = 4 })
                .transpose(.{ .e, .mo, .ko, .mi, .m4, .ki })
                .merge(.{ .flat = .{ .mo, .ko, .mi, .m4, .ki } });
        }
    };
    const SplitThenInterleave = struct {
        // qwen3_5's in_proj_qkv sliced into q, k and v, interleaved with z.
        const inputs = .{ .{ .row = 6, .col = 3 }, .{ .row = 2, .col = 3 } };
        fn build(comptime T: type, x: []const T) T {
            const qkv = x[0];
            const members = [4]T{
                qkv.slice(.row, .{ .start = 0, .end = 2 }),
                qkv.slice(.row, .{ .start = 2, .end = 4 }),
                qkv.slice(.row, .{ .start = 4, .end = 6 }),
                x[1],
            };
            return interleave(T, &members, 2);
        }
    };
    const ChunkThenStack = struct {
        const inputs = .{.{ .row = 6, .col = 4 }};
        fn build(comptime T: type, x: []const T) T {
            const chunks = x[0].chunkExact(.row, 3);
            return T.stack(&.{ chunks[2], chunks[0] }, .last, .pair).insertAxes(0, .{.one}).squeeze(0);
        }
    };
};

fn inputShape(comptime dims_: anytype) Shape {
    return Shape.init(dims_, .u16);
}

fn expectCase(comptime Case: type, name: []const u8) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var checkpoint: TestCheckpoint = undefined;
    checkpoint.init(allocator);
    defer checkpoint.deinit();
    const n = Case.inputs.len;
    var names: [n][]const u8 = undefined;
    var name_buffers: [n][16]u8 = undefined;
    var shapes: [n]Shape = undefined;
    inline for (Case.inputs, 0..) |dims_, i| {
        names[i] = try std.fmt.bufPrint(&name_buffers[i], "input{d}", .{i});
        shapes[i] = inputShape(dims_);
        try checkpoint.add(names[i], shapes[i]);
    }

    var sources: [n]Source = undefined;
    for (&sources, names, shapes) |*s, input_name, shape_| s.* = checkpoint.store.view().source(input_name, null).withTags(shape_);
    const bound = Case.build(Source, &sources).bind(.replicated, .{}, .{});
    const assembled = try checkpoint.assemble(bound);
    defer allocator.free(assembled);

    const platform = testing.env();
    var tensors: [n]Tensor = undefined;
    for (&tensors, shapes) |*t, shape_| t.* = .fromShape(shape_);
    const Program = struct {
        fn forward(x: []const Tensor) Tensor {
            return Case.build(Tensor, x);
        }
    };
    var exe = try platform.compileFn(allocator, io, Program.forward, .{&tensors}, .{});
    defer exe.deinit();
    var buffers: [n]Buffer = undefined;
    var created: usize = 0;
    defer for (buffers[0..created]) |*buffer| buffer.deinit();
    for (&buffers, shapes, names) |*buffer, shape_, input_name| {
        buffer.* = try .fromBytes(io, platform, shape_, checkpoint.bytes.get(input_name).?);
        created += 1;
    }
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{buffers[0..]});
    exe.call(io, args, &results, .{ .wait = true });
    var output = results.get(Buffer);
    defer output.deinit();
    const expected = try output.toSliceAlloc(allocator, io);
    defer expected.free(allocator);

    std.testing.expect(bound.shape().eqlWithTags(output.shape())) catch |err| {
        std.log.err("{s}: Source gives {f}, Tensor {f}", .{ name, bound.shape(), output.shape() });
        return err;
    };
    std.testing.expectEqualSlices(u8, expected.constData(), assembled) catch |err| {
        std.log.err("{s}: the loader's bytes differ from the program's", .{name});
        return err;
    };
}

test "every Source op places the bytes the same Tensor ops compute" {
    inline for (comptime std.meta.declarations(cases)) |decl| {
        const Case = @field(cases, decl.name);
        if (@TypeOf(Case) != type) continue;
        try expectCase(Case, decl.name);
    }
}

test "a whole checkpoint tensor binds plainly, any other source as byte parts" {
    var checkpoint: TestCheckpoint = undefined;
    checkpoint.init(std.testing.allocator);
    defer checkpoint.deinit();
    try checkpoint.add("q", .init(.{ 4, 3 }, .u16));
    try checkpoint.add("k", .init(.{ 2, 3 }, .u16));
    try checkpoint.add("v", .init(.{ 2, 3 }, .u16));
    const view = checkpoint.store.view();
    const q = view.source("q", .{ .row, .col });

    // Relabels keep the whole tensor; only `bind` records anything.
    try std.testing.expectEqual(@as(usize, 0), checkpoint.store.id_to_sources.count());
    const plain = q.splitAxis(.row, .{ .h = 2, .hd = 2 }).merge(.{ .row = .{ .h, .hd } }).bind(.replicated, .{}, .{ .memory = .host_pinned });
    const plain_binding = checkpoint.store.getSourcesById(plain.id).?;
    try std.testing.expect(plain_binding.layout == null);
    try std.testing.expectEqual(Memory.Kind.host_pinned, plain_binding.memory);
    try std.testing.expectEqualStrings("q", plain_binding.tensors[0].name);

    // q, k and v in two groups: each member is one part of two runs, its
    // rows of a group side by side with the others'.
    const members = [_]Source{ q, view.source("k", .{ .row, .col }), view.source("v", .{ .row, .col }) };
    const fused = cases.interleave(Source, &members, 2).bind(.replicated, .{}, .{});
    const binding = checkpoint.store.getSourcesById(fused.id).?;
    try std.testing.expectEqual(@as(usize, 3), binding.tensors.len);
    try std.testing.expectEqualDeep(&[_]TensorStore.Part{
        .{ .source = 0, .src_offset = 0, .dst_offset = 0, .width = 12, .dims = &.{.{ .count = 2, .src_stride = 12, .dst_stride = 24 }} },
        .{ .source = 1, .src_offset = 0, .dst_offset = 12, .width = 6, .dims = &.{.{ .count = 2, .src_stride = 6, .dst_stride = 24 }} },
        .{ .source = 2, .src_offset = 0, .dst_offset = 18, .width = 6, .dims = &.{.{ .count = 2, .src_stride = 6, .dst_stride = 24 }} },
    }, binding.layout.?.parts);

    // A transpose is one part of single elements.
    const transposed = q.transpose(.{}).bind(.replicated, .{}, .{});
    try std.testing.expectEqualDeep(&[_]TensorStore.Part{
        .{ .source = 0, .src_offset = 0, .dst_offset = 0, .width = 2, .dims = &.{
            .{ .count = 3, .src_stride = 2, .dst_stride = 8 },
            .{ .count = 4, .src_stride = 6, .dst_stride = 2 },
        } },
    }, checkpoint.store.getSourcesById(transposed.id).?.layout.?.parts);
}

test "lattices stay one piece per source at checkpoint sizes" {
    var checkpoint: TestCheckpoint = undefined;
    checkpoint.init(std.testing.allocator);
    defer checkpoint.deinit();
    // Registered without bytes: nothing is evaluated.
    try checkpoint.registry.registerTensor(.{ .file_uri = "memory", .name = "gate", .shape = .init(.{ 14336, 4096 }, .bf16), .offset = 0 });
    try checkpoint.registry.registerTensor(.{ .file_uri = "memory", .name = "up", .shape = .init(.{ 14336, 4096 }, .bf16), .offset = 0 });
    const view = checkpoint.store.view();
    const members = [_]Source{ view.source("gate", .{ .row, .col }), view.source("up", .{ .row, .col }) };

    // One row per group: two pieces, each 14336 runs of one row.
    const finest = cases.interleave(Source, &members, 14336);
    try std.testing.expectEqual(@as(usize, 2), finest.pieces.len);
    const finest_binding = checkpoint.store.getSourcesById(finest.bind(.replicated, .{}, .{}).id).?;
    for (finest_binding.layout.?.parts) |part| {
        try std.testing.expectEqual(@as(u64, 14336), part.chunks());
        try std.testing.expectEqual(@as(u64, 8192), part.width);
    }
    // One group per device of four: four runs of 28 MiB each.
    const per_shard = cases.interleave(Source, &members, 4);
    for (checkpoint.store.getSourcesById(per_shard.bind(.replicated, .{}, .{}).id).?.layout.?.parts) |part| {
        try std.testing.expectEqual(@as(u64, 4), part.chunks());
        try std.testing.expectEqual(@as(u64, 3584 * 8192), part.width);
    }
    // Slicing through the interleave cuts at most a few pieces.
    const sliced = finest.slice(.row, .{ .start = 1001, .end = 20001 });
    try std.testing.expect(sliced.pieces.len <= 6);
}
