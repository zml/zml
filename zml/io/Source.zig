//! A tensor the loader assembles from checkpoint bytes by moving them, never
//! by computing: a checkpoint tensor, or reshapes, concatenations and stacks
//! of checkpoint tensors. The ops are `zml.Tensor`'s shape ops, with the
//! same names, tags and argument conventions.
//!
//! A source comes from its store (`View.source`) and its pieces live in
//! the store's arena, where the ops that rewrite them allocate. Nothing is
//! freed before `TensorStore.deinit`, and the store must not move while
//! sources point to it. The ops touch neither the registry nor the
//! bindings. Sources of different stores do not mix: `concatenate` and
//! `stack` panic on them. Placement comes last: `withPartitioning` and `as`
//! return a `Placed`, which has no ops and drops its placement with
//! `unplaced`. A `Placed` is what binds: `Placed.bind` records the model
//! tensor the loader fills from it. Sources are built during model init.
const std = @import("std");

const stdx = @import("stdx");

const constants = @import("../constants.zig");
const DataType = @import("../dtype.zig").DataType;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;
const backend = @import("backend.zig");
const TensorStore = @import("TensorStore.zig");

const Source = @This();

store: *TensorStore,
/// Always replicated: placement comes last (`Placed`).
_shape: Shape,
/// Disjoint lattices of elements read from checkpoint tensors; elements no
/// piece covers are zero. zml/io's own: a binding keeps them (`Placed`)
/// and the loader lowers them to bytes (`Placed.lower`).
_pieces: []const Piece,

/// Checkpoint bits read through an affine lattice: for every `j` with
/// `j[k] < dims[k].count`, the element at row-major index
/// `origin + sum(j[k] * dims[k].step)` of the source's shape is the element
/// at bit `offset + sum(j[k] * dims[k].stride)` of `tensor`. Reshapes keep
/// the row-major order, so they never touch a piece.
const Piece = struct {
    tensor: *const safetensors.Tensor,
    offset: u64,
    origin: u64,
    /// Outermost first, none of count 1.
    dims: []const Dim,
};

const Dim = struct {
    count: u64,
    /// Row-major elements of the source's shape between two steps.
    step: u64,
    /// Bits of the checkpoint tensor between two steps; zero repeats them.
    stride: u64,
};

/// The checkpoint tensor `subkey` of `view`, with `tags` (null keeps the
/// checkpoint's); null when the checkpoint has no such tensor.
pub fn init(view: TensorStore.View, subkey: []const u8, tags: ?[]const Shape.Tag) ?Source {
    const tensor = view.get(subkey) orelse return null;
    const store = view.store;
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
    return .{ .store = store, ._shape = shape_, ._pieces = pieces };
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
    try writer.print("Source({f}, {d} pieces)", .{ self._shape, self._pieces.len });
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

// Ops that move elements.

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
        for (s._pieces) |piece| {
            if (a == 0) {
                // Along the outermost axis, a source's elements follow the previous one's.
                var flat: Flat = .fromPiece(piece);
                flat.origin += offset * outer_stride;
                ctx.emitFlat(flat);
                continue;
            }
            for (ctx.decompose(s._shape, piece)) |*axis_piece| {
                axis_piece.origin[a] += offset;
                ctx.emit(res_shape, axis_piece.*);
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
    const res_shape = first._shape.insertTag(axis_, @intCast(sources.len), tag);
    const r = first.rank();
    var ctx: Rewrite = .init(first.store);
    defer ctx.deinit();
    for (sources, 0..) |s, i| {
        for (s._pieces) |piece| {
            if (a == 0) {
                var flat: Flat = .fromPiece(piece);
                flat.origin += i * first.count();
                ctx.emitFlat(flat);
                continue;
            }
            for (ctx.decompose(s._shape, piece)) |axis_piece| {
                var moved = axis_piece;
                std.mem.copyBackwards(u64, moved.origin[a + 1 .. r + 1], axis_piece.origin[a..r]);
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

/// This source placed on `mesh_` by `partitioning`: a struct keyed by the
/// source's tags (axes it does not name are replicated), or a
/// `Sharding.Partitioning`. `.replicated` places it on every device without
/// a mesh, and takes only `.{}`. Placement comes last: a `Placed` only
/// binds (`Placed.bind`) or drops its placement (`Placed.unplaced`).
pub fn withPartitioning(self: Source, comptime mesh_: @EnumLiteral(), partitioning: anytype) Placed {
    var placed = self._shape;
    if (mesh_ == .replicated) {
        const T = @TypeOf(partitioning);
        if (comptime !(stdx.meta.isStruct(T) and std.meta.fieldNames(T).len == 0))
            @compileError("Source.withPartitioning(.replicated, ...) takes .{}: a replicated source has no partitioning, got " ++ @typeName(T));
    } else {
        placed = self._shape.withPartitioning(self.store.mesh(mesh_), partitioning);
    }
    return .{ .store = self.store, .shape = placed, ._pieces = self._pieces };
}

/// This source in `placed`, a shape built elsewhere: its dimensions, tags
/// and sharding. Like `reshape`, it moves no element: `placed` has the
/// source's element count and dtype. Placement comes last, as for
/// `withPartitioning`.
pub fn as(self: Source, placed: Shape) Placed {
    stdx.debug.assert(placed.dtype() == self.dtype() and placed.count() == self.count(), "Source {f} cannot be placed as {f}", .{ self._shape, placed });
    return .{ .store = self.store, .shape = placed, ._pieces = self._pieces };
}

/// A source with its placement (`withPartitioning`, `as`): the only thing
/// that binds, and what a binding keeps (`TensorStore.Binding`).
pub const Placed = struct {
    store: *TensorStore,
    /// The placed shape, whose row-major elements the pieces index.
    shape: Shape,
    /// See `Source._pieces`.
    _pieces: []const Piece,

    /// Makes the model tensor the loader fills from this source, in default
    /// memory. The binding keeps the source as it is; the loader lowers it
    /// to bytes (`lower`).
    pub fn bind(self: Placed) Tensor {
        return bindNew(self.store, self.shape, .{ .load = .{ .source = self } });
    }

    /// This source without its placement, for code that takes a placed
    /// source and reworks it by its placement (a layout per device), then
    /// places the result again.
    pub fn unplaced(self: Placed) Source {
        var shape_ = self.shape;
        shape_._sharding = .replicated;
        return .{ .store = self.store, ._shape = shape_, ._pieces = self._pieces };
    }

    /// The checkpoint tensor this source is, whole and alone, if it is: one
    /// piece that reads all of its bits in order.
    pub fn whole(self: Placed) ?*const safetensors.Tensor {
        if (self._pieces.len != 1) return null;
        const piece = self._pieces[0];
        if (piece.offset != 0 or piece.origin != 0 or piece.tensor.byteSize() != self.shape.byteSize()) return null;
        const contiguous = switch (piece.dims.len) {
            0 => self.shape.count() == 1,
            1 => piece.dims[0].count == self.shape.count() and piece.dims[0].step == 1 and piece.dims[0].stride == self.shape.dtype().bitSizeOf(),
            else => false,
        };
        return if (contiguous) piece.tensor else null;
    }

    /// The bytes the loader moves, one part per piece in piece order, in
    /// `arena`; none for a source that reads no bytes. The one rule from a
    /// source's bits to bytes: each part's dimensions outermost first, the
    /// innermost run that is contiguous on both sides as its width.
    /// `error.SubBytePosition` when a run starts or ends inside a byte.
    pub fn lower(self: Placed, arena: std.mem.Allocator) error{ OutOfMemory, SubBytePosition }![]backend.LoadPart {
        const parts = try arena.alloc(backend.LoadPart, self._pieces.len);
        for (parts, self._pieces) |*part, piece| part.* = try lowerPiece(arena, self.shape, piece);
        return parts;
    }
};

/// A new tensor of `shape_`, bound to `binding`: the one maker of
/// `store`'s bindings (`Placed.bind`), whose pieces are disjoint by
/// construction: the loader relies on that and does not check.
fn bindNew(store: *TensorStore, shape_: Shape, binding: TensorStore.Binding) Tensor {
    const tensor: Tensor = .fromShape(shape_);
    store.id_to_sources.putNoClobber(store.allocator, tensor.id, binding) catch oom();
    return tensor;
}

/// `piece` of a source placed in `shape_` as one part (`Placed.lower`), its
/// dimensions allocated with `allocator`, none for one run.
fn lowerPiece(allocator: std.mem.Allocator, shape_: Shape, piece: Piece) error{ OutOfMemory, SubBytePosition }!backend.LoadPart {
    const ebits: u64 = shape_.dtype().bitSizeOf();
    var lowered: stdx.BoundedArray(backend.Lattice.Dim, backend.max_part_dims) = .empty;
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
    if (!aligned) return error.SubBytePosition;
    return .{ .source = piece.tensor, .lattice = .{
        .src_offset = piece.offset / 8,
        .dst_offset = dst_bits / 8,
        .width = width / 8,
        .dims = if (lowered.len == 0) &.{} else try allocator.dupe(backend.Lattice.Dim, lowered.constSlice()),
    } };
}

// The machinery. A piece is kept flat; an op that addresses axes rewrites it
// in coordinates of the input shape (`AxisPiece`, every dimension moving one
// axis), moves the coordinates, then flattens it in the result's shape.

/// A piece while an op rewrites it, its dimensions inline.
const Flat = struct {
    tensor: *const safetensors.Tensor,
    offset: u64,
    origin: u64,
    dims: stdx.BoundedArray(Dim, backend.max_part_dims) = .empty,

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
const AxisPiece = struct {
    tensor: *const safetensors.Tensor,
    offset: u64,
    origin: [constants.MAX_RANK]u64 = @splat(0),
    dims: stdx.BoundedArray(AxisDim, backend.max_part_dims) = .empty,
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
    axis_pieces: std.ArrayList(AxisPiece) = .empty,
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
            ._pieces = self.store.arena.allocator().dupe(Piece, self.out.items) catch oom(),
        };
    }

    /// `piece` in coordinates of `shape_`, cut where a dimension would carry
    /// from one axis into the next; valid until the next call.
    fn decompose(self: *Rewrite, shape_: Shape, piece: Piece) []AxisPiece {
        const scratch = self.scratch.allocator();
        self.axis_pieces.clearRetainingCapacity();
        self.work.append(scratch, .fromPiece(piece)) catch oom();
        const strides = elementStrides(shape_);
        while (self.work.pop()) |flat| {
            switch (fitAxes(shape_, strides, flat)) {
                .piece => |axis_piece| self.axis_pieces.append(scratch, axis_piece) catch oom(),
                .overflow => |overflowing| self.cut(shape_, strides, flat, overflowing),
            }
        }
        return self.axis_pieces.items;
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

    /// Flattens `axis_piece` in `res_shape` into the result.
    fn emit(self: *Rewrite, res_shape: Shape, axis_piece: AxisPiece) void {
        const strides = elementStrides(res_shape);
        var flat: Flat = .{ .tensor = axis_piece.tensor, .offset = axis_piece.offset, .origin = 0 };
        for (0..res_shape.rank()) |b| flat.origin += axis_piece.origin[b] * strides[b];
        for (axis_piece.dims.constSlice()) |d| {
            if (d.count <= 1) continue;
            flat.dims.append(.{ .count = d.count, .step = d.step * strides[d.axis], .stride = d.stride }) catch tooManyDims();
        }
        self.emitFlat(flat);
    }

    /// Adds `flat` to the result, its dimensions outermost first and merged
    /// where one continues the other.
    fn emitFlat(self: *Rewrite, flat: Flat) void {
        var sorted: stdx.BoundedArray(Dim, backend.max_part_dims) = .empty;
        for (flat.dims.constSlice()) |d| {
            if (d.count > 1) sorted.appendAssumeCapacity(d);
        }
        std.mem.sort(Dim, sorted.slice(), {}, struct {
            fn moreOuter(_: void, lhs: Dim, rhs: Dim) bool {
                return lhs.step > rhs.step;
            }
        }.moreOuter);
        var merged: stdx.BoundedArray(Dim, backend.max_part_dims) = .empty;
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

const AxisFit = union(enum) {
    piece: AxisPiece,
    /// The axis some coordinate would run past.
    overflow: u3,
};

/// `flat` in coordinates of `shape_`, if no dimension carries into the next
/// axis: each dimension moves the outermost axis its step is a multiple of.
fn fitAxes(shape_: Shape, strides: [constants.MAX_RANK]u64, flat: Flat) AxisFit {
    const r = shape_.rank();
    var axis_piece: AxisPiece = .{ .tensor = flat.tensor, .offset = flat.offset };
    for (0..r) |b| axis_piece.origin[b] = (flat.origin / strides[b]) % @as(u64, @intCast(shape_.dim(b)));
    var extent = axis_piece.origin;
    for (flat.dims.constSlice()) |d| {
        if (d.count <= 1) continue;
        const b = axisOfStep(r, strides, d.step);
        const step = d.step / strides[b];
        axis_piece.dims.appendAssumeCapacity(.{ .count = d.count, .axis = b, .step = step, .stride = d.stride });
        extent[b] += (d.count - 1) * step;
    }
    for (0..r) |b| {
        if (extent[b] >= @as(u64, @intCast(shape_.dim(b)))) return .{ .overflow = @intCast(b) };
    }
    return .{ .piece = axis_piece };
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

fn oom() noreturn {
    @panic("OOM");
}

fn tooManyDims() noreturn {
    std.debug.panic("a Source piece needs more than {d} dimensions", .{backend.max_part_dims});
}

// Tests: every op is checked against the same function run as a program
// over `zml.Tensor`, the loader's byte placement replayed on the host.

const testing = @import("../testing.zig");
const Buffer = @import("../buffer.zig").Buffer;

/// A checkpoint of in-memory tensors (the registry points nowhere) and the
/// bytes each holds.
const TestCheckpoint = struct {
    store: TensorStore,
    bytes: std.StringHashMapUnmanaged([]u8) = .empty,
    allocator: std.mem.Allocator,

    const Entry = struct { name: []const u8, shape: Shape };

    /// Registers `entries` end to end, each holding distinct bytes.
    fn init(self: *TestCheckpoint, allocator: std.mem.Allocator, entries: []const Entry) !void {
        var registry: safetensors.TensorRegistry = .init(allocator);
        defer registry.deinit();
        var offset: u64 = 0;
        for (entries) |entry| {
            try registry.registerTensor(.{ .file_uri = "memory", .name = entry.name, .shape = entry.shape, .offset = offset });
            offset += entry.shape.byteSize();
        }
        self.* = .{ .store = .init(allocator, &registry, .{}), .allocator = allocator };
        errdefer self.deinit();
        for (entries, 0..) |entry, index| {
            const bytes = try allocator.alloc(u8, entry.shape.byteSize());
            errdefer allocator.free(bytes);
            for (bytes, 0..) |*byte, i| byte.* = @truncate(i *% 7 +% index *% 61 +% 1);
            try self.bytes.put(allocator, self.store.registry.tensors.getKey(entry.name).?, bytes);
        }
    }

    fn deinit(self: *TestCheckpoint) void {
        var it = self.bytes.valueIterator();
        while (it.next()) |bytes| self.allocator.free(bytes.*);
        self.bytes.deinit(self.allocator);
        self.store.deinit();
    }

    /// The bytes the loader would assemble for `tensor`, placed on the host:
    /// its source lowered as the loader lowers it (`Placed.lower`), the
    /// bytes no part covers zero. Fails when a part writes a byte another
    /// wrote, so that every byte is written exactly once, by a part or the
    /// zero fill: the loader relies on disjoint pieces and does not check
    /// them.
    fn assemble(self: *TestCheckpoint, tensor: Tensor) ![]u8 {
        var arena: std.heap.ArenaAllocator = .init(self.allocator);
        defer arena.deinit();
        const parts = try self.store.bindingOf(tensor.id).?.load.source.lower(arena.allocator());
        const out = try self.allocator.alloc(u8, tensor.shape().byteSize());
        errdefer self.allocator.free(out);
        @memset(out, 0);
        const written = try arena.allocator().alloc(bool, out.len);
        @memset(written, false);
        for (parts) |part| {
            const src = self.bytes.get(part.source.name).?;
            const width: usize = @intCast(part.lattice.width);
            var runs: backend.Lattice.Cursor = .init(part.lattice, 0, part.lattice.runs());
            while (runs.next()) |run| {
                const at: usize = @intCast(run.dst);
                for (written[at..][0..width]) |*byte| {
                    if (byte.*) return error.DoubleWrite;
                    byte.* = true;
                }
                @memcpy(out[at..][0..width], src[@intCast(run.src)..][0..width]);
            }
        }
        return out;
    }
};

fn interleave(comptime T: type, members: []const T, groups: i64) T {
    var parts: [4]T = undefined;
    for (members, parts[0..members.len]) |member, *part| part.* = member.splitAxis(.row, .{ .group = groups, .r = -1 });
    return T.concatenate(parts[0..members.len], .r).merge(.{ .row = .{ .group, .r } });
}

const qkv_inputs = .{ .{ .row = 4, .col = 3 }, .{ .row = 2, .col = 3 }, .{ .row = 2, .col = 3 } };
const expert_inputs = .{ .{ .row = 2, .col = 3 }, .{ .row = 2, .col = 3 }, .{ .row = 2, .col = 3 }, .{ .row = 2, .col = 3 } };

/// `build(T, x)` builds a tensor from `x` with shape ops only, the same code
/// for `Tensor` and `Source`; `inputs` holds the dims of each `u16` input.
fn expectCase(comptime inputs: anytype, comptime build: anytype) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const n = inputs.len;
    var names: [n][]const u8 = undefined;
    var name_buffers: [n][16]u8 = undefined;
    var shapes: [n]Shape = undefined;
    var entries: [n]TestCheckpoint.Entry = undefined;
    inline for (inputs, 0..) |dims_, i| {
        names[i] = try std.fmt.bufPrint(&name_buffers[i], "input{d}", .{i});
        shapes[i] = Shape.init(dims_, .u16);
        entries[i] = .{ .name = names[i], .shape = shapes[i] };
    }
    var checkpoint: TestCheckpoint = undefined;
    try checkpoint.init(allocator, &entries);
    defer checkpoint.deinit();

    var sources: [n]Source = undefined;
    for (&sources, names) |*s, input_name| s.* = checkpoint.store.view().source(input_name, null);
    const bound = build(Source, &sources).withPartitioning(.replicated, .{}).bind();
    const assembled = try checkpoint.assemble(bound);
    defer allocator.free(assembled);

    const platform = testing.env();
    var tensors: [n]Tensor = undefined;
    for (&tensors, shapes) |*t, shape_| t.* = .fromShape(shape_);
    const Program = struct {
        fn forward(x: []const Tensor) Tensor {
            return build(Tensor, x);
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
    try exe.call(io, args, &results, .{ .wait = true });
    var output = results.get(Buffer);
    defer output.deinit();
    const expected = try output.toSliceAlloc(allocator, io);
    defer expected.free(allocator);

    std.testing.expect(bound.shape().eqlWithTags(output.shape())) catch |err| {
        std.log.err("Source gives {f}, Tensor {f}", .{ bound.shape(), output.shape() });
        return err;
    };
    std.testing.expectEqualSlices(u8, expected.constData(), assembled) catch |err| {
        std.log.err("the loader's bytes differ from the program's", .{});
        return err;
    };
}

test "Source concatenates rows as Tensor does" {
    try expectCase(.{ .{ .row = 3, .col = 4 }, .{ .row = 2, .col = 4 }, .{ .row = 4, .col = 4 } }, struct {
        fn build(comptime T: type, x: []const T) T {
            return T.concatenate(x, .row);
        }
    }.build);
}

test "Source concatenates columns as Tensor does" {
    try expectCase(.{ .{ .row = 3, .col = 4 }, .{ .row = 3, .col = 2 } }, struct {
        fn build(comptime T: type, x: []const T) T {
            return T.concatenate(x, .col);
        }
    }.build);
}

test "Source interleaves q, k and v by heads as Tensor does" {
    try expectCase(qkv_inputs, struct {
        fn build(comptime T: type, x: []const T) T {
            return interleave(T, x, 2);
        }
    }.build);
}

test "Source interleaves rows as Tensor does" {
    try expectCase(.{ .{ .row = 4, .col = 3 }, .{ .row = 4, .col = 3 } }, struct {
        fn build(comptime T: type, x: []const T) T {
            return interleave(T, x, 4);
        }
    }.build);
}

test "Source stacks experts as Tensor does" {
    try expectCase(expert_inputs, struct {
        fn build(comptime T: type, x: []const T) T {
            const members = [2]T{ T.stack(x[0..2], 0, .expert), T.stack(x[2..4], 0, .expert) };
            return T.concatenate(&members, .row);
        }
    }.build);
}

test "Source interleaves stacked experts as Tensor does" {
    try expectCase(expert_inputs, struct {
        fn build(comptime T: type, x: []const T) T {
            const members = [2]T{ T.stack(x[0..2], 0, .expert), T.stack(x[2..4], 0, .expert) };
            return T.stack(&members, 2, .proj).merge(.{ .rows = .{ .row, .proj } });
        }
    }.build);
}

test "Source interleaves expert vectors as Tensor does" {
    // Per-expert vectors (biases): row r of projection p lands at r * 2 + p.
    try expectCase(.{ .{ .row = 3 }, .{ .row = 3 }, .{ .row = 3 }, .{ .row = 3 } }, struct {
        fn build(comptime T: type, x: []const T) T {
            const members = [2]T{ T.stack(x[0..2], 0, .expert), T.stack(x[2..4], 0, .expert) };
            return T.stack(&members, .last, .proj).merge(.{ .rows = .{ .row, .proj } });
        }
    }.build);
}

test "a source binds as its pieces, which the loader lowers to byte parts" {
    var checkpoint: TestCheckpoint = undefined;
    try checkpoint.init(std.testing.allocator, &.{
        .{ .name = "q", .shape = .init(.{ 4, 3 }, .u16) },
        .{ .name = "k", .shape = .init(.{ 2, 3 }, .u16) },
        .{ .name = "v", .shape = .init(.{ 2, 3 }, .u16) },
    });
    defer checkpoint.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const view = checkpoint.store.view();
    const q = view.source("q", .{ .row, .col });
    const entries = [_]*const safetensors.Tensor{ view.get("q").?, view.get("k").?, view.get("v").? };

    // Relabels keep the whole tensor, one run of it; only `bind` records
    // anything, and the binding keeps the placed source as it is.
    const relabeled = q.splitAxis(.row, .{ .h = 2, .hd = 2 }).merge(.{ .row = .{ .h, .hd } }).withPartitioning(.replicated, .{});
    try std.testing.expectEqual(@as(usize, 0), checkpoint.store.id_to_sources.count());
    const bound = checkpoint.store.bindingOf(relabeled.bind().id).?.load.source;
    try std.testing.expectEqual(relabeled._pieces, bound._pieces);
    try std.testing.expectEqual(entries[0], bound.whole().?);
    try std.testing.expectEqualDeep(&[_]backend.LoadPart{.whole(entries[0])}, try bound.lower(arena.allocator()));
    // One checkpoint tensor read twice is two pieces of it.
    const twice = Source.concatenate(&.{ q, view.source("q", .{ .row, .col }) }, .row).withPartitioning(.replicated, .{});
    try std.testing.expectEqual(null, twice.whole());
    try std.testing.expectEqualDeep(&[_]backend.LoadPart{
        .{ .source = entries[0], .lattice = .{ .dst_offset = 0, .width = 24 } },
        .{ .source = entries[0], .lattice = .{ .dst_offset = 24, .width = 24 } },
    }, try twice.lower(arena.allocator()));

    // q, k and v in two groups: each member is one part of two runs, its
    // rows of a group side by side with the others'.
    const members = [_]Source{ q, view.source("k", .{ .row, .col }), view.source("v", .{ .row, .col }) };
    const fused = interleave(Source, &members, 2).withPartitioning(.replicated, .{});
    try std.testing.expectEqualDeep(&[_]backend.LoadPart{
        .{ .source = entries[0], .lattice = .{ .src_offset = 0, .dst_offset = 0, .width = 12, .dims = &.{.{ .count = 2, .src_stride = 12, .dst_stride = 24 }} } },
        .{ .source = entries[1], .lattice = .{ .src_offset = 0, .dst_offset = 12, .width = 6, .dims = &.{.{ .count = 2, .src_stride = 6, .dst_stride = 24 }} } },
        .{ .source = entries[2], .lattice = .{ .src_offset = 0, .dst_offset = 18, .width = 6, .dims = &.{.{ .count = 2, .src_stride = 6, .dst_stride = 24 }} } },
    }, try fused.lower(arena.allocator()));
}

test "assembling fails on a byte two pieces write" {
    var checkpoint: TestCheckpoint = undefined;
    try checkpoint.init(std.testing.allocator, &.{.{ .name = "q", .shape = .init(.{ 4, 3 }, .u16) }});
    defer checkpoint.deinit();
    // Only a hand-built binding can repeat a piece.
    const q = checkpoint.store.view().source("q", .{ .row, .col });
    const pieces = try checkpoint.store.arena.allocator().dupe(Piece, &.{ q._pieces[0], q._pieces[0] });
    const tensor = bindNew(&checkpoint.store, q.shape(), .{ .load = .{ .source = .{ .store = &checkpoint.store, .shape = q.shape(), ._pieces = pieces } } });
    try std.testing.expectError(error.DoubleWrite, checkpoint.assemble(tensor));
}

test "a source is placed last, by as or withPartitioning, and binds in its placed shape" {
    const allocator = std.testing.allocator;
    var checkpoint: TestCheckpoint = undefined;
    try checkpoint.init(allocator, &.{.{ .name = "w", .shape = .init(.{ 4, 6 }, .u16) }});
    defer checkpoint.deinit();
    const mesh: Sharding.Mesh = .initTest(.tp, .{.model});
    const w = checkpoint.store.view().source("w", .{ .row, .col });

    // `as` takes a shape built elsewhere, here of another rank: its
    // dimensions, tags and sharding, for the same elements.
    const placed = Shape.init(.{ .flat = 24 }, .u16).withPartitioning(&mesh, .{ .flat = .model });
    const placed_source = w.as(placed);
    const flat = placed_source.bind();
    try std.testing.expect(flat.shape().eqlWithTagsAndPartitioning(placed));
    try std.testing.expect(checkpoint.store.bindingOf(flat.id).?.load.source.shape.eqlWithTagsAndPartitioning(placed));
    const replicated = w.withPartitioning(.replicated, .{}).bind();
    try std.testing.expect(replicated.shape().eqlWithTags(w.shape()));
    try std.testing.expectEqual(null, replicated.shape()._sharding.mesh);
    // `unplaced` keeps the placed dimensions and tags, without the sharding.
    const dropped = placed_source.unplaced();
    try std.testing.expect(dropped.shape().eqlWithTags(placed));
    try std.testing.expectEqual(null, dropped.shape()._sharding.mesh);

    const flat_bytes = try checkpoint.assemble(flat);
    defer allocator.free(flat_bytes);
    const replicated_bytes = try checkpoint.assemble(replicated);
    defer allocator.free(replicated_bytes);
    try std.testing.expectEqualSlices(u8, replicated_bytes, flat_bytes);
}

test "lattices stay one piece per source at checkpoint sizes" {
    // Registered without bytes: nothing is evaluated.
    var registry: safetensors.TensorRegistry = .init(std.testing.allocator);
    defer registry.deinit();
    try registry.registerTensor(.{ .file_uri = "memory", .name = "gate", .shape = .init(.{ 14336, 4096 }, .bf16), .offset = 0 });
    try registry.registerTensor(.{ .file_uri = "memory", .name = "up", .shape = .init(.{ 14336, 4096 }, .bf16), .offset = 0 });
    var store: TensorStore = .init(std.testing.allocator, &registry, .{});
    defer store.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const view = store.view();
    const members = [_]Source{ view.source("gate", .{ .row, .col }), view.source("up", .{ .row, .col }) };

    // One row per group: two pieces, each 14336 runs of one row.
    const finest = interleave(Source, &members, 14336);
    try std.testing.expectEqual(@as(usize, 2), finest._pieces.len);
    for (try finest.withPartitioning(.replicated, .{}).lower(arena.allocator())) |part| {
        try std.testing.expectEqual(@as(u64, 14336), part.lattice.runs());
        try std.testing.expectEqual(@as(u64, 8192), part.lattice.width);
    }
    // One group per device of four: four runs of 28 MiB each.
    const per_shard = interleave(Source, &members, 4);
    for (try per_shard.withPartitioning(.replicated, .{}).lower(arena.allocator())) |part| {
        try std.testing.expectEqual(@as(u64, 4), part.lattice.runs());
        try std.testing.expectEqual(@as(u64, 3584 * 8192), part.lattice.width);
    }
}
