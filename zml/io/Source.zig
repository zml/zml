//! A tensor the loader assembles from checkpoint bytes by moving them, never
//! by computing: a checkpoint tensor, its axes relabeled or reshaped. The
//! ops are `zml.Tensor`'s shape ops, with the same names, tags and argument
//! conventions.
//!
//! A source comes from its store (`View.source`) and its pieces live in
//! the store's arena. Nothing is freed before `TensorStore.deinit`, and the
//! store must not move while sources point to it. The ops touch neither the
//! registry nor the bindings. Placement comes last: `withPartitioning` and
//! `as` return a `Placed`, which has no ops and drops its placement with
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

fn oom() noreturn {
    @panic("OOM");
}

// Tests: the loader's byte placement of a binding, replayed on the host.

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

test "a source binds as its pieces, which the loader lowers to byte parts" {
    var checkpoint: TestCheckpoint = undefined;
    try checkpoint.init(std.testing.allocator, &.{
        .{ .name = "q", .shape = .init(.{ 4, 3 }, .u16) },
    });
    defer checkpoint.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const view = checkpoint.store.view();
    const q = view.source("q", .{ .row, .col });
    const entries = [_]*const safetensors.Tensor{view.get("q").?};

    // Relabels keep the whole tensor, one run of it; only `bind` records
    // anything, and the binding keeps the placed source as it is.
    const relabeled = q.splitAxis(.row, .{ .h = 2, .hd = 2 }).merge(.{ .row = .{ .h, .hd } }).withPartitioning(.replicated, .{});
    try std.testing.expectEqual(@as(usize, 0), checkpoint.store.id_to_sources.count());
    const bound = checkpoint.store.bindingOf(relabeled.bind().id).?.load.source;
    try std.testing.expectEqual(relabeled._pieces, bound._pieces);
    try std.testing.expectEqual(entries[0], bound.whole().?);
    try std.testing.expectEqualDeep(&[_]backend.LoadPart{.whole(entries[0])}, try bound.lower(arena.allocator()));
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
