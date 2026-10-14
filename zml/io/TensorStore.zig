//! Associates model tensors with checkpoint sources and prefixed views.
const std = @import("std");
const log = std.log.scoped(.@"zml/io/load");

const stdx = @import("stdx");

const Memory = @import("../platform.zig").Memory;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;

pub const ReaderError = std.Io.File.OpenError || std.Io.File.Reader.SeekError || error{TensorNotFound};

const TensorStore = @This();

pub const Binding = struct {
    tensors: []*safetensors.Tensor,
    transformed: bool,
    /// Memory space the loader must deliver this tensor into, on every device of its mesh.
    memory: Memory.Kind = .default,
    /// How `tensors` fill the tensor's logical bytes. Null is the identity:
    /// one source that is the whole tensor. Ignored for transformed bindings,
    /// whose sources are executable inputs.
    layout: ?*const Layout = null,
};

/// A composite destination: several source byte ranges placed in one
/// tensor's logical (row-major, packed) bytes, which the loader then places
/// on the devices like any tensor's. Bytes no part covers are `fill`.
/// The loader currently accepts contiguous parts (`Part.isContiguous`) with
/// a zero fill; it rejects strided parts and other fills with
/// `error.UnsupportedLayout`, so later layouts (interleaved or broadcast rows)
/// extend this without changing it.
pub const Layout = struct {
    /// One per source, in the binding's `tensors` order.
    parts: []const Part,
    /// The value of every byte no part covers.
    fill: u8 = 0,
};

/// A cudaMemcpy2D-style byte map of one source into the destination's
/// logical bytes: `rows` chunks of `width` bytes, chunk i read at
/// `src_offset + i * src_pitch` of the source and written at
/// `dst_offset + i * dst_pitch`. One contiguous range is `rows = 1` with
/// zero pitches; a zero `src_pitch` would repeat one chunk.
pub const Part = struct {
    src_offset: u64 = 0,
    src_pitch: u64 = 0,
    dst_offset: u64,
    dst_pitch: u64 = 0,
    width: u64,
    rows: u64 = 1,

    /// One source range written as one destination range.
    pub fn isContiguous(self: Part) bool {
        return self.rows == 1 and self.src_pitch == 0 and self.dst_pitch == 0;
    }
};

/// How `View.createConcatenated` sizes the destination's dimension 0, which
/// must hold every source's rows.
pub const ConcatenateOptions = struct {
    /// The destination's dimension 0; null is the sources' total rounded up
    /// to `pad_rows_to_multiple`. Either one may be set, not both.
    rows: ?i64 = null,
    pad_rows_to_multiple: i64 = 1,
    /// See `Binding.memory`.
    memory: Memory.Kind = .default,
};

registry: *safetensors.TensorRegistry,
meshes: []const *const Sharding.Mesh,
id_to_sources: std.AutoHashMapUnmanaged(Tensor.Id, Binding),
allocator: std.mem.Allocator,
arena: std.heap.ArenaAllocator,

pub fn fromRegistry(allocator: std.mem.Allocator, registry: *safetensors.TensorRegistry, meshes: []const *const Sharding.Mesh) TensorStore {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    return .{
        .registry = registry,
        .meshes = arena.allocator().dupe(*const Sharding.Mesh, meshes) catch @panic("OOM"),
        .id_to_sources = .empty,
        .allocator = allocator,
        .arena = arena,
    };
}

pub fn deinit(self: *TensorStore) void {
    self.id_to_sources.deinit(self.allocator);
    self.arena.deinit();
}

pub fn getReader(self: *const TensorStore, key: []const u8, io: std.Io, buffer: []u8) ReaderError!safetensors.TensorReader {
    const tensor = self.getPtrFromKey(key) orelse {
        log.debug("checkpoint has no tensor named {s}", .{key});
        return error.TensorNotFound;
    };
    return tensor.reader(io, buffer, .{});
}

pub fn getSourcesById(self: *const TensorStore, id: Tensor.Id) ?Binding {
    return self.id_to_sources.get(id);
}

pub fn getShape(self: *const TensorStore, key: []const u8) ?Shape {
    const entry_ptr = self.getPtrFromKey(key) orelse return null;
    return entry_ptr.shape;
}

/// Changes the memory space the loader delivers `tensor` into (see
/// `Binding.memory`), for a decision made after the tensor was created.
/// `tensor` must be bound in this store.
pub fn setMemory(self: *TensorStore, tensor: Tensor, kind: Memory.Kind) void {
    const binding = self.id_to_sources.getPtr(tensor.id) orelse
        stdx.debug.panic("Tensor {} {f} has no binding in this store", .{ tensor.id, tensor.shape() });
    binding.memory = kind;
}

pub fn view(self: *TensorStore) View {
    return .{ .store = self };
}

pub fn mesh(store: *const TensorStore, name: @EnumLiteral()) *const Sharding.Mesh {
    const name_slice = @tagName(name);
    for (store.meshes) |m| {
        if (std.mem.eql(u8, name_slice, m.name)) {
            return m;
        }
    }
    std.debug.panic(
        \\Found no meshes named {s} in TensorStore.
        \\Try passing more meshes to `zml.TensorStore.fromRegistry`.
        \\Known meshes: {f}
    , .{ name_slice, stdx.fmt.slice(store.meshes) });
}

pub const View = struct {
    store: *TensorStore,

    prefix_buffer: [256]u8 = undefined,
    prefix_length: usize = 0,

    pub fn root(self: *const View) View {
        return .{
            .store = self.store,
        };
    }

    pub fn withPrefix(self: *const View, prefix_: []const u8) View {
        var buffer: [256]u8 = undefined;
        const new_prefix = makeKey(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ });

        return .{
            .store = self.store,
            .prefix_buffer = buffer,
            .prefix_length = new_prefix.len,
        };
    }

    pub fn withLayer(self: *const View, index: usize) View {
        var buffer: [256]u8 = undefined;
        const new_prefix = makeKey(&buffer, "{s}{d}.", .{ self.prefix() orelse "", index });

        return .{
            .store = self.store,
            .prefix_buffer = buffer,
            .prefix_length = new_prefix.len,
        };
    }

    pub fn prefix(self: *const View) ?[]const u8 {
        return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
    }

    pub fn hasKey(self: *const View, subkey: []const u8) bool {
        var buffer: [256]u8 = undefined;
        const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        return for (self.store.registry.tensors.keys()) |k| {
            if (std.mem.startsWith(u8, k, key)) break true;
        } else false;
    }

    /// Creates a zml.Tensor from a specific entry in the store.
    pub fn createTensor(self: View, subkey: []const u8, tags: anytype, mesh_: @EnumLiteral(), partitioning: anytype) Tensor {
        return self.maybeCreateTensor(subkey, tags, mesh_, partitioning) orelse
            stdx.debug.panic("Checkpoint has no tensor named {s}{s}", .{ self.prefix() orelse "", subkey });
    }

    pub fn maybeCreateTensor(self: View, subkey: []const u8, comptime tags: anytype, mesh_: @EnumLiteral(), partitioning: anytype) ?Tensor {
        const has_tags: bool = comptime @TypeOf(tags) != @TypeOf(null);
        const parsed_tags: Shape.TagsArray = if (comptime has_tags) Shape.parseTags(tags) else undefined;
        if (mesh_ == .replicated) {
            return self.maybeCreateTensorInternal(subkey, if (has_tags) parsed_tags.constSlice() else null, .replicated);
        }
        const resolved_mesh = self.store.mesh(mesh_);
        // Unspecified tensor axes are treated as replicated
        const partition: Sharding.Partitioning = if (has_tags)
            .parse(parsed_tags.constSlice(), resolved_mesh, .replicated, partitioning)
        else
            partitioning;
        return self.maybeCreateTensorInternal(subkey, if (has_tags) parsed_tags.constSlice() else null, .{ .mesh = resolved_mesh, .partition = partition });
    }

    /// Like `createTensor`, but the loader delivers the tensor into each device's host pinned memory space.
    pub fn createHostPinnedTensor(self: View, subkey: []const u8, tags: anytype, mesh_: @EnumLiteral(), partitioning: anytype) Tensor {
        return self.maybeCreateHostPinnedTensor(subkey, tags, mesh_, partitioning) orelse
            stdx.debug.panic("Checkpoint has no tensor named {s}{s}", .{ self.prefix() orelse "", subkey });
    }

    pub fn maybeCreateHostPinnedTensor(self: View, subkey: []const u8, comptime tags: anytype, mesh_: @EnumLiteral(), comptime partitioning: anytype) ?Tensor {
        const tensor = self.maybeCreateTensor(subkey, tags, mesh_, partitioning);
        if (tensor) |t| self.store.setMemory(t, .host_pinned);
        return tensor;
    }

    /// See `TensorStore.setMemory`.
    pub fn setMemory(self: View, tensor: Tensor, kind: Memory.Kind) void {
        self.store.setMemory(tensor, kind);
    }

    /// Creates one tensor from `subkeys` concatenated along dimension 0, in
    /// order, which the loader assembles in the destination itself: no
    /// program, no per-source buffer. The sources must share the dtype and
    /// dimensions 1 and up; the tensor takes them, with dimension 0 sized by
    /// `opts`. Rows past the sources' are zero, so a single source with
    /// `.pad_rows_to_multiple` is a padded tensor. The loader checks the
    /// sources against the tensor (`error.InvalidLayout`).
    pub fn createConcatenated(self: View, subkeys: []const []const u8, comptime tags: anytype, mesh_: @EnumLiteral(), partitioning: anytype, opts: ConcatenateOptions) Tensor {
        return self.maybeCreateConcatenated(subkeys, tags, mesh_, partitioning, opts) orelse
            stdx.debug.panic("Checkpoint lacks one of the tensors {f} under {s}", .{ stdx.fmt.strings(subkeys), self.prefix() orelse "" });
    }

    /// Like `createConcatenated`, but null when any source is missing.
    pub fn maybeCreateConcatenated(self: View, subkeys: []const []const u8, comptime tags: anytype, mesh_: @EnumLiteral(), partitioning: anytype, opts: ConcatenateOptions) ?Tensor {
        const has_tags: bool = comptime @TypeOf(tags) != @TypeOf(null);
        const parsed_tags: Shape.TagsArray = if (comptime has_tags) Shape.parseTags(tags) else undefined;
        const user_tags: ?[]const Shape.Tag = if (has_tags) parsed_tags.constSlice() else null;
        if (mesh_ == .replicated) {
            return self.maybeCreateConcatenatedInternal(subkeys, user_tags, .replicated, opts);
        }
        const resolved_mesh = self.store.mesh(mesh_);
        // Unspecified tensor axes are treated as replicated
        const partition: Sharding.Partitioning = if (has_tags)
            .parse(parsed_tags.constSlice(), resolved_mesh, .replicated, partitioning)
        else
            partitioning;
        return self.maybeCreateConcatenatedInternal(subkeys, user_tags, .{ .mesh = resolved_mesh, .partition = partition }, opts);
    }

    fn maybeCreateConcatenatedInternal(self: View, subkeys: []const []const u8, tags: ?[]const Shape.Tag, sharding: Sharding, opts: ConcatenateOptions) ?Tensor {
        stdx.debug.assert(subkeys.len != 0, "createConcatenated expects at least one source", .{});
        stdx.debug.assert(opts.pad_rows_to_multiple >= 1, "createConcatenated expects pad_rows_to_multiple >= 1, got {}", .{opts.pad_rows_to_multiple});
        stdx.debug.assert(opts.rows == null or opts.pad_rows_to_multiple == 1, "createConcatenated takes rows or pad_rows_to_multiple, not both", .{});
        const arena = self.store.arena.allocator();
        const sources = arena.alloc(*safetensors.Tensor, subkeys.len) catch @panic("OOM");
        const parts = arena.alloc(Part, subkeys.len) catch @panic("OOM");
        var rows: i64 = 0;
        var offset: u64 = 0;
        var buffer: [256]u8 = undefined;
        for (subkeys, sources, parts) |subkey, *source, *part| {
            const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
            source.* = self.store.dupeSource(key) orelse return null;
            stdx.debug.assert(source.*.shape.rank() != 0, "createConcatenated expects sources of rank 1 or more, {s} is {f}", .{ key, source.*.shape });
            // Each source's bytes follow the previous one's: concatenating
            // along the outermost axis is concatenating the byte streams.
            part.* = .{ .dst_offset = offset, .width = source.*.byteSize() };
            offset += part.width;
            rows += source.*.shape.dim(0);
        }
        const multiple = opts.pad_rows_to_multiple;
        const shape = sources[0].shape.setDim(0, opts.rows orelse @divFloor(rows + multiple - 1, multiple) * multiple);
        const layout = arena.create(Layout) catch @panic("OOM");
        layout.* = .{ .parts = parts };

        const tensor: Tensor = .fromShape(placeShape(shape, tags, sharding, subkeys[0]));
        self.store.putSourcesNoClobber(tensor.id, .{
            .tensors = sources,
            .transformed = false,
            .memory = opts.memory,
            .layout = layout,
        }) catch |e| std.debug.panic("Not handling {} errors", .{e});
        return tensor;
    }

    /// Creates a tensor of `shape` (whose sharding places it) that the
    /// loader fills from `subkeys` through `layout`, one part per source, in
    /// the tensor's logical bytes. The general form of `createConcatenated`.
    pub fn createComposite(self: View, subkeys: []const []const u8, shape: Shape, layout: Layout, memory: Memory.Kind) Tensor {
        return self.maybeCreateComposite(subkeys, shape, layout, memory) orelse
            stdx.debug.panic("Checkpoint lacks one of the tensors {f} under {s}", .{ stdx.fmt.strings(subkeys), self.prefix() orelse "" });
    }

    /// Like `createComposite`, but null when any source is missing.
    pub fn maybeCreateComposite(self: View, subkeys: []const []const u8, shape: Shape, layout: Layout, memory: Memory.Kind) ?Tensor {
        stdx.debug.assert(layout.parts.len == subkeys.len, "createComposite expects one part per source, got {} parts for {} sources", .{ layout.parts.len, subkeys.len });
        const arena = self.store.arena.allocator();
        const sources = arena.alloc(*safetensors.Tensor, subkeys.len) catch @panic("OOM");
        var buffer: [256]u8 = undefined;
        for (subkeys, sources) |subkey, *source| {
            const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
            source.* = self.store.dupeSource(key) orelse return null;
        }
        const owned = arena.create(Layout) catch @panic("OOM");
        owned.* = .{
            .parts = arena.dupe(Part, layout.parts) catch @panic("OOM"),
            .fill = layout.fill,
        };

        const tensor: Tensor = .fromShape(shape);
        self.store.putSourcesNoClobber(tensor.id, .{
            .tensors = sources,
            .transformed = false,
            .memory = memory,
            .layout = owned,
        }) catch |e| std.debug.panic("Not handling {} errors", .{e});
        return tensor;
    }

    pub fn createReplicatedTensor(self: View, subkey: []const u8, tags: anytype) Tensor {
        const has_tags: bool = comptime @TypeOf(tags) != @TypeOf(null);
        const parsed_tags: Shape.TagsArray = if (comptime has_tags) Shape.parseTags(tags) else undefined;

        return self.maybeCreateTensorInternal(subkey, if (has_tags) parsed_tags.slice() else null, .replicated) orelse
            stdx.debug.panic("Checkpoint has no tensor named {s}{s}", .{ self.prefix() orelse "", subkey });
    }

    fn maybeCreateTensorInternal(self: View, subkey: []const u8, tags: ?[]const Shape.Tag, sharding: Sharding) ?Tensor {
        var buffer: [256]u8 = undefined;
        const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        const source = self.store.dupeSource(key) orelse return null;

        const sources = self.store.arena.allocator().alloc(*safetensors.Tensor, 1) catch |e| std.debug.panic("Not handling {} errors", .{e});
        errdefer self.store.arena.allocator().free(sources);
        sources[0] = source;

        const tensor: Tensor = .fromShape(placeShape(source.shape, tags, sharding, key));
        self.store.putSourcesNoClobber(tensor.id, .{ .tensors = sources, .transformed = false, .memory = .default }) catch |e| std.debug.panic("Not handling {} errors", .{e});

        return tensor;
    }

    /// `shape` with the caller's tags, then placed by `sharding`.
    fn placeShape(shape: Shape, tags: ?[]const Shape.Tag, sharding: Sharding, key: []const u8) Shape {
        var placed = shape;
        if (tags) |user_tags| {
            stdx.debug.assert(user_tags.len == placed.rank(), "tensor {s} from store has shape {f}, but `createTensor` got only {d} tags: {f}", .{ key, shape, user_tags.len, stdx.fmt.stringsZ(user_tags) });
            @memcpy(placed._tags.slice(), user_tags);
        }
        placed._sharding = sharding;
        if (sharding.mesh) |mesh_| placed = placed.withPartitioning(mesh_, sharding.partition);
        return placed;
    }

    pub fn maybeCreateBinding(self: View, sources: []const []const u8, shape: Shape) ?Tensor {
        const arena = self.store.arena.allocator();

        var tensor_list = std.ArrayList(*safetensors.Tensor).initCapacity(arena, sources.len) catch |e| std.debug.panic("Not handling {} errors", .{e});
        defer tensor_list.deinit(arena);

        var buffer: [256]u8 = undefined;
        for (sources) |subkey| {
            const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
            const tensor = self.store.dupeSource(key) orelse return null;
            tensor_list.appendAssumeCapacity(tensor);
        }

        const tensors = tensor_list.toOwnedSlice(arena) catch unreachable;
        errdefer arena.free(tensors);

        const tensor: Tensor = .fromShape(shape);
        self.store.putSourcesNoClobber(tensor.id, .{ .tensors = tensors, .transformed = true }) catch |e| std.debug.panic("Not handling {} errors", .{e});

        return tensor;
    }

    pub fn getShape(self: View, subkey: []const u8) ?Shape {
        var buffer: [256]u8 = undefined;
        const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        return self.store.getShape(key);
    }

    pub fn getReader(self: View, subkey: []const u8, io: std.Io, buffer: []u8) ReaderError!safetensors.TensorReader {
        var key_buffer: [256]u8 = undefined;
        const key = makeKey(&key_buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        return self.store.getReader(key, io, buffer);
    }

    pub fn count(self: View) usize {
        var count_: usize = 0;
        const prefix_ = self.prefix() orelse "";
        var it = self.store.registry.tensors.iterator();
        while (it.next()) |item| {
            const key = item.key_ptr.*;
            if (std.mem.startsWith(u8, key, prefix_)) {
                count_ += 1;
            }
        }
        return count_;
    }

    fn makeKey(buffer: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
        const key = std.fmt.bufPrint(buffer, fmt, args) catch
            std.debug.panic("Expected key to be less than {} characters", .{buffer.len});
        return key;
    }
};

fn putSourcesNoClobber(self: *TensorStore, id: Tensor.Id, sources: Binding) std.mem.Allocator.Error!void {
    const gop = try self.id_to_sources.getOrPut(self.allocator, id);
    if (gop.found_existing) {
        stdx.debug.panic("Id {} already has associated sources", .{id});
    }
    errdefer self.id_to_sources.removeByPtr(gop.key_ptr);

    gop.value_ptr.* = sources;
}

fn getPtrFromKey(self: *const TensorStore, key: []const u8) ?*safetensors.Tensor {
    const tensor_desc_ptr = self.registry.tensors.getPtr(key) orelse return null;
    return tensor_desc_ptr;
}

fn dupeSource(self: *TensorStore, key: []const u8) ?*safetensors.Tensor {
    const entry = self.getPtrFromKey(key) orelse return null;

    const copy = self.arena.allocator().create(safetensors.Tensor) catch @panic("OOM");
    copy.* = entry.*;

    return copy;
}

test "missing reader keys return TensorNotFound" {
    var registry = safetensors.TensorRegistry.init(std.testing.allocator);
    defer registry.deinit();
    var store = TensorStore.fromRegistry(std.testing.allocator, &registry, &.{});
    defer store.deinit();

    try std.testing.expectError(error.TensorNotFound, store.view().withPrefix("layer").getReader("weights", std.testing.io, &.{}));
}

test "setMemory changes the memory space of a created tensor's binding" {
    var registry = safetensors.TensorRegistry.init(std.testing.allocator);
    defer registry.deinit();
    try registry.registerTensor(.{ .file_uri = "weights", .name = "table", .shape = .init(.{ 4, 2 }, .u8), .offset = 0 });
    var store = TensorStore.fromRegistry(std.testing.allocator, &registry, &.{});
    defer store.deinit();

    const tensor = store.view().createReplicatedTensor("table", null);
    try std.testing.expectEqual(Memory.Kind.default, store.getSourcesById(tensor.id).?.memory);
    store.view().setMemory(tensor, .host_pinned);
    try std.testing.expectEqual(Memory.Kind.host_pinned, store.getSourcesById(tensor.id).?.memory);
    store.setMemory(tensor, .default);
    try std.testing.expectEqual(Memory.Kind.default, store.getSourcesById(tensor.id).?.memory);
}

test "createConcatenated lays the sources end to end and sizes dimension 0" {
    var registry = safetensors.TensorRegistry.init(std.testing.allocator);
    defer registry.deinit();
    try registry.registerTensor(.{ .file_uri = "first", .name = "layer.a", .shape = .init(.{ 3, 4 }, .u16), .offset = 0 });
    try registry.registerTensor(.{ .file_uri = "first", .name = "layer.b", .shape = .init(.{ 2, 4 }, .u16), .offset = 24 });
    try registry.registerTensor(.{ .file_uri = "second", .name = "layer.c", .shape = .init(.{ 4, 4 }, .u16), .offset = 8 });
    var store = TensorStore.fromRegistry(std.testing.allocator, &registry, &.{});
    defer store.deinit();
    const layer = store.view().withPrefix("layer");

    // Padded to a multiple of 8 rows: 9 rows of sources, 7 of zeros.
    const padded = layer.createConcatenated(&.{ "a", "b", "c" }, .{ .row, .col }, .replicated, .{}, .{
        .pad_rows_to_multiple = 8,
        .memory = .host_pinned,
    });
    try std.testing.expectEqualSlices(i64, &.{ 16, 4 }, padded.shape().dims());
    try std.testing.expectEqual(@as(?u3, 0), padded.shape().hasTag(.row));
    try std.testing.expectEqual(@as(?u3, 1), padded.shape().hasTag(.col));
    try std.testing.expectEqual(padded.shape().dtype(), registry.tensors.get("layer.a").?.shape.dtype());
    const binding = store.getSourcesById(padded.id).?;
    try std.testing.expect(!binding.transformed);
    try std.testing.expectEqual(Memory.Kind.host_pinned, binding.memory);
    try std.testing.expectEqual(@as(usize, 3), binding.tensors.len);
    try std.testing.expectEqualStrings("layer.b", binding.tensors[1].name);
    const layout = binding.layout.?;
    try std.testing.expectEqual(@as(u8, 0), layout.fill);
    try std.testing.expectEqualDeep(&[_]Part{
        .{ .dst_offset = 0, .width = 24 },
        .{ .dst_offset = 24, .width = 16 },
        .{ .dst_offset = 40, .width = 32 },
    }, layout.parts);

    // The sources' total by default, or an explicit dimension 0.
    const exact = layer.createConcatenated(&.{ "c", "a" }, null, .replicated, .{}, .{});
    try std.testing.expectEqualSlices(i64, &.{ 7, 4 }, exact.shape().dims());
    try std.testing.expectEqual(@as(u64, 32), store.getSourcesById(exact.id).?.layout.?.parts[1].dst_offset);
    const explicit = layer.createConcatenated(&.{"c"}, null, .replicated, .{}, .{ .rows = 6 });
    try std.testing.expectEqualSlices(i64, &.{ 6, 4 }, explicit.shape().dims());
    try std.testing.expectEqual(Memory.Kind.default, store.getSourcesById(explicit.id).?.memory);

    try std.testing.expect(layer.maybeCreateConcatenated(&.{ "a", "missing" }, null, .replicated, .{}, .{}) == null);
}

test "createComposite keeps its own copy of the layout" {
    var registry = safetensors.TensorRegistry.init(std.testing.allocator);
    defer registry.deinit();
    try registry.registerTensor(.{ .file_uri = "first", .name = "a", .shape = .init(.{ 2, 4 }, .u8), .offset = 0 });
    var store = TensorStore.fromRegistry(std.testing.allocator, &registry, &.{});
    defer store.deinit();

    var parts = [_]Part{.{ .dst_offset = 8, .width = 8 }};
    const tensor = store.view().createComposite(&.{"a"}, .init(.{ 4, 4 }, .u8), .{ .parts = &parts }, .default);
    parts[0].dst_offset = 0;
    const layout = store.getSourcesById(tensor.id).?.layout.?;
    try std.testing.expectEqual(@as(u64, 8), layout.parts[0].dst_offset);
    try std.testing.expect(layout.parts[0].isContiguous());
    try std.testing.expect(!(Part{ .dst_offset = 0, .width = 4, .rows = 2, .dst_pitch = 8 }).isContiguous());
    try std.testing.expect(store.view().maybeCreateComposite(&.{"missing"}, .init(.{4}, .u8), .{ .parts = &parts }, .default) == null);
}
