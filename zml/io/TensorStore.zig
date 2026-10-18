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
    /// one source that is the whole tensor. A transformed binding with a
    /// layout feeds the assembled tensor to its executable as the one input.
    layout: ?*const Layout = null,
};

/// A composite tensor: byte parts of checkpoint tensors placed in the
/// logical (row-major, packed) bytes of `shape`, every byte no part covers
/// zero. `Source.bind` makes them; the loader assembles them.
pub const Layout = struct {
    /// The logical shape the parts fill, unsharded.
    shape: Shape,
    parts: []const Part,
};

/// `width` bytes from `src_offset` of `Binding.tensors[source]` land at
/// `dst_offset`, repeated over `dims` (outermost first): each step of a
/// dimension adds its strides to both offsets. A zero `src_stride` repeats
/// the same bytes.
pub const Part = struct {
    source: u32 = 0,
    src_offset: u64 = 0,
    dst_offset: u64,
    width: u64,
    dims: []const Dim = &.{},

    pub const Dim = struct {
        count: u64,
        src_stride: u64,
        dst_stride: u64,
    };

    /// The number of `width`-byte runs the part copies.
    pub fn chunks(self: Part) u64 {
        var n: u64 = 1;
        for (self.dims) |d| n *= d.count;
        return n;
    }
};

pub const Source = @import("Source.zig");

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

    pub fn maybeCreateHostPinnedTensor(self: View, subkey: []const u8, comptime tags: anytype, mesh_: @EnumLiteral(), partitioning: anytype) ?Tensor {
        const tensor = self.maybeCreateTensor(subkey, tags, mesh_, partitioning);
        if (tensor) |t| self.store.setMemory(t, .host_pinned);
        return tensor;
    }

    /// See `TensorStore.setMemory`.
    pub fn setMemory(self: View, tensor: Tensor, kind: Memory.Kind) void {
        self.store.setMemory(tensor, kind);
    }

    /// The checkpoint tensor `subkey` as a `Source`, with `tags` (null keeps
    /// the checkpoint's). Panics when the checkpoint has no such tensor.
    pub fn source(self: View, subkey: []const u8, comptime tags: anytype) Source {
        return self.maybeSource(subkey, tags) orelse
            stdx.debug.panic("Checkpoint has no tensor named {s}{s}", .{ self.prefix() orelse "", subkey });
    }

    /// Like `source`, but null when the checkpoint has no such tensor.
    pub fn maybeSource(self: View, subkey: []const u8, comptime tags: anytype) ?Source {
        var buffer: [256]u8 = undefined;
        const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        const entry = self.store.dupeSource(key) orelse return null;
        const has_tags: bool = comptime @TypeOf(tags) != @TypeOf(null);
        const parsed_tags: Shape.TagsArray = if (comptime has_tags) Shape.parseTags(tags) else undefined;
        return .init(self.store, entry, if (has_tags) parsed_tags.constSlice() else null);
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
        const entry = self.store.dupeSource(key) orelse return null;

        const sources = self.store.arena.allocator().alloc(*safetensors.Tensor, 1) catch |e| std.debug.panic("Not handling {} errors", .{e});
        errdefer self.store.arena.allocator().free(sources);
        sources[0] = entry;

        const tensor: Tensor = .fromShape(placeShape(entry.shape, tags, sharding, key));
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

/// Records `sources` as `id`'s binding; an id is bound once.
pub fn putSourcesNoClobber(self: *TensorStore, id: Tensor.Id, sources: Binding) std.mem.Allocator.Error!void {
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
