//! Associates model tensors with checkpoint sources and prefixed views,
//! over a checkpoint registry the store owns.
//!
//! The store owns the registry, the bindings, and every source and program
//! value (its arena). Views and sources point to it: once `view` was
//! called it must not move, and it must outlive them and every load
//! started from it (`Loader.start` borrows it until `wait` or `cancel`).
//! It borrows the meshes of `Options.meshes`, which belong to the platform.
//! Binding cannot fail: a missing key panics with the key, and what the
//! loader cannot fill is its error (`Loader.start`'s `InvalidBinding`).
const std = @import("std");

const stdx = @import("stdx");

const Compiler = @import("../Compiler.zig");
const Exe = @import("../exe.zig").Exe;
const MemorySpace = @import("../platform.zig").Memory.Kind;
const Platform = @import("../platform.zig").Platform;
const Quantization = @import("../quantization.zig").Quantization;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;

const TensorStore = @This();

/// Where the loader delivers a loaded tensor, on every device of its mesh
/// (`View.setMemory`). The destination is allocated once per device
/// whatever its pieces; the host kinds take no device memory.
pub const Memory = enum {
    /// Device memory.
    default,
    /// Buffers in each device's pinned host memory space. An executable
    /// that takes them as device inputs copies them to the device first.
    host_pinned,
    /// Pinned host memory that executables read in place: each device's
    /// buffer is a device-kind view of it (`createViewOfDeviceBuffer`),
    /// and the memory goes with the last view. For a table too large for
    /// device memory, of which a step reads a few rows. XLA counts the
    /// views against device memory (`Compiler.Options.extra_device_memory_bytes`).
    host_mapped,

    /// Whether the bytes live in host memory, written by the CPU.
    pub fn isHost(self: Memory) bool {
        return self != .default;
    }

    /// The memory space of each device's delivered buffer.
    pub fn space(self: Memory) MemorySpace {
        return switch (self) {
            .default, .host_mapped => .default,
            .host_pinned => .host_pinned,
        };
    }
};

/// What the loader reads of a bound tensor (`bindingOf`): the source
/// it fills the tensor from, or the program that computes it.
pub const Binding = union(enum) {
    /// The loader fills the tensor from `source` (`Source.bind`), every
    /// byte no piece covers zero.
    load: Load,
    /// A program computes the tensor while loading (`Source.bindProgram`).
    /// Its output lands in default device memory.
    compute: Compute,

    pub const Load = struct {
        source: Source.Bound,
        /// Where the loader delivers this tensor (`View.setMemory`).
        memory: Memory = .default,
    };

    pub const Compute = struct {
        program: Program,
        /// The program's inputs in order, each landed in default device
        /// memory, placed by its shape.
        inputs: []const Source.Bound,

        /// Whether this computing `output` and `other` computing
        /// `other_output` compile to the same executable: programs of one
        /// type and equal values, over the same shapes placed alike.
        pub fn sameExecutable(self: Compute, output: Shape, other: Compute, other_output: Shape) bool {
            if (self.program.vtable != other.program.vtable or !self.program.vtable.eql(self.program.value, other.program.value)) return false;
            if (!samePlacedShape(output, other_output) or self.inputs.len != other.inputs.len) return false;
            for (self.inputs, other.inputs) |a, b| {
                if (!samePlacedShape(a.shape, b.shape)) return false;
            }
            return true;
        }
    };
};

fn samePlacedShape(a: Shape, b: Shape) bool {
    if (!a.eqlWithTags(b) or a._sharding.mesh != b._sharding.mesh) return false;
    for (0..a.rank()) |axis| {
        if (a.partition(axis) != b.partition(axis)) return false;
    }
    return true;
}

/// A program the loader compiles and runs to compute a tensor while loading
/// (`Binding.compute`): `forward(output, inputs)` of the program's value.
/// Model code makes one through `Source.bindProgram`, and reads it back
/// with `as`.
pub const Program = struct {
    /// The program's value, in the store's arena.
    value: *const anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The program's type name, which names its compiled module in
        /// traces, dumps and logs.
        name: []const u8,
        compile: *const fn (
            value: *const anyopaque,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const Platform,
            opts: Compiler.Options,
            output: Shape,
            inputs: []const Tensor,
        ) Compiler.Error!Exe,
        eql: *const fn (a: *const anyopaque, b: *const anyopaque) bool,
    };

    /// `program`, its value copied into `arena`. Its type declares
    /// `pub fn forward(self, output: Shape, inputs: []const Tensor) Tensor`
    /// and is plain data, without pointers, so equal values are found by
    /// value and compile once.
    pub fn init(arena: std.mem.Allocator, program: anytype) std.mem.Allocator.Error!Program {
        const T = @TypeOf(program);
        comptime {
            if (!isPlainData(T)) @compileError("TensorStore.Program: " ++ @typeName(T) ++ " must be plain data, without pointers");
        }
        const value: *const anyopaque = if (@sizeOf(T) == 0) &no_value else value: {
            const copy = try arena.create(T);
            copy.* = program;
            break :value copy;
        };
        return .{ .value = value, .vtable = &Erased(T).vtable };
    }

    /// The program's value, if it is a `T`.
    pub fn as(self: Program, comptime T: type) ?T {
        if (self.vtable != &Erased(T).vtable) return null;
        return Erased(T).typed(self.value);
    }

    /// What a program of a zero-sized type points to.
    const no_value: u8 = 0;

    /// The vtable of programs of type `T`, one per type.
    fn Erased(comptime T: type) type {
        return struct {
            const vtable: VTable = .{ .name = @typeName(T), .compile = compile, .eql = eql };

            fn typed(erased: *const anyopaque) T {
                if (@sizeOf(T) == 0) return undefined;
                return @as(*const T, @ptrCast(@alignCast(erased))).*;
            }

            fn compile(
                erased: *const anyopaque,
                allocator: std.mem.Allocator,
                io: std.Io,
                platform: *const Platform,
                opts: Compiler.Options,
                output: Shape,
                inputs: []const Tensor,
            ) Compiler.Error!Exe {
                return Compiler.compile(allocator, io, platform, T.forward, .{ typed(erased), output, inputs }, opts);
            }

            fn eql(a: *const anyopaque, b: *const anyopaque) bool {
                return std.meta.eql(typed(a), typed(b));
            }
        };
    }

    /// Values compared by value: no pointer anywhere inside.
    fn isPlainData(comptime T: type) bool {
        return switch (@typeInfo(T)) {
            .void, .bool, .int, .float, .@"enum" => true,
            .array => |info| isPlainData(info.child),
            .optional => |info| isPlainData(info.child),
            .@"struct" => |info| inline for (info.field_types) |Field| {
                if (!isPlainData(Field)) break false;
            } else true,
            .@"union" => |info| info.tag_type != null and inline for (info.field_types) |Field| {
                if (!isPlainData(Field)) break false;
            } else true,
            else => false,
        };
    }
};

pub const Source = @import("Source.zig");

registry: safetensors.TensorRegistry,
meshes: []const *const Sharding.Mesh,
quantization: Quantization.Target,
id_to_sources: std.AutoHashMapUnmanaged(Tensor.Id, Binding),
allocator: std.mem.Allocator,
arena: std.heap.ArenaAllocator,

pub const Options = struct {
    /// The meshes `Source.withPartitioning` and `View.mesh` name. The slice
    /// is copied; the meshes must outlive the store.
    meshes: []const *const Sharding.Mesh = &.{},
    /// How the platform's FP8 GEMMs want block-FP8 weights, for the code
    /// that binds them (`View.quantization`): usually `.of(platform)`.
    quantization: Quantization.Target = .{},
};

/// A store that owns `registry`, which must be complete: add every
/// checkpoint first (`TensorRegistry.addRepo`). The caller's copy is moved
/// in and must not be used or freed. The map stays locked until `deinit`,
/// so a later insert panics in safe builds, and an entry's address is its
/// checkpoint tensor's identity (`View.get`).
pub fn init(allocator: std.mem.Allocator, registry: safetensors.TensorRegistry, opts: Options) TensorStore {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    const meshes = arena.allocator().dupe(*const Sharding.Mesh, opts.meshes) catch @panic("OOM");
    var self: TensorStore = .{
        .registry = registry,
        .meshes = meshes,
        .quantization = opts.quantization,
        .id_to_sources = .empty,
        .allocator = allocator,
        .arena = arena,
    };
    self.registry.tensors.lockPointers();
    return self;
}

/// Frees the bindings, the sources and programs, then the registry. No load
/// started from the store may still run: `wait` or `cancel` it first.
pub fn deinit(self: *TensorStore) void {
    self.id_to_sources.deinit(self.allocator);
    self.arena.deinit();
    self.registry.tensors.unlockPointers();
    self.registry.deinit();
}

/// The binding of tensor `id`, as the loader reads it; null when this store
/// did not bind it.
pub fn bindingOf(self: *const TensorStore, id: Tensor.Id) ?Binding {
    return self.id_to_sources.get(id);
}

/// A new tensor of `shape`, bound to `binding`. Model code binds through
/// `Source` (`bind`, `bindProgram`), whose pieces are disjoint by
/// construction; the loader relies on it and does not check a hand-built
/// binding's.
pub fn bindNew(self: *TensorStore, shape: Shape, binding: Binding) Tensor {
    const tensor: Tensor = .fromShape(shape);
    self.id_to_sources.putNoClobber(self.allocator, tensor.id, binding) catch @panic("OOM");
    return tensor;
}

/// The root view, with no prefix. It points to this store, which must not
/// move afterwards.
pub fn view(self: *TensorStore) View {
    return .{ .store = self };
}

/// The mesh `name` among `Options.meshes`; model code asks its `View`.
pub fn mesh(store: *const TensorStore, name: @EnumLiteral()) *const Sharding.Mesh {
    const name_slice = @tagName(name);
    for (store.meshes) |m| {
        if (std.mem.eql(u8, name_slice, m.name)) {
            return m;
        }
    }
    std.debug.panic(
        \\Found no meshes named {s} in TensorStore.
        \\Try passing more meshes to `zml.io.TensorStore.init`.
        \\Known meshes: {f}
    , .{ name_slice, stdx.fmt.slice(store.meshes) });
}

/// What model code holds: a key prefix over the store's checkpoint, to look
/// tensors up, bind them and set their memory. A value that keeps its own
/// copy of the prefix (256 bytes at most, longer panics) and points to the
/// store.
pub const View = struct {
    store: *TensorStore,

    prefix_buffer: [256]u8 = undefined,
    prefix_length: usize = 0,

    /// The store's root view, with no prefix.
    pub fn root(self: *const View) View {
        return .{
            .store = self.store,
        };
    }

    /// This view under `prefix_` and a dot: `withPrefix("model")` then
    /// `get("embed")` looks up `model.embed`. It only scopes lookups;
    /// nothing is bound.
    pub fn withPrefix(self: *const View, prefix_: []const u8) View {
        var buffer: [256]u8 = undefined;
        const new_prefix = makeKey(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ });

        return .{
            .store = self.store,
            .prefix_buffer = buffer,
            .prefix_length = new_prefix.len,
        };
    }

    /// This view under `index` and a dot, as `withPrefix` of the number.
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

    /// The checkpoint tensor `subkey`, the registry's own entry, whose
    /// address is the tensor's identity. Null when the checkpoint has none.
    pub fn get(self: View, subkey: []const u8) ?*const safetensors.Tensor {
        var buffer: [256]u8 = undefined;
        const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        return self.store.registry.tensors.getPtr(key);
    }

    /// Whether some checkpoint key starts with `subkey` under this view:
    /// a scan of every key. `get` answers whether one key exists.
    pub fn hasPrefix(self: View, subkey: []const u8) bool {
        var buffer: [256]u8 = undefined;
        const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        return for (self.store.registry.tensors.keys()) |k| {
            if (std.mem.startsWith(u8, k, key)) break true;
        } else false;
    }

    /// The checkpoint tensor `subkey`, with `tags` (null keeps the
    /// checkpoint's), placed on `mesh_` by `partitioning` and bound:
    /// `source(subkey, tags).withPartitioning(mesh_, partitioning).bind()`.
    /// Panics when the checkpoint has no such tensor.
    pub fn createTensor(self: View, subkey: []const u8, comptime tags: anytype, comptime mesh_: @EnumLiteral(), partitioning: anytype) Tensor {
        return self.maybeCreateTensor(subkey, tags, mesh_, partitioning) orelse
            stdx.debug.panic("Checkpoint has no tensor named {s}{s}", .{ self.prefix() orelse "", subkey });
    }

    /// Like `createTensor`, but null when the checkpoint has no such tensor.
    pub fn maybeCreateTensor(self: View, subkey: []const u8, comptime tags: anytype, comptime mesh_: @EnumLiteral(), partitioning: anytype) ?Tensor {
        const source_ = self.maybeSource(subkey, tags) orelse return null;
        return self.keepEmpty(subkey, source_.withPartitioning(mesh_, partitioning)).bind();
    }

    /// `placed`, the whole checkpoint tensor `subkey`; when that tensor is
    /// empty, a source that keeps it (`Source.emptyOf`), so that
    /// `Loader.start` names the key it rejects.
    fn keepEmpty(self: View, subkey: []const u8, placed: Source) Source {
        return if (placed.count() != 0) placed else .emptyOf(self.store, self.get(subkey).?, placed.shape());
    }

    /// `createTensor` replicated on every device, without a mesh.
    pub fn createReplicatedTensor(self: View, subkey: []const u8, comptime tags: anytype) Tensor {
        return self.createTensor(subkey, tags, .replicated, .{});
    }

    /// Changes where the loader delivers `tensor` (`Memory`), for a
    /// decision made after the tensor was bound. `tensor` must be bound in
    /// this store, and not computed by a program: a program's output lands
    /// in default device memory.
    pub fn setMemory(self: View, tensor: Tensor, kind: Memory) void {
        const binding = self.store.id_to_sources.getPtr(tensor.id) orelse
            stdx.debug.panic("Tensor {} {f} has no binding in this store", .{ tensor.id, tensor.shape() });
        switch (binding.*) {
            .load => |*load| load.memory = kind,
            .compute => |compute| stdx.debug.panic("setMemory: program {s} computes tensor {} {f}, whose output lands in default device memory", .{ compute.program.vtable.name, tensor.id, tensor.shape() }),
        }
    }

    /// See `TensorStore.mesh`.
    pub fn mesh(self: View, name: @EnumLiteral()) *const Sharding.Mesh {
        return self.store.mesh(name);
    }

    /// See `Options.quantization`.
    pub fn quantization(self: View) Quantization.Target {
        return self.store.quantization;
    }

    /// The checkpoint tensor `subkey` as a `Source`, with `tags` (null keeps
    /// the checkpoint's). Panics when the checkpoint has no such tensor.
    pub fn source(self: View, subkey: []const u8, comptime tags: anytype) Source {
        return self.maybeSource(subkey, tags) orelse
            stdx.debug.panic("Checkpoint has no tensor named {s}{s}", .{ self.prefix() orelse "", subkey });
    }

    /// Like `source`, but null when the checkpoint has no such tensor.
    pub fn maybeSource(self: View, subkey: []const u8, comptime tags: anytype) ?Source {
        const entry = self.get(subkey) orelse return null;
        const has_tags: bool = comptime @TypeOf(tags) != @TypeOf(null);
        const parsed_tags: Shape.TagsArray = if (comptime has_tags) Shape.parseTags(tags) else undefined;
        return .init(self.store, entry, if (has_tags) parsed_tags.constSlice() else null);
    }

    /// A tensor of `shape` that `program` computes while loading
    /// (`Source.bindProgram`) from the checkpoint tensors `sources`, each
    /// whole and replicated, in this order. Null when one is missing.
    pub fn maybeCreateBinding(self: View, sources: []const []const u8, shape: Shape, program: anytype) ?Tensor {
        const inputs = self.store.allocator.alloc(Source, sources.len) catch |e| std.debug.panic("Not handling {} errors", .{e});
        defer self.store.allocator.free(inputs);
        for (sources, inputs) |subkey, *input| {
            const source_ = self.maybeSource(subkey, null) orelse return null;
            input.* = self.keepEmpty(subkey, source_.withPartitioning(.replicated, .{}));
        }
        return Source.bindProgram(inputs, shape, program);
    }

    /// The checkpoint tensors under this view's prefix: a scan of every key.
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

test "get finds exact keys, the registry's own entries, and hasPrefix any key under a prefix" {
    var store: TensorStore = store: {
        var registry: safetensors.TensorRegistry = .init(std.testing.allocator);
        errdefer registry.deinit();
        try registry.registerTensor(.{ .file_uri = "weights", .name = "layer.experts.0.down_proj.weight", .shape = .init(.{4}, .u8), .offset = 0 });
        break :store .init(std.testing.allocator, registry, .{});
    };
    defer store.deinit();

    const layer = store.view().withPrefix("layer");
    const entry = layer.get("experts.0.down_proj.weight").?;
    try std.testing.expectEqual(entry, store.registry.tensors.getPtr("layer.experts.0.down_proj.weight").?);
    try std.testing.expectEqual(null, layer.get("experts.0.down_proj"));
    try std.testing.expectEqual(null, layer.get("weights"));
    try std.testing.expect(layer.hasPrefix("experts"));
    try std.testing.expect(layer.withPrefix("experts").hasPrefix(""));
    try std.testing.expect(!layer.hasPrefix("gate"));

    // Every binding of the entry holds the entry itself.
    const tensor = layer.createReplicatedTensor("experts.0.down_proj.weight", null);
    try std.testing.expectEqual(entry, store.bindingOf(tensor.id).?.load.source.whole().?);
    try std.testing.expectEqual(entry, layer.source("experts.0.down_proj.weight", null)._pieces[0].tensor);
}

test "programs of one type and equal values over the same placed shapes share an executable" {
    var store: TensorStore = store: {
        var registry: safetensors.TensorRegistry = .init(std.testing.allocator);
        errdefer registry.deinit();
        try registry.registerTensor(.{ .file_uri = "weights", .name = "t", .shape = .init(.{4}, .u8), .offset = 0 });
        break :store .init(std.testing.allocator, registry, .{});
    };
    defer store.deinit();
    const Scale = struct {
        factor: u8,

        pub fn forward(_: @This(), _: Shape, inputs: []const Tensor) Tensor {
            return inputs[0];
        }
    };
    const Shift = struct {
        factor: u8,

        pub fn forward(_: @This(), _: Shape, inputs: []const Tensor) Tensor {
            return inputs[0];
        }
    };
    const shape: Shape = .init(.{4}, .u8);
    const root = store.view();
    const tensors = [_]Tensor{
        root.maybeCreateBinding(&.{"t"}, shape, Scale{ .factor = 1 }).?,
        root.maybeCreateBinding(&.{"t"}, shape, Scale{ .factor = 1 }).?,
        root.maybeCreateBinding(&.{"t"}, shape, Scale{ .factor = 2 }).?,
        root.maybeCreateBinding(&.{"t"}, shape, Shift{ .factor = 1 }).?,
    };
    var computes: [tensors.len]Binding.Compute = undefined;
    for (&computes, tensors) |*compute, tensor| compute.* = store.bindingOf(tensor.id).?.compute;
    try std.testing.expect(computes[0].sameExecutable(shape, computes[1], shape));
    try std.testing.expect(!computes[0].sameExecutable(shape, computes[2], shape));
    try std.testing.expect(!computes[0].sameExecutable(shape, computes[3], shape));
    try std.testing.expect(!computes[0].sameExecutable(shape, computes[1], shape.withDtype(.i8)));
    try std.testing.expectEqual(1, computes[3].program.as(Shift).?.factor);
    try std.testing.expect(computes[3].program.as(Scale) == null);
}

test "placed shapes on different meshes or partitionings differ" {
    var meshes: [2]Sharding.Mesh = undefined;
    meshes[0].name = "first";
    meshes[1].name = "second";
    const first = Shape.init(.{ .qout = 8, .qk = 16 }, .bf16).withPartitioning(&meshes[0], .{ .qout = .replicated, .qk = .replicated });
    const second = first.withPartitioning(&meshes[1], .{ .qout = .replicated, .qk = .replicated });
    try std.testing.expect(samePlacedShape(first, first));
    try std.testing.expect(!samePlacedShape(first, second));
    try std.testing.expect(!samePlacedShape(first, first.withPartitioning(first._sharding.mesh.?, .{ .qout = .open })));
}

test "setMemory changes where the loader delivers a bound tensor" {
    var store: TensorStore = store: {
        var registry: safetensors.TensorRegistry = .init(std.testing.allocator);
        errdefer registry.deinit();
        try registry.registerTensor(.{ .file_uri = "weights", .name = "table", .shape = .init(.{ 4, 2 }, .u8), .offset = 0 });
        break :store .init(std.testing.allocator, registry, .{});
    };
    defer store.deinit();

    const view_ = store.view();
    const tensor = view_.createReplicatedTensor("table", null);
    try std.testing.expectEqual(Memory.default, store.bindingOf(tensor.id).?.load.memory);
    view_.setMemory(tensor, .host_mapped);
    try std.testing.expectEqual(Memory.host_mapped, store.bindingOf(tensor.id).?.load.memory);
    view_.setMemory(tensor, .default);
    try std.testing.expectEqual(Memory.default, store.bindingOf(tensor.id).?.load.memory);
}
