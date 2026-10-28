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
const load_log = @import("log.zig").load;
const testing = @import("../testing.zig");

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
    /// views against device memory (`View.hostMappedBytes`).
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
    /// The loader fills the tensor from `source` (`Source.Placed.bind`),
    /// every byte no piece covers zero.
    load: Load,
    /// A program computes the tensor while loading (`Source.bindProgram`).
    /// Its output lands in default device memory.
    compute: Compute,

    pub const Load = struct {
        source: Source.Placed,
        /// Where the loader delivers this tensor (`View.setMemory`).
        memory: Memory = .default,
    };

    pub const Compute = struct {
        program: Program,
        /// The program's inputs in order, each landed in default device
        /// memory, placed by its shape.
        inputs: []const Source.Placed,

        /// Whether this computing `output` and `other` computing
        /// `other_output` compile to the same executable: equal programs
        /// over the same shapes placed alike.
        pub fn sameExecutable(self: Compute, output: Shape, other: Compute, other_output: Shape) bool {
            if (!self.program.eql(other.program)) return false;
            if (!output.eqlWithTagsAndPartitioning(other_output) or self.inputs.len != other.inputs.len) return false;
            for (self.inputs, other.inputs) |a, b| {
                if (!a.shape.eqlWithTagsAndPartitioning(b.shape)) return false;
            }
            return true;
        }

        /// Compiles the program computing a tensor of `output` from inputs
        /// of the shapes it binds, its module named after the program.
        pub fn compile(
            self: Compute,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const Platform,
            meshes: []const *const Sharding.Mesh,
            output: Shape,
        ) Compiler.Error!Exe {
            const inputs = try allocator.alloc(Tensor, self.inputs.len);
            defer allocator.free(inputs);
            for (inputs, self.inputs) |*input, source| input.* = .fromShape(source.shape);
            return self.program.compile(allocator, io, platform, meshes, output, inputs);
        }
    };
};

/// A program the loader compiles and runs to compute a tensor while loading
/// (`Binding.compute`): `forward(output, inputs)` of the program's value.
/// Model code makes one through `Source.bindProgram`, and reads it back
/// with `as`.
pub const Program = struct {
    /// The program's value, in the store's arena.
    value: *const anyopaque,
    vtable: *const VTable,

    const VTable = struct {
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
        const copy = try arena.create(T);
        copy.* = program;
        return .{ .value = copy, .vtable = &Erased(T).vtable };
    }

    /// The program's value, if it is a `T`.
    pub fn as(self: Program, comptime T: type) ?T {
        if (self.vtable != &Erased(T).vtable) return null;
        return Erased(T).typed(self.value);
    }

    /// The program's type name, which names its compiled module in traces,
    /// dumps and logs.
    pub fn name(self: Program) []const u8 {
        return self.vtable.name;
    }

    /// Whether `a` and `b` are programs of one type with equal values.
    pub fn eql(a: Program, b: Program) bool {
        return a.vtable == b.vtable and a.vtable.eql(a.value, b.value);
    }

    /// Compiles `forward(output, inputs)`, its module named after the program.
    fn compile(
        self: Program,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        meshes: []const *const Sharding.Mesh,
        output: Shape,
        inputs: []const Tensor,
    ) Compiler.Error!Exe {
        return self.vtable.compile(self.value, allocator, io, platform, .{ .meshes = meshes, .program_name = self.name() }, output, inputs);
    }

    /// The vtable of programs of type `T`, one per type.
    fn Erased(comptime T: type) type {
        return struct {
            const vtable: VTable = .{ .name = @typeName(T), .compile = compileValue, .eql = eqlValues };

            fn typed(erased: *const anyopaque) T {
                return @as(*const T, @ptrCast(@alignCast(erased))).*;
            }

            fn compileValue(
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

            fn eqlValues(a: *const anyopaque, b: *const anyopaque) bool {
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
/// `Source` (`Placed.bind`, `bindProgram`), whose pieces are disjoint by
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
        const new_prefix = makeKey(&buffer, "{s}{s}.", .{ self.prefix(), prefix_ });

        return .{
            .store = self.store,
            .prefix_buffer = buffer,
            .prefix_length = new_prefix.len,
        };
    }

    /// This view under `index` and a dot, as `withPrefix` of the number.
    pub fn withLayer(self: *const View, index: usize) View {
        var buffer: [256]u8 = undefined;
        const new_prefix = makeKey(&buffer, "{s}{d}.", .{ self.prefix(), index });

        return .{
            .store = self.store,
            .prefix_buffer = buffer,
            .prefix_length = new_prefix.len,
        };
    }

    /// The keys' prefix under this view, with its trailing dot; "" at the
    /// root.
    pub fn prefix(self: *const View) []const u8 {
        return self.prefix_buffer[0..self.prefix_length];
    }

    /// The checkpoint tensor `subkey`, the registry's own entry, whose
    /// address is the tensor's identity. Null when the checkpoint has none.
    pub fn get(self: View, subkey: []const u8) ?*const safetensors.Tensor {
        var buffer: [256]u8 = undefined;
        const key = makeKey(&buffer, "{s}{s}", .{ self.prefix(), subkey });
        return self.store.registry.tensors.getPtr(key);
    }

    /// The checkpoint tensor `subkey`, with `tags` (null keeps the
    /// checkpoint's), placed on `mesh_` by `partitioning` and bound:
    /// `source(subkey, tags).withPartitioning(mesh_, partitioning).bind()`.
    /// Panics when the checkpoint has no such tensor.
    pub fn createTensor(self: View, subkey: []const u8, comptime tags: anytype, comptime mesh_: @EnumLiteral(), partitioning: anytype) Tensor {
        return self.maybeCreateTensor(subkey, tags, mesh_, partitioning) orelse
            stdx.debug.panic("Checkpoint has no tensor named {s}{s}", .{ self.prefix(), subkey });
    }

    /// Like `createTensor`, but null when the checkpoint has no such tensor.
    pub fn maybeCreateTensor(self: View, subkey: []const u8, comptime tags: anytype, comptime mesh_: @EnumLiteral(), partitioning: anytype) ?Tensor {
        const source_ = self.maybeSource(subkey, tags) orelse return null;
        return source_.withPartitioning(mesh_, partitioning).bind();
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
            .compute => |compute| stdx.debug.panic("setMemory: program {s} computes tensor {} {f}, whose output lands in default device memory", .{ compute.program.name(), tensor.id, tensor.shape() }),
        }
    }

    /// The bytes on one device of every tensor the store delivers
    /// `.host_mapped`, each counted by its shard: what XLA counts against
    /// device memory for those views, which a model gives back through
    /// `Compiler.Options.extra_device_memory_bytes`. The whole store's,
    /// whatever this view's prefix; ask once every binding and `setMemory`
    /// is done. A tensor whose mesh partitions do not divide its dimensions
    /// is `IncompatibleSharding`, logged.
    pub fn hostMappedBytes(self: View) error{IncompatibleSharding}!u64 {
        var bytes: u64 = 0;
        var it = self.store.id_to_sources.iterator();
        while (it.next()) |entry| switch (entry.value_ptr.*) {
            .load => |load| if (load.memory == .host_mapped) {
                const shard = Sharding.shardedShape(load.source.shape) catch |err| {
                    load_log.err("tensor {} {f}, delivered host_mapped: its dimensions are not divisible by its mesh partitions", .{ entry.key_ptr.*, load.source.shape });
                    return err;
                };
                bytes += shard.byteSize();
            },
            .compute => {},
        };
        return bytes;
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
            stdx.debug.panic("Checkpoint has no tensor named {s}{s}", .{ self.prefix(), subkey });
    }

    /// Like `source`, but null when the checkpoint has no such tensor.
    pub fn maybeSource(self: View, subkey: []const u8, comptime tags: anytype) ?Source {
        const entry = self.get(subkey) orelse return null;
        const has_tags: bool = comptime @TypeOf(tags) != @TypeOf(null);
        const parsed_tags: Shape.TagsArray = if (comptime has_tags) Shape.parseTags(tags) else undefined;
        return .init(self.store, entry, if (has_tags) parsed_tags.constSlice() else null);
    }

    /// The checkpoint tensors under this view's prefix: a scan of every key.
    pub fn count(self: View) usize {
        var count_: usize = 0;
        const prefix_ = self.prefix();
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

test "get finds exact keys, the registry's own entries, and count the keys under a prefix" {
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
    try std.testing.expectEqualStrings("layer.", layer.prefix());
    try std.testing.expectEqualStrings("", layer.root().prefix());
    try std.testing.expectEqual(1, layer.withPrefix("experts").count());
    try std.testing.expectEqual(0, layer.withPrefix("gate").count());

    // Every binding of the entry holds the entry itself.
    const tensor = layer.createReplicatedTensor("experts.0.down_proj.weight", null);
    try std.testing.expectEqual(entry, store.bindingOf(tensor.id).?.load.source.whole().?);
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
    const input = store.view().source("t", null).withPartitioning(.replicated, .{});
    const tensors = [_]Tensor{
        Source.bindProgram(&.{input}, shape, Scale{ .factor = 1 }),
        Source.bindProgram(&.{input}, shape, Scale{ .factor = 1 }),
        Source.bindProgram(&.{input}, shape, Scale{ .factor = 2 }),
        Source.bindProgram(&.{input}, shape, Shift{ .factor = 1 }),
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

test "hostMappedBytes counts one device's shard of each host_mapped tensor" {
    const allocator = std.testing.allocator;
    var devices: testing.Devices = undefined;
    try devices.init(allocator, std.testing.io, 2);
    defer devices.deinit();
    const tp = try devices.mesh("tp");
    var store: TensorStore = store: {
        var registry: safetensors.TensorRegistry = .init(allocator);
        errdefer registry.deinit();
        try registry.registerTensor(.{ .file_uri = "weights", .name = "rows", .shape = .init(.{ 8, 2 }, .u8), .offset = 0 });
        try registry.registerTensor(.{ .file_uri = "weights", .name = "scale", .shape = .init(.{4}, .u8), .offset = 16 });
        try registry.registerTensor(.{ .file_uri = "weights", .name = "dense", .shape = .init(.{4}, .u8), .offset = 20 });
        try registry.registerTensor(.{ .file_uri = "weights", .name = "odd", .shape = .init(.{ 3, 2 }, .u8), .offset = 24 });
        break :store .init(allocator, registry, .{ .meshes = &.{&tp} });
    };
    defer store.deinit();

    const view_ = store.view();
    const rows = view_.createTensor("rows", .{ .row, .col }, .tp, .{ .row = .model });
    const scale = view_.createReplicatedTensor("scale", null);
    _ = view_.createReplicatedTensor("dense", null);
    try std.testing.expectEqual(0, try view_.hostMappedBytes());
    view_.setMemory(rows, .host_mapped);
    view_.setMemory(scale, .host_mapped);
    // Half the rows, and the whole replicated scale, from any prefix.
    try std.testing.expectEqual(8 + 4, try view_.withPrefix("rows").hostMappedBytes());

    // Three rows do not split over two devices.
    const odd = view_.createTensor("odd", .{ .row, .col }, .tp, .{ .row = .model });
    view_.setMemory(odd, .host_mapped);
    try std.testing.expectError(error.IncompatibleSharding, view_.hostMappedBytes());
}
