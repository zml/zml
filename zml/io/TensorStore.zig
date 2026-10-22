//! Associates model tensors with checkpoint sources and prefixed views,
//! over a checkpoint registry the store owns.
const std = @import("std");

const stdx = @import("stdx");

const Compiler = @import("../Compiler.zig");
const Exe = @import("../exe.zig").Exe;
const Memory = @import("../platform.zig").Memory;
const Platform = @import("../platform.zig").Platform;
const Quantization = @import("../quantization.zig").Quantization;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;

const TensorStore = @This();

pub const Binding = struct {
    /// Entries of the store's registry, each listed once.
    tensors: []const *const safetensors.Tensor,
    /// Memory space the loader must deliver this tensor into, on every
    /// device of its mesh (`View.setMemory`).
    memory: Memory.Kind = .default,
    /// How `tensors` fill the tensor's logical bytes. Null is the identity:
    /// one source that is the whole tensor.
    layout: ?*const Layout = null,
    /// Set when a program computes the tensor while loading
    /// (`Source.bindProgram`). The loader never fills such a tensor itself,
    /// so `layout` is unused, and its output lands in default device memory.
    program: ?*const Program = null,
};

/// A program the loader compiles and runs to compute a tensor while loading
/// (`Source.bindProgram`): `forward(output, inputs)` of the program's value,
/// over its inputs landed in default device memory, placed by their shapes.
pub const Program = struct {
    /// The program's inputs in order, each filled from `Binding.tensors`.
    inputs: []const Layout,
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

    /// The program `program` over `inputs`, its value copied into `arena`.
    /// Its type declares `pub fn forward(self, output: Shape, inputs: []const
    /// Tensor) Tensor` and is plain data, without pointers, so equal values
    /// are found by value and compile once.
    pub fn init(arena: std.mem.Allocator, inputs: []const Layout, program: anytype) std.mem.Allocator.Error!*const Program {
        const T = @TypeOf(program);
        comptime {
            if (!isPlainData(T)) @compileError("TensorStore.Program: " ++ @typeName(T) ++ " must be plain data, without pointers");
        }
        const value: *const anyopaque = if (@sizeOf(T) == 0) &no_value else value: {
            const copy = try arena.create(T);
            copy.* = program;
            break :value copy;
        };
        const self = try arena.create(Program);
        self.* = .{ .inputs = inputs, .value = value, .vtable = &Erased(T).vtable };
        return self;
    }

    /// The program's value, if it is a `T`.
    pub fn as(self: Program, comptime T: type) ?T {
        if (self.vtable != &Erased(T).vtable) return null;
        return Erased(T).typed(self.value);
    }

    /// Whether this program computing `output` and `other` computing
    /// `other_output` compile to the same executable: programs of one type
    /// and equal values, over the same shapes placed alike.
    pub fn sameExecutable(self: Program, output: Shape, other: Program, other_output: Shape) bool {
        if (self.vtable != other.vtable or !self.vtable.eql(self.value, other.value)) return false;
        if (!samePlacedShape(output, other_output) or self.inputs.len != other.inputs.len) return false;
        for (self.inputs, other.inputs) |a, b| {
            if (!samePlacedShape(a.shape, b.shape)) return false;
        }
        return true;
    }

    fn samePlacedShape(a: Shape, b: Shape) bool {
        if (!a.eqlWithTags(b) or a._sharding.mesh != b._sharding.mesh) return false;
        for (0..a.rank()) |axis| {
            if (a.partition(axis) != b.partition(axis)) return false;
        }
        return true;
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

/// A composite tensor: byte parts of checkpoint tensors placed in the
/// logical (row-major, packed) bytes of `shape`, every byte no part covers
/// zero. `Source.bind` makes them; the loader assembles them.
pub const Layout = struct {
    /// The logical shape the parts fill, placed like the tensor or program
    /// input they fill.
    shape: Shape,
    parts: []const Part,

    /// The checkpoint tensor of `tensors` that the parts copy whole and
    /// alone, if they do.
    pub fn whole(self: Layout, tensors: []const *const safetensors.Tensor) ?*const safetensors.Tensor {
        if (self.parts.len != 1) return null;
        const part = self.parts[0];
        if (part.dims.len != 0 or part.src_offset != 0 or part.dst_offset != 0 or part.width != self.shape.byteSize()) return null;
        const tensor = tensors[part.source];
        return if (tensor.byteSize() == part.width) tensor else null;
    }
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

registry: safetensors.TensorRegistry,
meshes: []const *const Sharding.Mesh,
quantization: Quantization.Target,
id_to_sources: std.AutoHashMapUnmanaged(Tensor.Id, Binding),
allocator: std.mem.Allocator,
arena: std.heap.ArenaAllocator,

pub const Options = struct {
    /// The meshes `Source.withPartitioning` and `View.mesh` name.
    meshes: []const *const Sharding.Mesh = &.{},
    /// How the platform's FP8 GEMMs want block-FP8 weights, for the code
    /// that binds them (`View.quantization`): usually `.of(platform)`.
    quantization: Quantization.Target = .{},
};

/// A store that owns `registry`, which must be complete: its map stays
/// locked until `deinit`, so a later insert panics in safe builds. An
/// entry's address is then its checkpoint tensor's identity (`View.get`).
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

/// Frees the bindings, then the registry.
pub fn deinit(self: *TensorStore) void {
    self.id_to_sources.deinit(self.allocator);
    self.arena.deinit();
    self.registry.tensors.unlockPointers();
    self.registry.deinit();
}

pub fn getSourcesById(self: *const TensorStore, id: Tensor.Id) ?Binding {
    return self.id_to_sources.get(id);
}

/// A new tensor of `shape`, bound to `binding`. Model code binds through
/// `Source` (`bind`, `bindProgram`), whose layouts are disjoint by
/// construction; `Loader.start` checks a hand-built one.
pub fn bindNew(self: *TensorStore, shape: Shape, binding: Binding) Tensor {
    const tensor: Tensor = .fromShape(shape);
    self.id_to_sources.putNoClobber(self.allocator, tensor.id, binding) catch @panic("OOM");
    return tensor;
}

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
        const placed = source_.withPartitioning(mesh_, partitioning);
        if (placed.count() != 0) return placed.bind();
        // An empty checkpoint tensor is a source of no piece, which `bind`
        // records as a composite without parts. Bound plainly, `Loader.start`
        // rejects it by its key.
        const entry = self.store.arena.allocator().dupe(*const safetensors.Tensor, &.{self.get(subkey).?}) catch @panic("OOM");
        return self.store.bindNew(placed.shape(), .{ .tensors = entry });
    }

    /// `createTensor` replicated on every device, without a mesh.
    pub fn createReplicatedTensor(self: View, subkey: []const u8, comptime tags: anytype) Tensor {
        return self.createTensor(subkey, tags, .replicated, .{});
    }

    /// Changes the memory space the loader delivers `tensor` into (see
    /// `Binding.memory`), for a decision made after the tensor was bound.
    /// `tensor` must be bound in this store, and not computed by a program:
    /// a program's output lands in default device memory.
    pub fn setMemory(self: View, tensor: Tensor, kind: Memory.Kind) void {
        const binding = self.store.id_to_sources.getPtr(tensor.id) orelse
            stdx.debug.panic("Tensor {} {f} has no binding in this store", .{ tensor.id, tensor.shape() });
        if (binding.program) |program|
            stdx.debug.panic("setMemory: program {s} computes tensor {} {f}, whose output lands in default device memory", .{ program.vtable.name, tensor.id, tensor.shape() });
        binding.memory = kind;
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
            input.* = source_.withPartitioning(.replicated, .{});
        }
        return Source.bindProgram(inputs, shape, program);
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
    try std.testing.expectEqual(entry, store.getSourcesById(tensor.id).?.tensors[0]);
    try std.testing.expectEqual(entry, layer.source("experts.0.down_proj.weight", null).pieces[0].tensor);
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
    var programs: [tensors.len]Program = undefined;
    for (&programs, tensors) |*program, tensor| program.* = store.getSourcesById(tensor.id).?.program.?.*;
    try std.testing.expect(programs[0].sameExecutable(shape, programs[1], shape));
    try std.testing.expect(!programs[0].sameExecutable(shape, programs[2], shape));
    try std.testing.expect(!programs[0].sameExecutable(shape, programs[3], shape));
    try std.testing.expect(!programs[0].sameExecutable(shape, programs[1], shape.withDtype(.i8)));
    try std.testing.expectEqual(1, programs[3].as(Shift).?.factor);
    try std.testing.expect(programs[3].as(Scale) == null);
}

test "placed shapes on different meshes or partitionings differ" {
    var meshes: [2]Sharding.Mesh = undefined;
    meshes[0].name = "first";
    meshes[1].name = "second";
    const first = Shape.init(.{ .qout = 8, .qk = 16 }, .bf16).withPartitioning(&meshes[0], .{ .qout = .replicated, .qk = .replicated });
    const second = first.withPartitioning(&meshes[1], .{ .qout = .replicated, .qk = .replicated });
    try std.testing.expect(Program.samePlacedShape(first, first));
    try std.testing.expect(!Program.samePlacedShape(first, second));
    try std.testing.expect(!Program.samePlacedShape(first, first.withPartitioning(first._sharding.mesh.?, .{ .qout = .open })));
}

test "setMemory changes the memory space of a bound tensor's binding" {
    var store: TensorStore = store: {
        var registry: safetensors.TensorRegistry = .init(std.testing.allocator);
        errdefer registry.deinit();
        try registry.registerTensor(.{ .file_uri = "weights", .name = "table", .shape = .init(.{ 4, 2 }, .u8), .offset = 0 });
        break :store .init(std.testing.allocator, registry, .{});
    };
    defer store.deinit();

    const view_ = store.view();
    const tensor = view_.createReplicatedTensor("table", null);
    try std.testing.expectEqual(Memory.Kind.default, store.getSourcesById(tensor.id).?.memory);
    view_.setMemory(tensor, .host_pinned);
    try std.testing.expectEqual(Memory.Kind.host_pinned, store.getSourcesById(tensor.id).?.memory);
    view_.setMemory(tensor, .default);
    try std.testing.expectEqual(Memory.Kind.default, store.getSourcesById(tensor.id).?.memory);
}
