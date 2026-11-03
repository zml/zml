//! The new loader's store: prefixed views over the entries it takes from a
//! checkpoint registry. zml does not export it: models and the loader they
//! use build `zml.io.TensorStore` (io.zig), which borrows its registry, so
//! production changes go there and the new loader's here.
//!
//! The store owns those entries. Views point to it: once `view` was called
//! it must not move, and it must outlive them. It borrows the meshes of
//! `Options.meshes`, which belong to the platform.
const std = @import("std");

const stdx = @import("stdx");

const MemorySpace = @import("../platform.zig").Memory.Kind;
const Quantization = @import("../quantization.zig").Quantization;
const safetensors = @import("../safetensors.zig");
const Sharding = @import("../Sharding.zig");

const TensorStore = @This();

/// Where the loader delivers a loaded tensor, on every device of its mesh.
/// The destination is allocated once per device; the host kinds take no
/// device memory.
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
    /// views against device memory.
    host_mapped,

    /// Whether the bytes live in host memory, written by the CPU.
    pub fn isHost(self: Memory) bool {
        return switch (self) {
            .default => false,
            .host_pinned, .host_mapped => true,
        };
    }

    /// The memory space of each device's delivered buffer.
    pub fn space(self: Memory) MemorySpace {
        return switch (self) {
            .default, .host_mapped => .default,
            .host_pinned => .host_pinned,
        };
    }
};

registry: safetensors.TensorRegistry,
meshes: []const *const Sharding.Mesh,
quantization: ?Quantization.Target,
arena: std.heap.ArenaAllocator,

pub const Options = struct {
    /// The meshes `View.mesh` names. The slice is copied; the meshes must
    /// outlive the store.
    meshes: []const *const Sharding.Mesh = &.{},
    /// How the platform's FP8 GEMMs want block-FP8 weights, for the code
    /// that binds them (`View.quantization`): `.of(platform)`. Required to
    /// bind block-FP8 weights.
    quantization: ?Quantization.Target = null,
};

/// A store over the entries of `registry`, which must be complete: add
/// every checkpoint first (`TensorRegistry.addRepo`). It takes the entries
/// and leaves `registry` empty. The store's map stays locked until
/// `deinit`, so a later insert panics in safe builds, and an entry's
/// address is its checkpoint tensor's identity (`View.get`).
pub fn init(allocator: std.mem.Allocator, registry: *safetensors.TensorRegistry, opts: Options) TensorStore {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    const meshes = arena.allocator().dupe(*const Sharding.Mesh, opts.meshes) catch @panic("OOM");
    var self: TensorStore = .{
        .registry = registry.*,
        .meshes = meshes,
        .quantization = opts.quantization,
        .arena = arena,
    };
    registry.* = .init(registry.arena.child_allocator);
    self.registry.tensors.lockPointers();
    return self;
}

/// Frees the registry entries `init` took.
pub fn deinit(self: *TensorStore) void {
    self.arena.deinit();
    self.registry.tensors.unlockPointers();
    self.registry.deinit();
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
/// tensors up. A value that keeps its own copy of the prefix (256 bytes at
/// most, longer panics) and points to the store.
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
    /// `get("embed")` looks up `model.embed`.
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

    /// See `TensorStore.mesh`.
    pub fn mesh(self: View, name: @EnumLiteral()) *const Sharding.Mesh {
        return self.store.mesh(name);
    }

    /// See `Options.quantization`. Panics when the store has none: binding
    /// block-FP8 weights requires it.
    pub fn quantization(self: View) Quantization.Target {
        return self.store.quantization orelse
            stdx.debug.panic("TensorStore.Options.quantization is unset: block-FP8 weights bind for a platform; pass .of(platform)", .{});
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
    var registry: safetensors.TensorRegistry = .init(std.testing.allocator);
    defer registry.deinit();
    try registry.registerTensor(.{ .file_uri = "weights", .name = "layer.experts.0.down_proj.weight", .shape = .init(.{4}, .u8), .offset = 0 });
    var store: TensorStore = .init(std.testing.allocator, &registry, .{});
    defer store.deinit();
    // The store took the entries.
    try std.testing.expectEqual(0, registry.tensors.count());

    const layer = store.view().withPrefix("layer");
    const entry = layer.get("experts.0.down_proj.weight").?;
    try std.testing.expectEqual(entry, store.registry.tensors.getPtr("layer.experts.0.down_proj.weight").?);
    try std.testing.expectEqual(null, layer.get("experts.0.down_proj"));
    try std.testing.expectEqual(null, layer.get("weights"));
    try std.testing.expectEqualStrings("layer.", layer.prefix());
    try std.testing.expectEqualStrings("", layer.root().prefix());
    try std.testing.expectEqualStrings("layer.experts.0.", layer.withPrefix("experts").withLayer(0).prefix());
    try std.testing.expectEqual(1, layer.withPrefix("experts").count());
    try std.testing.expectEqual(0, layer.withPrefix("gate").count());
}
