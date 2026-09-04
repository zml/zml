const std = @import("std");

const zml = @import("zml");

const weights = @import("../recipe/weights.zig");

// =============================================================================
// refine/load.zig — local safetensors open / first-existing path
// =============================================================================

pub const Store = struct {
    store: zml.io.TensorStore,
    registry: *zml.safetensors.TensorRegistry,
    allocator: std.mem.Allocator,

    pub fn open(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Store {
        var owned: []u8 = &.{};
        defer if (owned.len != 0) allocator.free(owned);
        const resolved = if (isUri(path) or std.fs.path.isAbsolute(path)) path else blk: {
            owned = try resolvePath(allocator, io, path);
            break :blk owned;
        };
        const registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(registry);
        registry.* = try zml.safetensors.TensorRegistry.fromPath(allocator, io, resolved);
        errdefer registry.deinit();
        return .{
            .store = .fromRegistry(allocator, registry),
            .registry = registry,
            .allocator = allocator,
        };
    }

    /// Private tensor-id bindings for one compile. TensorStore is not thread-safe.
    pub fn bind(self: *const Store, allocator: std.mem.Allocator) zml.io.TensorStore {
        return .fromRegistry(allocator, self.registry);
    }

    pub fn deinit(self: *Store) void {
        self.store.deinit();
        self.registry.deinit();
        self.allocator.destroy(self.registry);
    }

    pub fn view(self: *Store) zml.io.TensorStore.View {
        return self.store.view();
    }
};

pub fn resolvePath(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(path)) return allocator.dupe(u8, path);
    const cwd = try std.process.currentPathAlloc(io, allocator);
    defer allocator.free(cwd);
    return std.fs.path.join(allocator, &.{ cwd, path });
}

pub fn isUri(path: []const u8) bool {
    return std.mem.indexOf(u8, path, "://") != null;
}

pub fn fileExists(io: std.Io, path: []const u8) bool {
    if (isUri(path)) return false;
    if (std.fs.path.isAbsolute(path)) {
        var f = std.Io.Dir.openFileAbsolute(io, path, .{ .mode = .read_only }) catch return false;
        f.close(io);
        return true;
    }
    var f = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch return false;
    f.close(io);
    return true;
}

/// Local files win when present. Otherwise the first `hf://` / `https://` URI.
pub fn firstExisting(io: std.Io, paths: []const []const u8) ?[]const u8 {
    var remote: ?[]const u8 = null;
    for (paths) |p| {
        if (isUri(p)) {
            if (remote == null) remote = p;
            continue;
        }
        if (fileExists(io, p)) return p;
    }
    return remote;
}

/// Official Comfy/HF LTX dumps nest weights under `model.diffusion_model.*`.
/// Older stripped checkpoints keep the inner names at the root.
pub fn viewFor(store: zml.io.TensorStore.View, sentinel: []const u8, prefixes: []const []const u8) zml.io.TensorStore.View {
    if (store.hasKey(sentinel)) return store;
    for (prefixes) |p| {
        const v = store.withPrefix(p);
        if (v.hasKey(sentinel)) return v;
    }
    return store;
}

pub fn loadModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    comptime T: type,
    model: *const T,
    progress: *std.Progress.Node,
) !zml.Bufferized(T) {
    return weights.load(allocator, io, platform, store, shardings, T, model, progress, null);
}
