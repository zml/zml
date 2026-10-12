const std = @import("std");
const log = std.log;

const zml = @import("zml");
const block = @import("../blocks.zig");

const ActivationStore = @This();

store: zml.io.TensorStore = undefined,
registry: zml.safetensors.TensorRegistry = undefined,

pub fn load(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !ActivationStore {
    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, path);
    log.info("Found {} activations in {s}", .{ activations_registry.tensors.count(), path });

    const activations_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry);

    return ActivationStore{ .registry = activations_registry, .store = activations_store };
}

pub fn deinit(self: *ActivationStore) void {
    self.store.deinit();
    self.registry.deinit();
}
