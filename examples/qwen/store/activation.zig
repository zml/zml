const std = @import("std");
const log = std.log;

const zml = @import("zml");
const block = @import("../blocks.zig");

const ActivationStore = @This();

store: zml.io.TensorStore = undefined,
registry: zml.safetensors.TensorRegistry = undefined,

pub fn load(
    self: *ActivationStore,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !void {
    self.registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, path);
    log.info("Found {} activations in {s}", .{ self.registry.tensors.count(), path });

    self.store = .fromRegistry(allocator, &self.registry);
}

pub fn deinit(self: *ActivationStore) void {
    self.store.deinit();
    self.registry.deinit();
}
