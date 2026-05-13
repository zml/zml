const std = @import("std");
const log = std.log;

const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    const model_path, const activations_path = args[1..3].*;

    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, activations_path);
    defer activations_registry.deinit();
    log.info("Found {} activations in {s}", .{ activations_registry.tensors.count(), activations_path });

    var model_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, model_path);
    defer model_registry.deinit();
    log.info("Found {} activations in {s}", .{ model_registry.tensors.count(), model_path });
}
