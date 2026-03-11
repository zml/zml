const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const model_path = "/Users/oboulant/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors";
    var weights = try zml.safetensors.TensorRegistry.fromPath(allocator, io, model_path);
    defer weights.deinit();

    const keys = weights.tensors.keys();
    std.log.info("Found {d} tensors", .{keys.len});

    const limit = @min(keys.len, 200);
    for (keys[0..limit]) |k| {
        const t = weights.tensors.get(k).?;
        std.log.info("{s} {f}", .{ k, t.shape });
    }
}
