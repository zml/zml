const std = @import("std");
const zml = @import("zml");

/// Print safetensors metadata (tensor count, then first 200 tensor names and shapes).
/// Usage:
///   bazel run //examples/ltx:ltx
///   bazel run //examples/ltx:ltx -- /path/to/model.safetensors
/// The first positional argument is optional; when omitted, a local default path is used.
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var it = init.minimal.args.iterate();
    _ = it.next(); // skip executable name

    const model_path = it.next() orelse "/Users/oboulant/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors";
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
