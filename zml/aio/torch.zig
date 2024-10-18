const asynk = @import("async");
const std = @import("std");
const zml = @import("../zml.zig");

const eval = @import("torch/eval.zig");
const py = @import("torch/py.zig");
const File = @import("torch/file.zig").File;

const StringBuilder = std.ArrayListUnmanaged(u8);
const log = std.log.scoped(.zml_aio);

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(eval);
    std.testing.refAllDecls(py);
    std.testing.refAllDecls(File);
}

/// Opens and loads a BufferStore from the torch file at the given path.
pub fn open(allocator: std.mem.Allocator, path: []const u8) !zml.aio.BufferStore {
    const file = asynk.File.open(path, .{}) catch |err| {
        log.err("Failed to open {s}: {}", .{ path, err });
        return err;
    };
    errdefer file.close() catch unreachable;

    // Temporary memory needed to parse the pytorch file.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const tmp_alloc = arena.allocator();

    var torch_file = try File.init(tmp_alloc, file);
    const ops = try torch_file.parsePickle(tmp_alloc);
    const py_values = try eval.evaluate(tmp_alloc, ops, true);

    // file ownership is transferred to the BufferStore
    var res = try zml.aio.BufferStore.init(allocator, &.{torch_file.buffer_file});
    try torch_file.parseModel(py_values, &res);
    return res;
}
