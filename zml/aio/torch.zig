const std = @import("std");

const asynk = @import("async");

const zml = @import("../zml.zig");
const eval = @import("torch/eval.zig");
const File = @import("torch/file.zig").File;

const StringBuilder = std.ArrayListUnmanaged(u8);
const log = std.log.scoped(.@"zml/aio");

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(eval);
    std.testing.refAllDecls(@import("torch/py.zig"));
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

    const mmap_file = try zml.aio.MemoryMappedFile.init(file);
    var torch_file = try asynk.callBlocking(File.init, .{ tmp_alloc, mmap_file });

    const ops = try torch_file.parsePickle(tmp_alloc);
    const py_values = try eval.evaluate(tmp_alloc, ops, true);

    // file ownership is transferred to the BufferStore
    var res = try zml.aio.BufferStore.initWithFiles(allocator, &.{torch_file.buffer_file});
    try torch_file.parseModel(py_values, &res);
    return res;
}
