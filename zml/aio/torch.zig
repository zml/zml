const asynk = @import("async");
const std = @import("std");
const zml = @import("../zml.zig");

const eval = @import("torch/eval.zig");
const py_object = @import("torch/py_object.zig");
const File = @import("torch/file.zig").File;

const StringBuilder = std.ArrayListUnmanaged(u8);
const log = std.log.scoped(.zml_aio);

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(eval);
    std.testing.refAllDecls(py_object);
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

    // But we create the HostBuffer objects inside the result BufferStore arena.
    var res: zml.aio.BufferStore = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    const res_alloc = res.arena.allocator();
    res.files = try res_alloc.dupe(zml.aio.MemoryMappedFile, &.{torch_file.buffer_file});

    try torch_file.parseModel(res_alloc, py_values, &res);
    return res;
}
