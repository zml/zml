const std = @import("std");
const async = @import("async");
const zml = @import("zml");

pub fn main() !void {
    try async.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    zml.init();
    defer zml.deinit();

    var platform = try zml.Platform.init(.cpu, .{});
    defer platform.deinit();
}
