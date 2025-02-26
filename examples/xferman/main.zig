const asynk = @import("async");
const clap = @import("clap");
const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const log = std.log.scoped(.xferman);

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    log.info("   LLama was compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.c_allocator;
    _ = allocator;

    var context = try zml.Context.init();
    defer context.deinit();

    // Our weights and bias to use
    const weights = [4]f16{ 2.0, 2.0, 2.0, 2.0 };
    const input_shape = zml.Shape.init(.{4}, .f16);

    const hb = zml.HostBuffer.fromArray(&weights);
    log.info("hb memory: {}", .{hb._memory});

    const platform = context.autoPlatform(.{});

    const client = platform.pjrt_client;
    log.info("client : {}", .{client});

    var shapes: []zml.Shape = undefined;
    shapes.ptr = @constCast(@ptrCast(&input_shape));
    shapes.len = 1;

    log.debug("shape_arr = {any}", .{shapes});
    var manager = try zml.platform.TransferManager.init(
        platform,
        .unpinned_host,
        shapes,
    );
    defer manager.deinit();
    log.info("manager : {}", .{manager});
}
