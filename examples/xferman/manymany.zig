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
    log.info("   Compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.c_allocator;

    var context = try zml.Context.init();
    defer context.deinit();

    const NUM_BUFFERS: usize = 146;

    // Our weights and bias to use
    const weights = [_]f16{4} ** (256_000 * 1024);
    const input_shape = zml.Shape.init(.{(&weights).len}, .f16);

    const platform = context.autoPlatform(.{});
    const api = platform.pjrt_api;

    var shapes_list = try std.ArrayList(zml.Shape).initCapacity(allocator, NUM_BUFFERS);
    defer shapes_list.deinit();
    const weights_buffer = std.mem.sliceAsBytes(&weights);
    var buffers_list = try std.ArrayList([]const u8).initCapacity(allocator, NUM_BUFFERS);

    var total_bytesize: usize = 0;
    for (0..NUM_BUFFERS) |_| {
        shapes_list.appendAssumeCapacity(input_shape);
        buffers_list.appendAssumeCapacity(try allocator.dupe(u8, weights_buffer));
        total_bytesize += weights_buffer.len;
    }
    defer {
        for (buffers_list.items) |b| {
            allocator.free(b);
        }
    }

    const shapes = shapes_list.items;

    log.debug("input_shapes = {any}", .{shapes});
    var manager = try zml.platform.TransferManager.init(
        allocator,
        platform,
        .unpinned_host,
        shapes,
    );
    defer manager.deinit();
    const buffer_count = try manager.pjrt_transfer_manager.bufferCount(platform.pjrt_api);
    log.debug("transfer manager has {d} buffers", .{buffer_count});

    const start_time = std.time.nanoTimestamp();
    var event_cycle_counter: usize = 0;

    // transfer all slices in one call
    const events = try manager.transferDataMany(buffers_list.items, .{});
    for (events) |event| {
        while (!event.isReady(api)) : (event_cycle_counter += 1) {
            // this is faster than event.awaitt()
        }
    }

    const end_time = std.time.nanoTimestamp();
    log.info("Transferred {d} buffers ({d} bytes) in {d} cycles = {d} ns", .{
        buffer_count,
        total_bytesize,
        event_cycle_counter,
        end_time - start_time,
    });
}
