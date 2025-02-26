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

    var context = try zml.Context.init();
    defer context.deinit();

    // Our weights and bias to use
    const weights = [_]f16{4} ** (16 * 1024);
    const bias = [_]f16{4} ** (16 * 1024);
    const input_shape = zml.Shape.init(.{(&weights).len}, .f16);

    const platform = context.autoPlatform(.{});
    const api = platform.pjrt_api;
    const client = platform.pjrt_client;
    log.info("client : {}", .{client});

    for (client.getDevices(api), 0..) |device, device_index| {
        log.debug("{d} {s}", .{ device_index, device.getDescription(api).toString(api) });
        log.debug("{d} {any}", .{ device_index, device.addressableMemories(api) });
    }

    const shapes: []const zml.Shape = &.{ input_shape, input_shape };

    log.debug("shape_arr = {any}", .{shapes});
    var manager = try zml.platform.TransferManager.init(
        allocator,
        platform,
        .unpinned_host,
        shapes,
    );
    defer manager.deinit();
    const buffer_count = try manager.pjrt_transfer_manager.bufferCount(platform.pjrt_api);
    log.info("manager has {d} buffers", .{buffer_count});

    const weights_buffer = std.mem.sliceAsBytes(&weights);
    const bias_buffer = std.mem.sliceAsBytes(&bias);

    const start_time = std.time.nanoTimestamp();
    var event_cycle_counter: usize = 0;
    const do_all_at_once = false;

    if (do_all_at_once) {
        // const event = try manager.transferDataSingle(0, weights_buffer, 0, true);
        const events = try manager.transferDataMulti(&.{ weights_buffer, bias_buffer }, .{});
        for (events) |event| {
            while (!event.isReady(api)) : (event_cycle_counter += 1) {
                // this is faster than event.awaitt()
            }
        }
    } else {
        // first
        {
            const events = try manager.transferDataMulti(&.{weights_buffer}, .{
                .last_data_is_last_transfer = false,
            });

            for (events) |event| {
                while (!event.isReady(api)) : (event_cycle_counter += 1) {
                    // this is faster than event.awaitt()
                }
            }
        }
        // second
        {
            const events = try manager.transferDataMulti(&.{ weights_buffer, bias_buffer }, .{
                .last_data_is_last_transfer = true, // true is default but we are explicit here
            });

            for (events) |event| {
                while (!event.isReady(api)) : (event_cycle_counter += 1) {
                    // this is faster than event.awaitt()
                }
            }
        }
    }
    const end_time = std.time.nanoTimestamp();
    log.info("Transferred {d} buffers ({d} bytes) in {d} cycles = {d} ns", .{
        buffer_count,
        weights_buffer.len + bias_buffer.len,
        event_cycle_counter,
        end_time - start_time,
    });
}
