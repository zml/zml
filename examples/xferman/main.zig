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

    // Our weights and bias to use
    const weights = [_]f16{4} ** (16 * 1024);
    const bias = [_]f16{4} ** (16 * 1024);
    const input_shape = zml.Shape.init(.{(&weights).len}, .f16);

    const platform = context.autoPlatform(.{});
    const api = platform.pjrt_api;

    const shapes: []const zml.Shape = &.{ input_shape, input_shape };

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

    const weights_buffer = std.mem.sliceAsBytes(&weights);
    const bias_buffer = std.mem.sliceAsBytes(&bias);

    const start_time = std.time.nanoTimestamp();
    var event_cycle_counter: usize = 0;

    // transfer both slices in one call
    if (true) {
        const events = try manager.transferDataMany(&.{ weights_buffer, bias_buffer }, .{});
        for (events) |event| {
            while (!event.isReady(api)) : (event_cycle_counter += 1) {
                // this is faster than event.awaitt()
            }
        }
    }

    // transfer both buffers individually, but using transferDataMany to check
    // continuaton via opts: start_buffer_index, last_data_is_last_transfer
    if (false) {
        // first
        {
            // const event = try manager.transferDataSingle(0, weights_buffer, 0, true);
            const events = try manager.transferDataMany(&.{weights_buffer}, .{
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
            const events = try manager.transferDataMany(&.{bias_buffer}, .{
                .start_buffer_index = 1,
                .last_data_is_last_transfer = true, // true is default but we are explicit here
            });

            for (events) |event| {
                while (!event.isReady(api)) : (event_cycle_counter += 1) {
                    // this is faster than event.awaitt()
                }
            }
        }
    }

    // transfer all buffers as slices of one big buffer (as would be the case
    // with an mmapped file)
    if (false) {
        var big_buf = try allocator.alloc(u8, weights_buffer.len + bias_buffer.len);
        @memcpy(big_buf[0..weights_buffer.len], weights_buffer);
        @memcpy(big_buf[weights_buffer.len..], bias_buffer);

        const slice_specs: []const zml.platform.TransferManager.TransferDataSlicesSpec =
            &.{
                .{ .offset = 0, .len = weights_buffer.len },
                .{ .offset = weights_buffer.len, .len = bias_buffer.len },
            };
        const events = try manager.transferDataSlices(big_buf, slice_specs);
        _ = events; // we don't need them as we're going to query .progress()

        var dt: i128 = undefined;
        while (true) {
            event_cycle_counter += 1;
            dt = std.time.nanoTimestamp() - start_time;
            const progress = try manager.progress();
            log.debug("After {d} ns:  {}", .{ dt, progress });
            if (progress.transferred_buffers == progress.total_buffers) {
                break;
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
