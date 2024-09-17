const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");

const async_ = asynk.async_;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try asynk.AsyncThread.main(gpa.allocator(), asyncMain, .{});
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    // Skip executable path
    _ = args.next().?;

    const file = if (args.next()) |path| blk: {
        std.debug.print("File path: {s}\n", .{path});
        break :blk path;
    } else {
        std.debug.print("Missing file path argument\n", .{});
        std.debug.print("Try: bazel run -c opt //loader:safetensors -- /path/to/mymodel.safetensors or /path/to/model.safetensors.index.json \n", .{});
        std.process.exit(0);
    };

    var buffer_store = try zml.aio.safetensors.open(allocator, file);
    defer buffer_store.deinit();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform();
    const devices = platform.getDevices();

    for (devices) |device| {
        std.debug.print("Device visible: {s}\n", .{device.getDescription(platform.pjrt_api).debugString(platform.pjrt_api)});
    }

    var buffers = try gpa.allocator().alloc(zml.Buffer, buffer_store.buffers.count());
    defer {
        for (buffers) |*buf| {
            buf.deinit();
        }
        gpa.allocator().free(buffers);
    }

    var total_bytes: usize = 0;
    var timer = try std.time.Timer.start();

    var it = buffer_store.buffers.iterator();
    var i: usize = 0;
    std.debug.print("\nStart to read {d} buffers from store..\n", .{buffer_store.buffers.count()});

    while (it.next()) |entry| : (i += 1) {
        const host_buffer = entry.value_ptr.*;
        total_bytes += host_buffer.data.len;
        std.debug.print("Buffer: {any} / {any}\n", .{ i + 1, buffer_store.buffers.count() });
        buffers[i] = try zml.Buffer.from(platform, host_buffer);
    }

    const stop = timer.read();
    const time_in_s = zml.meta.divFloat(f64, stop, std.time.ns_per_s);
    const mbs = zml.meta.divFloat(f64, total_bytes, 1024 * 1024);

    std.debug.print("\nLoading speed: {d:.2} MB/s\n\n", .{mbs / time_in_s});
}
