const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");
const asynk = @import("async");

const asyncc = asynk.asyncc;

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
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
        std.debug.print("Try: bazel run -c opt //xferman:mmap -- /path/to/mymodel.safetensors or /path/to/model.safetensors.index.json \n", .{});
        std.process.exit(0);
    };

    var mapped_file = try zml.aio.MemoryMappedFile.init(try asynk.File.open(file, .{}));
    errdefer mapped_file.deinit();

    var buffer_store = try zml.aio.BufferStore.init(allocator, &.{mapped_file});
    defer buffer_store.deinit();

    const NUM_BUFFERS: usize = 146;
    const BUF_SIZE: usize = mapped_file.data.len / NUM_BUFFERS;
    const shape = zml.Shape.init(.{BUF_SIZE}, .u8);
    var free_list = std.ArrayList([]const u8).init(allocator);
    defer {
        for (free_list.items) |key| {
            allocator.free(key);
        }
        free_list.deinit();
    }
    var offset: usize = 0;
    for (0..NUM_BUFFERS) |buffer_index| {
        const filedata = mapped_file.mappedSlice(offset, BUF_SIZE);
        const key = try std.fmt.allocPrint(allocator, "buffer-{d}", .{buffer_index});
        try free_list.append(key);
        try buffer_store.registerBuffer(buffer_store.arena.allocator(), key, shape, filedata);
        offset += BUF_SIZE;
    }

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    var total_bytes: usize = 0;
    var timer = try std.time.Timer.start();

    var bit = buffer_store.buffers.iterator();
    while (bit.next()) |item| {
        const buffer = item.value_ptr;
        const key = item.key_ptr.*;
        std.log.info("Buffer {d} : {s} {} = {d} bytes @ {*}", .{ bit.index, key, buffer.shape, buffer.data.len, buffer.data.ptr });
    }

    const events = try buffer_store.starTransferToDevice(platform, .unpinned_host);
    std.debug.print("Received {d} events\n", .{events.len});

    var it = buffer_store.buffers.iterator();
    var i: usize = 0;
    std.debug.print("\nStart to read {d} buffers from store..\n", .{buffer_store.buffers.count()});

    while (it.next()) |entry| : (i += 1) {
        total_bytes += entry.value_ptr.*.data.len;
        std.debug.print("Buffer: {s} ({any} / {any})\n", .{ entry.key_ptr.*, i + 1, buffer_store.buffers.count() });
    }

    const stop = timer.read();
    const time_in_s = stdx.math.divFloat(f64, stop, std.time.ns_per_s);
    const mbs = stdx.math.divFloat(f64, total_bytes, 1024 * 1024);

    std.debug.print("\nLoading speed: {d:.2} MB/s\n\n", .{mbs / time_in_s});
}
