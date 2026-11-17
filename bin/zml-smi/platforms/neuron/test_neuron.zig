const std = @import("std");
const c = @import("c");
const model = @import("zml-smi/model");
const source = @import("zml-smi/source");
const native = @import("zml-smi/native");
const Neuron = @import("Neuron.zig");

const Api = Neuron.testing.Api;
const findPrivate = Neuron.testing.findPrivate;

test findPrivate {
    const io = std.testing.io;
    const api = try native.dynlib.openBundled(Api, io, "libnrt.so.1");
    const nrt = try findPrivate(io, api.nrt_get_version.function);
    // Listing devices works without any, so the call checks the address.
    var indices: [c.MAX_NEURON_DEVICE_COUNT]c_int = undefined;
    try std.testing.expect(nrt.ndl_available_devices(&indices, indices.len) >= 0);
}

test "utilization and process rows" {
    const fake = struct {
        var busy_us: u64 = 1000;

        fn apps(device: [*c]c.ndl_device_t, info: [*c][*c]c.struct_neuron_app_info, count: [*c]usize, _: u16) callconv(.c) c_int {
            if (@intFromPtr(device) == 0x1000) return -1;
            const list: [*]c.struct_neuron_app_info = @ptrCast(@alignCast(std.c.malloc(3 * @sizeOf(c.struct_neuron_app_info)).?));
            // pid 0 is skipped, and pid 8 holds less than 1 KiB.
            for (list[0..3], [_]i32{ 0, 7, 8 }, [_]usize{ 1 << 20, 1 << 20, 512 }) |*app, pid, bytes| {
                app.* = std.mem.zeroes(c.struct_neuron_app_info);
                app.pid = pid;
                app.device_mem_size = bytes;
            }
            info.* = list;
            count.* = 3;
            return 0;
        }

        fn open(_: [*c]c.ndl_device_t, pid: c.pid_t, instance: [*c]?*c.nds_instance_t) callconv(.c) c_int {
            instance.* = @ptrFromInt(@as(usize, @intCast(pid)));
            return 0;
        }

        fn close(_: ?*c.nds_instance_t) callconv(.c) c_int {
            return 0;
        }

        /// Every process is busy on the first core of a chip.
        fn counter(_: ?*c.nds_instance_t, core: c_int, _: u32, value: [*c]u64) callconv(.c) c_int {
            value.* = if (core == 0) busy_us else 0;
            return 0;
        }
    };

    const devices: [4]model.Device = @splat(.{ .target = .neuron });
    var neuron: Neuron = .{
        .devices = &devices,
        .io = std.testing.io,
        .nrt = .{
            .ndl_available_devices = undefined,
            .ndl_open_device = undefined,
            .ndl_get_all_apps_info = .{ .function = &fake.apps },
            .nds_open = .{ .function = &fake.open },
            .nds_close = &fake.close,
            .nds_get_nc_counter = .{ .function = &fake.counter },
        },
        // The first chip cannot be read.
        .chips = &.{
            .{ .handle = @ptrFromInt(0x1000), .mem_total_bytes = null, .memory = &.{ null, null } },
            .{ .handle = @ptrFromInt(0x2000), .mem_total_bytes = null, .memory = &.{ null, null } },
        },
        .busy = .init(std.testing.allocator),
    };
    defer neuron.busy.deinit();
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    var samples: [4]model.Sample = @splat(.init(.neuron));
    var reading: source.Reading = .{ .samples = &samples, .allocator = std.testing.allocator };
    defer reading.processes.deinit(std.testing.allocator);

    // The first reading has no growth; the second grows by far more than the time between them.
    for ([_]u8{ 0, 100 }) |util| {
        @memset(&samples, .init(.neuron));
        reading.processes.clearRetainingCapacity();
        try neuron.poll(scratch.allocator(), &reading);

        try std.testing.expectEqual(null, samples[0].neuron.util_percent);
        try std.testing.expectEqual(util, samples[2].neuron.util_percent);
        try std.testing.expectEqual(0, samples[3].neuron.util_percent);
        const row: model.ProcessInfo = .{
            .pid = 7,
            .device_idx = 2,
            .dev_util_percent = util,
            .dev_mem_kib = 1024,
        };
        try std.testing.expectEqualDeep(
            @as([]const model.ProcessInfo, &.{row}),
            reading.processes.items,
        );
        fake.busy_us += 1_000_000_000;
    }
}
