const std = @import("std");
const contract = @import("zml-smi-2/contract");
const Snapshot = @import("zml-smi-2/snapshot").Snapshot;
const Reading = @import("zml-smi-2/source").Reading;

const Remote = @import("remote.zig").Remote;

test "peer samples and processes use inventory order" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    var remote: Remote = .{
        .devices = &.{ .{ .target = .cuda, .name = "first" }, .{ .target = .metal, .name = "second" } },
        .io = std.testing.io,
        .url = "http://peer/metrics",
        .timeout = .fromSeconds(1),
        .warned = true,
    };

    var samples = [_]contract.Metrics{ .{}, .{} };
    var owned: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer owned.deinit();
    var reading: Reading = .{ .samples = &samples, .allocator = owned.allocator() };
    const response = try parsedBody(scratch.allocator(),
        \\{"devices":[{"target":"cuda","name":"first"},{"target":"metal","name":"second"}],
        \\ "samples":[{"util_percent":41,"pcie_tx_kbps":3},{"util_percent":0,"mem_used_bytes":null}],
        \\ "processes":[{"pid":7,"device_idx":2,"username":"ignored"},
        \\              {"pid":42,"device_idx":1,"username":"alice","comm":"worker"}]}
    );
    try Remote.testing.readResponse(&remote, response, &reading);
    _ = scratch.reset(.free_all);
    try std.testing.expectEqual(41, samples[0].get(.util_percent));
    try std.testing.expectEqual(3, samples[0].get(.pcie_tx_kbps));
    try std.testing.expectEqual(0, samples[1].get(.util_percent));
    try std.testing.expectEqual(null, samples[1].get(.mem_used_bytes));
    try std.testing.expectEqual(1, reading.processes.items.len);
    const process = reading.processes.items[0];
    try std.testing.expectEqual(1, process.device_idx);
    try std.testing.expectEqual(42, process.pid);
    try std.testing.expect(process.remote);
    try std.testing.expectEqualStrings("alice", process.username);
    try std.testing.expectEqualStrings("worker", process.comm);

    reading.processes.clearRetainingCapacity();
    const changed = try parsedBody(scratch.allocator(),
        \\{"devices":[{"target":"cuda","name":"first"},{"target":"metal","name":"replacement"}],
        \\ "samples":[{"util_percent":99},{}],"processes":[{"pid":18,"device_idx":0}]}
    );
    try std.testing.expectError(error.Unavailable, Remote.testing.readResponse(
        &remote,
        changed,
        &reading,
    ));
    try std.testing.expectEqual(41, samples[0].get(.util_percent));
    try std.testing.expectEqual(3, samples[0].get(.pcie_tx_kbps));
    try std.testing.expectEqual(0, samples[1].get(.util_percent));
    try std.testing.expectEqual(0, reading.processes.items.len);
}

test "peer device inventory owns metadata after the initial response" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    var owned: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer owned.deinit();

    const response = try parsedBody(scratch.allocator(),
        \\{"devices":[{"target":"tpu","name":"remote TPU","driver_version":"1.9",
        \\  "details":[{"key":"peer_version","value":"2.1"}]}],"samples":[{}]}
    );
    const devices = try Remote.testing.ownInventory(owned.allocator(), response.devices);
    _ = scratch.reset(.free_all);

    try std.testing.expectEqual(1, devices.len);
    const device = devices[0];
    try std.testing.expectEqual(.tpu, device.target);
    try std.testing.expect(device.remote);
    try std.testing.expectEqualStrings("remote TPU", device.name.?);
    try std.testing.expectEqualStrings("1.9", device.driver_version.?);
    try std.testing.expectEqual(1, device.details.len);
    try std.testing.expectEqualStrings("peer_version", device.details[0].key);
    try std.testing.expectEqualStrings("2.1", device.details[0].value);
}

test "peer snapshot rejects incompatible inventory and sample lengths" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    try std.testing.expectError(error.InvalidSnapshot, parsedBody(scratch.allocator(),
        \\{"devices":[{"target":"metal"}],"samples":[]}
    ));
}

fn parsedBody(scratch: std.mem.Allocator, body: []const u8) !Snapshot {
    return Snapshot.parse(scratch, try scratch.dupe(u8, body));
}
