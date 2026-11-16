const std = @import("std");
const smi = @import("smi.zig");

const Remote = smi.Remote;

test "remote transactions map indices, own strings, and reject inventory changes" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    var remote: Remote = .{
        .devices = &.{ .{ .target = .cuda, .name = "first" }, .{ .target = .cuda, .name = "second" } },
        .io = std.testing.io,
        .url = "http://peer/metrics",
        .positions = &.{ null, 0, 1 },
        .timeout = .fromSeconds(1),
        .warned = true,
    };
    var samples = [_]smi.Sample{ .init(.cuda), .init(.cuda) };
    var owned: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer owned.deinit();
    var reading: smi.Reading = .{ .samples = &samples, .allocator = owned.allocator() };
    const response = try std.json.parseFromSliceLeaky(std.json.Value, scratch.allocator(),
        \\{"devices":[
        \\ {"device":{"target":"unknown"}},
        \\ {"device":{"target":"cuda","name":"first"},
        \\  "sample":{"cuda":{"util_percent":41,"pcie_tx_kbps":3}}},
        \\ {"device":{"target":"cuda","name":"second"},
        \\  "sample":{"cuda":{"util_percent":0,"mem_used_bytes":null}}}
        \\],"processes":[
        \\ {"pid":7,"device_idx":0,"username":"ignored"},
        \\ {"pid":42,"device_idx":1,"username":"alice","comm":"worker"}
        \\]}
    , .{});
    try Remote.testing.readResponse(&remote, scratch.allocator(), response, &reading);
    _ = scratch.reset(.free_all);
    try std.testing.expectEqual(41, samples[0].cuda.util_percent);
    try std.testing.expectEqual(3, samples[0].cuda.pcie_tx_kbps);
    try std.testing.expectEqual(0, samples[1].cuda.util_percent);
    try std.testing.expectEqual(null, samples[1].cuda.mem_used_bytes);
    try std.testing.expectEqual(1, reading.processes.items.len);
    const process = reading.processes.items[0];
    try std.testing.expectEqual(0, process.device_idx);
    try std.testing.expectEqual(42, process.pid);
    try std.testing.expect(process.remote);
    try std.testing.expectEqualStrings("alice", process.username);
    try std.testing.expectEqualStrings("worker", process.comm);

    reading.processes.clearRetainingCapacity();
    const changed = try std.json.parseFromSliceLeaky(std.json.Value, scratch.allocator(),
        \\{"devices":[
        \\ {"device":{"target":"unknown"}},
        \\ {"device":{"target":"cuda","name":"first"},"sample":{"cuda":{"util_percent":99}}},
        \\ {"device":{"target":"cuda","name":"replacement"},"sample":{"cuda":{}}}
        \\]}
    , .{});
    try std.testing.expectError(error.Unavailable, Remote.testing.readResponse(
        &remote,
        scratch.allocator(),
        changed,
        &reading,
    ));
    for (samples) |sample| {
        try std.testing.expectEqual(null, sample.cuda.util_percent);
        try std.testing.expectEqual(null, sample.cuda.pcie_tx_kbps);
    }
}
