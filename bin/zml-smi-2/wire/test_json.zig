const std = @import("std");
const contract = @import("zml-smi-2/contract");
const Snapshot = @import("zml-smi-2/snapshot").Snapshot;
const json = @import("json.zig");

test "snapshot JSON writes flat samples and omits unavailable metrics" {
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const devices = [_]contract.Device{.{ .target = .tpu, .remote = true, .name = "TPU" }};
    var samples = [_]contract.Metrics{.{}};
    samples[0].put(.util_percent, 0);
    samples[0].put(.mem_total_bytes, 4096);
    const processes = [_]contract.ProcessInfo{.{
        .pid = 42,
        .dev_util_percent = 0,
        .username = "alice",
        .comm = "worker",
        .remote = true,
    }};

    try json.write(&output.writer, .{
        .devices = &devices,
        .samples = &samples,
        .processes = &processes,
        .host = .{ .hostname = "node" },
    });
    try std.testing.expectEqualStrings(
        "{\"devices\":[{\"target\":\"tpu\",\"remote\":true,\"name\":\"TPU\",\"details\":[]}]," ++
            "\"samples\":[{\"util_percent\":0,\"mem_total_bytes\":4096}],\"processes\":[{\"pid\":42," ++
            "\"device_idx\":0,\"dev_util_percent\":0,\"uid\":0," ++
            "\"username\":\"alice\",\"comm\":\"worker\",\"cpu_percent\":0,\"rss_kib\":0," ++
            "\"remote\":true}],\"host\":{\"hostname\":\"node\"}}\n",
        output.written(),
    );
}

test "native snapshot round trips all targets and exact integer readings" {
    const targets = std.enums.values(contract.Target);
    var devices: [targets.len]contract.Device = undefined;
    var samples: [targets.len]contract.Metrics = @splat(.{});
    for (targets, &devices, &samples) |target, *device, *sample| {
        device.* = .{ .target = target, .details = &.{.{ .key = "serial", .value = "S1" }} };
        sample.put(.util_percent, 0);
        sample.put(.mem_used_bytes, std.math.maxInt(u64));
    }

    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try json.write(&output.writer, .{ .devices = &devices, .samples = &samples });
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const parsed = try Snapshot.parse(arena.allocator(), output.written());
    try std.testing.expectEqual(devices.len, parsed.devices.len);
    for (devices, parsed.devices, parsed.samples) |device, received, sample| {
        try std.testing.expectEqual(device.target, received.target);
        try std.testing.expectEqualStrings("S1", received.details[0].value);
        try std.testing.expectEqual(0, sample.get(.util_percent));
        try std.testing.expectEqual(std.math.maxInt(u64), sample.get(.mem_used_bytes));
        try std.testing.expectEqual(null, sample.get(.power_mw));
    }
}

test "native snapshot parser checks indices and accepts additional metric keys" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const snapshot = try Snapshot.parse(allocator,
        \\{"devices":[{"target":"cuda"}],"samples":[{"power_mw":1250,"new_field":99}]}
    );
    try std.testing.expectEqual(1250, snapshot.samples[0].get(.power_mw));
    try std.testing.expectError(
        error.InvalidSnapshot,
        Snapshot.parse(allocator, "{\"devices\":[],\"samples\":[{}]}"),
    );
    try std.testing.expectError(
        error.MissingField,
        Snapshot.parse(allocator, "{\"devices\":[],\"processes\":[]}"),
    );
}
