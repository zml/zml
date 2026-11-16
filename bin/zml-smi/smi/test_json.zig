const std = @import("std");
const smi = @import("smi.zig");
const json = @import("json.zig");

test "snapshot JSON preserves target shape, nulls, zeroes, and process fields" {
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const devices = [_]smi.Device{.{ .target = .tpu, .remote = true, .name = "TPU" }};
    const samples = [_]smi.Sample{.{ .tpu = .{
        .util_percent = 0,
        .mem_total_bytes = 4096,
    } }};
    const processes = [_]smi.ProcessInfo{.{
        .pid = 42,
        .dev_util_percent = 0,
        .username = "alice",
        .comm = "worker",
        .remote = true,
    }};

    try json.write(&output.writer, &devices, .{
        .samples = &samples,
        .processes = &processes,
        .host = .{},
        .histories = &.{},
    });
    try std.testing.expectEqualStrings(
        "{\"devices\":[{\"device\":{\"target\":\"tpu\",\"remote\":true,\"name\":\"TPU\"," ++
            "\"driver_version\":null,\"details\":[]},\"sample\":{\"tpu\":{\"util_percent\":0," ++
            "\"mem_used_bytes\":null,\"mem_total_bytes\":4096}}}],\"processes\":[{\"pid\":42," ++
            "\"device_idx\":0,\"dev_util_percent\":0,\"dev_mem_kib\":null,\"uid\":0," ++
            "\"username\":\"alice\",\"comm\":\"worker\",\"cpu_percent\":0,\"rss_kib\":0," ++
            "\"remote\":true}]}\n",
        output.written(),
    );
}
