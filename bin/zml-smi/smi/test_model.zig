const std = @import("std");
const model = @import("model.zig");

test "every target preserves nulls, zeroes and all values through JSON" {
    inline for (std.enums.values(model.Target)) |target| {
        var sample: model.Sample = .init(target);
        try expectJsonRoundTrip(sample);
        const values = &@field(sample, @tagName(target));
        inline for (@typeInfo(@TypeOf(values.*)).@"struct".field_names, 0..) |field, index| {
            @field(values.*, field) = index * 101;
        }
        try expectJsonRoundTrip(sample);
    }
    try expectJsonRoundTrip(.{ .furiosa = .{ .governor = 99, .throttle_reasons = 1 << 63 } });
}

fn expectJsonRoundTrip(sample: model.Sample) !void {
    const encoded = try std.json.Stringify.valueAlloc(std.testing.allocator, sample, .{});
    defer std.testing.allocator.free(encoded);
    const decoded = try std.json.parseFromSlice(model.Sample, std.testing.allocator, encoded, .{});
    defer decoded.deinit();
    try std.testing.expectEqualDeep(sample, decoded.value);
}

test "common projection preserves zero, units and unsupported fields" {
    const gpu: model.Sample = .{ .cuda = .{ .util_percent = 0, .power_mw = 125_500 } };
    try std.testing.expectEqualDeep(model.Common{ .util_percent = 0, .power_mw = 125_500 }, gpu.common());
    const chip: model.Sample = .{ .tpu = .{ .mem_used_bytes = 0, .mem_total_bytes = 4096 } };
    try std.testing.expectEqualDeep(model.Common{ .mem_used_bytes = 0, .mem_total_bytes = 4096 }, chip.common());
}

test "wire records retain historical field order and raw optional values" {
    try expectFields(model.Gpu, "util_percent,encoder_util_percent,decoder_util_percent,power_mw,power_limit_mw," ++
        "temperature,fan_speed_percent,clock_graphics_mhz,clock_sm_mhz,clock_soc_mhz," ++
        "clock_mem_mhz,clock_graphics_max_mhz,clock_mem_max_mhz,mem_used_bytes," ++
        "mem_total_bytes,mem_bus_width,pcie_tx_kbps,pcie_rx_kbps,pcie_bandwidth_mbps," ++
        "pcie_link_gen,pcie_link_width");
    try expectFields(model.Neuron, "util_percent,mem_used_bytes,mem_total_bytes,nc_tensors,nc_constants,nc_model_code," ++
        "nc_shared_scratchpad,nc_nonshared_scratchpad,nc_runtime,nc_driver,nc_dma_rings," ++
        "nc_collectives,nc_notifications,nc_uncategorized");
    try expectFields(model.Tpu, "util_percent,mem_used_bytes,mem_total_bytes");
    try expectFields(model.Furiosa, "util_percent,mem_used_bytes,mem_total_bytes,power_mw,power_limit_mw,temperature," ++
        "ambient_temperature,governor,liveness,throttle_reasons");
    try expectFields(model.Metal, "util_percent,mem_used_bytes,mem_total_bytes,power_mw,temperature,fan_speed_percent," ++
        "clock_graphics_mhz,clock_graphics_max_mhz,ane_power_mw");

    var sample: model.Sample = .init(.cuda);
    sample.cuda.power_mw = 125_500;
    const flat = sample.cuda;
    try std.testing.expectEqual(@as(?u64, 125_500), flat.power_mw);
    try std.testing.expectEqual(@as(?u64, null), flat.temperature);
}

fn expectFields(comptime Record: type, expected: []const u8) !void {
    var names = std.mem.splitScalar(u8, expected, ',');
    for (@typeInfo(Record).@"struct".field_names) |name| {
        try std.testing.expectEqualStrings(names.next() orelse return error.MissingField, name);
    }
    try std.testing.expect(names.next() == null);
}

test "poll reads each query once and fills only its declared fields" {
    var reader: PollReader = .{};
    var sample: model.Gpu = .{ .temperature = 70, .pcie_tx_kbps = 1000 };
    sample.poll(.{&reader}, .{
        .read = PollReader.readScalar,
        .fields = .{ .temperature = .sensor },
        .query = PollReader.readMemory,
    });
    try std.testing.expectEqual(1, reader.memory_reads);
    try std.testing.expectEqualDeep(model.Gpu{
        .pcie_tx_kbps = 1000,
        .mem_used_bytes = 4096,
        .mem_total_bytes = 8192,
    }, sample);
}

const PollReader = struct {
    memory_reads: usize = 0,

    fn readScalar(_: *PollReader, comptime _: anytype) ?u64 {
        return null;
    }

    fn readMemory(self: *PollReader) struct { mem_used_bytes: u64, mem_total_bytes: u64 } {
        self.memory_reads += 1;
        return .{ .mem_used_bytes = 4096, .mem_total_bytes = 8192 };
    }
};
