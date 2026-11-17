const std = @import("std");

pub const Target = enum { cuda, rocm, oneapi, neuron, tpu, furiosa, metal };

pub const Device = struct {
    target: Target,
    remote: bool = false,
    name: ?[]const u8 = null,
    driver_version: ?[]const u8 = null,
    details: []const Detail = &.{},

    pub const Detail = struct {
        key: []const u8,
        value: []const u8,
    };
};

/// Metric names carry the canonical units. Temperature is in degrees Celsius.
/// The enum order has no external meaning.
pub const Metric = enum {
    util_percent,
    encoder_util_percent,
    decoder_util_percent,
    power_mw,
    power_limit_mw,
    temperature,
    fan_speed_percent,
    clock_graphics_mhz,
    clock_sm_mhz,
    clock_soc_mhz,
    clock_mem_mhz,
    clock_graphics_max_mhz,
    clock_mem_max_mhz,
    mem_used_bytes,
    mem_total_bytes,
    mem_bus_width,
    pcie_tx_kbps,
    pcie_rx_kbps,
    pcie_bandwidth_mbps,
    pcie_link_gen,
    pcie_link_width,
    nc_tensors,
    nc_constants,
    nc_model_code,
    nc_shared_scratchpad,
    nc_nonshared_scratchpad,
    nc_runtime,
    nc_driver,
    nc_dma_rings,
    nc_collectives,
    nc_notifications,
    nc_uncategorized,
    ambient_temperature,
    governor,
    liveness,
    throttle_reasons,
    ane_power_mw,
};

pub const Metrics = struct {
    values: std.EnumMap(Metric, u64) = .empty,

    const Fields = std.enums.EnumFieldStruct(Metric, ?u64, @as(?u64, null));

    pub fn get(self: *const Metrics, metric: Metric) ?u64 {
        return self.values.get(metric);
    }

    pub fn put(self: *Metrics, metric: Metric, value: ?u64) void {
        if (value) |number| self.values.put(metric, number) else self.values.remove(metric);
    }

    pub fn jsonParse(
        allocator: std.mem.Allocator,
        source: anytype,
        options: std.json.ParseOptions,
    ) !Metrics {
        return .{ .values = .init(try std.json.innerParse(Fields, allocator, source, options)) };
    }

    pub fn jsonStringify(self: *const Metrics, output: *std.json.Stringify) !void {
        try output.beginObject();
        for (std.enums.values(Metric)) |metric| {
            const value = self.get(metric) orelse continue;
            try output.objectField(@tagName(metric));
            try output.write(value);
        }
        try output.endObject();
    }
};

pub fn Reply(comptime fields: []const Metric) type {
    var names: [fields.len][:0]const u8 = undefined;
    for (fields, &names) |field, *name| name.* = @tagName(field);
    const types: [fields.len]type = @splat(?u64);
    const attributes: [fields.len]std.builtin.Type.Struct.FieldAttributes = @splat(.{
        .default_value_ptr = &@as(?u64, null),
    });
    return @Struct(.auto, null, &names, &types, &attributes);
}

pub const ProcessInfo = struct {
    pid: u32 = 0,
    device_idx: u16 = 0,
    dev_util_percent: ?u16 = null,
    dev_mem_kib: ?u64 = null,
    uid: u32 = 0,
    username: []const u8 = "",
    comm: []const u8 = "",
    cpu_percent: u16 = 0,
    rss_kib: u64 = 0,
    remote: bool = false,
};

pub const HostData = struct {
    hostname: ?[]const u8 = null,
    kernel: ?[]const u8 = null,
    cpu_name: ?[]const u8 = null,
    cpu_cores: ?u64 = null,
    mem_total_kib: ?u64 = null,
    mem_available_kib: ?u64 = null,
    load_1: ?f32 = null,
    load_5: ?f32 = null,
    load_15: ?f32 = null,
    uptime_seconds: ?u64 = null,
    cpu_power_mw: ?u64 = null,
    system_power_mw: ?u64 = null,
};

pub const Governor = enum(u64) { performance, powersave };
pub const Liveness = enum(u64) { unknown, alive };
pub const ThrottleReason = enum(u6) {
    idle,
    thermal_slowdown,
    app_power_cap,
    app_clock_cap,
    hw_clock_cap,
    hw_bus_limit,
    hw_power_cap,
    gov_clock_cap,
    other,
};

pub const HistoryMetric = enum { util_percent, mem_used_bytes, mem_total_bytes, temperature, power_mw };
pub const HistoryMetrics = std.EnumArray(HistoryMetric, ?u64);

pub fn historyMetrics(sample: *const Metrics) HistoryMetrics {
    var values: HistoryMetrics = undefined;
    inline for (@typeInfo(HistoryMetric).@"enum".field_names) |field| {
        values.set(@field(HistoryMetric, field), sample.get(@field(Metric, field)));
    }
    return values;
}

pub const HistoryPoint = struct { at: std.Io.Timestamp, values: HistoryMetrics };

/// Captured samples, process strings, and history belong to the frame allocator.
/// Samples, process rows, and history use the indices of the fixed inventory.
pub const Frame = struct {
    samples: []const Metrics = &.{},
    processes: []const ProcessInfo = &.{},
    host: HostData = .{},
    history: ?History = null,

    pub const History = struct {
        points: []const []const HistoryPoint,
        captured_at: std.Io.Timestamp,
        interval_ms: u32,
        capacity: usize,
    };
};
