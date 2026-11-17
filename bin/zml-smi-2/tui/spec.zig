//! Static device presentation in the same order as the sampled devices.
const contract = @import("zml-smi-2/contract");

pub const DeviceView = struct {
    title: []const u8,
    heading: []const u8,
    remote: bool,
    profile: *const Profile,
    logo: ?usize = null,
};

pub const Profile = struct {
    util_label: []const u8,
    util_title: []const u8,
    memory_prefix: []const u8,
    environment: enum { whole_aligned, tenths, hidden } = .whole_aligned,
    memory_suffix: union(enum) { none, literal: []const u8, bus_width: contract.Metric } = .none,
    util_note: ?[2]Note = null,
    temperature_note: ?Note = null,
    power_note: ?Note = null,
    sections: []const Section = &.{},
};

pub const Note = struct {
    label: []const u8,
    metric: contract.Metric,
    suffix: []const u8 = "",
};

pub const Section = struct { title: []const u8, rows: []const Row };
pub const Named = struct { metric: contract.Metric, labels: []const [:0]const u8 };

pub const Row = struct {
    label: []const u8,
    value: Value,

    pub const Value = union(enum) {
        clock_pair: [2]contract.Metric,
        unit: struct { metric: contract.Metric, suffix: []const u8 },
        pcie_link: [2]contract.Metric,
        traffic: [2]contract.Metric,
        watts: contract.Metric,
        host_watts: enum { cpu_power_mw, system_power_mw },
        named: Named,
        flags: Named,
        memory: contract.Metric,
    };
};
