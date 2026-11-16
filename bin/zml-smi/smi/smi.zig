//! The core of zml-smi: the model, the collector, the blocks platforms are made of, and the outputs.
const builtin = @import("builtin");
const std = @import("std");

const model = @import("model.zig");
pub const Target = model.Target;
pub const Device = model.Device;
pub const Sample = model.Sample;
pub const Common = model.Common;
pub const Gpu = model.Gpu;
pub const Neuron = model.Neuron;
pub const Tpu = model.Tpu;
pub const Furiosa = model.Furiosa;
pub const Metal = model.Metal;
pub const Row = model.Row;
pub const ProcessInfo = model.ProcessInfo;
pub const HostData = model.HostData;

const collector = @import("collector.zig");
pub const Collector = collector.Collector;
pub const Reading = collector.Reading;
pub const Remote = @import("remote.zig").Remote;
pub const Host = switch (builtin.os.tag) {
    .linux => @import("host/linux.zig").Host,
    .macos => @import("host/macos.zig").Host,
    else => @compileError("zml-smi runs on Linux and macOS"),
};

pub const sysfs = @import("sysfs.zig");
pub const hwmon = @import("hwmon.zig");
pub const pci = @import("pci.zig");
pub const fdinfo = @import("fdinfo.zig");
pub const proc = @import("proc.zig");
pub const dynlib = @import("dynlib.zig");
pub const units = @import("units.zig");
const counter = @import("counter.zig");
pub const Counter = counter.Counter;
pub const Counters = counter.Counters;
pub const Growth = counter.Growth;
pub const DoubleBuffer = @import("double_buffer.zig").DoubleBuffer;

pub const csv = @import("csv.zig");
pub const json = @import("json.zig");
pub const prometheus = @import("prometheus.zig");
pub const api = @import("api.zig");

const Serializer = *const fn (*std.Io.Writer, []const Device, Collector.Snapshot) std.Io.Writer.Error!void;

pub fn write(writer: *std.Io.Writer, source: *Collector, serialize: Serializer) !void {
    var frame: std.heap.ArenaAllocator = .init(source.gpa);
    defer frame.deinit();
    try serialize(writer, source.devices.items, try source.capture(frame.allocator(), false));
}
