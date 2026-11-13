const builtin = @import("builtin");

const model = @import("model.zig");
pub const Target = model.Target;
pub const Device = model.Device;
pub const Sample = model.Sample;
pub const Row = model.Row;
pub const Gpu = model.Gpu;
pub const Neuron = model.Neuron;
pub const Tpu = model.Tpu;
pub const ProcessInfo = model.ProcessInfo;
pub const HostData = model.HostData;
pub const Platform = model.Platform;
pub const ProcessList = model.ProcessList;

pub const Collector = @import("collector.zig").Collector;
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

test {
    _ = @import("collector.zig");
    _ = counter;
    _ = @import("double_buffer.zig");
    _ = dynlib;
    _ = fdinfo;
    _ = hwmon;
    _ = pci;
    _ = proc;
    _ = @import("sandbox.zig");
    _ = sysfs;
    _ = units;
}
