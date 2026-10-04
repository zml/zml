const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const ProcessInfo = smi_info.process_info.ProcessInfo;

pub fn write(writer: *std.Io.Writer, devices: []const *DeviceInfo, processes: []const ProcessInfo, host: ?*const smi_info.host_info.HostInfo) !void {
    var jw: std.json.Stringify = .{ .writer = writer };
    try jw.write(Response{ .devices = devices, .processes = processes, .host = host });
    try writer.writeAll("\n");
}

const Response = struct {
    host: ?*const smi_info.host_info.HostInfo,
    devices: []const *DeviceInfo,
    processes: []const ProcessInfo,
};
