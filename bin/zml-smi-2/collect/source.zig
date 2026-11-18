const std = @import("std");
const contract = @import("zml-smi-2/contract");

pub const HostPower = struct {
    cpu_power_mw: ?u64 = null,
    system_power_mw: ?u64 = null,
};

/// One backend poll fills its samples and may append processes or measure host power.
pub const Reading = struct {
    samples: []contract.Metrics,
    allocator: std.mem.Allocator,
    processes: std.ArrayList(contract.ProcessInfo) = .empty,
    host_power: HostPower = .{},

    /// Any strings in the row must live in the publication's allocator.
    pub fn addProcess(self: *Reading, row: contract.ProcessInfo) std.mem.Allocator.Error!void {
        try self.processes.append(self.allocator, row);
    }
};

pub const Source = struct {
    devices: []const contract.Device,
    two_readings: bool,
    context: *anyopaque,
    poll: *const fn (*anyopaque, std.mem.Allocator, *Reading) anyerror!void,
    close: *const fn (*anyopaque) void,

    /// The backend and its immutable device inventory must live through close.
    pub fn from(backend: anytype) Source {
        const Backend = @TypeOf(backend.*);
        const Adapter = struct {
            fn poll(context: *anyopaque, scratch: std.mem.Allocator, reading: *Reading) !void {
                const typed: *Backend = @ptrCast(@alignCast(context));
                return Backend.poll(typed, scratch, reading);
            }

            fn close(context: *anyopaque) void {
                if (@hasDecl(Backend, "close")) Backend.close(@ptrCast(@alignCast(context)));
            }
        };
        return .{
            .devices = backend.devices,
            .two_readings = @hasDecl(Backend, "two_readings") and Backend.two_readings,
            .context = backend,
            .poll = Adapter.poll,
            .close = Adapter.close,
        };
    }
};
