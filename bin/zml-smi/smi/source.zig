const std = @import("std");
const model = @import("zml-smi/model");

pub const HostPower = struct {
    cpu_power_mw: ?u64 = null,
    system_power_mw: ?u64 = null,
};

/// One backend poll fills its samples and may append processes or measure host power.
pub const Reading = struct {
    samples: []model.Sample,
    allocator: std.mem.Allocator,
    processes: std.ArrayList(model.ProcessInfo) = .empty,
    host_power: HostPower = .{},

    /// Any strings in the row must live in the publication's allocator.
    pub fn addProcess(self: *Reading, row: model.ProcessInfo) std.mem.Allocator.Error!void {
        try self.processes.append(self.allocator, row);
    }
};

pub const Source = struct {
    devices: []const model.Device,
    two_readings: bool,
    context: *anyopaque,
    poll: *const fn (*anyopaque, std.mem.Allocator, *Reading) anyerror!void,
    close: *const fn (*anyopaque) void,

    /// The backend and its immutable device inventory must live through close.
    pub fn from(backend: anytype) Source {
        const Backend = @TypeOf(backend.*);
        const Adapter = struct {
            fn poll(context: *anyopaque, scratch: std.mem.Allocator, reading: *Reading) !void {
                const instance: *Backend = @ptrCast(@alignCast(context));
                return instance.poll(scratch, reading);
            }

            fn close(context: *anyopaque) void {
                const instance: *Backend = @ptrCast(@alignCast(context));
                if (@hasDecl(Backend, "close")) instance.close();
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

pub const HostReader = struct {
    context: *anyopaque,
    read: *const fn (*anyopaque, std.mem.Allocator, []model.ProcessInfo, *model.HostData) void,
    close: *const fn (*anyopaque) void,

    /// The host backend must live through close.
    pub fn from(host: anytype) HostReader {
        const Backend = @TypeOf(host.*);
        const Adapter = struct {
            fn read(
                context: *anyopaque,
                arena: std.mem.Allocator,
                rows: []model.ProcessInfo,
                data: *model.HostData,
            ) void {
                const instance: *Backend = @ptrCast(@alignCast(context));
                instance.enrich(arena, rows);
                instance.sample(data);
            }

            fn close(context: *anyopaque) void {
                const instance: *Backend = @ptrCast(@alignCast(context));
                instance.deinit();
            }
        };
        return .{ .context = host, .read = Adapter.read, .close = Adapter.close };
    }
};
