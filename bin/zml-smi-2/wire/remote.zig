const std = @import("std");
const contract = @import("zml-smi-2/contract");
const Snapshot = @import("zml-smi-2/snapshot").Snapshot;
const Reading = @import("zml-smi-2/source").Reading;

const log = std.log.scoped(.@"zml-smi-2/remote");

/// The fixed device inventory of a peer, read from its snapshot endpoint.
pub const Remote = struct {
    devices: []const contract.Device,
    io: std.Io,
    url: []const u8,
    timeout: std.Io.Duration,
    warned: bool = false,

    /// Fetches the devices once. Later fetches time out after at least one second.
    pub fn open(
        arena: std.mem.Allocator,
        gpa: std.mem.Allocator,
        io: std.Io,
        host: []const u8,
        interval: std.Io.Duration,
    ) !Remote {
        const startsWith = std.mem.startsWith;
        if (!startsWith(u8, host, "http://") and !startsWith(u8, host, "https://"))
            return error.InvalidUrl;
        const url = try std.fmt.allocPrint(arena, "{s}/metrics", .{host});
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const snapshot = try fetch(scratch.allocator(), io, url, .fromSeconds(3));
        return .{
            .devices = try ownDevices(arena, snapshot.devices),
            .io = io,
            .url = url,
            .timeout = .fromNanoseconds(@max(interval.nanoseconds, std.time.ns_per_s)),
        };
    }

    pub fn poll(self: *Remote, scratch: std.mem.Allocator, reading: *Reading) !void {
        try self.readResponse(try fetch(scratch, self.io, self.url, self.timeout), reading);
    }

    pub const testing = if (@import("builtin").is_test) struct {
        pub const readResponse = Remote.readResponse;
        pub const ownInventory = ownDevices;
    } else struct {};

    fn readResponse(self: *Remote, snapshot: Snapshot, reading: *Reading) !void {
        if (snapshot.devices.len != self.devices.len) return self.changed();
        for (snapshot.devices, self.devices) |received, expected| {
            if (received.target != expected.target or
                !std.mem.eql(u8, received.name orelse "", expected.name orelse ""))
            {
                return self.changed();
            }
        }
        @memcpy(reading.samples, snapshot.samples);
        for (snapshot.processes) |row| {
            if (row.device_idx >= self.devices.len) continue;
            var copy = row;
            copy.remote = true;
            copy.username = try reading.allocator.dupe(u8, row.username);
            copy.comm = try reading.allocator.dupe(u8, row.comm);
            try reading.addProcess(copy);
        }
    }

    fn changed(self: *Remote) error{Unavailable} {
        if (!self.warned) log.warn("{s}: the devices changed while monitoring", .{self.url});
        self.warned = true;
        return error.Unavailable;
    }
};

/// Retains only inventory metadata, independently of the initial snapshot's arena.
fn ownDevices(arena: std.mem.Allocator, inventory: []const contract.Device) ![]const contract.Device {
    const devices = try arena.dupe(contract.Device, inventory);
    for (devices) |*device| {
        device.remote = true;
        if (device.name) |name| device.name = try arena.dupe(u8, name);
        if (device.driver_version) |version| device.driver_version = try arena.dupe(u8, version);
        const details = try arena.dupe(contract.Device.Detail, device.details);
        for (details) |*detail| {
            detail.key = try arena.dupe(u8, detail.key);
            detail.value = try arena.dupe(u8, detail.value);
        }
        device.details = details;
    }
    return devices;
}

/// GETs `url` and parses its body into `scratch`, giving up after `timeout`.
/// std.http has no timeout, so the fetch races a timer and the loser is canceled.
fn fetch(
    scratch: std.mem.Allocator,
    io: std.Io,
    url: []const u8,
    timeout: std.Io.Duration,
) !Snapshot {
    const Race = union(enum) { body: anyerror![]const u8, timer: std.Io.Cancelable!void };
    var results: [2]Race = undefined;
    var race: std.Io.Select(Race) = .init(io, &results);
    defer race.cancelDiscard();

    try race.concurrent(.body, get, .{ scratch, io, url });
    try race.concurrent(.timer, std.Io.sleep, .{ io, timeout, .awake });
    const body = switch (try race.await()) {
        .body => |body| try body,
        .timer => return error.Timeout,
    };
    return Snapshot.parse(scratch, body);
}

fn get(scratch: std.mem.Allocator, io: std.Io, url: []const u8) ![]const u8 {
    var client: std.http.Client = .{ .allocator = scratch, .io = io };
    defer client.deinit();

    var body: std.Io.Writer.Allocating = .init(scratch);
    const result = try client.fetch(.{
        .location = .{ .url = url },
        .response_writer = &body.writer,
    });
    if (result.status != .ok) return error.HttpError;
    return body.written();
}
