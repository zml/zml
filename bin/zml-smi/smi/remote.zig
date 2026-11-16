const std = @import("std");
const smi = @import("smi.zig");

const log = std.log.scoped(.@"zml-smi/remote");

/// The fixed device inventory of another zml-smi, read from its JSON API.
pub const Remote = struct {
    devices: []const smi.Device,
    io: std.Io,
    url: []const u8,
    /// Peer row to local device, or null when this build does not know its target.
    positions: []const ?u16,
    timeout: std.Io.Duration,
    warned: bool = false,

    /// Fetches the devices once. Later fetches time out after at least one second.
    pub fn open(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, host: []const u8, interval: std.Io.Duration) !Remote {
        if (!std.mem.startsWith(u8, host, "http://") and !std.mem.startsWith(u8, host, "https://")) {
            return error.InvalidUrl;
        }
        const url = try std.fmt.allocPrint(arena, "{s}/metrics", .{host});
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const rows = items(try fetch(scratch.allocator(), io, url, .fromSeconds(3)), "devices");
        var devices: std.ArrayList(smi.Device) = try .initCapacity(arena, rows.len);
        const positions = try arena.alloc(?u16, rows.len);
        @memset(positions, null);
        for (rows, positions) |item, *position| {
            // FromValue owns the metadata strings in arena, independently of the response.
            var row = std.json.parseFromValueLeaky(smi.Row, arena, item, .{
                .ignore_unknown_fields = true,
            }) catch |err| {
                if (err == error.OutOfMemory) return err;
                continue;
            };
            if (std.meta.activeTag(row.sample) != row.device.target) continue;
            position.* = std.math.cast(u16, devices.items.len) orelse return error.TooManyDevices;
            row.device.remote = true;
            devices.appendAssumeCapacity(row.device);
        }
        return .{
            .devices = devices.items,
            .io = io,
            .url = url,
            .positions = positions,
            .timeout = if (interval.nanoseconds < std.time.ns_per_s) .fromSeconds(1) else interval,
        };
    }

    pub fn poll(self: *Remote, scratch: std.mem.Allocator, reading: *smi.Reading) !void {
        try self.readResponse(scratch, try fetch(scratch, self.io, self.url, self.timeout), reading);
    }

    pub const testing = if (@import("builtin").is_test) struct {
        pub const readResponse = Remote.readResponse;
    } else struct {};

    fn readResponse(
        self: *Remote,
        scratch: std.mem.Allocator,
        root: std.json.Value,
        reading: *smi.Reading,
    ) !void {
        {
            // Nothing is published during this transaction. Roll back partial inventory reads.
            errdefer for (reading.samples, self.devices) |*sample, device| {
                sample.* = .init(device.target);
            };
            const rows = items(root, "devices");
            if (rows.len != self.positions.len) return self.changed();
            for (rows, self.positions) |item, position| {
                const index = position orelse continue;
                const row = std.json.parseFromValueLeaky(smi.Row, scratch, item, .{
                    .ignore_unknown_fields = true,
                }) catch |err| return if (err == error.OutOfMemory) err else self.changed();
                const device = self.devices[index];
                if (row.device.target != device.target or
                    std.meta.activeTag(row.sample) != device.target or
                    !std.mem.eql(u8, row.device.name orelse "", device.name orelse ""))
                {
                    return self.changed();
                }
                reading.samples[index] = row.sample;
            }
        }
        for (items(root, "processes")) |item| {
            var row = std.json.parseFromValueLeaky(smi.ProcessInfo, reading.allocator, item, .{
                .ignore_unknown_fields = true,
            }) catch |err| {
                if (err == error.OutOfMemory) return err;
                continue;
            };
            if (row.device_idx >= self.positions.len) continue;
            row.device_idx = self.positions[row.device_idx] orelse continue;
            row.remote = true;
            try reading.addProcess(row);
        }
    }

    fn changed(self: *Remote) error{Unavailable} {
        if (!self.warned) log.warn("{s}: the devices changed since zml-smi started", .{self.url});
        self.warned = true;
        return error.Unavailable;
    }
};

fn items(root: std.json.Value, key: []const u8) []const std.json.Value {
    if (root != .object) return &.{};
    const value = root.object.get(key) orelse return &.{};
    return if (value == .array) value.array.items else &.{};
}

/// GETs `url` and parses its body into `scratch`, giving up after `timeout`.
/// std.http has no timeout, so the fetch races a timer and the loser is canceled.
fn fetch(scratch: std.mem.Allocator, io: std.Io, url: []const u8, timeout: std.Io.Duration) !std.json.Value {
    const Race = union(enum) {
        body: @typeInfo(@TypeOf(get)).@"fn".return_type.?,
        timer: std.Io.Cancelable!void,
    };
    var results: [2]Race = undefined;
    var race: std.Io.Select(Race) = .init(io, &results);
    defer race.cancelDiscard();

    try race.concurrent(.body, get, .{ scratch, io, url });
    try race.concurrent(.timer, std.Io.sleep, .{ io, timeout, .awake });
    const body = switch (try race.await()) {
        .body => |body| try body,
        .timer => return error.Timeout,
    };
    return std.json.parseFromSliceLeaky(std.json.Value, scratch, body, .{});
}

fn get(scratch: std.mem.Allocator, io: std.Io, url: []const u8) ![]const u8 {
    var client: std.http.Client = .{ .allocator = scratch, .io = io };
    defer client.deinit();

    var body: std.Io.Writer.Allocating = .init(scratch);
    const result = try client.fetch(.{ .location = .{ .url = url }, .response_writer = &body.writer });
    if (result.status != .ok) return error.HttpError;
    return body.written();
}
