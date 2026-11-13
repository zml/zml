const std = @import("std");
const smi = @import("smi.zig");

const log = std.log.scoped(.@"zml-smi/remote");

/// The devices of another zml-smi, read from its API.
pub const Remote = struct {
    platform: smi.Platform,
    io: std.Io,
    /// "http://host:9090/metrics".
    url: []const u8,
    /// For each row of the peer, the position of its device here, or null when this build does not know its target.
    positions: []const ?u16,
    timeout: std.Io.Duration,
    /// What the poll of this round fetched, in scratch, for `processes`.
    fetched: ?std.json.Value = null,
    warned: bool = false,

    const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes };

    /// Fetches the devices once. `interval` bounds every later fetch, to at least a second.
    pub fn open(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, host: []const u8, interval: std.Io.Duration) !Remote {
        if (!std.mem.startsWith(u8, host, "http://") and !std.mem.startsWith(u8, host, "https://")) {
            log.err("{s}: a remote host starts with http:// or https://", .{host});
            return error.InvalidUrl;
        }
        const url = try std.fmt.allocPrint(arena, "{s}/metrics", .{host});

        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const rows = items(try fetch(scratch.allocator(), io, url, .fromSeconds(3)), "devices");

        var devices: std.ArrayList(smi.Device) = try .initCapacity(arena, rows.len);
        const positions = try arena.alloc(?u16, rows.len);
        for (rows, positions) |item, *position| {
            const row = std.json.parseFromValueLeaky(smi.Row, scratch.allocator(), item, .{ .ignore_unknown_fields = true }) catch {
                position.* = null;
                continue;
            };
            position.* = @intCast(devices.items.len);
            devices.appendAssumeCapacity(try copy(arena, row.device));
        }

        return .{
            .platform = .{ .devices = devices.items, .vtable = &vtable },
            .io = io,
            .url = url,
            .positions = positions,
            .timeout = if (interval.nanoseconds < std.time.ns_per_s) .fromSeconds(1) else interval,
        };
    }

    fn poll(platform: *smi.Platform, scratch: std.mem.Allocator, samples: []smi.Sample) void {
        const self: *Remote = @alignCast(@fieldParentPtr("platform", platform));
        self.fetched = null;
        const root = fetch(scratch, self.io, self.url, self.timeout) catch return;

        const rows = items(root, "devices");
        if (rows.len != self.positions.len) return self.changed(samples);
        for (rows, self.positions) |item, maybe_position| {
            const position = maybe_position orelse continue;
            const row = std.json.parseFromValueLeaky(smi.Row, scratch, item, .{ .ignore_unknown_fields = true }) catch return self.changed(samples);
            const device = self.platform.devices[position];
            if (row.device.target != device.target or !std.mem.eql(u8, row.device.name orelse "", device.name orelse "")) {
                return self.changed(samples);
            }
            samples[position] = row.sample;
        }
        self.fetched = root;
    }

    /// The peer's devices are not the ones it had when it was opened: show nothing.
    fn changed(self: *Remote, samples: []smi.Sample) void {
        for (samples, self.platform.devices) |*sample, device| sample.* = .init(device.target);
        if (!self.warned) log.warn("{s}: the devices changed since zml-smi started", .{self.url});
        self.warned = true;
    }

    fn processes(platform: *smi.Platform, scratch: std.mem.Allocator, _: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
        const self: *Remote = @alignCast(@fieldParentPtr("platform", platform));
        const root = self.fetched orelse return error.Unavailable;
        for (items(root, "processes")) |item| {
            var row = std.json.parseFromValueLeaky(smi.ProcessInfo, scratch, item, .{ .ignore_unknown_fields = true }) catch continue;
            if (row.device_idx >= self.positions.len) continue;
            row.device_idx = self.positions[row.device_idx] orelse continue;
            row.remote = true;
            row.username = try out.dupe(row.username);
            row.comm = try out.dupe(row.comm);
            try out.append(row);
        }
    }
};

fn copy(arena: std.mem.Allocator, device: smi.Device) !smi.Device {
    const details = try arena.alloc(smi.Device.Detail, device.details.len);
    for (details, device.details) |*detail, source| {
        detail.* = .{
            .key = try arena.dupe(u8, source.key),
            .label = try arena.dupe(u8, source.label),
            .value = try arena.dupe(u8, source.value),
            .in_title = source.in_title,
        };
    }
    return .{
        .target = device.target,
        .remote = true,
        .name = if (device.name) |name| try arena.dupe(u8, name) else null,
        .driver_version = if (device.driver_version) |version| try arena.dupe(u8, version) else null,
        .details = details,
    };
}

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
