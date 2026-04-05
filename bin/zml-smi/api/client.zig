const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const Collector = @import("zml-smi/collector").Collector;

const JsonDeviceInfo = struct {
    device: DeviceInfo,

    pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, _: std.json.ParseOptions) !JsonDeviceInfo {
        const type_str = source.object.get("type").?.string;

        inline for (@typeInfo(DeviceInfo).@"union".fields) |field| {
            if (std.mem.eql(u8, type_str, field.name)) {
                const val = try std.json.innerParseFromValue(field.type.Value, allocator, source, .{
                    .ignore_unknown_fields = true,
                });

                return .{ .device = @unionInit(DeviceInfo, field.name, .{ .values = .{ val, val } }) };
            }
        }
        return error.UnexpectedToken;
    }
};

pub fn addRemotes(collector: *Collector, hosts: []const u8) !void {
    var it = std.mem.splitScalar(u8, hosts, ',');
    while (it.next()) |host| {
        addHost(collector, host) catch |err| {
            std.log.err("{s}: {s}", .{ host, @errorName(err) });
        };
    }
}

fn addHost(collector: *Collector, host: []const u8) !void {
    const body = try httpGet(collector.gpa, collector.io, host);
    defer collector.gpa.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, collector.gpa, body, .{});
    defer parsed.deinit();

    const json_items = parsed.value.array.items;

    var devices: std.ArrayList(*DeviceInfo) = .empty;
    for (json_items) |item| {
        const jdi = std.json.parseFromValueLeaky(JsonDeviceInfo, collector.arena, item, .{}) catch continue;
        const info = try collector.addDevice(jdi.device);
        try devices.append(collector.arena, info);
    }

    if (devices.items.len == 0) {
        return;
    }

    const url = try collector.arena.dupe(u8, host);
    const poll_arena = try collector.createPollArena();
    try collector.spawnPoll(pollOnce, .{ poll_arena, collector.gpa, collector.io, url, devices.items });
}

fn pollOnce(poll_arena: *std.heap.ArenaAllocator, gpa: std.mem.Allocator, io: std.Io, host: []const u8, devices: []const *DeviceInfo) void {
    _ = poll_arena.reset(.retain_capacity);

    const body = httpGet(gpa, io, host) catch return;
    defer gpa.free(body);

    const parsed = std.json.parseFromSlice(std.json.Value, gpa, body, .{}) catch return;
    defer parsed.deinit();

    const json_items = parsed.value.array.items;

    for (json_items, 0..) |item, i| {
        const jdi = std.json.parseFromValueLeaky(JsonDeviceInfo, poll_arena.allocator(), item, .{}) catch continue;
        if (i >= devices.len) {
            break;
        }

        inline for (@typeInfo(DeviceInfo).@"union".fields) |field| {
            if (devices[i].* == @field(smi_info.device_info.Target, field.name)) {
                @field(devices[i], field.name).back().* = @field(jdi.device, field.name).front().*;
                @field(devices[i], field.name).swap();
            }
        }
    }
}

fn httpGet(allocator: std.mem.Allocator, io: std.Io, host: []const u8) ![]const u8 {
    const url = try std.fmt.allocPrint(allocator, "{s}/metrics", .{host});
    defer allocator.free(url);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();

    const result = try client.fetch(.{
        .location = .{ .url = url },
        .response_writer = &aw.writer,
    });

    if (result.status != .ok) {
        aw.deinit();
        return error.HttpError;
    }

    return try aw.toOwnedSlice();
}
