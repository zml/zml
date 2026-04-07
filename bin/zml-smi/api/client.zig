const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const pi = smi_info.process_info;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Collector = @import("zml-smi/collector").Collector;

const JsonDeviceInfo = struct {
    device: DeviceInfo,

    pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, _: std.json.ParseOptions) !JsonDeviceInfo {
        const type_str = source.object.get("type").?.string;

        inline for (@typeInfo(DeviceInfo).@"union".fields) |field| {
            if (std.mem.eql(u8, type_str, field.name)) {
                var val = try std.json.innerParseFromValue(field.type.Value, allocator, source, .{
                    .ignore_unknown_fields = true,
                });
                val.remote = true;

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

    const root = parsed.value.object;
    const json_devices = if (root.get("devices")) |v| v.array.items else &.{};

    const dev_offset: u8 = @intCast(collector.device_infos.items.len);

    var devices: std.ArrayList(*DeviceInfo) = .empty;
    for (json_devices) |item| {
        const jdi = std.json.parseFromValueLeaky(JsonDeviceInfo, collector.arena, item, .{}) catch continue;
        const info = try collector.addDevice(jdi.device);
        try devices.append(collector.arena, info);
    }

    if (devices.items.len == 0) {
        return;
    }

    const processes = try collector.createProcessList();
    const str_arenas: [2]*std.heap.ArenaAllocator = .{ try collector.createPollArena(), try collector.createPollArena() };

    const json_processes = if (root.get("processes")) |v| v.array.items else &.{};
    const back_idx: usize = 1 - processes.current.load(.acquire);
    const back = processes.back();
    for (json_processes) |item| {
        var proc = std.json.parseFromValueLeaky(pi.ProcessInfo, str_arenas[back_idx].allocator(), item, .{ .ignore_unknown_fields = true }) catch continue;
        proc.device_idx +|= dev_offset;
        proc.remote = true;
        back.append(collector.gpa, proc) catch continue;
    }
    processes.swap();

    const url = try collector.arena.dupe(u8, host);
    const poll_arena = try collector.createPollArena();
    try collector.spawnPoll(pollOnce, .{ poll_arena, collector.gpa, collector.io, url, devices.items, processes, dev_offset, str_arenas });
}

fn pollOnce(poll_arena: *std.heap.ArenaAllocator, gpa: std.mem.Allocator, io: std.Io, host: []const u8, devices: []const *DeviceInfo, processes: *ProcessDoubleBuffer, dev_offset: u8, str_arenas: [2]*std.heap.ArenaAllocator) void {
    _ = poll_arena.reset(.retain_capacity);

    const body = httpGet(gpa, io, host) catch return;
    defer gpa.free(body);

    const parsed = std.json.parseFromSlice(std.json.Value, gpa, body, .{}) catch return;
    defer parsed.deinit();

    const root = parsed.value.object;
    const json_devices = if (root.get("devices")) |v| v.array.items else &.{};

    for (json_devices, 0..) |item, i| {
        const jdi = std.json.parseFromValueLeaky(JsonDeviceInfo, poll_arena.allocator(), item, .{}) catch continue;
        if (i >= devices.len) {
            break;
        }

        inline for (@typeInfo(DeviceInfo).@"union".fields) |field| {
            if (devices[i].* == @field(smi_info.device_info.Target, field.name)) {
                @field(devices[i], field.name).back().* = @field(jdi.device, field.name).front().*;
                @field(devices[i], field.name).back().remote = true;
                @field(devices[i], field.name).swap();
            }
        }
    }

    const json_processes = if (root.get("processes")) |v| v.array.items else &.{};
    const back_idx: usize = 1 - processes.current.load(.acquire);
    _ = str_arenas[back_idx].reset(.retain_capacity);
    const alloc = str_arenas[back_idx].allocator();
    const back = processes.back();
    back.clearRetainingCapacity();
    for (json_processes) |item| {
        var proc = std.json.parseFromValueLeaky(pi.ProcessInfo, alloc, item, .{ .ignore_unknown_fields = true }) catch continue;
        proc.device_idx +|= dev_offset;
        proc.remote = true;
        back.append(gpa, proc) catch continue;
    }
    processes.swap();
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
