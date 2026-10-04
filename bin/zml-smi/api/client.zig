const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const hi = smi_info.host_info;
const pi = smi_info.process_info;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Collector = @import("zml-smi/collector").Collector;

fn setRemote(device: *DeviceInfo) void {
    inline for (@typeInfo(DeviceInfo).@"union".fields) |field| {
        if (device.* == @field(smi_info.device_info.Target, field.name)) {
            @field(device, field.name).values[0].remote = true;
            @field(device, field.name).values[1].remote = true;
        }
    }
}

pub fn addRemotes(collector: *Collector, hosts: []const u8) ![]const hi.HostSummary {
    var summaries: std.ArrayList(hi.HostSummary) = .empty;
    var it = std.mem.splitScalar(u8, hosts, ',');
    while (it.next()) |host| {
        const summary = addHost(collector, host) catch |err| {
            std.log.err("{s}: {s}", .{ host, @errorName(err) });
            continue;
        };
        try summaries.append(collector.arena, summary);
    }
    return summaries.toOwnedSlice(collector.arena);
}

fn addHost(collector: *Collector, host: []const u8) !hi.HostSummary {
    const body = try httpGet(collector.gpa, collector.io, host);
    defer collector.gpa.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, collector.gpa, body, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    const json_devices = if (root.get("devices")) |v| v.array.items else &.{};

    const dev_offset: u16 = @intCast(collector.device_infos.items.len);

    var devices: std.ArrayList(*DeviceInfo) = .empty;
    for (json_devices) |item| {
        var device = std.json.parseFromValueLeaky(DeviceInfo, collector.arena, item, .{ .ignore_unknown_fields = true }) catch continue;
        setRemote(&device);
        const info = try collector.addDevice(device);
        try devices.append(collector.arena, info);
    }

    const host_info = try collector.arena.create(hi.HostInfo);
    const initial_host = try parseHost(collector.arena, root, host);
    host_info.* = .{ .values = .{ initial_host, initial_host } };

    const processes = try collector.createProcessList();
    const str_arenas: [2]*std.heap.ArenaAllocator = .{ try collector.createPollArena(), try collector.createPollArena() };

    const json_processes = if (root.get("processes")) |v| v.array.items else &.{};
    const back_idx: usize = 1 - processes.current.load(.acquire);
    const back = processes.back();
    for (json_processes) |item| {
        var proc = std.json.parseFromValueLeaky(pi.ProcessInfo, str_arenas[back_idx].allocator(), item, .{ .ignore_unknown_fields = true }) catch continue;
        proc.device_idx += dev_offset;
        proc.remote = true;
        back.append(collector.gpa, proc) catch continue;
    }
    processes.swap();

    const url = try collector.arena.dupe(u8, host);
    const poll_arena = try collector.createPollArena();
    try collector.spawnPoll(pollOnce, .{ poll_arena, collector.gpa, collector.io, url, devices.items, processes, dev_offset, str_arenas, host_info });
    return .{ .host = host_info, .device_count = devices.items.len };
}

fn pollOnce(poll_arena: *std.heap.ArenaAllocator, gpa: std.mem.Allocator, io: std.Io, host: []const u8, devices: []const *DeviceInfo, processes: *ProcessDoubleBuffer, dev_offset: u16, str_arenas: [2]*std.heap.ArenaAllocator, host_info: *hi.HostInfo) void {
    _ = poll_arena.reset(.retain_capacity);

    const body = httpGet(gpa, io, host) catch return;
    defer gpa.free(body);

    const parsed = std.json.parseFromSlice(std.json.Value, gpa, body, .{}) catch return;
    defer parsed.deinit();

    const root = parsed.value.object;
    const json_devices = if (root.get("devices")) |v| v.array.items else &.{};

    for (json_devices, 0..) |item, i| {
        if (i >= devices.len) {
            std.log.warn("{s}: remote has more devices than expected ({d} > {d})", .{ host, json_devices.len, devices.len });
            break;
        }
        var device = std.json.parseFromValueLeaky(DeviceInfo, poll_arena.allocator(), item, .{ .ignore_unknown_fields = true }) catch continue;
        setRemote(&device);

        inline for (@typeInfo(DeviceInfo).@"union".fields) |field| {
            if (devices[i].* == @field(smi_info.device_info.Target, field.name)) {
                @field(devices[i], field.name).back().* = @field(device, field.name).front().*;
                @field(devices[i], field.name).swap();
            }
        }
    }

    const json_processes = if (root.get("processes")) |v| v.array.items else &.{};
    const back_idx: usize = 1 - processes.current.load(.acquire);
    _ = str_arenas[back_idx].reset(.retain_capacity);
    const alloc = str_arenas[back_idx].allocator();
    host_info.back().* = parseHost(alloc, root, host) catch .{};
    host_info.swap();
    const back = processes.back();
    back.clearRetainingCapacity();
    for (json_processes) |item| {
        var proc = std.json.parseFromValueLeaky(pi.ProcessInfo, alloc, item, .{ .ignore_unknown_fields = true }) catch continue;
        proc.device_idx += dev_offset;
        proc.remote = true;
        back.append(gpa, proc) catch continue;
    }
    processes.swap();
}

fn httpGet(allocator: std.mem.Allocator, io: std.Io, host: []const u8) ![]const u8 {
    if (!std.mem.startsWith(u8, host, "http://") and !std.mem.startsWith(u8, host, "https://")) {
        std.log.err("remote host must start with http:// or https://: {s}", .{host});
        return error.InvalidUrl;
    }

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

fn parseHost(allocator: std.mem.Allocator, root: std.json.ObjectMap, url: []const u8) !hi.HostData {
    if (root.get("host")) |value| {
        if (value != .null) return std.json.parseFromValueLeaky(hi.HostData, allocator, value, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    }
    return .{ .hostname = try allocator.dupe(u8, url) };
}

test "remote host survives response cleanup" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const host: hi.HostData = .{ .hostname = "remote-node", .kernel = "remote-kernel", .cpu_name = "remote-cpu", .cpu_cores = 96, .mem_total_kib = 1024, .mem_available_kib = 512, .uptime_seconds = 1234, .load_1 = 2.5 };
    const host_info: hi.HostInfo = .{ .values = .{ host, host } };
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer writer.deinit();
    try @import("zml-smi/json").write(&writer.writer, &.{}, &.{}, &host_info);
    const remote = blk: {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, writer.written(), .{});
        defer parsed.deinit();
        break :blk try parseHost(arena.allocator(), parsed.value.object, "http://remote");
    };
    try std.testing.expectEqualDeep(host, remote);
}

test "legacy server host data stays unknown" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"devices\":[],\"processes\":[]}", .{});
    defer parsed.deinit();
    const remote = try parseHost(arena.allocator(), parsed.value.object, "http://legacy");
    try std.testing.expectEqualDeep(hi.HostData{ .hostname = "http://legacy" }, remote);
}
