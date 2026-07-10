const std = @import("std");
const smi_sysfs = @import("zml-smi/sysfs");
const Monitor = @This();

pub const bus_device_identifier_len = "0000:00:00.0".len;
pub const Handle = enum(u32) { _ };

const Device = struct {
    render_idx: usize,
    drm_path: []const u8,
    dev_path: []const u8,
    pcie_path: []const u8,
    hwmon_path: ?[]const u8,
    bus_device_identifier: ?[bus_device_identifier_len]u8,
    activity_prev: ?EngineSample = null,
    encoder_prev: ?EngineSample = null,
    decoder_prev: ?EngineSample = null,
    energy_prev: ?EnergySample = null,
};

const EnergySample = struct {
    micro_joules: u64,
    timestamp_ns: u64,
};

pub const EngineSample = struct {
    engine_ns: u64,
    timestamp_ns: u64,
};

const DeviceSample = struct {
    mem_kib: u64 = 0,
    engine: ?EngineSample = null,
    encoder: ?EngineSample = null,
    decoder: ?EngineSample = null,
};

pub const ProcessSample = struct {
    pid: u32,
    device_idx: u16,
    mem_kib: ?u64 = null,
    engine: ?EngineSample = null,
    encoder: ?EngineSample = null,
    decoder: ?EngineSample = null,
};

pub const ProcessUsage = std.AutoHashMapUnmanaged(u64, ProcessSample);
pub const ProcessPrevious = std.AutoHashMapUnmanaged(u64, EngineSample);
const DeviceUsage = std.AutoHashMapUnmanaged(u16, DeviceSample);

allocator: std.mem.Allocator,
io: std.Io,
devices: []Device,

pub fn init(allocator: std.mem.Allocator, io: std.Io) !Monitor {
    var monitor: Monitor = .{
        .allocator = allocator,
        .io = io,
        .devices = try discoverIntelDevices(allocator, io),
    };
    try monitor.seedDeviceMetrics();
    return monitor;
}

pub fn deviceId(self: *const Monitor, allocator: std.mem.Allocator, handle: Handle) ![]const u8 {
    const dev = try self.device(handle);
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/device", .{dev.dev_path});
    const raw = try smi_sysfs.readString(allocator, self.io, path);
    return std.mem.trim(u8, raw, &std.ascii.whitespace);
}

inline fn seedDeviceMetrics(self: *Monitor) !void {
    var device_usage = collectDeviceUsage(self.allocator, self.io, self.devices) catch DeviceUsage{};
    defer device_usage.deinit(self.allocator);

    for (self.devices, 0..) |*dev, idx| {
        dev.energy_prev = readEnergy(self.allocator, self.io, dev) catch null;
        if (device_usage.get(@intCast(idx))) |sample| {
            dev.activity_prev = sample.engine;
            dev.encoder_prev = sample.encoder;
            dev.decoder_prev = sample.decoder;
        }
    }
}

pub fn saveProcessPrevious(allocator: std.mem.Allocator, previous: *ProcessPrevious, usage: *const ProcessUsage) !void {
    previous.clearRetainingCapacity();
    try previous.ensureTotalCapacity(allocator, @intCast(usage.count()));

    var it = usage.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.engine) |engine| {
            previous.putAssumeCapacity(entry.key_ptr.*, engine);
        }
    }
}

inline fn discoverIntelDevices(allocator: std.mem.Allocator, io: std.Io) ![]Device {
    var devices: std.ArrayList(Device) = .empty;
    errdefer devices.deinit(allocator);

    var dir = std.Io.Dir.openDirAbsolute(io, "/dev/dri", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return &.{},
        else => |e| return e,
    };
    defer dir.close(io);

    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        const render_idx = parseRenderIndex(entry.name) orelse continue;
        const dev = openDevice(allocator, io, render_idx) catch continue;
        try devices.append(allocator, dev);
    }

    std.mem.sort(Device, devices.items, {}, struct {
        fn lessThan(_: void, lhs: Device, rhs: Device) bool {
            return lhs.render_idx < rhs.render_idx;
        }
    }.lessThan);
    return devices.toOwnedSlice(allocator);
}

inline fn openDevice(allocator: std.mem.Allocator, io: std.Io, render_idx: usize) !Device {
    const drm_path = try std.fmt.allocPrint(allocator, "/sys/class/drm/renderD{d}", .{render_idx});
    errdefer allocator.free(drm_path);

    const dev_path = try std.fmt.allocPrint(allocator, "{s}/device", .{drm_path});
    errdefer allocator.free(dev_path);

    var vendor_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const vendor_path = try std.fmt.bufPrint(&vendor_path_buf, "{s}/vendor", .{dev_path});
    const vendor = try smi_sysfs.readString(allocator, io, vendor_path);
    if (!std.mem.eql(u8, std.mem.trim(u8, vendor, &std.ascii.whitespace), "0x8086")) {
        return error.not_intel;
    }

    const pcie_path = try findPcieLinkPath(allocator, io, dev_path);
    errdefer if (pcie_path.ptr != dev_path.ptr) allocator.free(pcie_path);

    return .{
        .render_idx = render_idx,
        .drm_path = drm_path,
        .dev_path = dev_path,
        .pcie_path = pcie_path,
        .hwmon_path = findHwmon(allocator, io, dev_path) catch null,
        .bus_device_identifier = readBusDeviceId(allocator, io, dev_path) catch null,
    };
}

pub fn device(self: *const Monitor, handle: Handle) !*const Device {
    const idx = @intFromEnum(handle);
    if (idx >= self.devices.len) return error.NotFound;
    return &self.devices[idx];
}

pub fn readTemperature(allocator: std.mem.Allocator, io: std.Io, dev: *const Device) !u64 {
    const hwmon = dev.hwmon_path orelse return error.NotFound;
    const milli_c = readHwmonInput(allocator, io, hwmon, "temp", "pkg") catch
        readHwmonInput(allocator, io, hwmon, "temp", "gpu") catch
        readNumberedInput(allocator, io, hwmon, "temp", 2) catch
        try readNumberedInput(allocator, io, hwmon, "temp", 1);
    return (milli_c + 500) / 1000;
}

pub fn readEnergy(allocator: std.mem.Allocator, io: std.Io, dev: *const Device) !EnergySample {
    const hwmon = dev.hwmon_path orelse return error.NotFound;
    const micro_joules = readHwmonInput(allocator, io, hwmon, "energy", "card") catch
        readHwmonInput(allocator, io, hwmon, "energy", "pkg") catch
        try readNumberedInput(allocator, io, hwmon, "energy", 1);
    return .{
        .micro_joules = micro_joules,
        .timestamp_ns = @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds),
    };
}
pub fn readPowerLimit(allocator: std.mem.Allocator, io: std.Io, dev: *const Device) !u64 {
    const hwmon = dev.hwmon_path orelse return error.NotFound;
    const microwatts = readNumberedSuffix(allocator, io, hwmon, "power", 1, "cap") catch
        try readNumberedSuffix(allocator, io, hwmon, "power", 1, "max");
    return microwatts / 1000;
}

pub fn readFanSpeed(allocator: std.mem.Allocator, io: std.Io, dev: *const Device) !u64 {
    const hwmon = dev.hwmon_path orelse return error.NotFound;
    return readPwmFanSpeed(allocator, io, hwmon) catch readRpmFanSpeedPercent(allocator, io, hwmon);
}

pub fn readClockGraphics(allocator: std.mem.Allocator, io: std.Io, dev: *const Device) !u64 {
    return readFreq(allocator, io, dev, "act_freq") catch readFreq(allocator, io, dev, "cur_freq");
}

pub fn readMaxClockGraphics(allocator: std.mem.Allocator, io: std.Io, dev: *const Device) !u64 {
    return readFreq(allocator, io, dev, "max_freq");
}

pub fn readMemTotal(allocator: std.mem.Allocator, io: std.Io, dev: *const Device) !u64 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/resource", .{dev.dev_path});
    return readLargestResource(allocator, io, path);
}

pub fn readPcieLinkGen(allocator: std.mem.Allocator, io: std.Io, dev: *const Device) !u64 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/current_link_speed", .{dev.pcie_path});
    const raw = try smi_sysfs.readString(allocator, io, path);
    return pcieGenFromSpeed(raw);
}

pub fn readPcieLinkWidth(allocator: std.mem.Allocator, io: std.Io, dev: *const Device) !u64 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/current_link_width", .{dev.pcie_path});
    return smi_sysfs.readInt(allocator, io, path);
}

pub fn readPcieBandwidth(allocator: std.mem.Allocator, io: std.Io, dev: *const Device) !u64 {
    const gen = try readPcieLinkGen(allocator, io, dev);
    const width = try readPcieLinkWidth(allocator, io, dev);
    return pcieBandwidthFromLink(gen, width) orelse error.NotFound;
}

pub fn collectDeviceUsage(allocator: std.mem.Allocator, io: std.Io, devices: []const Device) !DeviceUsage {
    var processes = try collectProcessUsage(allocator, io, devices);
    errdefer processes.deinit(allocator);

    var result: DeviceUsage = .{};
    errdefer result.deinit(allocator);

    var it = processes.iterator();
    while (it.next()) |entry| {
        const sample = entry.value_ptr.*;
        const gop = try result.getOrPut(allocator, sample.device_idx);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        if (sample.mem_kib) |mem| gop.value_ptr.mem_kib += mem;
        addEngineSample(&gop.value_ptr.engine, sample.engine);
        addEngineSample(&gop.value_ptr.encoder, sample.encoder);
        addEngineSample(&gop.value_ptr.decoder, sample.decoder);
    }

    return result;
}

pub fn collectProcessUsage(allocator: std.mem.Allocator, io: std.Io, devices: []const Device) !ProcessUsage {
    var result: ProcessUsage = .{};
    errdefer result.deinit(allocator);

    var seen_device_clients: std.AutoHashMapUnmanaged(u128, void) = .{};
    defer seen_device_clients.deinit(allocator);

    var proc_dir = try std.Io.Dir.openDirAbsolute(io, "/proc", .{ .iterate = true });
    defer proc_dir.close(io);

    const now_ns: u64 = @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds);
    var proc_it = proc_dir.iterate();
    while (proc_it.next(io) catch null) |proc_entry| {
        const pid = std.fmt.parseInt(u32, proc_entry.name, 10) catch continue;

        var fdinfo_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const fdinfo_path = std.fmt.bufPrint(&fdinfo_path_buf, "/proc/{d}/fdinfo", .{pid}) catch continue;
        var fdinfo_dir = std.Io.Dir.openDirAbsolute(io, fdinfo_path, .{ .iterate = true }) catch continue;
        defer fdinfo_dir.close(io);

        var fd_it = fdinfo_dir.iterate();
        while (fd_it.next(io) catch null) |fd_entry| {
            var file_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const file_path = std.fmt.bufPrint(&file_path_buf, "{s}/{s}", .{ fdinfo_path, fd_entry.name }) catch continue;
            const parsed = parseFdinfoFile(io, file_path, now_ns) catch continue;
            const bus_device_identifier = parsed.bus_device_identifier orelse continue;
            const dev_idx = matchDevice(devices, bus_device_identifier) orelse continue;

            // Skip fd if DRM client has been seen for this device.
            // Summing after dup() / fork() over-reports memory and compute.
            if (parsed.drm_client_id) |cid| {
                if (try markDrmClientSeen(allocator, &seen_device_clients, dev_idx, cid)) continue;
            }

            const key = processKey(pid, dev_idx);
            const gop = try result.getOrPut(allocator, key);
            if (!gop.found_existing) {
                gop.value_ptr.* = .{ .pid = pid, .device_idx = dev_idx };
            }
            mergeProcessSample(gop.value_ptr, parsed.sample);
        }
    }

    return result;
}

pub fn processUtil(previous: ?EngineSample, current: ?EngineSample) ?u16 {
    const prev = previous orelse return null;
    const cur = current orelse return null;
    if (cur.timestamp_ns <= prev.timestamp_ns or cur.engine_ns < prev.engine_ns) return null;
    const engine_delta = cur.engine_ns - prev.engine_ns;
    const time_delta = cur.timestamp_ns - prev.timestamp_ns;
    if (time_delta == 0) return null;
    const pct = @min(engine_delta * 100 / time_delta, 100);
    return @intCast(if (pct == 0 and engine_delta > 0) 0 else pct);
}

pub fn powerMilliwatts(previous: ?EnergySample, current: ?EnergySample) ?u64 {
    const prev = previous orelse return null;
    const cur = current orelse return null;
    return milliwattsFromEnergyDelta(prev.micro_joules, cur.micro_joules, prev.timestamp_ns, cur.timestamp_ns);
}

inline fn milliwattsFromEnergyDelta(prev_uj: u64, cur_uj: u64, prev_ns: u64, cur_ns: u64) ?u64 {
    if (cur_uj < prev_uj or cur_ns <= prev_ns) return null;
    const energy_delta = cur_uj - prev_uj;
    const time_delta = cur_ns - prev_ns;
    if (time_delta == 0) return null;
    return energy_delta * 1_000_000 / time_delta;
}

inline fn readFreq(allocator: std.mem.Allocator, io: std.Io, dev: *const Device, file: []const u8) !u64 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/tile0/gt0/freq0/{s}", .{ dev.dev_path, file });
    return smi_sysfs.readInt(allocator, io, path);
}

inline fn parseRenderIndex(name: []const u8) ?usize {
    if (!std.mem.startsWith(u8, name, "renderD")) return null;
    return std.fmt.parseInt(usize, name["renderD".len..], 10) catch null;
}

inline fn findHwmon(allocator: std.mem.Allocator, io: std.Io, dev_path: []const u8) !?[]const u8 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const hwmon_root = try std.fmt.bufPrint(&path_buf, "{s}/hwmon", .{dev_path});
    var dir = std.Io.Dir.openDirAbsolute(io, hwmon_root, .{ .iterate = true }) catch return null;
    defer dir.close(io);

    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!std.mem.startsWith(u8, entry.name, "hwmon")) continue;
        const candidate = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ hwmon_root, entry.name });
        var name_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const name_path = try std.fmt.bufPrint(&name_buf, "{s}/name", .{candidate});
        const name = smi_sysfs.readString(allocator, io, name_path) catch return candidate;
        const trimmed = std.mem.trim(u8, name, &std.ascii.whitespace);
        if (std.mem.eql(u8, trimmed, "xe") or std.mem.eql(u8, trimmed, "i915")) return candidate;
    }
    return null;
}

inline fn readBusDeviceId(allocator: std.mem.Allocator, io: std.Io, dev_path: []const u8) ![bus_device_identifier_len]u8 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/uevent", .{dev_path});
    const raw = try smi_sysfs.readFieldString(allocator, io, path, "PCI_SLOT_NAME=");
    const trimmed = std.mem.trim(u8, raw, &std.ascii.whitespace);
    if (trimmed.len != bus_device_identifier_len) return error.NotFound;
    var out: [bus_device_identifier_len]u8 = undefined;
    @memcpy(&out, trimmed);
    return out;
}

inline fn findPcieLinkPath(allocator: std.mem.Allocator, io: std.Io, dev_path: []const u8) ![]const u8 {
    var candidate = dev_path;
    var best_path: ?[]const u8 = null;

    var best_bandwidth: u64 = 0;
    var depth: usize = 0;
    while (depth < 8) : (depth += 1) {
        if (!try isIntelPciDevice(allocator, io, candidate)) break;

        const bandwidth = readMaxPcieBandwidth(allocator, io, candidate) catch 0;
        if (bandwidth > best_bandwidth) {
            best_path = candidate;
            best_bandwidth = bandwidth;
        }

        const parent = try std.fmt.allocPrint(allocator, "{s}/..", .{candidate});
        candidate = parent;
    }

    return best_path orelse dev_path;
}

inline fn isIntelPciDevice(allocator: std.mem.Allocator, io: std.Io, dev_path: []const u8) !bool {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const vendor_path = try std.fmt.bufPrint(&path_buf, "{s}/vendor", .{dev_path});
    const vendor_raw = smi_sysfs.readString(allocator, io, vendor_path) catch return false;

    const vendor = std.mem.trim(u8, vendor_raw, &std.ascii.whitespace);
    return std.mem.eql(u8, vendor, "0x8086");
}

inline fn readMaxPcieBandwidth(allocator: std.mem.Allocator, io: std.Io, dev_path: []const u8) !u64 {
    var speed_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const speed_path = try std.fmt.bufPrint(&speed_path_buf, "{s}/max_link_speed", .{dev_path});
    const raw_speed = try smi_sysfs.readString(allocator, io, speed_path);

    var width_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const width_path = try std.fmt.bufPrint(&width_path_buf, "{s}/max_link_width", .{dev_path});
    const width = try smi_sysfs.readInt(allocator, io, width_path);
    const gen = try pcieGenFromSpeed(raw_speed);
    return pcieBandwidthFromLink(gen, width) orelse error.NotFound;
}

inline fn readHwmonInput(allocator: std.mem.Allocator, io: std.Io, hwmon: []const u8, comptime prefix: []const u8, label: []const u8) !u64 {
    for (1..16) |idx| {
        var label_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const label_path = try std.fmt.bufPrint(&label_path_buf, "{s}/{s}{d}_label", .{ hwmon, prefix, idx });
        const raw = smi_sysfs.readString(allocator, io, label_path) catch continue;
        const trimmed = std.mem.trim(u8, raw, &std.ascii.whitespace);
        if (!std.ascii.eqlIgnoreCase(trimmed, label)) continue;
        return readNumberedInput(allocator, io, hwmon, prefix, idx);
    }
    return error.NotFound;
}

inline fn readNumberedInput(allocator: std.mem.Allocator, io: std.Io, hwmon: []const u8, comptime prefix: []const u8, idx: usize) !u64 {
    return readNumberedSuffix(allocator, io, hwmon, prefix, idx, "input");
}

inline fn readNumberedSuffix(allocator: std.mem.Allocator, io: std.Io, hwmon: []const u8, comptime prefix: []const u8, idx: usize, suffix: []const u8) !u64 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}{d}_{s}", .{ hwmon, prefix, idx, suffix });
    return smi_sysfs.readInt(allocator, io, path);
}

inline fn readNumberedRaw(allocator: std.mem.Allocator, io: std.Io, hwmon: []const u8, comptime prefix: []const u8, idx: usize) !u64 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}{d}", .{ hwmon, prefix, idx });
    return smi_sysfs.readInt(allocator, io, path);
}

inline fn readPwmFanSpeed(allocator: std.mem.Allocator, io: std.Io, hwmon: []const u8) !u64 {
    for (1..16) |idx| {
        const pwm = readNumberedRaw(allocator, io, hwmon, "pwm", idx) catch continue;
        return pwmToPercent(pwm);
    }
    return error.NotFound;
}

inline fn readRpmFanSpeedPercent(allocator: std.mem.Allocator, io: std.Io, hwmon: []const u8) !u64 {
    for (1..16) |idx| {
        const input = readNumberedSuffix(allocator, io, hwmon, "fan", idx, "input") catch continue;
        const max = readNumberedSuffix(allocator, io, hwmon, "fan", idx, "max") catch continue;
        if (max == 0) continue;
        return @min(input * 100 / max, 100);
    }
    return error.NotFound;
}

inline fn pwmToPercent(pwm: u64) u64 {
    return @min((@min(pwm, 255) * 100 + 127) / 255, 100);
}

inline fn readLargestResource(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !u64 {
    var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    var line_writer: std.Io.Writer.Allocating = .init(allocator);
    defer line_writer.deinit();

    var largest: u64 = 0;
    while (true) {
        line_writer.clearRetainingCapacity();
        _ = reader.interface.streamDelimiter(&line_writer.writer, '\n') catch |err| switch (err) {
            error.EndOfStream => break,
            else => |e| return e,
        };
        reader.interface.toss(1);
        largest = @max(largest, resourceLineSize(line_writer.written()) orelse 0);
    }
    if (largest == 0) return error.NotFound;
    return largest;
}

inline fn resourceLineSize(line: []const u8) ?u64 {
    var it = std.mem.tokenizeAny(u8, line, " \t");
    const base = parseHex(it.next() orelse return null) orelse return null;
    const end = parseHex(it.next() orelse return null) orelse return null;
    if (end < base) return null;
    const size = end - base + 1;
    return if (size >= 256 * 1024 * 1024) size else null;
}

inline fn parseHex(raw: []const u8) ?u64 {
    const trimmed = std.mem.trim(u8, raw, &std.ascii.whitespace);
    const digits = if (std.mem.startsWith(u8, trimmed, "0x")) trimmed[2..] else trimmed;
    return std.fmt.parseInt(u64, digits, 16) catch null;
}

inline fn pcieGenFromSpeed(raw: []const u8) !u64 {
    const trimmed = std.mem.trim(u8, raw, &std.ascii.whitespace);
    if (std.mem.startsWith(u8, trimmed, "2.5")) return 1;
    if (std.mem.startsWith(u8, trimmed, "5.0") or std.mem.startsWith(u8, trimmed, "5 ")) return 2;
    if (std.mem.startsWith(u8, trimmed, "8.0") or std.mem.startsWith(u8, trimmed, "8 ")) return 3;
    if (std.mem.startsWith(u8, trimmed, "16.0") or std.mem.startsWith(u8, trimmed, "16 ")) return 4;
    if (std.mem.startsWith(u8, trimmed, "32.0") or std.mem.startsWith(u8, trimmed, "32 ")) return 5;
    if (std.mem.startsWith(u8, trimmed, "64.0") or std.mem.startsWith(u8, trimmed, "64 ")) return 6;
    return error.NotFound;
}

inline fn pcieBandwidthFromLink(gen: u64, width: u64) ?u64 {
    const lane_mb_s: u64 = switch (gen) {
        1 => 250,
        2 => 500,
        3 => 985,
        4 => 1969,
        5 => 3938,
        6 => 7563,
        else => return null,
    };
    return lane_mb_s * width;
}

const FdinfoSample = struct {
    mem_kib: ?u64 = null,
    engine: ?EngineSample = null,
    encoder: ?EngineSample = null,
    decoder: ?EngineSample = null,
};

const ParsedFdinfo = struct {
    bus_device_identifier: ?[bus_device_identifier_len]u8 = null,
    drm_client_id: ?u64 = null,
    sample: FdinfoSample = .{},
    resident_vram_kib: u64 = 0,
    total_vram_kib: u64 = 0,
    resident_memory_kib: u64 = 0,
    total_memory_kib: u64 = 0,
    cycle_counter: u64 = 0,
    total_cycle_counter: u64 = 0,
    engine_time_seen: bool = false,
};

inline fn parseFdinfoFile(io: std.Io, path: []const u8, timestamp_ns: u64) !ParsedFdinfo {
    var read_buf: [8192]u8 = undefined;
    const data = std.Io.Dir.readFile(.cwd(), io, path, &read_buf) catch return error.FileUnavailable;

    var parsed: ParsedFdinfo = .{};

    var lines = std.mem.tokenizeScalar(u8, data, '\n');
    while (lines.next()) |line| {
        parseFdinfoLine(&parsed, line, timestamp_ns);
    }

    finishFdinfo(&parsed);
    return parsed;
}

inline fn finishFdinfo(self: *ParsedFdinfo) void {
    if (self.resident_vram_kib > 0) {
        self.sample.mem_kib = self.resident_vram_kib;
    } else if (self.total_vram_kib > 0) {
        self.sample.mem_kib = self.total_vram_kib;
    } else if (self.resident_memory_kib > 0) {
        self.sample.mem_kib = self.resident_memory_kib;
    } else if (self.total_memory_kib > 0) {
        self.sample.mem_kib = self.total_memory_kib;
    }

    if (!self.engine_time_seen and self.total_cycle_counter > 0) {
        self.sample.engine = .{
            .engine_ns = self.cycle_counter,
            .timestamp_ns = self.total_cycle_counter,
        };
    }
}

inline fn parseFdinfoLine(parsed: *ParsedFdinfo, line: []const u8, timestamp_ns: u64) void {
    if (std.mem.startsWith(u8, line, "drm-pdev:")) {
        parsed.bus_device_identifier = parseBdf(line["drm-pdev:".len..]);
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-client-id:")) {
        parsed.drm_client_id = firstInt(line["drm-client-id:".len..]);
        return;
    }

    if (std.mem.startsWith(u8, line, "vram mem:")) {
        parsed.total_vram_kib += memoryKiB(line["vram mem:".len..]);
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-cycles-")) {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return;
        parsed.cycle_counter += firstInt(line[colon + 1 ..]);
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-total-cycles-")) {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return;
        parsed.total_cycle_counter = @max(parsed.total_cycle_counter, firstInt(line[colon + 1 ..]));
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-engine-capacity-")) {
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-engine-")) {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return;
        const engine_label = line["drm-engine-".len..colon];
        const value = firstInt(line[colon + 1 ..]);
        parsed.engine_time_seen = true;
        addEngineTime(&parsed.sample.engine, value, timestamp_ns);
        if (containsAsciiIgnoreCase(engine_label, "encode")) {
            addEngineTime(&parsed.sample.encoder, value, timestamp_ns);
        } else if (containsAsciiIgnoreCase(engine_label, "decode")) {
            addEngineTime(&parsed.sample.decoder, value, timestamp_ns);
        }
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-")) {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return;
        const key = line[0..colon];
        const value = memoryKiB(line[colon + 1 ..]);
        if (value == 0) return;

        if (std.mem.startsWith(u8, key, "drm-resident-vram")) {
            parsed.resident_vram_kib += value;
            return;
        }
        if (std.mem.startsWith(u8, key, "drm-total-vram") or std.mem.startsWith(u8, key, "drm-memory-vram")) {
            parsed.total_vram_kib += value;
            return;
        }
        if (std.mem.startsWith(u8, key, "drm-resident-")) {
            parsed.resident_memory_kib += value;
            return;
        }
        if (std.mem.startsWith(u8, key, "drm-total-")) {
            parsed.total_memory_kib += value;
            return;
        }
        if (std.mem.eql(u8, key, "drm-resident-memory")) {
            parsed.resident_memory_kib += value;
            return;
        }
        if (std.mem.eql(u8, key, "drm-total-memory")) {
            parsed.total_memory_kib += value;
            return;
        }
    }
}

fn mergeProcessSample(dst: *ProcessSample, src: FdinfoSample) void {
    if (src.mem_kib) |mem| dst.mem_kib = (dst.mem_kib orelse 0) + mem;
    addEngineSample(&dst.engine, src.engine);
    addEngineSample(&dst.encoder, src.encoder);
    addEngineSample(&dst.decoder, src.decoder);
}

fn addEngineSample(dst: *?EngineSample, src: ?EngineSample) void {
    if (src) |engine| addEngineTime(dst, engine.engine_ns, engine.timestamp_ns);
}

fn addEngineTime(dst: *?EngineSample, engine_ns: u64, timestamp_ns: u64) void {
    const current = dst.* orelse EngineSample{ .engine_ns = 0, .timestamp_ns = timestamp_ns };
    dst.* = .{
        .engine_ns = current.engine_ns + engine_ns,
        .timestamp_ns = timestamp_ns,
    };
}

fn matchDevice(devices: []const Device, bus_device_identifier: [bus_device_identifier_len]u8) ?u16 {
    for (devices, 0..) |dev, idx| {
        if (dev.bus_device_identifier) |device_bus_device_identifier| {
            if (std.mem.eql(u8, &device_bus_device_identifier, &bus_device_identifier)) return @intCast(idx);
        }
    }
    return null;
}

fn processKey(pid: u32, device_idx: u16) u64 {
    return (@as(u64, device_idx) << 32) | pid;
}

inline fn drmClientKey(device_idx: u16, client_id: u64) u128 {
    return (@as(u128, device_idx) << 64) | @as(u128, client_id);
}

fn markDrmClientSeen(
    allocator: std.mem.Allocator,
    seen_device_clients: *std.AutoHashMapUnmanaged(u128, void),
    device_idx: u16,
    client_id: u64,
) !bool {
    return (try seen_device_clients.getOrPut(allocator, drmClientKey(device_idx, client_id))).found_existing;
}

pub const BdfParts = struct {
    domain: u32,
    bus: u32,
    device: u32,
    function: u32,
};

pub fn formatBdf(parts: BdfParts) [bus_device_identifier_len]u8 {
    var buf: [bus_device_identifier_len]u8 = undefined;
    _ = std.fmt.bufPrint(&buf, "{x:0>4}:{x:0>2}:{x:0>2}.{x}", .{
        parts.domain,
        parts.bus,
        parts.device,
        parts.function,
    }) catch unreachable;
    return buf;
}

inline fn parseBdf(raw: []const u8) ?[bus_device_identifier_len]u8 {
    const trimmed = std.mem.trim(u8, raw, " \t\n");
    var out: [bus_device_identifier_len]u8 = undefined;
    if (trimmed.len == bus_device_identifier_len) {
        @memcpy(&out, trimmed[0..bus_device_identifier_len]);
        return out;
    }
    if (trimmed.len == "00:00.0".len) {
        _ = std.fmt.bufPrint(&out, "0000:{s}", .{trimmed}) catch return null;
        return out;
    }
    return null;
}

inline fn firstInt(raw: []const u8) u64 {
    var iter = std.mem.tokenizeAny(u8, raw, " \t");
    return std.fmt.parseInt(u64, iter.next() orelse return 0, 10) catch 0;
}

inline fn memoryKiB(raw: []const u8) u64 {
    var iter = std.mem.tokenizeAny(u8, raw, " \t");
    const value = std.fmt.parseInt(u64, iter.next() orelse return 0, 10) catch return 0;
    const unit = iter.next() orelse return value;
    if (std.ascii.eqlIgnoreCase(unit, "KiB")) return value;
    if (std.ascii.eqlIgnoreCase(unit, "MiB")) return value * 1024;
    if (std.ascii.eqlIgnoreCase(unit, "GiB")) return value * 1024 * 1024;
    if (std.ascii.eqlIgnoreCase(unit, "B")) return value / 1024;
    return value;
}

inline fn containsAsciiIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    for (0..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

test "oneAPI fdinfo parser" {
    var parsed: ParsedFdinfo = .{};
    parseFdinfoLine(&parsed, "drm-pdev:\t0000:03:00.0", 10);
    parseFdinfoLine(&parsed, "drm-client-id:\t42", 10);
    parseFdinfoLine(&parsed, "drm-total-memory:\t2048 KiB", 10);
    parseFdinfoLine(&parsed, "drm-engine-render:\t100 ns", 10);
    parseFdinfoLine(&parsed, "drm-engine-compute:\t50 ns", 10);
    finishFdinfo(&parsed);

    try std.testing.expect(parsed.bus_device_identifier != null);
    try std.testing.expectEqualSlices(u8, "0000:03:00.0", &parsed.bus_device_identifier.?);
    try std.testing.expectEqual(@as(?u64, 42), parsed.drm_client_id);
    try std.testing.expectEqual(@as(?u64, 2048), parsed.sample.mem_kib);
    try std.testing.expectEqual(@as(u64, 150), parsed.sample.engine.?.engine_ns);
    try std.testing.expectEqual(@as(?EngineSample, null), parsed.sample.encoder);
    try std.testing.expectEqual(@as(?EngineSample, null), parsed.sample.decoder);
}

test "oneAPI fdinfo parser classifies explicit media engines" {
    var parsed: ParsedFdinfo = .{};
    parseFdinfoLine(&parsed, "drm-engine-video-decode:\t100 ns", 10);
    parseFdinfoLine(&parsed, "drm-engine-media-encode:\t50 ns", 10);
    parseFdinfoLine(&parsed, "drm-engine-VIDEO-DECODE1:\t25 ns", 10);
    finishFdinfo(&parsed);

    try std.testing.expectEqual(@as(u64, 175), parsed.sample.engine.?.engine_ns);
    try std.testing.expectEqual(@as(u64, 50), parsed.sample.encoder.?.engine_ns);
    try std.testing.expectEqual(@as(u64, 125), parsed.sample.decoder.?.engine_ns);
}

test "oneAPI fdinfo parser leaves generic video engines unclassified" {
    var parsed: ParsedFdinfo = .{};
    parseFdinfoLine(&parsed, "drm-engine-video:\t100 ns", 10);
    parseFdinfoLine(&parsed, "drm-engine-vcs0:\t50 ns", 10);
    parseFdinfoLine(&parsed, "drm-engine-vecs0:\t25 ns", 10);
    finishFdinfo(&parsed);

    try std.testing.expectEqual(@as(u64, 175), parsed.sample.engine.?.engine_ns);
    try std.testing.expectEqual(@as(?EngineSample, null), parsed.sample.encoder);
    try std.testing.expectEqual(@as(?EngineSample, null), parsed.sample.decoder);
}

test "oneAPI fdinfo parser cycle counters" {
    var parsed: ParsedFdinfo = .{};
    parseFdinfoLine(&parsed, "drm-pdev:\t0000:03:00.0", 10);
    parseFdinfoLine(&parsed, "drm-cycles-rcs:\t10", 10);
    parseFdinfoLine(&parsed, "drm-total-cycles-rcs:\t100", 10);
    parseFdinfoLine(&parsed, "drm-engine-capacity-vcs:\t2", 10);
    parseFdinfoLine(&parsed, "drm-cycles-ccs:\t30", 10);
    parseFdinfoLine(&parsed, "drm-total-cycles-ccs:\t100", 10);
    finishFdinfo(&parsed);

    try std.testing.expectEqual(@as(u64, 40), parsed.sample.engine.?.engine_ns);
    try std.testing.expectEqual(@as(u64, 100), parsed.sample.engine.?.timestamp_ns);
}

test "oneAPI fdinfo parser xe vram regions" {
    var parsed: ParsedFdinfo = .{};
    parseFdinfoLine(&parsed, "drm-pdev:\t0000:03:00.0", 10);
    parseFdinfoLine(&parsed, "drm-total-system:\t184 MiB", 10);
    parseFdinfoLine(&parsed, "drm-total-vram0:\t4096 KiB", 10);
    parseFdinfoLine(&parsed, "drm-resident-vram0:\t3072 KiB", 10);
    finishFdinfo(&parsed);

    try std.testing.expectEqual(@as(?u64, 3072), parsed.sample.mem_kib);
}

test "oneAPI fdinfo parser falls back to total vram" {
    var parsed: ParsedFdinfo = .{};
    parseFdinfoLine(&parsed, "drm-total-vram0:\t2 MiB", 10);
    finishFdinfo(&parsed);

    try std.testing.expectEqual(@as(?u64, 2048), parsed.sample.mem_kib);
}

test "oneAPI DRM clients dedup per device" {
    var seen_device_clients: std.AutoHashMapUnmanaged(u128, void) = .{};
    defer seen_device_clients.deinit(std.testing.allocator);

    try std.testing.expectEqual(false, try markDrmClientSeen(std.testing.allocator, &seen_device_clients, 0, 42));
    try std.testing.expectEqual(true, try markDrmClientSeen(std.testing.allocator, &seen_device_clients, 0, 42));
    try std.testing.expectEqual(false, try markDrmClientSeen(std.testing.allocator, &seen_device_clients, 1, 42));
    try std.testing.expectEqual(false, try markDrmClientSeen(std.testing.allocator, &seen_device_clients, 0, 43));
}

test "oneAPI process utilization delta" {
    try std.testing.expectEqual(@as(?u16, 50), processUtil(.{ .engine_ns = 100, .timestamp_ns = 1000 }, .{ .engine_ns = 600, .timestamp_ns = 2000 }));
    try std.testing.expectEqual(@as(?u16, 1), processUtil(.{ .engine_ns = 100, .timestamp_ns = 1000 }, .{ .engine_ns = 101, .timestamp_ns = 2000 }));
    try std.testing.expectEqual(@as(?u16, null), processUtil(null, .{ .engine_ns = 600, .timestamp_ns = 2000 }));
}

test "oneAPI PWM fan speed conversion" {
    try std.testing.expectEqual(@as(u64, 0), pwmToPercent(0));
    try std.testing.expectEqual(@as(u64, 50), pwmToPercent(128));
    try std.testing.expectEqual(@as(u64, 100), pwmToPercent(255));
    try std.testing.expectEqual(@as(u64, 100), pwmToPercent(300));
}

test "oneAPI process previous keeps engine samples" {
    var usage: ProcessUsage = .{};
    defer usage.deinit(std.testing.allocator);

    const engine_key = processKey(10, 1);
    const mem_only_key = processKey(20, 1);
    try usage.put(std.testing.allocator, engine_key, .{
        .pid = 10,
        .device_idx = 1,
        .engine = .{ .engine_ns = 100, .timestamp_ns = 1000 },
    });
    try usage.put(std.testing.allocator, mem_only_key, .{
        .pid = 20,
        .device_idx = 1,
        .mem_kib = 2048,
    });

    var previous: ProcessPrevious = .{};
    defer previous.deinit(std.testing.allocator);

    try saveProcessPrevious(std.testing.allocator, &previous, &usage);

    try std.testing.expectEqual(@as(u32, 1), previous.count());
    try std.testing.expectEqual(@as(?EngineSample, .{ .engine_ns = 100, .timestamp_ns = 1000 }), previous.get(engine_key));
    try std.testing.expectEqual(@as(?EngineSample, null), previous.get(mem_only_key));
}

test "oneAPI power delta from hwmon energy" {
    try std.testing.expectEqual(@as(?u64, 20_000), milliwattsFromEnergyDelta(1_000, 11_000, 1_000, 501_000));
    try std.testing.expectEqual(@as(?u64, null), milliwattsFromEnergyDelta(11_000, 1_000, 1_000, 501_000));
    try std.testing.expectEqual(@as(?u64, null), milliwattsFromEnergyDelta(1_000, 11_000, 501_000, 1_000));
}

test "oneAPI resource line chooses large memory BARs" {
    try std.testing.expectEqual(@as(?u64, 34_359_738_368), resourceLineSize("0x0000048800000000 0x0000048fffffffff 0x000000000014220c"));
    try std.testing.expectEqual(@as(?u64, null), resourceLineSize("0x00000000f6c00000 0x00000000f6dfffff 0x0000000000046200"));
}

test "oneAPI PCIe speed maps to generation" {
    try std.testing.expectEqual(@as(u64, 1), try pcieGenFromSpeed("2.5 GT/s PCIe"));
    try std.testing.expectEqual(@as(u64, 4), try pcieGenFromSpeed("16.0 GT/s PCIe"));
    try std.testing.expectError(error.NotFound, pcieGenFromSpeed("Unknown"));
}

test "oneAPI PCIe bandwidth derives from generation and width" {
    try std.testing.expectEqual(@as(?u64, 250), pcieBandwidthFromLink(1, 1));
    try std.testing.expectEqual(@as(?u64, 63_008), pcieBandwidthFromLink(5, 16));
    try std.testing.expectEqual(@as(?u64, null), pcieBandwidthFromLink(0, 16));
}
