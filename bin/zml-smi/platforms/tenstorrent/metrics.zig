const std = @import("std");
const Umd = @import("tenstorrent.zig").Umd;
const Sysfs = @import("tenstorrent.zig").Sysfs;
const device_info = @import("zml-smi/info").device_info;
const TenstorrentInfo = device_info.TenstorrentInfo;
const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;
const Collector = @import("zml-smi/collector").Collector;
const poll_metrics = @import("zml-smi/info").poll_metrics;
const process = @import("process.zig");

pub fn start(collector: *Collector) !void {
    const umd = try collector.arena.create(Umd);
    umd.* = Umd.init() catch |err| switch (err) {
        error.TtUmdUnavailable => return,
    };

    const sysfs = try collector.arena.create(Sysfs);
    sysfs.* = Sysfs.init(collector.arena, collector.io) catch .{
        .allocator = collector.arena,
        .io = collector.io,
        .devices = &.{},
    };

    const dev_offset: u16 = @intCast(collector.device_infos.items.len);

    const board_power_limit_mw: ?u64 = if (sysfs.devices.len > 0)
        (sysfs.powerLimit(collector.arena, @enumFromInt(0)) catch null)
    else
        null;

    // Host kernel-driver version (TT-KMD), shared by every chip.
    const driver_version = sysfs.driverVersion(collector.arena);

    var chip: u32 = 0;
    while (chip < umd.chipCount()) : (chip += 1) {
        const poll_arena = try collector.createPollArena();
        const is_remote = umd.isRemote(chip);

        // Reused across the identity strings below: each is duped into the arena
        // before the next call overwrites it.
        var str_buf: [Umd.str_buf_len]u8 = undefined;
        const initial: TenstorrentInfo = .{
            .name = if (umd.name(chip, &str_buf)) |s| collector.arena.dupe(u8, s) catch null else |_| null,
            .driver_version = driver_version,
            .fw_bundle_version = if (umd.fwBundle(chip, &str_buf)) |s| collector.arena.dupe(u8, s) catch null else |_| null,
            .cm_fw_version = if (umd.cmFw(chip, &str_buf)) |s| collector.arena.dupe(u8, s) catch null else |_| null,
            .eth_fw_version = if (umd.ethFw(chip, &str_buf)) |s| collector.arena.dupe(u8, s) catch null else |_| null,
            .dm_app_version = if (umd.dmApp(chip, &str_buf)) |s| collector.arena.dupe(u8, s) catch null else |_| null,
            .board_serial = umd.boardSerial(collector.arena, chip) catch null,
            .asic_id = umd.asicId(collector.arena, chip) catch null,
            .asic_location = umd.asicLocation(chip) catch null,
            .mem_total_bytes = umd.memTotal(chip) catch null,
        };
        const info = try collector.addDevice(.{ .tenstorrent = .{ .values = .{ initial, initial } } });

        const dev: Device = .{
            .umd = umd,
            .chip = chip,
            .sysfs = if (!is_remote and sysfs.devices.len > 0) sysfs else null, // PCIe attributes exist only for the local chip
            .board_power_limit_mw = board_power_limit_mw,
            .arena = poll_arena,
        };
        try collector.spawnPoll(pollOnce, .{ poll_arena, &info.tenstorrent, dev });
    }

    const processes = try collector.createProcessList();
    try process.init(collector, processes, sysfs, dev_offset);
}

const pollOnce = poll_metrics.poll(*DoubleBuffer(TenstorrentInfo), Device, metrics);

const Device = struct {
    umd: *Umd,
    sysfs: ?*const Sysfs,

    chip: u32,
    board_power_limit_mw: ?u64,

    arena: *std.heap.ArenaAllocator,

    pub fn temperature(self: Device) !u64 {
        return self.umd.temperature(self.chip);
    }
    pub fn temperatureMax(self: Device) !u64 {
        return self.umd.temperatureMax(self.chip);
    }
    pub fn boardTemperature(self: Device) !u64 {
        return self.umd.boardTemperature(self.chip);
    }
    pub fn dramTemperature(self: Device) !u64 {
        return self.umd.dramTemperature(self.chip);
    }
    pub fn powerUsage(self: Device) !u64 {
        return self.umd.powerUsage(self.chip);
    }
    pub fn powerLimit(self: Device) !u64 {
        // tt-umd's board power limit, with the sysfs hwmon cap as fallback.
        return self.umd.powerLimit(self.chip) catch
            (self.board_power_limit_mw orelse return error.Unavailable);
    }
    pub fn voltage(self: Device) !u64 {
        return self.umd.voltage(self.chip);
    }
    pub fn current(self: Device) !u64 {
        return self.umd.current(self.chip);
    }
    pub fn clockAi(self: Device) !u64 {
        return self.umd.clockAi(self.chip);
    }
    pub fn clockArc(self: Device) !u64 {
        return self.umd.clockArc(self.chip);
    }
    pub fn clockAxi(self: Device) !u64 {
        return self.umd.clockAxi(self.chip);
    }
    pub fn clockMem(self: Device) !u64 {
        return self.umd.clockMem(self.chip);
    }
    pub fn heartbeat(self: Device) !u64 {
        return self.umd.heartbeat(self.chip);
    }
    pub fn thermTripCount(self: Device) !u64 {
        return self.umd.thermTripCount(self.chip);
    }
    pub fn fanRpm(self: Device) !u64 {
        return self.umd.fanRpm(self.chip);
    }

    pub fn pcieLinkGen(self: Device) !u64 {
        const s = self.sysfs orelse return error.Unavailable;
        return s.pcieLinkGen(self.arena.allocator(), @enumFromInt(0));
    }
    pub fn pcieLinkWidth(self: Device) !u64 {
        const s = self.sysfs orelse return error.Unavailable;
        return s.pcieLinkWidth(self.arena.allocator(), @enumFromInt(0));
    }
    pub fn pcieBandwidth(self: Device) !u64 {
        const s = self.sysfs orelse return error.Unavailable;
        return s.pcieBandwidth(self.arena.allocator(), @enumFromInt(0));
    }
};

const metrics = .{
    .{ .field = "temperature", .query = Device.temperature },
    .{ .field = "temperature_max", .query = Device.temperatureMax },
    .{ .field = "board_temperature", .query = Device.boardTemperature },
    .{ .field = "dram_temperature", .query = Device.dramTemperature },
    .{ .field = "power_mw", .query = Device.powerUsage },
    .{ .field = "power_limit_mw", .query = Device.powerLimit },
    .{ .field = "voltage_mv", .query = Device.voltage },
    .{ .field = "current_ma", .query = Device.current },
    .{ .field = "clock_ai_mhz", .query = Device.clockAi },
    .{ .field = "clock_arc_mhz", .query = Device.clockArc },
    .{ .field = "clock_axi_mhz", .query = Device.clockAxi },
    .{ .field = "clock_mem_mhz", .query = Device.clockMem },
    .{ .field = "heartbeat", .query = Device.heartbeat },
    .{ .field = "therm_trip_count", .query = Device.thermTripCount },
    .{ .field = "fan_rpm", .query = Device.fanRpm },
    .{ .field = "pcie_link_gen", .query = Device.pcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.pcieLinkWidth },
    .{ .field = "pcie_bandwidth_mbps", .query = Device.pcieBandwidth },
};
