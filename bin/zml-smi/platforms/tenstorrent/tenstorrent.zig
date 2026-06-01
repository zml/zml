const std = @import("std");
const c = @import("c");
const DynLib = @import("zml-smi/dynlib");
const sandbox = @import("zml-smi/sandbox");
const smi_sysfs = @import("zml-smi/sysfs");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

// ── tt-umd binding ──────────────────────────────────────────────────────────

pub const Umd = struct {
    pub const Error = error{TtUmdUnavailable};

    const Fns = struct {
        ttumd_open: *const @TypeOf(c.ttumd_open),
        ttumd_close: *const @TypeOf(c.ttumd_close),
        ttumd_chip_count: *const @TypeOf(c.ttumd_chip_count),
        ttumd_is_remote: *const @TypeOf(c.ttumd_is_remote),
        ttumd_device_name: *const @TypeOf(c.ttumd_device_name),
        ttumd_asic_location: *const @TypeOf(c.ttumd_asic_location),
        ttumd_board_id: *const @TypeOf(c.ttumd_board_id),
        ttumd_asic_id: *const @TypeOf(c.ttumd_asic_id),
        ttumd_mem_total_bytes: *const @TypeOf(c.ttumd_mem_total_bytes),
        ttumd_fw_bundle: *const @TypeOf(c.ttumd_fw_bundle),
        ttumd_eth_fw: *const @TypeOf(c.ttumd_eth_fw),
        ttumd_cm_fw: *const @TypeOf(c.ttumd_cm_fw),
        ttumd_dm_app: *const @TypeOf(c.ttumd_dm_app),
        ttumd_temperature_mc: *const @TypeOf(c.ttumd_temperature_mc),
        ttumd_temperature_limit_mc: *const @TypeOf(c.ttumd_temperature_limit_mc),
        ttumd_board_temperature_mc: *const @TypeOf(c.ttumd_board_temperature_mc),
        ttumd_dram_temperature_mc: *const @TypeOf(c.ttumd_dram_temperature_mc),
        ttumd_power_mw: *const @TypeOf(c.ttumd_power_mw),
        ttumd_power_limit_mw: *const @TypeOf(c.ttumd_power_limit_mw),
        ttumd_voltage_mv: *const @TypeOf(c.ttumd_voltage_mv),
        ttumd_current_ma: *const @TypeOf(c.ttumd_current_ma),
        ttumd_aiclk_mhz: *const @TypeOf(c.ttumd_aiclk_mhz),
        ttumd_arcclk_mhz: *const @TypeOf(c.ttumd_arcclk_mhz),
        ttumd_axiclk_mhz: *const @TypeOf(c.ttumd_axiclk_mhz),
        ttumd_dram_mhz: *const @TypeOf(c.ttumd_dram_mhz),
        ttumd_heartbeat: *const @TypeOf(c.ttumd_heartbeat),
        ttumd_therm_trip_count: *const @TypeOf(c.ttumd_therm_trip_count),
        ttumd_fan_rpm: *const @TypeOf(c.ttumd_fan_rpm),
    };

    lib: Fns,
    ctx: *c.ttumd_ctx,

    pub fn init() Error!Umd {
        _ = setenv("TT_LOGGER_LEVEL", "error", 1); // Quiet tt-logger

        var sandbox_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const sandbox_path = sandbox.path(&sandbox_buf) orelse return error.TtUmdUnavailable;

        var lib_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const lib_path = std.fmt.bufPrintZ(&lib_buf, "{s}/lib/libtt_umd.so", .{sandbox_path}) catch
            return error.TtUmdUnavailable;

        var dynlib: std.DynLib = .{ .inner = .{
            .handle = std.c.dlopen(lib_path, .{ .LAZY = true, .GLOBAL = true, .NODELETE = true }) orelse {
                if (std.c.dlerror()) |err| std.log.debug("tt-umd: dlopen: {s}", .{err});
                return error.TtUmdUnavailable;
            },
        } };
        const fns = DynLib.lookupStruct(&dynlib, Fns) catch return error.TtUmdUnavailable;

        const ctx = fns.ttumd_open() orelse return error.TtUmdUnavailable;
        if (fns.ttumd_chip_count(ctx) == 0) {
            fns.ttumd_close(ctx);
            return error.TtUmdUnavailable;
        }

        return .{ .lib = fns, .ctx = ctx };
    }

    pub fn chipCount(self: Umd) u32 {
        return self.lib.ttumd_chip_count(self.ctx);
    }

    pub fn isRemote(self: Umd, index: u32) bool {
        return self.lib.ttumd_is_remote(self.ctx, index) == 1;
    }

    /// Caller buffer size for the string readers below (mirrors nvml's name_buf_len).
    pub const str_buf_len = c.TTUMD_STR_BUF_LEN;

    /// Fills `buf` with the shim's display name (e.g. "Wormhole n300 · ASIC 1") and returns a slice of it.
    pub fn name(self: Umd, index: u32, buf: *[str_buf_len]u8) ![:0]const u8 {
        if (self.lib.ttumd_device_name(self.ctx, index, buf, buf.len) != 0) return error.Unavailable;
        return std.mem.span(@as([*c]const u8, @ptrCast(buf)));
    }

    pub fn asicLocation(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_asic_location(self.ctx, index));
    }

    pub fn boardSerial(self: Umd, allocator: std.mem.Allocator, index: u32) ![]const u8 {
        return hexId(allocator, self.lib.ttumd_board_id(self.ctx, index));
    }

    pub fn asicId(self: Umd, allocator: std.mem.Allocator, index: u32) ![]const u8 {
        return hexId(allocator, self.lib.ttumd_asic_id(self.ctx, index));
    }

    pub fn memTotal(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_mem_total_bytes(self.ctx, index));
    }

    pub fn fwBundle(self: Umd, index: u32, buf: *[str_buf_len]u8) ![:0]const u8 {
        if (self.lib.ttumd_fw_bundle(self.ctx, index, buf, buf.len) != 0) return error.Unavailable;
        return std.mem.span(@as([*c]const u8, @ptrCast(buf)));
    }
    pub fn ethFw(self: Umd, index: u32, buf: *[str_buf_len]u8) ![:0]const u8 {
        if (self.lib.ttumd_eth_fw(self.ctx, index, buf, buf.len) != 0) return error.Unavailable;
        return std.mem.span(@as([*c]const u8, @ptrCast(buf)));
    }
    pub fn cmFw(self: Umd, index: u32, buf: *[str_buf_len]u8) ![:0]const u8 {
        if (self.lib.ttumd_cm_fw(self.ctx, index, buf, buf.len) != 0) return error.Unavailable;
        return std.mem.span(@as([*c]const u8, @ptrCast(buf)));
    }
    pub fn dmApp(self: Umd, index: u32, buf: *[str_buf_len]u8) ![:0]const u8 {
        if (self.lib.ttumd_dm_app(self.ctx, index, buf, buf.len) != 0) return error.Unavailable;
        return std.mem.span(@as([*c]const u8, @ptrCast(buf)));
    }

    // Live telemetry readers. Each crosses into the shim for one field and
    // returns error.Unavailable when the reading is absent.
    pub fn temperature(self: Umd, index: u32) !u64 {
        return milliToWhole(self.lib.ttumd_temperature_mc(self.ctx, index));
    }
    pub fn temperatureMax(self: Umd, index: u32) !u64 {
        return milliToWhole(self.lib.ttumd_temperature_limit_mc(self.ctx, index));
    }
    pub fn boardTemperature(self: Umd, index: u32) !u64 {
        return milliToWhole(self.lib.ttumd_board_temperature_mc(self.ctx, index));
    }
    pub fn dramTemperature(self: Umd, index: u32) !u64 {
        return milliToWhole(self.lib.ttumd_dram_temperature_mc(self.ctx, index));
    }
    pub fn powerUsage(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_power_mw(self.ctx, index));
    }
    pub fn powerLimit(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_power_limit_mw(self.ctx, index));
    }
    pub fn voltage(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_voltage_mv(self.ctx, index));
    }
    pub fn current(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_current_ma(self.ctx, index));
    }
    pub fn clockAi(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_aiclk_mhz(self.ctx, index));
    }
    pub fn clockArc(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_arcclk_mhz(self.ctx, index));
    }
    pub fn clockAxi(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_axiclk_mhz(self.ctx, index));
    }
    pub fn clockMem(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_dram_mhz(self.ctx, index));
    }
    pub fn heartbeat(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_heartbeat(self.ctx, index));
    }
    pub fn thermTripCount(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_therm_trip_count(self.ctx, index));
    }
    pub fn fanRpm(self: Umd, index: u32) !u64 {
        return value(self.lib.ttumd_fan_rpm(self.ctx, index));
    }

    fn hexId(allocator: std.mem.Allocator, id: u64) ![]const u8 {
        if (id == 0) return error.NotFound;
        return std.fmt.allocPrint(allocator, "0x{x}", .{id});
    }

    /// The C ABI marks absent with a negative
    fn value(raw: i64) !u64 {
        return if (raw < 0) error.Unavailable else @intCast(raw);
    }

    fn milliToWhole(raw: i64) !u64 {
        if (raw < 0) return error.Unavailable;
        return @intFromFloat(@round(@as(f64, @floatFromInt(raw)) / 1000.0));
    }
};

// ── sysfs / procfs helpers ──────────────────────────────────────────────────
// TODO: create a generic helper for sysfs access and metrics retrieval once we have enough platforms relying on sysfs

pub const Sysfs = struct {
    const class_root = "/sys/class/tenstorrent";

    pub const Handle = enum(u32) { _ };

    pub const Device = struct {
        index: u32,
        base: []const u8,
        device_path: []const u8,
        hwmon_path: ?[]const u8,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    devices: []Device,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Sysfs {
        return .{
            .allocator = allocator,
            .io = io,
            .devices = try discover(allocator, io),
        };
    }

    fn discover(allocator: std.mem.Allocator, io: std.Io) ![]Device {
        var devices: std.ArrayList(Device) = .empty;
        errdefer devices.deinit(allocator);

        var dir = std.Io.Dir.openDirAbsolute(io, class_root, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => return &.{},
            else => |e| return e,
        };
        defer dir.close(io);

        var it = dir.iterate();
        while (it.next(io) catch null) |entry| {
            // sysfs class entries are named "tenstorrent!N" (the "!" maps to "/").
            const sep = std.mem.indexOfScalar(u8, entry.name, '!') orelse continue;
            const index = std.fmt.parseInt(u32, entry.name[sep + 1 ..], 10) catch continue;

            const base = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ class_root, entry.name });
            const device_path = try std.fmt.allocPrint(allocator, "{s}/device", .{base});
            try devices.append(allocator, .{
                .index = index,
                .base = base,
                .device_path = device_path,
                .hwmon_path = findHwmon(allocator, io, device_path) catch null,
            });
        }

        std.mem.sort(Device, devices.items, {}, struct {
            fn lessThan(_: void, lhs: Device, rhs: Device) bool {
                return lhs.index < rhs.index;
            }
        }.lessThan);
        return devices.toOwnedSlice(allocator);
    }

    pub fn handleByIndex(self: *const Sysfs, device_id: usize) !Handle {
        if (device_id >= self.devices.len) return error.NotFound;
        return @enumFromInt(@as(u32, @intCast(device_id)));
    }

    fn device(self: *const Sysfs, handle: Handle) !*const Device {
        const idx = @intFromEnum(handle);
        if (idx >= self.devices.len) return error.NotFound;
        return &self.devices[idx];
    }

    /// TT-KMD (host kernel driver) version, e.g. "TT-KMD 2.7.1-pre". Host-global,
    /// read from the tenstorrent module's sysfs node (not a per-chip property).
    pub fn driverVersion(self: *const Sysfs, allocator: std.mem.Allocator) ?[]const u8 {
        const raw = smi_sysfs.readString(allocator, self.io, "/sys/module/tenstorrent/version") catch return null;
        const text = std.mem.trim(u8, raw, &std.ascii.whitespace);
        if (text.len == 0) return null;
        return std.fmt.allocPrint(allocator, "TT-KMD {s}", .{text}) catch null;
    }

    /// Board-level power cap from hwmon; fallback when tt-umd omits it.
    pub fn powerLimit(self: *const Sysfs, allocator: std.mem.Allocator, handle: Handle) !u64 {
        const dev = try self.device(handle);
        const hwmon = dev.hwmon_path orelse return error.NotFound;
        return (try readHwmon(allocator, self.io, hwmon, "power1_max")) / 1000;
    }

    // PCIe link (local chip only; not in tt-umd).

    pub fn pcieLinkGen(self: *const Sysfs, allocator: std.mem.Allocator, handle: Handle) !u64 {
        const dev = try self.device(handle);
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/current_link_speed", .{dev.device_path});
        const raw = try smi_sysfs.readString(allocator, self.io, path);
        return pcieGenFromSpeed(raw);
    }

    pub fn pcieLinkWidth(self: *const Sysfs, allocator: std.mem.Allocator, handle: Handle) !u64 {
        const dev = try self.device(handle);
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/current_link_width", .{dev.device_path});
        return smi_sysfs.readInt(allocator, self.io, path);
    }

    pub fn pcieBandwidth(self: *const Sysfs, allocator: std.mem.Allocator, handle: Handle) !u64 {
        const gen = try self.pcieLinkGen(allocator, handle);
        const width = try self.pcieLinkWidth(allocator, handle);
        return pcieBandwidthFromLink(gen, width) orelse error.NotFound;
    }

    fn readHwmon(allocator: std.mem.Allocator, io: std.Io, hwmon: []const u8, file: []const u8) !u64 {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ hwmon, file });
        return smi_sysfs.readInt(allocator, io, path);
    }

    fn findHwmon(allocator: std.mem.Allocator, io: std.Io, device_path: []const u8) !?[]const u8 {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const hwmon_root = try std.fmt.bufPrint(&path_buf, "{s}/hwmon", .{device_path});
        var dir = std.Io.Dir.openDirAbsolute(io, hwmon_root, .{ .iterate = true }) catch return null;
        defer dir.close(io);

        var it = dir.iterate();
        while (it.next(io) catch null) |entry| {
            if (!std.mem.startsWith(u8, entry.name, "hwmon")) continue;
            return try std.fmt.allocPrint(allocator, "{s}/{s}", .{ hwmon_root, entry.name });
        }
        return null;
    }

    fn pcieGenFromSpeed(raw: []const u8) !u64 {
        const trimmed = std.mem.trim(u8, raw, &std.ascii.whitespace);
        if (std.mem.startsWith(u8, trimmed, "2.5")) return 1;
        if (std.mem.startsWith(u8, trimmed, "5.0") or std.mem.startsWith(u8, trimmed, "5 ")) return 2;
        if (std.mem.startsWith(u8, trimmed, "8.0") or std.mem.startsWith(u8, trimmed, "8 ")) return 3;
        if (std.mem.startsWith(u8, trimmed, "16.0") or std.mem.startsWith(u8, trimmed, "16 ")) return 4;
        if (std.mem.startsWith(u8, trimmed, "32.0") or std.mem.startsWith(u8, trimmed, "32 ")) return 5;
        if (std.mem.startsWith(u8, trimmed, "64.0") or std.mem.startsWith(u8, trimmed, "64 ")) return 6;
        return error.NotFound;
    }

    fn pcieBandwidthFromLink(gen: u64, width: u64) ?u64 {
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
};
