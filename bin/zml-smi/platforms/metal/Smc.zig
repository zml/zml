//! Inspired by mactop.

const std = @import("std");
const iokit = @import("iokit.zig");

const Smc = @This();

api: iokit.Api,
connection: iokit.Object,
temperatures: []const Key,
fans: []const Fan,
/// Whole-machine power: the system total (laptops), the DC-in total, then the DC-in rail,
/// which is the power supply's output on a Mac Studio. The first that reads a value wins.
power: []const Key,

const Key = struct { code: u32, info: Info };
const Fan = struct { speed: Key, min_rpm: f64, max_rpm: f64 };
const Info = extern struct { size: u32 = 0, type: u32 = 0, attributes: u8 = 0 };

/// SMCKeyData_t: what each call sends and receives.
const Call = extern struct {
    key: u32 = 0,
    version: extern struct { major: u8 = 0, minor: u8 = 0, build: u8 = 0, reserved: u8 = 0, release: u16 = 0 } = .{},
    power_limits: extern struct { version: u16 = 0, length: u16 = 0, cpu: u32 = 0, gpu: u32 = 0, memory: u32 = 0 } = .{},
    info: Info = .{},
    result: u8 = 0,
    status: u8 = 0,
    command: Command = .read_bytes,
    index: u32 = 0,
    bytes: [32]u8 = @splat(0),

    const Command = enum(u8) { read_bytes = 5, read_index = 8, read_info = 9, _ };
};

comptime {
    std.debug.assert(@sizeOf(Call) == 80);
}

/// A GPU sensor at or below this has not started: it reads about 2 °C right after boot.
const min_celsius = 10;
const max_celsius = 150;
const max_watts = 1000;

pub fn open(arena: std.mem.Allocator, api: iokit.Api) !Smc {
    const service = api.IOServiceGetMatchingService(iokit.main_port, api.IOServiceMatching("AppleSMC"));
    if (service == 0) return error.SmcNotFound;
    defer _ = api.IOObjectRelease(service);

    var connection: iokit.Object = 0;
    try iokit.check(api.IOServiceOpen(service, std.c.mach_task_self(), 0, &connection));

    var smc: Smc = .{
        .api = api,
        .connection = connection,
        .temperatures = &.{},
        .fans = &.{},
        .power = &.{},
    };
    errdefer smc.close();

    smc.temperatures = try smc.gpuSensors(arena);
    smc.fans = try smc.findFans(arena);
    smc.power = try smc.powerKeys(arena);

    return smc;
}

pub fn close(self: Smc) void {
    _ = self.api.IOServiceClose(self.connection);
}

pub fn temperature(self: Smc) ?u64 {
    var hottest: ?f64 = null;

    for (self.temperatures) |sensor| {
        const celsius = self.read(sensor) catch continue;
        if (!(celsius > min_celsius and celsius <= max_celsius)) continue;
        hottest = @max(hottest orelse celsius, celsius);
    }

    return if (hottest) |celsius| @intFromFloat(@round(celsius)) else null;
}

pub fn fanPercent(self: Smc) ?u64 {
    var fastest: ?u64 = null;

    for (self.fans) |fan| {
        const rpm = self.read(fan.speed) catch continue;
        if (!std.math.isFinite(rpm)) continue;
        // A fan that cannot stop and reads 0 hides its speed from this user.
        if (rpm <= 0 and fan.min_rpm > 0) continue;

        const percent: u64 = @intFromFloat(@round(std.math.clamp(rpm / fan.max_rpm * 100, 0, 100)));
        fastest = @max(fastest orelse percent, percent);
    }

    return fastest;
}

pub fn systemPower(self: Smc) ?u64 {
    for (self.power) |key| {
        const watts = self.read(key) catch continue;
        if (watts > 0 and watts <= max_watts) return @intFromFloat(@round(watts * 1000));
    }

    return null;
}

fn gpuSensors(self: Smc, arena: std.mem.Allocator) ![]const Key {
    const count = self.readCode(code("#KEY")) catch return &.{};
    var sensors: std.ArrayList(Key) = .empty;

    for (0..@intFromFloat(count)) |index| {
        const name = (self.call(.{ .command = .read_index, .index = @intCast(index) }) catch continue).key;
        if (name >> 16 != std.mem.readInt(u16, "Tg", .big)) continue;
        const key = self.find(name) catch continue;
        if (key.info.type == code("flt ")) try sensors.append(arena, key);
    }

    return sensors.items;
}

fn findFans(self: Smc, arena: std.mem.Allocator) ![]const Fan {
    const count = self.readCode(code("FNum")) catch return &.{};
    var fans: std.ArrayList(Fan) = .empty;

    for (0..@intFromFloat(@min(count, 10))) |index| {
        const max_rpm = self.readCode(fanKey(index, "Mx")) catch continue;
        if (max_rpm <= 0) continue;
        try fans.append(arena, .{
            .speed = self.find(fanKey(index, "Ac")) catch continue,
            .min_rpm = self.readCode(fanKey(index, "Mn")) catch 0,
            .max_rpm = max_rpm,
        });
    }

    return fans.items;
}

fn powerKeys(self: Smc, arena: std.mem.Allocator) ![]const Key {
    var keys: std.ArrayList(Key) = .empty;

    for ([_]*const [4]u8{ "PSTR", "PDTR", "PD0R" }) |name| {
        const key = self.find(code(name)) catch continue;
        if (key.info.type == code("flt ")) try keys.append(arena, key);
    }

    return keys.items;
}

fn call(self: Smc, args: struct { key: u32 = 0, size: u32 = 0, command: Call.Command, index: u32 = 0 }) !Call {
    var request = std.mem.zeroes(Call);
    request.key = args.key;
    request.info.size = args.size;
    request.command = args.command;
    request.index = args.index;

    var output = std.mem.zeroes(Call);
    var size: usize = @sizeOf(Call);
    try iokit.check(self.api.IOConnectCallStructMethod(self.connection, 2, &request, @sizeOf(Call), &output, &size));
    if (output.result != 0) return error.SmcError;

    return output;
}

fn find(self: Smc, name: u32) !Key {
    return .{ .code = name, .info = (try self.call(.{ .key = name, .command = .read_info })).info };
}

fn readCode(self: Smc, name: u32) !f64 {
    return self.read(try self.find(name));
}

fn read(self: Smc, key: Key) !f64 {
    const bytes = (try self.call(.{ .key = key.code, .size = key.info.size, .command = .read_bytes })).bytes;

    return switch (key.info.type) {
        // Native byte order, unlike the integers.
        code("flt ") => @as(f32, @bitCast(bytes[0..4].*)),
        code("ui8 ") => @floatFromInt(bytes[0]),
        code("ui16") => @floatFromInt(std.mem.readInt(u16, bytes[0..2], .big)),
        code("ui32") => @floatFromInt(std.mem.readInt(u32, bytes[0..4], .big)),
        else => error.UnknownType,
    };
}

fn code(name: *const [4]u8) u32 {
    return std.mem.readInt(u32, name, .big);
}

/// "F0Ac": the actual speed of fan 0.
fn fanKey(index: usize, suffix: *const [2]u8) u32 {
    return code(&.{ 'F', '0' + @as(u8, @intCast(index)), suffix[0], suffix[1] });
}
