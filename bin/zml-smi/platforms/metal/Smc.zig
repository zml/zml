//! The System Management Controller of an Apple silicon Mac: GPU temperature, fans and system power.
//! Ported from mactop v2.1.6 (8dbfaae): internal/app/smc.c and the SMC readers of internal/app/ioreport.m.

// MIT License
//
// Copyright (c) 2024-2026 Carsen Klock
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

const std = @import("std");
const iokit = @import("iokit.zig");

const Smc = @This();

api: iokit.Api,
connection: iokit.Object,
/// The GPU sensors.
temperatures: []const Key,
fans: []const Fan,
/// Whole-machine power: the system total (laptops), the DC-in total, then the DC-in rail,
/// which is the power supply's output on a Mac Studio. The first that reads a value wins.
power: []const Key,

/// A key and the type and size of its value.
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

    const Command = enum(u8) { read_bytes = 5, read_index = 8, read_info = 9 };
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

    var smc: Smc = .{
        .api = api,
        .connection = try api.IOServiceOpen.get(.{ service, std.c.mach_task_self(), 0 }),
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

/// The hottest GPU sensor, in degrees Celsius.
pub fn temperature(self: Smc) ?u64 {
    var hottest: ?f64 = null;
    for (self.temperatures) |sensor| {
        const celsius = self.read(sensor) catch continue;
        if (celsius <= min_celsius or celsius > max_celsius) continue;
        hottest = @max(hottest orelse celsius, celsius);
    }
    return if (hottest) |celsius| @intFromFloat(@round(celsius)) else null;
}

/// The fastest fan, as a share of its top speed.
pub fn fanPercent(self: Smc) ?u64 {
    var fastest: ?u64 = null;
    for (self.fans) |fan| {
        const rpm = self.read(fan.speed) catch continue;
        // A fan that cannot stop and reads 0 hides its speed from this user.
        if (rpm <= 0 and fan.min_rpm > 0) continue;
        const percent: u64 = @intFromFloat(@round(std.math.clamp(rpm / fan.max_rpm * 100, 0, 100)));
        fastest = @max(fastest orelse percent, percent);
    }
    return fastest;
}

/// The power of the whole machine, in milliwatts.
pub fn systemPower(self: Smc) ?u64 {
    for (self.power) |key| {
        const watts = self.read(key) catch continue;
        if (watts > 0 and watts <= max_watts) return @intFromFloat(@round(watts * 1000));
    }
    return null;
}

/// The GPU sensors, "Tg" keys that hold a float. Finding them reads the name of every key, about 3800 on an M5 Max.
fn gpuSensors(self: Smc, arena: std.mem.Allocator) ![]const Key {
    const count = self.read(self.find(code("#KEY".*)) catch return &.{}) catch return &.{};
    var sensors: std.ArrayList(Key) = .empty;
    for (0..@intFromFloat(count)) |index| {
        const name = (self.call(.{ .command = .read_index, .index = @intCast(index) }) catch continue).key;
        if (name >> 16 != code("Tg\x00\x00".*) >> 16) continue;
        const key = self.find(name) catch continue;
        if (key.info.type == code("flt ".*)) try sensors.append(arena, key);
    }
    return sensors.items;
}

fn findFans(self: Smc, arena: std.mem.Allocator) ![]const Fan {
    const count = self.read(self.find(code("FNum".*)) catch return &.{}) catch return &.{};
    var fans: std.ArrayList(Fan) = .empty;
    for (0..@intFromFloat(@min(count, 10))) |index| {
        const max_rpm = self.read(self.find(fanKey(index, "Mx".*)) catch continue) catch continue;
        if (max_rpm <= 0) continue;
        try fans.append(arena, .{
            .speed = self.find(fanKey(index, "Ac".*)) catch continue,
            .min_rpm = if (self.find(fanKey(index, "Mn".*))) |key| self.read(key) catch 0 else |_| 0,
            .max_rpm = max_rpm,
        });
    }
    return fans.items;
}

fn powerKeys(self: Smc, arena: std.mem.Allocator) ![]const Key {
    var keys: std.ArrayList(Key) = .empty;
    for ([_][4]u8{ "PSTR".*, "PDTR".*, "PD0R".* }) |name| {
        const key = self.find(code(name)) catch continue;
        if (key.info.type == code("flt ".*)) try keys.append(arena, key);
    }
    return keys.items;
}

fn call(self: Smc, input: Call) !Call {
    var output: Call = .{};
    var size: usize = @sizeOf(Call);
    try self.api.IOConnectCallStructMethod.call(.{ self.connection, 2, &input, @sizeOf(Call), &output, &size });
    if (output.result != 0) return error.SmcError;
    return output;
}

fn find(self: Smc, name: u32) !Key {
    return .{ .code = name, .info = (try self.call(.{ .key = name, .command = .read_info })).info };
}

/// The value of a key that holds a float or an unsigned integer.
fn read(self: Smc, key: Key) !f64 {
    const bytes = (try self.call(.{ .key = key.code, .info = .{ .size = key.info.size }, .command = .read_bytes })).bytes;
    return switch (key.info.type) {
        // Native byte order, unlike the integers.
        code("flt ".*) => @as(f32, @bitCast(bytes[0..4].*)),
        code("ui8 ".*) => @floatFromInt(bytes[0]),
        code("ui16".*) => @floatFromInt(std.mem.readInt(u16, bytes[0..2], .big)),
        code("ui32".*) => @floatFromInt(std.mem.readInt(u32, bytes[0..4], .big)),
        else => error.UnknownType,
    };
}

fn code(name: [4]u8) u32 {
    return std.mem.readInt(u32, &name, .big);
}

/// "F0Ac": the actual speed of fan 0.
fn fanKey(index: usize, suffix: [2]u8) u32 {
    return code(.{ 'F', '0' + @as(u8, @intCast(index)), suffix[0], suffix[1] });
}
