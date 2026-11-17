//! Apple GPUs: utilization and memory from the IORegistry, power and clock from IOReport,
//! temperature and fans from the SMC. It also measures the power of the CPU and of the whole machine.
//! The IOReport readers and the GPU time of each client are ported from mactop v2.1.6 (8dbfaae):
//! internal/app/ioreport.m, internal/app/pmp_names.h and internal/app/native_stats.go.

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

const builtin = @import("builtin");
const std = @import("std");
const contract = @import("zml-smi-2/contract");
const native = @import("zml-smi-2/native");
const source = @import("zml-smi-2/source");
const iokit = @import("iokit.zig");
const Smc = @import("Smc.zig");

const Metal = @This();
const log = std.log.scoped(.@"zml-smi-2/metal");

devices: []const contract.Device = &.{},
io: std.Io,
cf: CoreFoundation,
io_kit: iokit.Api,
ioreport: IoReport,
keys: std.EnumArray(Key, ?*const String) = .initFill(null),
accelerator: iokit.Object = 0,
smc: ?Smc = null,
/// The GPU shares the host's memory.
mem_total_bytes: ?u64,
/// The frequency of each active GPU performance state, in MHz, in order.
frequencies: []const u64 = &.{},
channels: ?*Dictionary = null,
subscription: ?*const Subscription = null,
/// The sample the next poll makes its delta against.
last: ?Reading = null,
/// macOS 27 reads 0 for the CPU in the Energy Model. Once it does while the PMP
/// histograms show power, they give the CPU power for the rest of the run.
cpu_from_pmp: bool = false,
/// GPU nanoseconds of each process.
busy: native.Counters(u32),

const Reading = struct { sample: *const Dictionary, at: std.Io.Timestamp };

pub const two_readings = true;

/// The keys Metal looks up, named by their text. Made once in `open`.
const Key = enum {
    IOReportChannels,
    PerformanceStatistics,
    @"Device Utilization %",
    @"In use system memory",
    IOUserClientCreator,
    AppUsage,
    accumulatedGPUTime,
    model,
    IOSourceVersion,
    @"voltage-states9-sram",
    @"voltage-states9",
};

pub fn open(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !Metal {
    var self: Metal = .{
        .io = io,
        .cf = try native.dynlib.open(CoreFoundation, &.{
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
        }),
        .io_kit = try iokit.open(),
        .ioreport = try native.dynlib.open(IoReport, &.{"/usr/lib/libIOReport.dylib"}),
        .mem_total_bytes = native.sysctl(u64, "hw.memsize"),
        .busy = .init(gpa),
    };
    errdefer self.close();

    self.accelerator = self.io_kit.IOServiceGetMatchingService(iokit.main_port, self.io_kit.IOServiceMatching("AGXAccelerator"));
    if (self.accelerator == 0) {
        log.warn("no Apple GPU", .{});
        return error.GpuNotFound;
    }
    for (std.enums.values(Key), &self.keys.values) |key, *value| {
        value.* = self.cf.CFStringCreateWithCString(null, @tagName(key), utf8) orelse
            return error.OutOfMemory;
    }

    const channels = try self.copyChannels();
    self.channels = channels;
    var subscribed: ?*Dictionary = null;
    self.subscription = self.ioreport.IOReportCreateSubscription(null, channels, &subscribed, 0, null) orelse {
        log.warn("IOReport refused the subscription", .{});
        return error.IoReportUnavailable;
    };
    if (subscribed) |registered| self.cf.CFRelease(registered);

    self.frequencies = try self.gpuFrequencies(arena);
    self.smc = Smc.open(arena, self.io_kit) catch |err| smc_blk: {
        log.warn("no SMC, so no temperature, fan or system power: {t}", .{err});
        break :smc_blk null;
    };
    self.devices = try arena.dupe(contract.Device, &.{.{
        .target = .metal,
        .name = try self.propertyText(arena, .model),
        .driver_version = try self.propertyText(arena, .IOSourceVersion),
    }});
    return self;
}

pub fn close(self: *Metal) void {
    if (self.last) |last| self.cf.CFRelease(last.sample);
    if (self.subscription) |subscription| self.cf.CFRelease(subscription);
    if (self.channels) |channels| self.cf.CFRelease(channels);
    for (self.keys.values) |key| if (key) |string| self.cf.CFRelease(string);
    if (self.smc) |smc| smc.close();
    if (self.accelerator != 0) _ = self.io_kit.IOObjectRelease(self.accelerator);
    self.busy.deinit();
}

pub fn poll(self: *Metal, scratch: std.mem.Allocator, reading: *source.Reading) !void {
    const gpu = &reading.samples[0];
    gpu.put(.mem_total_bytes, self.mem_total_bytes);
    const frequencies = self.frequencies;
    gpu.put(.clock_graphics_max_mhz, if (frequencies.len > 0) std.mem.max(u64, frequencies) else null);
    {
        const stats = self.property(Dictionary, self.accelerator, .PerformanceStatistics);
        defer if (stats) |dictionary| self.cf.CFRelease(dictionary);
        gpu.put(.util_percent, self.numberAt(stats, .@"Device Utilization %"));
        gpu.put(.mem_used_bytes, self.numberAt(stats, .@"In use system memory"));
    }

    self.readReport(gpu, &reading.host);
    if (self.smc) |smc| {
        gpu.put(.temperature, smc.temperature());
        gpu.put(.fan_speed_percent, smc.fanPercent());
        reading.host.system_power_mw = smc.systemPower();
    }

    // One row per process with GPU time, which has no GPU memory of its own.
    defer self.busy.sweep();
    const children = self.io_kit.IORegistryEntryGetChildIterator.get(.{ self.accelerator, "IOService" }) catch return error.Unavailable;
    defer _ = self.io_kit.IOObjectRelease(children);

    // A process may have several clients.
    const ClientTime = struct { nanoseconds: u64 = 0, growth: ?native.Growth = null };
    var times: std.AutoArrayHashMapUnmanaged(u32, ClientTime) = .empty;
    while (true) {
        const client = self.io_kit.IOIteratorNext(children);
        if (client == 0) break;
        defer _ = self.io_kit.IOObjectRelease(client);
        if (self.io_kit.IOObjectConformsTo(client, "AGXDeviceUserClient") == 0) continue;
        const pid = self.clientPid(client) orelse continue;
        const time = try times.getOrPutValue(scratch, pid, .{});
        time.value_ptr.nanoseconds +|= self.gpuTime(client);
    }

    var total: u64 = 0;
    for (times.keys(), times.values()) |pid, *time| {
        if (time.nanoseconds == 0) continue;
        time.growth = try self.busy.advance(pid, self.io, time.nanoseconds);
        if (time.growth) |busy| total +|= busy.amount;
    }
    const utilization = gpu.get(.util_percent);
    for (times.keys(), times.values()) |pid, time| {
        if (time.nanoseconds == 0) continue;
        try reading.addProcess(.{
            .pid = pid,
            .dev_util_percent = if (time.growth) |busy| share(busy, total, utilization) else null,
        });
    }
}

/// A process's part of the device's utilization, in proportion to its GPU time, as mactop does:
/// the GPU time of the clients adds up to less than the device's utilization.
/// Its own share of the elapsed time when the device's utilization is unknown or 0.
fn share(busy: native.Growth, total: u64, device: ?u64) u16 {
    const percent = device orelse 0;
    if (percent == 0 or total == 0) return busy.percentOfTime();
    return @intCast(@as(u128, @min(percent, 100)) * busy.amount / total);
}

/// The process that opened a GPU client, from its creator: "pid 682, WindowServer".
fn clientPid(self: *Metal, client: iokit.Object) ?u32 {
    const creator = self.property(String, client, .IOUserClientCreator) orelse return null;
    defer self.cf.CFRelease(creator);
    var buffer: [128]u8 = undefined;
    const creator_text = self.text(creator, &buffer);
    const rest = std.mem.cutPrefix(u8, creator_text, "pid ") orelse return null;
    const pid, _ = std.mem.cutScalar(u8, rest, ',') orelse return null;
    return std.fmt.parseInt(u32, pid, 10) catch null;
}

/// The GPU nanoseconds of a client, summed over the APIs it used.
fn gpuTime(self: *Metal, client: iokit.Object) u64 {
    const uses = self.property(Array, client, .AppUsage) orelse return 0;
    defer self.cf.CFRelease(uses);
    var total: u64 = 0;
    for (0..@intCast(@max(0, self.cf.CFArrayGetCount(uses)))) |index| {
        const use = self.as(Dictionary, self.cf.CFArrayGetValueAtIndex(uses, @intCast(index))) orelse continue;
        total +|= self.numberAt(use, .accumulatedGPUTime) orelse 0;
    }
    return total;
}

/// Takes a sample and reads its delta against the last one: GPU, ANE and CPU power, and the GPU clock.
fn readReport(self: *Metal, gpu: *contract.Metrics, host: *contract.HostData) void {
    const report = &self.ioreport;
    const sample = report.IOReportCreateSamples(self.subscription.?, self.channels.?, null) orelse return;
    const previous = self.last;
    const now: std.Io.Timestamp = .now(self.io, .awake);
    self.last = .{ .sample = sample, .at = now };
    const last = previous orelse return;
    defer self.cf.CFRelease(last.sample);

    const delta = report.IOReportCreateSamplesDelta(last.sample, sample, null) orelse return;
    defer self.cf.CFRelease(delta);
    const elapsed = last.at.durationTo(now);
    if (elapsed.nanoseconds <= 0) return;
    const channels = self.channelsOf(delta) orelse return;

    var energy: std.EnumArray(Energy, u64) = .initFill(0);
    var pmp_cpu_mw: u64 = 0;
    for (0..@intCast(@max(0, self.cf.CFArrayGetCount(channels)))) |index| {
        const channel = self.as(Dictionary, self.cf.CFArrayGetValueAtIndex(channels, @intCast(index))) orelse continue;
        var group_buffer: [64]u8 = undefined;
        var name_buffer: [128]u8 = undefined;
        const group = self.text(report.IOReportChannelGetGroup(channel), &group_buffer);
        const name = self.text(report.IOReportChannelGetChannelName(channel), &name_buffer);

        if (std.mem.eql(u8, group, "Energy Model")) {
            const kind = Energy.of(name) orelse continue;
            // Negative is an invalid value.
            const amount = std.math.cast(u64, report.IOReportSimpleGetIntegerValue(channel, 0)) orelse continue;
            var unit_buffer: [16]u8 = undefined;
            const unit = self.text(report.IOReportChannelGetUnitLabel(channel), &unit_buffer);
            energy.getPtr(kind).* +|= milliwatts(amount, unit, elapsed);
        } else if (std.mem.eql(u8, group, "GPU Stats") and std.mem.eql(u8, name, "GPUPH")) {
            gpu.put(.clock_graphics_mhz, self.clock(channel));
        } else if (isPmpGroup(group) and isPmpCpuPower(name)) {
            pmp_cpu_mw +|= self.histogramMilliwatts(channel);
        }
    }

    // Whole-CPU totals, else the clusters: an OS with both would count them twice.
    const cpu = Energy.preferred(&energy, .cpu, .cpu_cluster);
    if (cpu == 0 and pmp_cpu_mw > 0) self.cpu_from_pmp = true;
    host.cpu_power_mw = if (self.cpu_from_pmp) pmp_cpu_mw else cpu;
    gpu.put(.power_mw, Energy.preferred(&energy, .gpu, .gpu_alias));
    gpu.put(.ane_power_mw, Energy.preferred(&energy, .ane_block, .ane));
}

/// The Energy Model channels Metal reads, by name. Where an OS has two names for one
/// thing, the first of each pair is read, and the second only when the first reads 0.
const Energy = enum {
    cpu,
    cpu_cluster,
    gpu,
    gpu_alias,
    ane_block,
    ane,

    fn preferred(values: *const std.EnumArray(Energy, u64), first: Energy, fallback: Energy) u64 {
        const primary = values.get(first);
        return if (primary > 0) primary else values.get(fallback);
    }

    fn of(name: []const u8) ?Energy {
        const clusters = [_][]const u8{ "ECPU Energy", "PCPU Energy", "MCPU Energy", "eCPUs Energy", "pCPUs Energy", "mCPUs Energy" };
        for (clusters) |cluster| {
            if (std.mem.indexOf(u8, name, cluster) != null) return .cpu_cluster;
        }
        if (std.mem.indexOf(u8, name, "CPU Energy") != null) return .cpu;
        if (std.mem.eql(u8, name, "GPU Energy")) return .gpu;
        if (std.mem.eql(u8, name, "GPU")) return .gpu_alias;
        for ([_][]const u8{ "ANE", "NPU", "Neural", "ane" }) |engine| {
            if (std.mem.indexOf(u8, name, engine) == null) continue;
            return if (std.mem.indexOf(u8, name, "Energy") != null) .ane else .ane_block;
        }
        return null;
    }
};

/// Energy over the time it took, in milliwatts. mactop reads a unit that is not mJ or nJ as µJ.
fn milliwatts(amount: u64, unit: []const u8, elapsed: std.Io.Duration) u64 {
    const prefixes = std.StaticStringMap(native.units.Prefix).initComptime(.{
        .{ "mJ", .milli }, .{ "nJ", .nano },
    });
    const prefix = prefixes.get(std.mem.sliceTo(unit, ' ')) orelse .micro;
    const growth: native.Growth = .{ .amount = amount, .elapsed = elapsed };
    return native.units.scale(growth.perSecond(), prefix, .milli);
}

/// The mean frequency of the active GPU performance states, weighted by the time spent in each.
/// 0 when the GPU was off or idle all along.
fn clock(self: *Metal, channel: *const Dictionary) ?u64 {
    if (self.frequencies.len == 0) return null;
    var active: u64 = 0;
    var weighted: u128 = 0;
    var state_index: usize = 0;
    for (0..@intCast(@max(0, self.ioreport.IOReportStateGetCount(channel)))) |state| {
        var buffer: [32]u8 = undefined;
        const name = self.text(self.ioreport.IOReportStateGetNameForIndex(channel, @intCast(state)), &buffer);
        if (std.mem.eql(u8, name, "OFF") or std.mem.eql(u8, name, "IDLE") or std.mem.eql(u8, name, "DOWN")) continue;
        const residency = std.math.cast(u64, self.ioreport.IOReportStateGetResidency(channel, @intCast(state))) orelse 0;
        active += residency;
        if (state_index < self.frequencies.len) weighted += @as(u128, self.frequencies[state_index]) * residency;
        state_index += 1;
    }
    return if (active > 0) @intCast(weighted / active) else 0;
}

/// The power of a CPU cluster from its histogram, whose states are named by watts ("   2W", " 0.250W"):
/// the mean of the states weighted by the time spent in each, the lowest reading 0, where an idle cluster sits.
fn histogramMilliwatts(self: *Metal, channel: *const Dictionary) u64 {
    var total: u64 = 0;
    var weighted: f64 = 0;
    for (0..@intCast(@max(0, self.ioreport.IOReportStateGetCount(channel)))) |state| {
        const residency = std.math.cast(u64, self.ioreport.IOReportStateGetResidency(channel, @intCast(state))) orelse 0;
        total += residency;
        if (state == 0) continue;
        var buffer: [32]u8 = undefined;
        weighted += stateWatts(self.text(self.ioreport.IOReportStateGetNameForIndex(channel, @intCast(state)), &buffer)) * @as(f64, @floatFromInt(residency));
    }
    if (total == 0) return 0;
    return @intFromFloat(@round(weighted / @as(f64, @floatFromInt(total)) * 1000));
}

fn stateWatts(name: []const u8) f64 {
    return std.fmt.parseFloat(f64, std.mem.trimEnd(u8, std.mem.trim(u8, name, " "), "W")) catch 0;
}

/// "PMP" on older chips, "PMP0", "PMP1"... one per die on an M5 Max.
fn isPmpGroup(group: []const u8) bool {
    const suffix = std.mem.cutPrefix(u8, group, "PMP") orelse return false;
    return std.mem.findNone(u8, suffix, "0123456789") == null;
}

/// A CPU cluster's power histogram: "PACC", "MACC1", "PACC0 SRAM", but not "AGX", the GPU.
fn isPmpCpuPower(name: []const u8) bool {
    if (name.len < 4 or !std.mem.eql(u8, name[1..4], "ACC")) return false;
    const rest = std.mem.trimStart(u8, name[4..], "0123456789");
    return rest.len == 0 or std.mem.eql(u8, rest, " SRAM");
}

/// The channels Metal subscribes to: the Energy Model, the GPU performance states, and the
/// power histograms of PMP, the CPU clusters' among them.
fn copyChannels(self: *Metal) !*Dictionary {
    const energy = self.channelsIn("Energy Model", null) orelse {
        log.warn("IOReport has no Energy Model", .{});
        return error.IoReportUnavailable;
    };
    defer self.cf.CFRelease(energy);
    const channels = self.cf.CFDictionaryCreateMutableCopy(null, 0, energy) orelse return error.OutOfMemory;

    const groups = [_][:0]const u8{
        "GPU Stats", "PMP", "PMP0", "PMP1", "PMP2", "PMP3", "PMP4", "PMP5", "PMP6", "PMP7",
    };
    for (groups, 0..) |group, index| {
        const subgroup: [:0]const u8 = if (index == 0) "GPU Performance States" else "Energy";
        // The GPU group and legacy PMP are optional. Numbered dies are contiguous.
        const extra = self.channelsIn(group, subgroup) orelse if (index < 2) continue else break;
        defer self.cf.CFRelease(extra);
        self.ioreport.IOReportMergeChannels(channels, extra, null);
    }
    return channels;
}

/// The channels of an IOReport group, or null when it has none.
fn channelsIn(self: *Metal, group: [:0]const u8, subgroup: ?[:0]const u8) ?*const Dictionary {
    const group_string = self.cf.CFStringCreateWithCString(null, group, utf8) orelse return null;
    defer self.cf.CFRelease(group_string);
    const subgroup_string = if (subgroup) |name| (self.cf.CFStringCreateWithCString(null, name, utf8) orelse return null) else null;
    defer if (subgroup_string) |string| self.cf.CFRelease(string);

    const channels = self.ioreport.IOReportCopyChannelsInGroup(group_string, subgroup_string, 0, 0, 0) orelse return null;
    if (self.channelsOf(channels)) |list| {
        if (self.cf.CFArrayGetCount(list) != 0) return channels;
    }
    self.cf.CFRelease(channels);
    return null;
}

fn channelsOf(self: *Metal, dictionary: *const Dictionary) ?*const Array {
    return self.as(Array, self.cf.CFDictionaryGetValue(dictionary, self.keys.get(.IOReportChannels).?));
}

/// The frequency of each GPU performance state, from the power manager's voltage states:
/// pairs of a frequency and a voltage, in Hz up to the M4 and for the M5 GPU, in kHz for the M5 CPU.
fn gpuFrequencies(self: *Metal, arena: std.mem.Allocator) ![]const u64 {
    const manager = self.io_kit.IOServiceGetMatchingService(iokit.main_port, self.io_kit.IOServiceNameMatching("pmgr"));
    if (manager == 0) return &.{};
    defer _ = self.io_kit.IOObjectRelease(manager);

    for ([_]Key{ .@"voltage-states9-sram", .@"voltage-states9" }) |key| {
        const data = self.property(Data, manager, key) orelse continue;
        defer self.cf.CFRelease(data);
        const bytes = self.cf.CFDataGetBytePtr(data)[0..@intCast(@max(0, self.cf.CFDataGetLength(data)))];

        var frequencies: std.ArrayList(u64) = .empty;
        var pairs = std.mem.window(u8, bytes, 8, 8);
        while (pairs.next()) |pair| {
            if (pair.len < 4) break;
            const frequency = std.mem.readInt(u32, pair[0..4], .little);
            const mhz = if (frequency >= 100_000_000) frequency / 1_000_000 else if (frequency >= 100_000) frequency / 1000 else 0;
            if (mhz > 0) try frequencies.append(arena, mhz);
        }
        return frequencies.items;
    }
    log.warn("no GPU frequencies, so no clock", .{});
    return &.{};
}

/// A string property of the GPU, copied into `arena`.
fn propertyText(self: *Metal, arena: std.mem.Allocator, key: Key) !?[]const u8 {
    const value = self.property(String, self.accelerator, key) orelse return null;
    defer self.cf.CFRelease(value);
    var buffer: [128]u8 = undefined;
    const value_text = self.text(value, &buffer);
    return if (value_text.len > 0) try arena.dupe(u8, value_text) else null;
}

/// A property of a registry entry, which the caller releases.
fn property(self: *Metal, comptime T: type, entry: iokit.Object, key: Key) ?*const T {
    const value = self.io_kit.IORegistryEntryCreateCFProperty(entry, self.keys.get(key).?, null, 0) orelse return null;
    if (self.as(T, value)) |typed| return typed;
    self.cf.CFRelease(value);
    return null;
}

/// `object` as a T, or null when it is another kind of object.
fn as(self: *Metal, comptime T: type, object: ?*const anyopaque) ?*const T {
    const value = object orelse return null;
    const id = switch (T) {
        String => self.cf.CFStringGetTypeID(),
        Number => self.cf.CFNumberGetTypeID(),
        Array => self.cf.CFArrayGetTypeID(),
        Dictionary => self.cf.CFDictionaryGetTypeID(),
        Data => self.cf.CFDataGetTypeID(),
        else => @compileError("not a CoreFoundation type"),
    };
    return if (self.cf.CFGetTypeID(value) == id) @ptrCast(value) else null;
}

fn numberAt(self: *Metal, dictionary: ?*const Dictionary, key: Key) ?u64 {
    const object = self.cf.CFDictionaryGetValue(dictionary orelse return null, self.keys.get(key).?);
    const value = self.as(Number, object) orelse return null;
    var result: i64 = 0;
    if (self.cf.CFNumberGetValue(value, number_sint64, &result) == 0) return null;
    return std.math.cast(u64, result);
}

/// The text of a string, in `buffer`. Empty when there is none or it does not fit.
fn text(self: *Metal, string: ?*const String, buffer: []u8) []const u8 {
    if (self.cf.CFStringGetCString(string orelse return "", buffer.ptr, @intCast(buffer.len), utf8) == 0) return "";
    return std.mem.sliceTo(buffer, 0);
}

const String = opaque {};
const Number = opaque {};
const Array = opaque {};
const Dictionary = opaque {};
const Data = opaque {};
const Subscription = opaque {};

/// kCFStringEncodingUTF8.
const utf8: u32 = 0x08000100;
/// kCFNumberSInt64Type.
const number_sint64: isize = 4;

/// Create and Copy functions return an object the caller releases; Get functions borrow it.
const CoreFoundation = struct {
    CFRelease: *const fn (*const anyopaque) callconv(.c) void,
    CFGetTypeID: *const fn (*const anyopaque) callconv(.c) usize,
    CFStringGetTypeID: *const fn () callconv(.c) usize,
    CFNumberGetTypeID: *const fn () callconv(.c) usize,
    CFArrayGetTypeID: *const fn () callconv(.c) usize,
    CFDictionaryGetTypeID: *const fn () callconv(.c) usize,
    CFDataGetTypeID: *const fn () callconv(.c) usize,
    CFStringCreateWithCString: *const fn (
        ?*const anyopaque,
        [*:0]const u8,
        u32,
    ) callconv(.c) ?*const String,
    CFStringGetCString: *const fn (*const String, [*]u8, isize, u32) callconv(.c) u8,
    CFNumberGetValue: *const fn (*const Number, isize, *anyopaque) callconv(.c) u8,
    CFArrayGetCount: *const fn (*const Array) callconv(.c) isize,
    CFArrayGetValueAtIndex: *const fn (*const Array, isize) callconv(.c) ?*const anyopaque,
    CFDictionaryGetValue: *const fn (*const Dictionary, *const anyopaque) callconv(.c) ?*const anyopaque,
    CFDictionaryCreateMutableCopy: *const fn (
        ?*const anyopaque,
        isize,
        *const Dictionary,
    ) callconv(.c) ?*Dictionary,
    CFDataGetLength: *const fn (*const Data) callconv(.c) isize,
    CFDataGetBytePtr: *const fn (*const Data) callconv(.c) [*]const u8,
};

/// The private library behind powermetrics.
const IoReport = struct {
    IOReportCopyChannelsInGroup: *const fn (
        ?*const String,
        ?*const String,
        u64,
        u64,
        u64,
    ) callconv(.c) ?*const Dictionary,
    /// Adds the channels of the second argument to the first.
    IOReportMergeChannels: *const fn (*Dictionary, *const Dictionary, ?*const anyopaque) callconv(.c) void,
    IOReportCreateSubscription: *const fn (
        ?*const anyopaque,
        *Dictionary,
        *?*Dictionary,
        u64,
        ?*const anyopaque,
    ) callconv(.c) ?*const Subscription,
    IOReportCreateSamples: *const fn (
        *const Subscription,
        *Dictionary,
        ?*const anyopaque,
    ) callconv(.c) ?*const Dictionary,
    IOReportCreateSamplesDelta: *const fn (
        *const Dictionary,
        *const Dictionary,
        ?*const anyopaque,
    ) callconv(.c) ?*const Dictionary,
    IOReportChannelGetGroup: *const fn (*const Dictionary) callconv(.c) ?*const String,
    IOReportChannelGetChannelName: *const fn (*const Dictionary) callconv(.c) ?*const String,
    IOReportChannelGetUnitLabel: *const fn (*const Dictionary) callconv(.c) ?*const String,
    IOReportSimpleGetIntegerValue: *const fn (*const Dictionary, i32) callconv(.c) i64,
    IOReportStateGetCount: *const fn (*const Dictionary) callconv(.c) i32,
    IOReportStateGetNameForIndex: *const fn (*const Dictionary, i32) callconv(.c) ?*const String,
    IOReportStateGetResidency: *const fn (*const Dictionary, i32) callconv(.c) i64,
};

pub const testing = if (builtin.is_test) struct {
    pub const energyOf = Energy.of;
    pub const isPmpGroup = Metal.isPmpGroup;
    pub const isPmpCpuPower = Metal.isPmpCpuPower;
    pub const stateWatts = Metal.stateWatts;
    pub const milliwatts = Metal.milliwatts;
    pub const share = Metal.share;
} else struct {};
