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

const std = @import("std");
const smi = @import("zml-smi");
const iokit = @import("iokit.zig");
const Smc = @import("Smc.zig");

const Metal = @This();
const log = std.log.scoped(.@"zml-smi/metal");

platform: smi.Platform,
io: std.Io,
cf: CoreFoundation,
io_kit: iokit.Api,
ioreport: IoReport,
keys: std.EnumArray(Key, *const String),
accelerator: iokit.Object,
smc: ?Smc,
/// The GPU shares the host's memory.
mem_total_bytes: ?u64,
/// The frequency of each active GPU performance state, in MHz, in order.
frequencies: []const u64,
channels: *Dictionary,
subscription: *const Subscription,
/// The sample the next poll makes its delta against.
last: ?Reading = null,
/// macOS 27 reads 0 for the CPU in the Energy Model. Once it does while the PMP
/// histograms show power, they give the CPU power for the rest of the run.
cpu_from_pmp: bool = false,
/// What this round measured of the host, for `host`.
cpu_power_mw: ?u64 = null,
system_power_mw: ?u64 = null,
/// GPU nanoseconds of each process.
busy: smi.Counters(u32),

const Reading = struct { sample: *const Dictionary, at: std.Io.Timestamp };

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .deinit = deinit, .host = host };

/// The keys Metal looks up, named by their text. Made once in `init`.
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

/// Every Apple silicon Mac has a GPU, which `init` finds.
pub fn detect(_: std.Io) bool {
    return true;
}

pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !Metal {
    var self: Metal = .{
        .platform = undefined,
        .io = io,
        .cf = try smi.dynlib.open(CoreFoundation, &.{"/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"}),
        .io_kit = try iokit.open(),
        .ioreport = try smi.dynlib.open(IoReport, &.{"/usr/lib/libIOReport.dylib"}),
        .keys = undefined,
        .accelerator = 0,
        .smc = null,
        .mem_total_bytes = memoryTotal(),
        .frequencies = &.{},
        .channels = undefined,
        .subscription = undefined,
        .busy = .init(gpa),
    };

    self.accelerator = self.io_kit.IOServiceGetMatchingService(iokit.main_port, self.io_kit.IOServiceMatching("AGXAccelerator"));
    if (self.accelerator == 0) {
        log.warn("no Apple GPU", .{});
        return error.GpuNotFound;
    }
    errdefer _ = self.io_kit.IOObjectRelease(self.accelerator);

    for (std.enums.values(Key), 0..) |key, made| {
        self.keys.set(key, self.cf.CFStringCreateWithCString(null, @tagName(key), utf8) orelse {
            for (std.enums.values(Key)[0..made]) |done| self.cf.CFRelease(self.keys.get(done));
            return error.OutOfMemory;
        });
    }
    errdefer for (self.keys.values) |string| self.cf.CFRelease(string);

    self.channels = try self.copyChannels();
    errdefer self.cf.CFRelease(self.channels);
    var subscribed: ?*Dictionary = null;
    self.subscription = self.ioreport.IOReportCreateSubscription(null, self.channels, &subscribed, 0, null) orelse {
        log.warn("IOReport refused the subscription", .{});
        return error.IoReportUnavailable;
    };
    if (subscribed) |channels| self.cf.CFRelease(channels);
    errdefer self.cf.CFRelease(self.subscription);

    self.frequencies = try self.gpuFrequencies(arena);
    self.smc = Smc.open(arena, self.io_kit) catch |err| smc: {
        log.warn("no SMC, so no temperature, fan or system power: {t}", .{err});
        break :smc null;
    };
    errdefer if (self.smc) |smc| smc.close();

    const devices = try arena.alloc(smi.Device, 1);
    devices[0] = .{
        .target = .metal,
        .name = try self.propertyText(arena, .model),
        .driver_version = try self.propertyText(arena, .IOSourceVersion),
    };
    self.platform = .{ .devices = devices, .two_readings = true, .vtable = &vtable };
    return self;
}

fn deinit(platform: *smi.Platform) void {
    const self: *Metal = @alignCast(@fieldParentPtr("platform", platform));
    if (self.last) |last| self.cf.CFRelease(last.sample);
    self.cf.CFRelease(self.subscription);
    self.cf.CFRelease(self.channels);
    for (self.keys.values) |string| self.cf.CFRelease(string);
    if (self.smc) |smc| smc.close();
    _ = self.io_kit.IOObjectRelease(self.accelerator);
    self.busy.deinit();
}

fn poll(platform: *smi.Platform, _: std.mem.Allocator, samples: []smi.Sample) void {
    const self: *Metal = @alignCast(@fieldParentPtr("platform", platform));
    const gpu = &samples[0].metal;

    gpu.mem_total_bytes = self.mem_total_bytes;
    gpu.clock_graphics_max_mhz = if (self.frequencies.len > 0) std.mem.max(u64, self.frequencies) else null;
    if (self.property(self.accelerator, .PerformanceStatistics)) |statistics| {
        defer self.cf.CFRelease(statistics);
        if (self.as(Dictionary, statistics)) |values| {
            gpu.util_percent = self.number(self.cf.CFDictionaryGetValue(values, self.keys.get(.@"Device Utilization %")));
            gpu.mem_used_bytes = self.number(self.cf.CFDictionaryGetValue(values, self.keys.get(.@"In use system memory")));
        }
    }

    self.cpu_power_mw = null;
    self.readReport(gpu);

    self.system_power_mw = null;
    if (self.smc) |smc| {
        gpu.temperature = smc.temperature();
        gpu.fan_speed_percent = smc.fanPercent();
        self.system_power_mw = smc.systemPower();
    }
}

fn host(platform: *smi.Platform, data: *smi.HostData) void {
    const self: *Metal = @alignCast(@fieldParentPtr("platform", platform));
    data.cpu_power_mw = self.cpu_power_mw;
    data.system_power_mw = self.system_power_mw;
}

/// One row per process with GPU time, which has no GPU memory of its own.
fn processes(platform: *smi.Platform, scratch: std.mem.Allocator, samples: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *Metal = @alignCast(@fieldParentPtr("platform", platform));
    defer self.busy.sweep();
    const children = self.io_kit.IORegistryEntryGetChildIterator.get(.{ self.accelerator, "IOService" }) catch return error.Unavailable;
    defer _ = self.io_kit.IOObjectRelease(children);

    // A process may have several clients.
    var times: std.AutoArrayHashMapUnmanaged(u32, u64) = .empty;
    while (true) {
        const client = self.io_kit.IOIteratorNext(children);
        if (client == 0) break;
        defer _ = self.io_kit.IOObjectRelease(client);
        if (self.io_kit.IOObjectConformsTo(client, "AGXDeviceUserClient") == 0) continue;
        const pid = self.clientPid(client) orelse continue;
        const time = try times.getOrPutValue(scratch, pid, 0);
        time.value_ptr.* +|= self.gpuTime(client);
    }

    const growths = try scratch.alloc(?smi.Growth, times.count());
    var total: u64 = 0;
    for (times.keys(), times.values(), growths) |pid, nanoseconds, *growth| {
        growth.* = if (nanoseconds == 0) null else try self.busy.advance(pid, self.io, nanoseconds);
        if (growth.*) |busy| total +|= busy.amount;
    }
    const device = samples[0].metal.util_percent;
    for (times.keys(), times.values(), growths) |pid, nanoseconds, growth| {
        if (nanoseconds == 0) continue;
        try out.append(.{ .pid = pid, .dev_util_percent = if (growth) |busy| share(busy, total, device) else null });
    }
}

/// A process's part of the device's utilization, in proportion to its GPU time, as mactop does:
/// the GPU time of the clients adds up to less than the device's utilization.
/// Its own share of the elapsed time when the device's utilization is unknown or 0.
fn share(busy: smi.Growth, total: u64, device: ?u64) u16 {
    const percent = device orelse 0;
    if (percent == 0 or total == 0) return busy.percentOfTime();
    return @intCast(@as(u128, @min(percent, 100)) * busy.amount / total);
}

/// The process that opened a GPU client, from its creator: "pid 682, WindowServer".
fn clientPid(self: *Metal, client: iokit.Object) ?u32 {
    const creator = self.property(client, .IOUserClientCreator) orelse return null;
    defer self.cf.CFRelease(creator);
    var buffer: [128]u8 = undefined;
    const creator_text = self.text(self.as(String, creator), &buffer);
    if (!std.mem.startsWith(u8, creator_text, "pid ")) return null;
    const end = std.mem.indexOfScalar(u8, creator_text, ',') orelse return null;
    return std.fmt.parseInt(u32, creator_text[4..end], 10) catch null;
}

/// The GPU nanoseconds of a client, summed over the APIs it used.
fn gpuTime(self: *Metal, client: iokit.Object) u64 {
    const usage = self.property(client, .AppUsage) orelse return 0;
    defer self.cf.CFRelease(usage);
    const uses = self.as(Array, usage) orelse return 0;
    var total: u64 = 0;
    for (0..@intCast(@max(0, self.cf.CFArrayGetCount(uses)))) |index| {
        const use = self.as(Dictionary, self.cf.CFArrayGetValueAtIndex(uses, @intCast(index))) orelse continue;
        total +|= self.number(self.cf.CFDictionaryGetValue(use, self.keys.get(.accumulatedGPUTime))) orelse 0;
    }
    return total;
}

/// Takes a sample and reads its delta against the last one: GPU, ANE and CPU power, and the GPU clock.
fn readReport(self: *Metal, gpu: *smi.Metal) void {
    const sample = self.ioreport.IOReportCreateSamples(self.subscription, self.channels, null) orelse return;
    const reading: Reading = .{ .sample = sample, .at = .now(self.io, .awake) };
    const last = self.last orelse {
        self.last = reading;
        return;
    };
    self.last = reading;
    defer self.cf.CFRelease(last.sample);

    const delta = self.ioreport.IOReportCreateSamplesDelta(last.sample, sample, null) orelse return;
    defer self.cf.CFRelease(delta);
    const elapsed = last.at.durationTo(reading.at);
    if (elapsed.nanoseconds <= 0) return;
    const channels = self.as(Array, self.cf.CFDictionaryGetValue(delta, self.keys.get(.IOReportChannels))) orelse return;

    var energy: std.EnumArray(Energy, u64) = .initFill(0);
    var pmp_cpu_mw: u64 = 0;
    for (0..@intCast(@max(0, self.cf.CFArrayGetCount(channels)))) |index| {
        const channel = self.as(Dictionary, self.cf.CFArrayGetValueAtIndex(channels, @intCast(index))) orelse continue;
        var group_buffer: [64]u8 = undefined;
        var name_buffer: [128]u8 = undefined;
        const group = self.text(self.ioreport.IOReportChannelGetGroup(channel), &group_buffer);
        const name = self.text(self.ioreport.IOReportChannelGetChannelName(channel), &name_buffer);

        if (std.mem.eql(u8, group, "Energy Model")) {
            const kind = Energy.of(name) orelse continue;
            // Negative is an invalid value.
            const amount = std.math.cast(u64, self.ioreport.IOReportSimpleGetIntegerValue(channel, 0)) orelse continue;
            var unit_buffer: [16]u8 = undefined;
            const unit = self.text(self.ioreport.IOReportChannelGetUnitLabel(channel), &unit_buffer);
            energy.getPtr(kind).* +|= milliwatts(amount, unit, elapsed);
        } else if (std.mem.eql(u8, group, "GPU Stats") and std.mem.eql(u8, name, "GPUPH")) {
            gpu.clock_graphics_mhz = self.clock(channel);
        } else if (isPmpGroup(group) and isPmpCpuPower(name)) {
            pmp_cpu_mw +|= self.histogramMilliwatts(channel);
        }
    }

    // Whole-CPU totals, else the clusters: an OS with both would count them twice.
    const cpu = if (energy.get(.cpu) > 0) energy.get(.cpu) else energy.get(.cpu_cluster);
    if (cpu == 0 and pmp_cpu_mw > 0) self.cpu_from_pmp = true;
    self.cpu_power_mw = if (self.cpu_from_pmp) pmp_cpu_mw else cpu;
    gpu.power_mw = if (energy.get(.gpu) > 0) energy.get(.gpu) else energy.get(.gpu_alias);
    gpu.ane_power_mw = if (energy.get(.ane_block) > 0) energy.get(.ane_block) else energy.get(.ane);
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
    const symbol = unit[0 .. std.mem.indexOfScalar(u8, unit, ' ') orelse unit.len];
    const prefix: smi.units.Prefix = if (std.mem.eql(u8, symbol, "mJ")) .milli else if (std.mem.eql(u8, symbol, "nJ")) .nano else .micro;
    const growth: smi.Growth = .{ .amount = amount, .elapsed = elapsed };
    return smi.units.scale(growth.perSecond(), prefix, .milli);
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
    if (!std.mem.startsWith(u8, group, "PMP")) return false;
    for (group[3..]) |char| if (!std.ascii.isDigit(char)) return false;
    return true;
}

/// A CPU cluster's power histogram: "PACC", "MACC1", "PACC0 SRAM", but not "AGX", the GPU.
fn isPmpCpuPower(name: []const u8) bool {
    if (name.len < 4 or !std.mem.eql(u8, name[1..4], "ACC")) return false;
    var rest = name[4..];
    while (rest.len > 0 and std.ascii.isDigit(rest[0])) rest = rest[1..];
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

    if (self.channelsIn("GPU Stats", "GPU Performance States")) |gpu| {
        defer self.cf.CFRelease(gpu);
        self.ioreport.IOReportMergeChannels(channels, gpu, null);
    }
    // The dies are numbered contiguously, so the first without channels ends the search.
    for (0..9) |die| {
        var buffer: [8]u8 = undefined;
        const group = if (die == 0) "PMP" else std.fmt.bufPrintSentinel(&buffer, "PMP{d}", .{die - 1}, 0) catch unreachable;
        const pmp = self.channelsIn(group, "Energy") orelse if (die == 0) continue else break;
        defer self.cf.CFRelease(pmp);
        self.ioreport.IOReportMergeChannels(channels, pmp, null);
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
    const list = self.as(Array, self.cf.CFDictionaryGetValue(channels, self.keys.get(.IOReportChannels)));
    if (list == null or self.cf.CFArrayGetCount(list.?) == 0) {
        self.cf.CFRelease(channels);
        return null;
    }
    return channels;
}

/// The frequency of each GPU performance state, from the power manager's voltage states:
/// pairs of a frequency and a voltage, in Hz up to the M4 and for the M5 GPU, in kHz for the M5 CPU.
fn gpuFrequencies(self: *Metal, arena: std.mem.Allocator) ![]const u64 {
    const manager = self.io_kit.IOServiceGetMatchingService(iokit.main_port, self.io_kit.IOServiceNameMatching("pmgr"));
    if (manager == 0) return &.{};
    defer _ = self.io_kit.IOObjectRelease(manager);

    for ([_]Key{ .@"voltage-states9-sram", .@"voltage-states9" }) |key| {
        const states = self.property(manager, key) orelse continue;
        defer self.cf.CFRelease(states);
        const data = self.as(Data, states) orelse continue;
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
    const value = self.property(self.accelerator, key) orelse return null;
    defer self.cf.CFRelease(value);
    var buffer: [128]u8 = undefined;
    const value_text = self.text(self.as(String, value), &buffer);
    return if (value_text.len > 0) try arena.dupe(u8, value_text) else null;
}

/// A property of a registry entry, which the caller releases.
fn property(self: *Metal, entry: iokit.Object, key: Key) ?*const anyopaque {
    return self.io_kit.IORegistryEntryCreateCFProperty(entry, self.keys.get(key), null, 0);
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

fn number(self: *Metal, object: ?*const anyopaque) ?u64 {
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

fn memoryTotal() ?u64 {
    var bytes: u64 = 0;
    var size: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &bytes, &size, null, 0) != 0) return null;
    return bytes;
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

const CoreFoundation = struct {
    CFRelease: *const @TypeOf(decls.CFRelease),
    CFGetTypeID: *const @TypeOf(decls.CFGetTypeID),
    CFStringGetTypeID: *const @TypeOf(decls.CFStringGetTypeID),
    CFNumberGetTypeID: *const @TypeOf(decls.CFNumberGetTypeID),
    CFArrayGetTypeID: *const @TypeOf(decls.CFArrayGetTypeID),
    CFDictionaryGetTypeID: *const @TypeOf(decls.CFDictionaryGetTypeID),
    CFDataGetTypeID: *const @TypeOf(decls.CFDataGetTypeID),
    CFStringCreateWithCString: *const @TypeOf(decls.CFStringCreateWithCString),
    CFStringGetCString: *const @TypeOf(decls.CFStringGetCString),
    CFNumberGetValue: *const @TypeOf(decls.CFNumberGetValue),
    CFArrayGetCount: *const @TypeOf(decls.CFArrayGetCount),
    CFArrayGetValueAtIndex: *const @TypeOf(decls.CFArrayGetValueAtIndex),
    CFDictionaryGetValue: *const @TypeOf(decls.CFDictionaryGetValue),
    CFDictionaryCreateMutableCopy: *const @TypeOf(decls.CFDictionaryCreateMutableCopy),
    CFDataGetLength: *const @TypeOf(decls.CFDataGetLength),
    CFDataGetBytePtr: *const @TypeOf(decls.CFDataGetBytePtr),
};

/// The private library behind powermetrics.
const IoReport = struct {
    IOReportCopyChannelsInGroup: *const @TypeOf(decls.IOReportCopyChannelsInGroup),
    IOReportMergeChannels: *const @TypeOf(decls.IOReportMergeChannels),
    IOReportCreateSubscription: *const @TypeOf(decls.IOReportCreateSubscription),
    IOReportCreateSamples: *const @TypeOf(decls.IOReportCreateSamples),
    IOReportCreateSamplesDelta: *const @TypeOf(decls.IOReportCreateSamplesDelta),
    IOReportChannelGetGroup: *const @TypeOf(decls.IOReportChannelGetGroup),
    IOReportChannelGetChannelName: *const @TypeOf(decls.IOReportChannelGetChannelName),
    IOReportChannelGetUnitLabel: *const @TypeOf(decls.IOReportChannelGetUnitLabel),
    IOReportSimpleGetIntegerValue: *const @TypeOf(decls.IOReportSimpleGetIntegerValue),
    IOReportStateGetCount: *const @TypeOf(decls.IOReportStateGetCount),
    IOReportStateGetNameForIndex: *const @TypeOf(decls.IOReportStateGetNameForIndex),
    IOReportStateGetResidency: *const @TypeOf(decls.IOReportStateGetResidency),
};

/// Never called directly: `CoreFoundation` and `IoReport` take their types.
/// The functions named Create or Copy return an object the caller releases; Get functions do not.
const decls = struct {
    extern fn CFRelease(object: *const anyopaque) void;
    extern fn CFGetTypeID(object: *const anyopaque) usize;
    extern fn CFStringGetTypeID() usize;
    extern fn CFNumberGetTypeID() usize;
    extern fn CFArrayGetTypeID() usize;
    extern fn CFDictionaryGetTypeID() usize;
    extern fn CFDataGetTypeID() usize;
    extern fn CFStringCreateWithCString(allocator: ?*const anyopaque, text: [*:0]const u8, encoding: u32) ?*const String;
    extern fn CFStringGetCString(string: *const String, buffer: [*]u8, size: isize, encoding: u32) u8;
    extern fn CFNumberGetValue(number: *const Number, kind: isize, value: *anyopaque) u8;
    extern fn CFArrayGetCount(array: *const Array) isize;
    extern fn CFArrayGetValueAtIndex(array: *const Array, index: isize) ?*const anyopaque;
    extern fn CFDictionaryGetValue(dictionary: *const Dictionary, key: *const anyopaque) ?*const anyopaque;
    extern fn CFDictionaryCreateMutableCopy(allocator: ?*const anyopaque, capacity: isize, dictionary: *const Dictionary) ?*Dictionary;
    extern fn CFDataGetLength(data: *const Data) isize;
    extern fn CFDataGetBytePtr(data: *const Data) [*]const u8;

    extern fn IOReportCopyChannelsInGroup(group: ?*const String, subgroup: ?*const String, a: u64, b: u64, c: u64) ?*const Dictionary;
    /// Adds the channels of `from` to `into`.
    extern fn IOReportMergeChannels(into: *Dictionary, from: *const Dictionary, unused: ?*const anyopaque) void;
    extern fn IOReportCreateSubscription(a: ?*const anyopaque, channels: *Dictionary, subscribed: *?*Dictionary, d: u64, e: ?*const anyopaque) ?*const Subscription;
    extern fn IOReportCreateSamples(subscription: *const Subscription, channels: *Dictionary, unused: ?*const anyopaque) ?*const Dictionary;
    extern fn IOReportCreateSamplesDelta(previous: *const Dictionary, current: *const Dictionary, unused: ?*const anyopaque) ?*const Dictionary;
    extern fn IOReportChannelGetGroup(channel: *const Dictionary) ?*const String;
    extern fn IOReportChannelGetChannelName(channel: *const Dictionary) ?*const String;
    extern fn IOReportChannelGetUnitLabel(channel: *const Dictionary) ?*const String;
    extern fn IOReportSimpleGetIntegerValue(channel: *const Dictionary, index: i32) i64;
    extern fn IOReportStateGetCount(channel: *const Dictionary) i32;
    extern fn IOReportStateGetNameForIndex(channel: *const Dictionary, index: i32) ?*const String;
    extern fn IOReportStateGetResidency(channel: *const Dictionary, index: i32) i64;
};

test "channel names" {
    try std.testing.expectEqual(.cpu, Energy.of("CPU Energy"));
    try std.testing.expectEqual(.cpu, Energy.of("DIE_0_CPU Energy"));
    try std.testing.expectEqual(.cpu_cluster, Energy.of("PCPU Energy"));
    try std.testing.expectEqual(.gpu, Energy.of("GPU Energy"));
    try std.testing.expectEqual(null, Energy.of("GPU0"));
    try std.testing.expectEqual(.ane_block, Energy.of("ANE0"));
    try std.testing.expectEqual(.ane, Energy.of("ANE Energy"));
    try std.testing.expectEqual(null, Energy.of("PACC_0"));

    try std.testing.expect(isPmpGroup("PMP") and isPmpGroup("PMP1") and !isPmpGroup("PMPX"));
    try std.testing.expect(isPmpCpuPower("PACC") and isPmpCpuPower("MACC1") and isPmpCpuPower("PACC0 SRAM"));
    try std.testing.expect(!isPmpCpuPower("AGX") and !isPmpCpuPower("AGX AFR") and !isPmpCpuPower("PACC SRAM2"));
    try std.testing.expectEqual(0.25, stateWatts(" 0.250W"));
    try std.testing.expectEqual(64, stateWatts("  64W"));

    // 1500 mJ in 500 ms.
    try std.testing.expectEqual(3000, milliwatts(1500, "mJ", .fromMilliseconds(500)));
    try std.testing.expectEqual(450, milliwatts(450_000_000, "nJ", .fromSeconds(1)));
}

test share {
    const second: std.Io.Duration = .fromSeconds(1);
    // Two processes with 300 ms and 100 ms of GPU time share a device at 80%.
    try std.testing.expectEqual(60, share(.{ .amount = 300_000_000, .elapsed = second }, 400_000_000, 80));
    try std.testing.expectEqual(20, share(.{ .amount = 100_000_000, .elapsed = second }, 400_000_000, 80));
    try std.testing.expectEqual(30, share(.{ .amount = 300_000_000, .elapsed = second }, 400_000_000, null));
}

test "a GPU of this Mac" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var metal = Metal.init(arena.allocator(), std.testing.allocator, io) catch return error.SkipZigTest;
    defer deinit(&metal.platform);

    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    var list: smi.ProcessList = .init(std.testing.allocator);
    defer list.deinit();
    var data: smi.HostData = .{};

    var samples: [1]smi.Sample = .{.init(.metal)};
    for (0..2) |round| {
        samples = .{.init(.metal)};
        list.clear();
        if (round > 0) try io.sleep(.fromMilliseconds(100), .awake);
        poll(&metal.platform, scratch.allocator(), &samples);
        try processes(&metal.platform, scratch.allocator(), &samples, &list);
        host(&metal.platform, &data);
    }
    const gpu = samples[0].metal;
    try std.testing.expect(gpu.util_percent.? <= 100);
    try std.testing.expect(gpu.mem_used_bytes.? < gpu.mem_total_bytes.?);
    try std.testing.expect(gpu.clock_graphics_mhz.? <= gpu.clock_graphics_max_mhz.?);
    try std.testing.expect(gpu.power_mw != null and data.cpu_power_mw != null);
    // WindowServer draws the screen with the GPU.
    try std.testing.expect(list.rows.items.len > 0);
}
