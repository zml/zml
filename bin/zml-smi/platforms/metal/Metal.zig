//! Inspired by mactop.

const std = @import("std");
const smi = @import("zml-smi");
const iokit = @import("iokit.zig");
const Smc = @import("Smc.zig");

const Metal = @This();
const log = std.log.scoped(.@"zml-smi/metal");

platform: smi.Platform,
io: std.Io,
gpa: std.mem.Allocator,
cf: CoreFoundation,
io_kit: iokit.Api,
ioreport: IoReport,
keys: std.EnumArray(Key, *const String),
accelerator: iokit.Object,
smc: ?Smc,
mem_total_bytes: ?u64,
frequencies: []const u64,
channels: *Dictionary,
subscription: *const Subscription,
last: ?Reading = null,
/// macOS 27 reads 0 for the CPU in the Energy Model. Once it does while the PMP
/// histograms show power, they give the CPU power for the rest of the run.
cpu_from_pmp: bool = false,
prev_busy: std.AutoHashMapUnmanaged(u32, std.Io.Duration) = .empty,
curr_busy: std.AutoHashMapUnmanaged(u32, std.Io.Duration) = .empty,
last_processes: ?std.Io.Timestamp = null,

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .deinit = deinit };

const CoreFoundation = struct {
    CFRelease: *const fn (object: *const anyopaque) callconv(.c) void,
    CFGetTypeID: *const fn (object: *const anyopaque) callconv(.c) usize,
    CFStringGetTypeID: *const fn () callconv(.c) usize,
    CFNumberGetTypeID: *const fn () callconv(.c) usize,
    CFArrayGetTypeID: *const fn () callconv(.c) usize,
    CFDictionaryGetTypeID: *const fn () callconv(.c) usize,
    CFDataGetTypeID: *const fn () callconv(.c) usize,
    CFStringCreateWithCString: *const fn (allocator: ?*const anyopaque, text: [*:0]const u8, encoding: u32) callconv(.c) ?*const String,
    CFStringGetCString: *const fn (string: *const String, buffer: [*]u8, size: isize, encoding: u32) callconv(.c) u8,
    CFNumberGetValue: *const fn (number: *const Number, kind: isize, value: *anyopaque) callconv(.c) u8,
    CFArrayGetCount: *const fn (array: *const Array) callconv(.c) isize,
    CFArrayGetValueAtIndex: *const fn (array: *const Array, index: isize) callconv(.c) ?*const anyopaque,
    CFDictionaryGetValue: *const fn (dictionary: *const Dictionary, key: *const anyopaque) callconv(.c) ?*const anyopaque,
    CFDictionaryCreateMutableCopy: *const fn (allocator: ?*const anyopaque, capacity: isize, dictionary: *const Dictionary) callconv(.c) ?*Dictionary,
    CFDataGetLength: *const fn (data: *const Data) callconv(.c) isize,
    CFDataGetBytePtr: *const fn (data: *const Data) callconv(.c) ?[*]const u8,
};

const IoReport = struct {
    IOReportCopyChannelsInGroup: *const fn (group: ?*const String, subgroup: ?*const String, a: u64, b: u64, c: u64) callconv(.c) ?*const Dictionary,
    IOReportMergeChannels: *const fn (into: *Dictionary, from: *const Dictionary, unused: ?*const anyopaque) callconv(.c) void,
    IOReportCreateSubscription: *const fn (a: ?*const anyopaque, channels: *Dictionary, subscribed: *?*Dictionary, d: u64, e: ?*const anyopaque) callconv(.c) ?*const Subscription,
    IOReportCreateSamples: *const fn (subscription: *const Subscription, channels: *Dictionary, unused: ?*const anyopaque) callconv(.c) ?*const Dictionary,
    IOReportCreateSamplesDelta: *const fn (previous: *const Dictionary, current: *const Dictionary, unused: ?*const anyopaque) callconv(.c) ?*const Dictionary,
    IOReportChannelGetGroup: *const fn (channel: *const Dictionary) callconv(.c) ?*const String,
    IOReportChannelGetChannelName: *const fn (channel: *const Dictionary) callconv(.c) ?*const String,
    IOReportChannelGetUnitLabel: *const fn (channel: *const Dictionary) callconv(.c) ?*const String,
    IOReportSimpleGetIntegerValue: *const fn (channel: *const Dictionary, index: i32) callconv(.c) i64,
    IOReportStateGetCount: *const fn (channel: *const Dictionary) callconv(.c) i32,
    IOReportStateGetNameForIndex: *const fn (channel: *const Dictionary, index: i32) callconv(.c) ?*const String,
    IOReportStateGetResidency: *const fn (channel: *const Dictionary, index: i32) callconv(.c) i64,
};

pub fn detect(_: std.Io) bool {
    return true;
}

pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !Metal {
    var self: Metal = .{
        .platform = undefined,
        .io = io,
        .gpa = gpa,
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

    self.readReport(gpu);

    if (self.smc) |smc| {
        gpu.temperature = smc.temperature();
        gpu.fan_speed_percent = smc.fanPercent();
        gpu.system_power_mw = smc.systemPower();
    }
}

fn processes(platform: *smi.Platform, scratch: std.mem.Allocator, samples: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
    const self: *Metal = @alignCast(@fieldParentPtr("platform", platform));

    var children: iokit.Object = 0;
    if (self.io_kit.IORegistryEntryGetChildIterator(self.accelerator, "IOService", &children) != 0) return error.Unavailable;
    defer _ = self.io_kit.IOObjectRelease(children);

    self.curr_busy.clearRetainingCapacity();
    defer std.mem.swap(@TypeOf(self.prev_busy), &self.prev_busy, &self.curr_busy);

    const now: std.Io.Timestamp = .now(self.io, .awake);
    const elapsed: ?std.Io.Duration = blk: {
        const last = self.last_processes orelse break :blk null;
        if (now.nanoseconds <= last.nanoseconds) break :blk null;

        break :blk last.durationTo(now);
    };
    self.last_processes = now;

    const Client = struct { busy: std.Io.Duration = .zero, delta: ?std.Io.Duration = null };
    var clients: std.AutoArrayHashMapUnmanaged(u32, Client) = .empty;

    while (true) {
        const client = self.io_kit.IOIteratorNext(children);
        if (client == 0) break;
        defer _ = self.io_kit.IOObjectRelease(client);

        if (self.io_kit.IOObjectConformsTo(client, "AGXDeviceUserClient") == 0) continue;
        const pid = self.clientPid(client) orelse continue;
        const busy = self.gpuTime(client);
        if (busy.nanoseconds == 0) continue;
        const entry = try clients.getOrPutValue(scratch, pid, .{});
        entry.value_ptr.busy.nanoseconds +|= busy.nanoseconds;
    }

    var total: std.Io.Duration = .zero;
    for (clients.keys(), clients.values()) |pid, *client| {
        try self.curr_busy.put(self.gpa, pid, client.busy);
        const prev = self.prev_busy.get(pid) orelse continue;
        if (client.busy.nanoseconds < prev.nanoseconds) continue;
        const delta: std.Io.Duration = .fromNanoseconds(client.busy.nanoseconds - prev.nanoseconds);
        client.delta = delta;
        total.nanoseconds +|= delta.nanoseconds;
    }

    const device = samples[0].metal.util_percent;
    for (clients.keys(), clients.values()) |pid, client| {
        try out.append(.{ .pid = pid, .dev_util_percent = if (client.delta) |delta| share(delta, total, elapsed, device) else null });
    }
}

fn deinit(platform: *smi.Platform) void {
    const self: *Metal = @alignCast(@fieldParentPtr("platform", platform));

    if (self.last) |last| self.cf.CFRelease(last.sample);
    self.cf.CFRelease(self.subscription);
    self.cf.CFRelease(self.channels);
    for (self.keys.values) |string| self.cf.CFRelease(string);

    if (self.smc) |smc| smc.close();
    _ = self.io_kit.IOObjectRelease(self.accelerator);
    self.prev_busy.deinit(self.gpa);
    self.curr_busy.deinit(self.gpa);
}

const Reading = struct { sample: *const Dictionary, at: std.Io.Timestamp };

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

/// A process's part of the device's utilization, in proportion to its GPU time, as mactop does:
/// the GPU time of the clients adds up to less than the device's utilization.
fn share(busy: std.Io.Duration, total: std.Io.Duration, elapsed: ?std.Io.Duration, device: ?u64) ?u16 {
    const percent = device orelse 0;
    if (percent > 0 and total.nanoseconds > 0) {
        return @intCast(@divTrunc(@as(i96, @min(percent, 100)) * busy.nanoseconds, total.nanoseconds));
    }

    const dt = elapsed orelse return null;
    if (dt.nanoseconds <= 0) return null;

    return @intCast(@min(@divTrunc(busy.nanoseconds * 100, dt.nanoseconds), 100));
}

fn clientPid(self: *Metal, client: iokit.Object) ?u32 {
    const creator = self.property(client, .IOUserClientCreator) orelse return null;
    defer self.cf.CFRelease(creator);

    var buffer: [128]u8 = undefined;
    const creator_text = self.text(self.as(String, creator), &buffer);
    const after_pid = std.mem.cutPrefix(u8, creator_text, "pid ") orelse return null;
    const digits, _ = std.mem.cutScalar(u8, after_pid, ',') orelse return null;

    return std.fmt.parseInt(u32, digits, 10) catch null;
}

fn gpuTime(self: *Metal, client: iokit.Object) std.Io.Duration {
    const usage = self.property(client, .AppUsage) orelse return .zero;
    defer self.cf.CFRelease(usage);

    const uses = self.as(Array, usage) orelse return .zero;
    var total: u64 = 0;

    var i: isize = 0;
    while (i < self.cf.CFArrayGetCount(uses)) : (i += 1) {
        const use = self.as(Dictionary, self.cf.CFArrayGetValueAtIndex(uses, i)) orelse continue;
        total +|= self.number(self.cf.CFDictionaryGetValue(use, self.keys.get(.accumulatedGPUTime))) orelse 0;
    }

    return .fromNanoseconds(total);
}

fn readReport(self: *Metal, gpu: *smi.Sample.Metal) void {
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

    var i: isize = 0;
    while (i < self.cf.CFArrayGetCount(channels)) : (i += 1) {
        const channel = self.as(Dictionary, self.cf.CFArrayGetValueAtIndex(channels, i)) orelse continue;
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

    gpu.cpu_power_mw = if (self.cpu_from_pmp) pmp_cpu_mw else cpu;
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
            if (std.mem.find(u8, name, cluster) != null) return .cpu_cluster;
        }

        if (std.mem.find(u8, name, "CPU Energy") != null) return .cpu;
        if (std.mem.eql(u8, name, "GPU Energy")) return .gpu;
        if (std.mem.eql(u8, name, "GPU")) return .gpu_alias;

        for ([_][]const u8{ "ANE", "NPU", "Neural", "ane" }) |engine| {
            if (std.mem.find(u8, name, engine) == null) continue;
            return if (std.mem.find(u8, name, "Energy") != null) .ane else .ane_block;
        }

        return null;
    }
};

/// Energy over the time it took, in milliwatts. mactop reads a unit that is not mJ or nJ as µJ.
fn milliwatts(amount: u64, unit: []const u8, elapsed: std.Io.Duration) u64 {
    const symbol = std.mem.sliceTo(unit, ' ');
    const scale: u64 = blk: {
        if (std.mem.eql(u8, symbol, "mJ")) break :blk std.time.ns_per_s;
        if (std.mem.eql(u8, symbol, "nJ")) break :blk std.time.ns_per_us;

        break :blk std.time.ns_per_ms;
    };

    return std.math.lossyCast(u64, @as(u128, amount) * scale / @as(u128, @intCast(elapsed.nanoseconds)));
}

fn clock(self: *Metal, channel: *const Dictionary) ?u64 {
    if (self.frequencies.len == 0) return null;

    var active: u64 = 0;
    var weighted: u128 = 0;
    var state_index: usize = 0;

    var state: i32 = 0;
    while (state < self.ioreport.IOReportStateGetCount(channel)) : (state += 1) {
        var buffer: [32]u8 = undefined;
        const name = self.text(self.ioreport.IOReportStateGetNameForIndex(channel, state), &buffer);
        if (std.mem.eql(u8, name, "OFF") or std.mem.eql(u8, name, "IDLE") or std.mem.eql(u8, name, "DOWN")) continue;

        const residency = std.math.cast(u64, self.ioreport.IOReportStateGetResidency(channel, state)) orelse 0;
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

    var state: i32 = 0;
    while (state < self.ioreport.IOReportStateGetCount(channel)) : (state += 1) {
        const residency = std.math.cast(u64, self.ioreport.IOReportStateGetResidency(channel, state)) orelse 0;
        total +|= residency;
        if (state == 0 or residency == 0) continue;

        var buffer: [32]u8 = undefined;
        weighted += stateWatts(self.text(self.ioreport.IOReportStateGetNameForIndex(channel, state), &buffer)) * @as(f64, @floatFromInt(residency));
    }

    if (total == 0) return 0;

    const mw = @round(weighted / @as(f64, @floatFromInt(total)) * 1000);
    return if (std.math.isFinite(mw) and mw > 0) std.math.lossyCast(u64, mw) else 0;
}

fn stateWatts(name: []const u8) f64 {
    const watts = std.fmt.parseFloat(f64, std.mem.trimEnd(u8, std.mem.trim(u8, name, " "), "W")) catch return 0;

    return if (std.math.isFinite(watts) and watts > 0) watts else 0;
}

fn isPmpGroup(group: []const u8) bool {
    const rest = std.mem.cutPrefix(u8, group, "PMP") orelse return false;

    return std.mem.findNone(u8, rest, "0123456789") == null;
}

/// A CPU cluster's power histogram: "PACC", "MACC1", "PACC0 SRAM", but not "AGX", the GPU.
fn isPmpCpuPower(name: []const u8) bool {
    if (name.len < 4 or !std.mem.eql(u8, name[1..4], "ACC")) return false;

    const rest = std.mem.trimStart(u8, name[4..], "0123456789");

    return rest.len == 0 or std.mem.eql(u8, rest, " SRAM");
}

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

    if (self.channelsIn("PMP", "Energy")) |pmp| {
        defer self.cf.CFRelease(pmp);
        self.ioreport.IOReportMergeChannels(channels, pmp, null);
    }

    // The dies are numbered contiguously, so the first without channels ends the search.
    for (0..8) |die| {
        var buffer: [8]u8 = undefined;
        const group = std.fmt.bufPrintSentinel(&buffer, "PMP{d}", .{die}, 0) catch unreachable;
        const pmp = self.channelsIn(group, "Energy") orelse break;
        defer self.cf.CFRelease(pmp);
        self.ioreport.IOReportMergeChannels(channels, pmp, null);
    }

    return channels;
}

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
        const len = self.cf.CFDataGetLength(data);
        if (len <= 0) continue;

        const ptr = self.cf.CFDataGetBytePtr(data) orelse continue;
        const bytes = ptr[0..@intCast(len)];

        var frequencies: std.ArrayList(u64) = .empty;
        var pairs = std.mem.window(u8, bytes, 8, 8);

        while (pairs.next()) |pair| {
            if (pair.len < 4) break;
            const frequency = std.mem.readInt(u32, pair[0..4], .little);
            const mhz: u64 = blk: {
                if (frequency >= 100_000_000) break :blk frequency / 1_000_000;
                if (frequency >= 100_000) break :blk frequency / 1000;

                break :blk 0;
            };
            if (mhz > 0) try frequencies.append(arena, mhz);
        }

        if (frequencies.items.len > 0) return frequencies.items;
    }

    log.warn("no GPU frequencies, so no clock", .{});
    return &.{};
}

fn propertyText(self: *Metal, arena: std.mem.Allocator, key: Key) !?[]const u8 {
    const value = self.property(self.accelerator, key) orelse return null;
    defer self.cf.CFRelease(value);

    var buffer: [128]u8 = undefined;
    const value_text = self.text(self.as(String, value), &buffer);

    return if (value_text.len > 0) try arena.dupe(u8, value_text) else null;
}

fn property(self: *Metal, entry: iokit.Object, key: Key) ?*const anyopaque {
    return self.io_kit.IORegistryEntryCreateCFProperty(entry, self.keys.get(key), null, 0);
}

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

test share {
    const second: std.Io.Duration = .fromSeconds(1);
    const total: std.Io.Duration = .fromMilliseconds(400);

    try std.testing.expectEqual(60, share(.fromMilliseconds(300), total, second, 80));
    try std.testing.expectEqual(20, share(.fromMilliseconds(100), total, second, 80));
    try std.testing.expectEqual(30, share(.fromMilliseconds(300), total, second, null));
    try std.testing.expectEqual(null, share(.fromMilliseconds(300), .zero, .zero, null));
}
