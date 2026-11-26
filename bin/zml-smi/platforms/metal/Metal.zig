//! Inspired by mactop.

const std = @import("std");
const c = @import("c.zig");
const smi = @import("zml-smi");
const Smc = @import("Smc.zig");

const Metal = @This();
const log = std.log.scoped(.@"zml-smi/metal");

platform: smi.Platform,
io: std.Io,
gpa: std.mem.Allocator,
keys: std.EnumArray(Key, *const c.CFString),
accelerator: c.io_service_t,
smc: ?Smc,
mem_total_bytes: ?u64,
frequencies: []const u64,
channels: *c.CFDictionary,
subscription: *c.IOReportSubscription,
last: ?Reading = null,
/// macOS 27 reads 0 for the CPU in the Energy Model. Once it does while the PMP
/// histograms show power, they give the CPU power for the rest of the run.
cpu_from_pmp: bool = false,
prev_busy: std.AutoHashMapUnmanaged(u32, std.Io.Duration) = .empty,
curr_busy: std.AutoHashMapUnmanaged(u32, std.Io.Duration) = .empty,
last_processes: ?std.Io.Timestamp = null,

const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .deinit = deinit };

pub fn detect(_: std.Io) bool {
    return true;
}

pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io) !Metal {
    var self: Metal = .{
        .platform = undefined,
        .io = io,
        .gpa = gpa,
        .keys = undefined,
        .accelerator = 0,
        .smc = null,
        .mem_total_bytes = memoryTotal(),
        .frequencies = &.{},
        .channels = undefined,
        .subscription = undefined,
    };

    self.accelerator = c.IOServiceGetMatchingService(c.kIOMainPortDefault, c.IOServiceMatching("AGXAccelerator"));
    if (self.accelerator == 0) {
        log.warn("no Apple GPU", .{});
        return error.GpuNotFound;
    }
    errdefer _ = c.IOObjectRelease(self.accelerator);

    for (std.enums.values(Key), 0..) |key, made| {
        self.keys.set(key, c.CFStringCreateWithCString(null, @tagName(key), c.kCFStringEncodingUTF8) orelse {
            for (std.enums.values(Key)[0..made]) |done| c.CFRelease(self.keys.get(done));
            return error.OutOfMemory;
        });
    }
    errdefer for (self.keys.values) |string| c.CFRelease(string);

    self.channels = try self.copyChannels();
    errdefer c.CFRelease(self.channels);
    var subscribed: c.CFMutableDictionaryRef = null;
    self.subscription = c.IOReportCreateSubscription(null, self.channels, &subscribed, 0, null) orelse {
        log.warn("IOReport refused the subscription", .{});
        return error.IoReportUnavailable;
    };
    if (subscribed) |channels| c.CFRelease(channels);
    errdefer c.CFRelease(self.subscription);

    self.frequencies = try self.gpuFrequencies(arena);
    self.smc = Smc.open(arena) catch |err| smc: {
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
        defer c.CFRelease(statistics);
        if (as(c.CFDictionary, statistics)) |values| {
            gpu.util_percent = number(c.CFDictionaryGetValue(values, self.keys.get(.@"Device Utilization %")));
            gpu.mem_used_bytes = number(c.CFDictionaryGetValue(values, self.keys.get(.@"In use system memory")));
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

    var children: c.io_iterator_t = 0;
    if (c.IORegistryEntryGetChildIterator(self.accelerator, "IOService", &children) != c.KERN_SUCCESS) return error.Unavailable;
    defer _ = c.IOObjectRelease(children);

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
        const client = c.IOIteratorNext(children);
        if (client == 0) break;
        defer _ = c.IOObjectRelease(client);

        if (c.IOObjectConformsTo(client, "AGXDeviceUserClient") == 0) continue;
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

    if (self.last) |last| c.CFRelease(last.sample);
    c.CFRelease(self.subscription);
    c.CFRelease(self.channels);
    for (self.keys.values) |string| c.CFRelease(string);

    if (self.smc) |smc| smc.close();
    _ = c.IOObjectRelease(self.accelerator);
    self.prev_busy.deinit(self.gpa);
    self.curr_busy.deinit(self.gpa);
}

const Reading = struct { sample: *const c.CFDictionary, at: std.Io.Timestamp };

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

fn clientPid(self: *Metal, client: c.io_object_t) ?u32 {
    const creator = self.property(client, .IOUserClientCreator) orelse return null;
    defer c.CFRelease(creator);

    var buffer: [128]u8 = undefined;
    const creator_text = text(as(c.CFString, creator), &buffer);
    const after_pid = std.mem.cutPrefix(u8, creator_text, "pid ") orelse return null;
    const digits, _ = std.mem.cutScalar(u8, after_pid, ',') orelse return null;

    return std.fmt.parseInt(u32, digits, 10) catch null;
}

fn gpuTime(self: *Metal, client: c.io_object_t) std.Io.Duration {
    const usage = self.property(client, .AppUsage) orelse return .zero;
    defer c.CFRelease(usage);

    const uses = as(c.CFArray, usage) orelse return .zero;
    var total: u64 = 0;

    var i: c.CFIndex = 0;
    while (i < c.CFArrayGetCount(uses)) : (i += 1) {
        const use = as(c.CFDictionary, c.CFArrayGetValueAtIndex(uses, i)) orelse continue;
        total +|= number(c.CFDictionaryGetValue(use, self.keys.get(.accumulatedGPUTime))) orelse 0;
    }

    return .fromNanoseconds(total);
}

fn readReport(self: *Metal, gpu: *smi.Sample.Metal) void {
    const sample = c.IOReportCreateSamples(self.subscription, self.channels, null) orelse return;
    const reading: Reading = .{ .sample = sample, .at = .now(self.io, .awake) };
    const last = self.last orelse {
        self.last = reading;
        return;
    };
    self.last = reading;
    defer c.CFRelease(last.sample);

    const delta = c.IOReportCreateSamplesDelta(last.sample, sample, null) orelse return;
    defer c.CFRelease(delta);

    const elapsed = last.at.durationTo(reading.at);
    if (elapsed.nanoseconds <= 0) return;

    const channels = as(c.CFArray, c.CFDictionaryGetValue(delta, self.keys.get(.IOReportChannels))) orelse return;

    var energy: std.EnumArray(Energy, u64) = .initFill(0);
    var pmp_cpu_mw: u64 = 0;

    var i: c.CFIndex = 0;
    while (i < c.CFArrayGetCount(channels)) : (i += 1) {
        const channel = as(c.CFDictionary, c.CFArrayGetValueAtIndex(channels, i)) orelse continue;
        var group_buffer: [64]u8 = undefined;
        var name_buffer: [128]u8 = undefined;
        const group = text(c.IOReportChannelGetGroup(channel), &group_buffer);
        const name = text(c.IOReportChannelGetChannelName(channel), &name_buffer);

        if (std.mem.eql(u8, group, "Energy Model")) {
            const kind = Energy.of(name) orelse continue;
            // Negative is an invalid value.
            const amount = std.math.cast(u64, c.IOReportSimpleGetIntegerValue(channel, 0)) orelse continue;
            var unit_buffer: [16]u8 = undefined;
            const unit = text(c.IOReportChannelGetUnitLabel(channel), &unit_buffer);
            energy.getPtr(kind).* +|= milliwatts(amount, unit, elapsed);
        } else if (std.mem.eql(u8, group, "GPU Stats") and std.mem.eql(u8, name, "GPUPH")) {
            gpu.clock_graphics_mhz = self.clock(channel);
        } else if (isPmpGroup(group) and isPmpCpuPower(name)) {
            pmp_cpu_mw +|= histogramMilliwatts(channel);
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

fn clock(self: *Metal, channel: *const c.CFDictionary) ?u64 {
    if (self.frequencies.len == 0) return null;

    var active: u64 = 0;
    var weighted: u128 = 0;
    var state_index: usize = 0;

    var state: i32 = 0;
    while (state < c.IOReportStateGetCount(channel)) : (state += 1) {
        var buffer: [32]u8 = undefined;
        const name = text(c.IOReportStateGetNameForIndex(channel, state), &buffer);
        if (std.mem.eql(u8, name, "OFF") or std.mem.eql(u8, name, "IDLE") or std.mem.eql(u8, name, "DOWN")) continue;

        const residency = std.math.cast(u64, c.IOReportStateGetResidency(channel, state)) orelse 0;
        active += residency;
        if (state_index < self.frequencies.len) weighted += @as(u128, self.frequencies[state_index]) * residency;
        state_index += 1;
    }

    return if (active > 0) @intCast(weighted / active) else 0;
}

/// The power of a CPU cluster from its histogram, whose states are named by watts ("   2W", " 0.250W"):
/// the mean of the states weighted by the time spent in each, the lowest reading 0, where an idle cluster sits.
fn histogramMilliwatts(channel: *const c.CFDictionary) u64 {
    var total: u64 = 0;
    var weighted: f64 = 0;

    var state: i32 = 0;
    while (state < c.IOReportStateGetCount(channel)) : (state += 1) {
        const residency = std.math.cast(u64, c.IOReportStateGetResidency(channel, state)) orelse 0;
        total +|= residency;
        if (state == 0 or residency == 0) continue;

        var buffer: [32]u8 = undefined;
        weighted += stateWatts(text(c.IOReportStateGetNameForIndex(channel, state), &buffer)) * @as(f64, @floatFromInt(residency));
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

fn copyChannels(self: *Metal) !*c.CFDictionary {
    const energy = self.channelsIn("Energy Model", null) orelse {
        log.warn("IOReport has no Energy Model", .{});
        return error.IoReportUnavailable;
    };
    defer c.CFRelease(energy);

    const channels = c.CFDictionaryCreateMutableCopy(null, 0, energy) orelse return error.OutOfMemory;

    if (self.channelsIn("GPU Stats", "GPU Performance States")) |gpu| {
        defer c.CFRelease(gpu);
        c.IOReportMergeChannels(channels, gpu, null);
    }

    if (self.channelsIn("PMP", "Energy")) |pmp| {
        defer c.CFRelease(pmp);
        c.IOReportMergeChannels(channels, pmp, null);
    }

    // The dies are numbered contiguously, so the first without channels ends the search.
    for (0..8) |die| {
        var buffer: [8]u8 = undefined;
        const group = std.fmt.bufPrintSentinel(&buffer, "PMP{d}", .{die}, 0) catch unreachable;
        const pmp = self.channelsIn(group, "Energy") orelse break;
        defer c.CFRelease(pmp);
        c.IOReportMergeChannels(channels, pmp, null);
    }

    return channels;
}

fn channelsIn(self: *Metal, group: [:0]const u8, subgroup: ?[:0]const u8) c.CFDictionaryRef {
    const group_string = c.CFStringCreateWithCString(null, group, c.kCFStringEncodingUTF8) orelse return null;
    defer c.CFRelease(group_string);

    const subgroup_string = if (subgroup) |name| (c.CFStringCreateWithCString(null, name, c.kCFStringEncodingUTF8) orelse return null) else null;
    defer if (subgroup_string) |string| c.CFRelease(string);

    const channels = c.IOReportCopyChannelsInGroup(group_string, subgroup_string, 0, 0, 0) orelse return null;
    const list = as(c.CFArray, c.CFDictionaryGetValue(channels, self.keys.get(.IOReportChannels)));
    if (list == null or c.CFArrayGetCount(list.?) == 0) {
        c.CFRelease(channels);
        return null;
    }

    return channels;
}

/// The frequency of each GPU performance state, from the power manager's voltage states:
/// pairs of a frequency and a voltage, in Hz up to the M4 and for the M5 GPU, in kHz for the M5 CPU.
fn gpuFrequencies(self: *Metal, arena: std.mem.Allocator) ![]const u64 {
    const manager = c.IOServiceGetMatchingService(c.kIOMainPortDefault, c.IOServiceNameMatching("pmgr"));
    if (manager == 0) return &.{};
    defer _ = c.IOObjectRelease(manager);

    for ([_]Key{ .@"voltage-states9-sram", .@"voltage-states9" }) |key| {
        const states = self.property(manager, key) orelse continue;
        defer c.CFRelease(states);

        const data = as(c.CFData, states) orelse continue;
        const len = c.CFDataGetLength(data);
        if (len <= 0) continue;

        const ptr = c.CFDataGetBytePtr(data) orelse continue;
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
    defer c.CFRelease(value);

    var buffer: [128]u8 = undefined;
    const value_text = text(as(c.CFString, value), &buffer);

    return if (value_text.len > 0) try arena.dupe(u8, value_text) else null;
}

fn property(self: *Metal, entry: c.io_registry_entry_t, key: Key) c.CFTypeRef {
    return c.IORegistryEntryCreateCFProperty(entry, self.keys.get(key), null, 0);
}

fn as(comptime T: type, object: c.CFTypeRef) ?*const T {
    const value = object orelse return null;
    const id = switch (T) {
        c.CFString => c.CFStringGetTypeID(),
        c.CFNumber => c.CFNumberGetTypeID(),
        c.CFArray => c.CFArrayGetTypeID(),
        c.CFDictionary => c.CFDictionaryGetTypeID(),
        c.CFData => c.CFDataGetTypeID(),
        else => @compileError("not a CoreFoundation type"),
    };

    return if (c.CFGetTypeID(value) == id) @ptrCast(value) else null;
}

fn number(object: c.CFTypeRef) ?u64 {
    const value = as(c.CFNumber, object) orelse return null;
    var result: i64 = 0;
    if (c.CFNumberGetValue(value, c.kCFNumberSInt64Type, &result) == 0) return null;

    return std.math.cast(u64, result);
}

fn text(string: c.CFStringRef, buffer: []u8) []const u8 {
    if (c.CFStringGetCString(string orelse return "", buffer.ptr, @intCast(buffer.len), c.kCFStringEncodingUTF8) == 0) return "";

    return std.mem.sliceTo(buffer, 0);
}

fn memoryTotal() ?u64 {
    var bytes: u64 = 0;
    var size: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &bytes, &size, null, 0) != 0) return null;

    return bytes;
}

test share {
    const second: std.Io.Duration = .fromSeconds(1);
    const total: std.Io.Duration = .fromMilliseconds(400);

    try std.testing.expectEqual(60, share(.fromMilliseconds(300), total, second, 80));
    try std.testing.expectEqual(20, share(.fromMilliseconds(100), total, second, 80));
    try std.testing.expectEqual(30, share(.fromMilliseconds(300), total, second, null));
    try std.testing.expectEqual(null, share(.fromMilliseconds(300), .zero, .zero, null));
}
