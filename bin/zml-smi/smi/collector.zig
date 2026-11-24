const std = @import("std");
const smi = @import("smi.zig");

pub const Platform = struct {
    devices: []const smi.Device,
    two_readings: bool = false,
    vtable: *const VTable,

    pub const VTable = struct {
        poll: *const fn (platform: *Platform, scratch: std.mem.Allocator, samples: []smi.Sample) void,
        /// Called right after `poll`, with the samples it just filled. A row's device is its position in `devices`.
        processes: ?*const fn (platform: *Platform, scratch: std.mem.Allocator, samples: []const smi.Sample, out: *ProcessList) ProcessList.Error!void = null,
        deinit: ?*const fn (platform: *Platform) void = null,
    };
};

pub const ProcessList = struct {
    rows: std.ArrayList(smi.ProcessInfo) = .empty,
    strings: std.heap.ArenaAllocator,
    gpa: std.mem.Allocator,

    pub const Error = error{ OutOfMemory, Unavailable };

    pub fn init(gpa: std.mem.Allocator) ProcessList {
        return .{ .strings = .init(gpa), .gpa = gpa };
    }

    pub fn deinit(list: *ProcessList) void {
        list.rows.deinit(list.gpa);
        list.strings.deinit();
    }

    pub fn clear(list: *ProcessList) void {
        list.rows.clearRetainingCapacity();
        _ = list.strings.reset(.retain_capacity);
    }

    pub fn append(list: *ProcessList, row: smi.ProcessInfo) error{OutOfMemory}!void {
        try list.rows.append(list.gpa, row);
    }

    pub fn dupe(list: *ProcessList, string: []const u8) error{OutOfMemory}![]const u8 {
        return list.strings.allocator().dupe(u8, string);
    }
};

pub const Collector = struct {
    devices: std.ArrayList(smi.Device) = .empty,
    processes: smi.DoubleBuffer(ProcessList),
    host: smi.DoubleBuffer(smi.HostData),
    /// A source derives values from two polls, so a one-shot print runs two rounds.
    two_readings: bool = false,

    sources: std.ArrayList(*Source) = .empty,
    host_reader: smi.Host,
    host_scratch: std.heap.ArenaAllocator,
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    interval: std.Io.Duration,
    tasks: std.Io.Group = .init,
    stopping: std.atomic.Value(bool) = .init(false),

    pub const Options = struct { interval: std.Io.Duration = .fromMilliseconds(500) };

    const Source = struct {
        platform: *Platform,
        first: u16,
        sides: smi.DoubleBuffer(Side),
        scratch: std.heap.ArenaAllocator,
    };

    const Side = struct { samples: []smi.Sample, processes: ProcessList };

    pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, options: Options) Collector {
        var host: smi.HostData = .{};
        const host_reader: smi.Host = .init(arena, gpa, io, &host);

        return .{
            .processes = .{ .values = .{ .init(gpa), .init(gpa) } },
            .host = .{ .values = .{ host, host } },
            .host_reader = host_reader,
            .host_scratch = .init(gpa),
            .arena = arena,
            .gpa = gpa,
            .io = io,
            .interval = options.interval,
        };
    }

    pub fn deinit(self: *Collector) void {
        self.stopping.store(true, .release);
        self.tasks.cancel(self.io);

        for (self.sources.items) |source| {
            if (source.platform.vtable.deinit) |deinitPlatform| deinitPlatform(source.platform);

            for (&source.sides.values) |*side| {
                side.processes.deinit();
            }
            source.scratch.deinit();
        }

        for (&self.processes.values) |*list| {
            list.deinit();
        }

        self.host_reader.deinit();
        self.host_scratch.deinit();
    }

    const scratch_retain_limit: usize = 1024 * 1024;

    pub fn add(self: *Collector, platform: *smi.Platform) !void {
        const count = platform.devices.len;
        const max_devices = @as(usize, std.math.maxInt(u16)) + 1;
        if (self.devices.items.len > std.math.maxInt(u16) or count > max_devices - self.devices.items.len) {
            return error.TooManyDevices;
        }

        const source = try self.arena.create(Source);

        source.* = .{
            .platform = platform,
            .first = @intCast(self.devices.items.len),
            .sides = .{ .values = .{
                .{ .samples = try self.arena.alloc(smi.Sample, count), .processes = .init(self.gpa) },
                .{ .samples = try self.arena.alloc(smi.Sample, count), .processes = .init(self.gpa) },
            } },
            .scratch = .init(self.gpa),
        };

        for (&source.sides.values) |side| {
            for (side.samples, platform.devices) |*sample_, device| sample_.* = .init(device.target);
        }

        try self.sources.ensureUnusedCapacity(self.arena, 1);
        try self.devices.appendSlice(self.arena, platform.devices);
        self.sources.appendAssumeCapacity(source);

        self.two_readings = self.two_readings or platform.two_readings;
    }

    pub fn round(self: *Collector) void {
        for (self.sources.items) |source| self.poll(source);
        self.hostStep();
    }

    pub fn start(self: *Collector) !void {
        const begin: std.Io.Timestamp = .now(self.io, .awake);
        self.round();

        for (self.sources.items) |source| {
            try self.tasks.concurrent(self.io, run, .{ self, source, begin });
        }

        try self.tasks.concurrent(self.io, run, .{ self, null, begin });
    }

    pub fn sample(self: *Collector, index: u16) smi.Sample {
        for (self.sources.items) |source| {
            const position = index -% source.first;

            if (index < source.first or position >= source.platform.devices.len) continue;
            defer source.sides.unlock(self.io);

            return source.sides.lock(self.io).samples[position];
        }

        unreachable;
    }

    fn run(self: *Collector, source: ?*Source, first_round: std.Io.Timestamp) std.Io.Cancelable!void {
        var begin = first_round;
        while (!self.stopping.load(.acquire)) {
            const elapsed = begin.untilNow(self.io, .awake);

            if (elapsed.nanoseconds < self.interval.nanoseconds) {
                try self.io.sleep(.fromNanoseconds(self.interval.nanoseconds - elapsed.nanoseconds), .awake);
            }

            try self.io.checkCancel();
            if (self.stopping.load(.acquire)) return;

            begin = .now(self.io, .awake);
            if (source) |s| self.poll(s) else self.hostStep();
        }
    }

    fn poll(self: *Collector, source: *Source) void {
        const platform = source.platform;
        const back = source.sides.back();
        const scratch = source.scratch.allocator();
        defer _ = source.scratch.reset(.{ .retain_with_limit = scratch_retain_limit });

        for (back.samples, platform.devices) |*sample_, device| sample_.* = .init(device.target);
        back.processes.clear();

        platform.vtable.poll(platform, scratch, back.samples);
        for (back.samples, platform.devices) |*sample_, device| {
            if (std.meta.activeTag(sample_.*) != device.target) sample_.* = .init(device.target);
        }

        if (platform.vtable.processes) |processes| {
            if (processes(platform, scratch, back.samples, &back.processes)) {
                const rows = &back.processes.rows;
                var kept: usize = 0;

                for (rows.items) |row| {
                    if (row.device_idx >= platform.devices.len) continue;
                    rows.items[kept] = row;
                    rows.items[kept].device_idx += source.first;
                    kept += 1;
                }

                rows.shrinkRetainingCapacity(kept);
            } else |_| {
                back.processes.clear();
            }
        }

        source.sides.swap(self.io);
    }

    fn hostStep(self: *Collector) void {
        const merged = self.processes.back();
        merged.clear();

        for (self.sources.items) |source| {
            const side = source.sides.lock(self.io);
            defer source.sides.unlock(self.io);

            for (side.processes.rows.items) |row| {
                var copy = row;
                copy.username = merged.dupe(row.username) catch "";
                copy.comm = merged.dupe(row.comm) catch "";
                merged.append(copy) catch break;
            }
        }

        self.host_reader.enrich(merged.strings.allocator(), self.host_scratch.allocator(), merged.rows.items);
        _ = self.host_scratch.reset(.{ .retain_with_limit = scratch_retain_limit });

        const host = self.host.back();
        host.* = self.host.front().*;
        self.host_reader.sample(host);

        self.processes.swap(self.io);
        self.host.swap(self.io);
    }
};

test "Collector.add enforces u16 device cap" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const dummy_vtable: smi.Platform.VTable = .{
        .poll = struct {
            fn f(_: *smi.Platform, _: std.mem.Allocator, _: []smi.Sample) void {}
        }.f,
    };

    var collector: Collector = .init(arena.allocator(), std.testing.allocator, std.testing.io, .{});
    defer collector.deinit();

    const one = try arena.allocator().alloc(smi.Device, 1);
    one[0] = .{ .target = .cuda };
    var first: smi.Platform = .{ .devices = one, .vtable = &dummy_vtable };
    try collector.add(&first);

    const overflow = try arena.allocator().alloc(smi.Device, @as(usize, std.math.maxInt(u16)) + 1);
    @memset(overflow, .{ .target = .cuda });
    var second: smi.Platform = .{ .devices = overflow, .vtable = &dummy_vtable };
    try std.testing.expectError(error.TooManyDevices, collector.add(&second));
}
