const std = @import("std");
const smi = @import("smi.zig");

pub const Collector = struct {
    devices: std.ArrayList(smi.Device) = .empty,
    processes: smi.DoubleBuffer(smi.ProcessList),
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

    pub const Options = struct { poll_interval_ms: u16 = 500 };

    const Source = struct {
        platform: *smi.Platform,
        first: u16,
        sides: smi.DoubleBuffer(Side),
        scratch: std.heap.ArenaAllocator,
    };

    const Side = struct { samples: []smi.Sample, processes: smi.ProcessList };

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
            .interval = .fromMilliseconds(options.poll_interval_ms),
        };
    }

    pub fn deinit(self: *Collector) void {
        self.stopping.store(true, .release);
        self.tasks.cancel(self.io);
        for (self.sources.items) |source| {
            if (source.platform.vtable.deinit) |deinitPlatform| deinitPlatform(source.platform);
            for (&source.sides.values) |*side| side.processes.deinit();
            source.scratch.deinit();
        }
        for (&self.processes.values) |*list| list.deinit();
        self.host_reader.deinit();
        self.host_scratch.deinit();
    }

    pub fn add(self: *Collector, platform: *smi.Platform) !void {
        const count = platform.devices.len;
        const source = try self.arena.create(Source);
        source.* = .{
            .platform = platform,
            .first = std.math.cast(u16, self.devices.items.len) orelse return error.TooManyDevices,
            .sides = .{ .values = .{
                .{ .samples = try self.arena.alloc(smi.Sample, count), .processes = .init(self.gpa) },
                .{ .samples = try self.arena.alloc(smi.Sample, count), .processes = .init(self.gpa) },
            } },
            .scratch = .init(self.gpa),
        };
        for (&source.sides.values) |side| {
            for (side.samples, platform.devices) |*sample_, device| sample_.* = .init(device.target);
        }
        try self.devices.appendSlice(self.arena, platform.devices);
        try self.sources.append(self.arena, source);
        self.two_readings = self.two_readings or platform.two_readings;
    }

    pub fn round(self: *Collector) void {
        for (self.sources.items) |source| self.poll(source);
        self.hostStep();
    }

    pub fn start(self: *Collector) !void {
        const begin: std.Io.Timestamp = .now(self.io, .awake);
        self.round();
        for (self.sources.items) |source| try self.tasks.concurrent(self.io, run, .{ self, source, begin });
        try self.tasks.concurrent(self.io, run, .{ self, null, begin });
    }

    pub fn sample(self: *Collector, device: smi.Device.Index) smi.Sample {
        const index = @backingInt(device);
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
        while (true) {
            const elapsed = begin.untilNow(self.io, .awake);
            if (elapsed.nanoseconds < self.interval.nanoseconds) {
                try self.io.sleep(.fromNanoseconds(self.interval.nanoseconds - elapsed.nanoseconds), .awake);
            }
            begin = .now(self.io, .awake);
            if (source) |s| self.poll(s) else self.hostStep();
            if (self.stopping.load(.acquire)) return;
        }
    }

    fn poll(self: *Collector, source: *Source) void {
        const platform = source.platform;
        const back = source.sides.back();
        const scratch = source.scratch.allocator();
        defer _ = source.scratch.reset(.retain_capacity);

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
        _ = self.host_scratch.reset(.retain_capacity);

        const host = self.host.back();
        host.* = self.host.front().*;
        self.host_reader.sample(host);

        self.processes.swap(self.io);
        self.host.swap(self.io);
    }
};
