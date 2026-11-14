const std = @import("std");
const smi = @import("smi.zig");

/// Owns everything a platform does not: tasks, global device indices, snapshots
/// and the merged process list.
pub const Collector = struct {
    /// Every device of every platform and remote host, in the order they were added.
    devices: std.ArrayList(smi.Device) = .empty,
    /// Every process list merged into one, with host details. Written by the host step.
    processes: smi.DoubleBuffer(smi.ProcessList),
    host: smi.DoubleBuffer(smi.HostData),
    /// What the host reader read once. Each host step starts from it, so a value a platform did not measure is null.
    host_once: smi.HostData,
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
        /// The global index of its first device.
        first: u16,
        sides: smi.DoubleBuffer(Side),
        scratch: std.heap.ArenaAllocator,
    };

    const Side = struct {
        samples: []smi.Sample,
        processes: smi.ProcessList,
        /// What the platform measured of the host.
        host: smi.HostData = .{},
    };

    pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, options: Options) Collector {
        var host: smi.HostData = .{};
        const host_reader: smi.Host = .init(arena, gpa, io, &host);
        return .{
            .processes = .{ .values = .{ .init(gpa), .init(gpa) } },
            .host = .{ .values = .{ host, host } },
            .host_once = host,
            .host_reader = host_reader,
            .host_scratch = .init(gpa),
            .arena = arena,
            .gpa = gpa,
            .io = io,
            .interval = .fromMilliseconds(options.poll_interval_ms),
        };
    }

    /// Stops the tasks, then calls `deinit` on the platforms that have one.
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

    /// Registers an opened platform or remote host. Both sides of its samples start empty.
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

    /// One round: every source once, then the host step.
    pub fn round(self: *Collector) void {
        for (self.sources.items) |source| self.poll(source);
        self.hostStep();
    }

    /// Runs one round, then starts one task per source and the host task. No `add` may follow.
    pub fn start(self: *Collector) !void {
        const begin: std.Io.Timestamp = .now(self.io, .awake);
        self.round();
        for (self.sources.items) |source| try self.tasks.concurrent(self.io, run, .{ self, source, begin });
        try self.tasks.concurrent(self.io, run, .{ self, null, begin });
    }

    /// The latest sample of a device, by value.
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
                // A row that names a device the source does not have is dropped.
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
        if (platform.vtable.host) |measureHost| {
            back.host = .{};
            measureHost(platform, &back.host);
        }
        source.sides.swap(self.io);
    }

    /// Merges the rows of every source, adds the host details, and reads the host.
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
        host.* = self.host_once;
        self.host_reader.sample(host);
        for (self.sources.items) |source| {
            if (source.platform.vtable.host == null) continue;
            const measured = source.sides.lock(self.io).host;
            source.sides.unlock(self.io);
            inline for (@typeInfo(smi.HostData).@"struct".field_names) |field| {
                if (@field(measured, field)) |value| @field(host, field) = value;
            }
        }

        self.processes.swap(self.io);
        self.host.swap(self.io);
    }
};

const Fake = struct {
    platform: smi.Platform,
    polls: u64 = 0,

    const vtable: smi.Platform.VTable = .{ .poll = poll, .processes = processes, .host = host };

    fn poll(platform: *smi.Platform, _: std.mem.Allocator, samples: []smi.Sample) void {
        const self: *Fake = @alignCast(@fieldParentPtr("platform", platform));
        self.polls += 1;
        samples[0].cuda.util_percent = self.polls;
        // A tag that is not the device's target.
        if (samples.len > 1) samples[1] = .{ .tpu = .{ .util_percent = 1 } };
    }

    fn processes(platform: *smi.Platform, _: std.mem.Allocator, _: []const smi.Sample, out: *smi.ProcessList) smi.ProcessList.Error!void {
        const self: *Fake = @alignCast(@fieldParentPtr("platform", platform));
        if (self.polls % 2 == 0) return error.Unavailable;
        // A device the platform does not have.
        try out.append(.{ .pid = 7, .device_idx = @intCast(self.platform.devices.len) });
        try out.append(.{ .pid = 42, .device_idx = @intCast(self.platform.devices.len - 1), .remote = true, .username = try out.dupe("alice") });
    }

    /// Measures the CPU power on odd polls only.
    fn host(platform: *smi.Platform, data: *smi.HostData) void {
        const self: *Fake = @alignCast(@fieldParentPtr("platform", platform));
        if (self.polls % 2 == 1) data.cpu_power_mw = self.polls;
    }
};

test "collector rounds" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    var collector: Collector = .init(arena.allocator(), std.testing.allocator, io, .{});
    defer collector.deinit();

    var gpus: Fake = .{ .platform = .{ .devices = &.{ .{ .target = .cuda }, .{ .target = .cuda } }, .vtable = &Fake.vtable } };
    var other: Fake = .{ .platform = .{ .devices = &.{.{ .target = .cuda }}, .two_readings = true, .vtable = &Fake.vtable } };
    try collector.add(&gpus.platform);
    try std.testing.expect(!collector.two_readings);
    try collector.add(&other.platform);
    try std.testing.expect(collector.two_readings);
    try std.testing.expectEqual(3, collector.devices.items.len);

    // Before any round, samples are empty.
    try std.testing.expectEqual(null, collector.sample(@fromBackingInt(0)).cuda.util_percent);

    collector.round();
    try std.testing.expectEqual(1, collector.sample(@fromBackingInt(0)).cuda.util_percent);
    // The wrong tag was replaced with an empty sample of the device's target.
    try std.testing.expectEqual(null, collector.sample(@fromBackingInt(1)).cuda.util_percent);
    try std.testing.expectEqual(1, collector.sample(@fromBackingInt(2)).cuda.util_percent);
    {
        const merged = collector.processes.lock(io);
        defer collector.processes.unlock(io);
        // A row naming a missing device is dropped. The others get the global index
        // of their device, and keep their strings.
        try std.testing.expectEqual(2, merged.rows.items.len);
        try std.testing.expectEqual(1, merged.rows.items[0].device_idx);
        try std.testing.expectEqual(2, merged.rows.items[1].device_idx);
        try std.testing.expectEqualStrings("alice", merged.rows.items[1].username);
    }
    try std.testing.expectEqual(1, collector.host.read(io).cpu_power_mw);

    // `processes` fails: no rows this round, not the previous ones.
    collector.round();
    try std.testing.expectEqual(2, collector.sample(@fromBackingInt(0)).cuda.util_percent);
    // Host values a platform did not measure this round are null, not the previous ones.
    try std.testing.expectEqual(null, collector.host.read(io).cpu_power_mw);
    const merged = collector.processes.lock(io);
    defer collector.processes.unlock(io);
    try std.testing.expectEqual(0, merged.rows.items.len);
}

test "collector tasks stop" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    var collector: Collector = .init(arena.allocator(), std.testing.allocator, io, .{ .poll_interval_ms = 1 });
    var fake: Fake = .{ .platform = .{ .devices = &.{.{ .target = .cuda }}, .vtable = &Fake.vtable } };
    try collector.add(&fake.platform);
    try collector.start();
    while (collector.sample(@fromBackingInt(0)).cuda.util_percent.? < 3) try io.sleep(.fromMilliseconds(1), .awake);
    collector.deinit();
}
