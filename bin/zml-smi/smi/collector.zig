const std = @import("std");
const smi = @import("smi.zig");

/// One backend poll fills its samples and may append processes or measure host power.
pub const Reading = struct {
    samples: []smi.Sample,
    allocator: std.mem.Allocator,
    processes: std.ArrayList(smi.ProcessInfo) = .empty,
    host: smi.HostData = .{},

    /// Any strings in the row must live in the publication's allocator.
    pub fn addProcess(self: *Reading, row: smi.ProcessInfo) std.mem.Allocator.Error!void {
        try self.processes.append(self.allocator, row);
    }
};

pub const Collector = struct {
    devices: std.ArrayList(smi.Device) = .empty,
    two_readings: bool = false,
    sources: std.ArrayList(*Source) = .empty,
    host: smi.DoubleBuffer(Host),
    host_reader: smi.Host,
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    interval: std.Io.Duration,
    keep_history: bool,
    tasks: std.Io.Group = .init,
    stopping: std.atomic.Value(bool) = .init(false),

    pub const history_capacity = 1024;
    pub const Options = struct { poll_interval_ms: u16 = 500, history: bool = false };
    pub const HistoryPoint = struct { at: std.Io.Timestamp, common: smi.Common };
    pub const Snapshot = struct {
        samples: []const smi.Sample,
        processes: []const smi.ProcessInfo,
        host: smi.HostData,
        histories: []const []const HistoryPoint,
    };

    const Host = struct {
        data: smi.HostData,
        processes: std.AutoArrayHashMapUnmanaged(u32, smi.ProcessInfo) = .empty,
    };

    const Source = struct {
        context: *anyopaque,
        read: *const fn (*anyopaque, std.mem.Allocator, *Reading) anyerror!void,
        close: *const fn (*anyopaque) void,
        first: usize,
        scratch: std.heap.ArenaAllocator,
        readings: smi.DoubleBuffer(Reading),
        history: [][history_capacity]HistoryPoint,
        next: usize = 0,
        count: usize = 0,
    };

    pub fn init(
        arena: std.mem.Allocator,
        gpa: std.mem.Allocator,
        io: std.Io,
        options: Options,
    ) Collector {
        var host: smi.HostData = .{};
        const host_reader: smi.Host = .init(arena, gpa, io, &host);
        return .{
            .host = .{ .values = @splat(.{ .value = .{ .data = host }, .arena = .init(gpa) }) },
            .host_reader = host_reader,
            .arena = arena,
            .gpa = gpa,
            .io = io,
            .interval = .fromMilliseconds(options.poll_interval_ms),
            .keep_history = options.history,
        };
    }

    pub fn deinit(self: *Collector) void {
        self.stopping.store(true, .release);
        self.tasks.cancel(self.io);
        for (self.sources.items) |source| {
            source.close(source.context);
            source.readings.deinit();
            source.scratch.deinit();
        }
        self.host.deinit();
        self.host_reader.deinit();
    }

    /// Registers a native reader before starting the workers.
    pub fn open(self: *Collector, comptime Backend: type, args: anytype) !void {
        if (@hasDecl(Backend, "detect") and !Backend.detect(self.io)) return;
        const Adapter = struct {
            fn read(context: *anyopaque, scratch: std.mem.Allocator, reading: *Reading) !void {
                const backend: *Backend = @ptrCast(@alignCast(context));
                try backend.poll(scratch, reading);
            }

            fn close(context: *anyopaque) void {
                const backend: *Backend = @ptrCast(@alignCast(context));
                if (@hasDecl(Backend, "close")) backend.close();
            }
        };
        const backend = try self.arena.create(Backend);
        backend.* = try @call(.auto, Backend.open, .{ self.arena, self.gpa, self.io } ++ args);
        errdefer Adapter.close(backend);
        const count = backend.devices.len;
        if (count == 0) return Adapter.close(backend);
        const first = self.devices.items.len;
        if (count > std.math.maxInt(u16) - first) return error.TooManyDevices;

        const source = try self.arena.create(Source);
        source.* = .{
            .context = backend,
            .read = Adapter.read,
            .close = Adapter.close,
            .first = first,
            .scratch = .init(self.gpa),
            .readings = .{ .values = undefined },
            .history = if (self.keep_history)
                try self.arena.alloc([history_capacity]HistoryPoint, count)
            else
                &.{},
        };
        for (&source.readings.values) |*slot| {
            const samples = try self.arena.alloc(smi.Sample, count);
            slot.arena = .init(self.gpa);
            slot.value = .{ .samples = samples, .allocator = slot.arena.allocator() };
            for (samples, backend.devices) |*sample, device| sample.* = .init(device.target);
        }
        try self.devices.appendSlice(self.arena, backend.devices);
        errdefer self.devices.items.len = first;
        try self.sources.append(self.arena, source);
        if (@hasDecl(Backend, "two_readings")) self.two_readings = self.two_readings or Backend.two_readings;
    }

    pub fn round(self: *Collector) void {
        for (self.sources.items) |source| self.poll(source);
        self.hostStep();
    }

    pub fn start(self: *Collector) !void {
        for (self.sources.items) |source| try self.tasks.concurrent(self.io, run, .{ self, source });
        try self.tasks.concurrent(self.io, run, .{ self, null });
    }

    fn run(self: *Collector, source: ?*Source) std.Io.Cancelable!void {
        while (!self.stopping.load(.acquire)) {
            const start_at: std.Io.Timestamp = .now(self.io, .awake);
            if (source) |reader| self.poll(reader) else self.hostStep();
            const remaining = self.interval.nanoseconds - start_at.untilNow(self.io, .awake).nanoseconds;
            if (remaining > 0) try self.io.sleep(.fromNanoseconds(remaining), .awake);
        }
    }

    fn poll(self: *Collector, source: *Source) void {
        const slot = source.readings.back();
        const back = &slot.value;
        const devices = self.devices.items[source.first..][0..back.samples.len];
        defer _ = source.scratch.reset(.retain_capacity);
        for (back.samples, devices) |*sample, device| sample.* = .init(device.target);
        _ = slot.arena.reset(.retain_capacity);
        back.* = .{ .samples = back.samples, .allocator = back.allocator };
        source.read(source.context, source.scratch.allocator(), back) catch
            back.processes.clearRetainingCapacity();
        for (back.samples, devices) |*sample, device| {
            if (std.meta.activeTag(sample.*) != device.target) sample.* = .init(device.target);
        }

        const completed: std.Io.Timestamp = .now(self.io, .awake);
        _ = source.readings.lock(self.io);
        defer source.readings.unlock(self.io);
        if (source.history.len != 0) {
            for (source.history, back.samples) |*ring, sample| {
                ring[source.next] = .{ .at = completed, .common = sample.common() };
            }
            source.next = (source.next + 1) % history_capacity;
            source.count = @min(source.count + 1, history_capacity);
        }
        source.readings.publishLocked();
    }

    fn hostStep(self: *Collector) void {
        const slot = self.host.back();
        const host = &slot.value;
        _ = slot.arena.reset(.retain_capacity);
        host.processes = .empty;
        const storage = slot.arena.allocator();
        for (self.sources.items) |source| {
            const front = &source.readings.lock(self.io).value;
            defer source.readings.unlock(self.io);
            for (front.processes.items) |row| {
                if (row.remote or row.device_idx >= front.samples.len) continue;
                _ = host.processes.getOrPutValue(storage, row.pid, .{ .pid = row.pid }) catch continue;
            }
        }
        self.host_reader.enrich(storage, host.processes.values());
        self.host_reader.sample(&host.data);
        _ = self.host.lock(self.io);
        self.host.publishLocked();
        self.host.unlock(self.io);
    }

    /// Copies each source's readings, process rows, and history under the same lock.
    /// OS process enrichment has its own cadence. The frame allocator owns the copies.
    pub fn capture(
        self: *Collector,
        frame: std.mem.Allocator,
        with_history: bool,
    ) std.mem.Allocator.Error!Snapshot {
        const samples = try frame.alloc(smi.Sample, self.devices.items.len);
        const histories: [][]const HistoryPoint = if (with_history)
            try frame.alloc([]const HistoryPoint, samples.len)
        else
            &.{};
        @memset(histories, &.{});
        var processes: std.ArrayList(smi.ProcessInfo) = .empty;
        var power: smi.HostData = .{};
        for (self.sources.items) |source| {
            const front = &source.readings.lock(self.io).value;
            defer source.readings.unlock(self.io);
            @memcpy(samples[source.first..][0..front.samples.len], front.samples);
            for (front.processes.items) |row| {
                if (row.device_idx >= front.samples.len) continue;
                var copy = row;
                copy.device_idx += @intCast(source.first);
                copy.username = try frame.dupe(u8, row.username);
                copy.comm = try frame.dupe(u8, row.comm);
                try processes.append(frame, copy);
            }
            if (front.host.cpu_power_mw) |value| power.cpu_power_mw = value;
            if (front.host.system_power_mw) |value| power.system_power_mw = value;

            if (with_history and source.count != 0) {
                const first = (source.next + history_capacity - source.count) % history_capacity;
                const split = @min(source.count, history_capacity - first);
                for (histories[source.first..][0..front.samples.len], source.history) |*history, *ring| {
                    history.* = try std.mem.concat(frame, HistoryPoint, &.{
                        ring[first..][0..split], ring[0 .. source.count - split],
                    });
                }
            }
        }

        const enriched = &self.host.lock(self.io).value;
        defer self.host.unlock(self.io);
        var host = enriched.data;
        host.cpu_power_mw = power.cpu_power_mw;
        host.system_power_mw = power.system_power_mw;
        for (processes.items) |*row| {
            if (row.remote) continue;
            const details = enriched.processes.get(row.pid) orelse continue;
            row.uid = details.uid;
            row.username = try frame.dupe(u8, details.username);
            row.comm = try frame.dupe(u8, details.comm);
            row.cpu_percent = details.cpu_percent;
            row.rss_kib = details.rss_kib;
        }
        return .{ .samples = samples, .processes = processes.items, .host = host, .histories = histories };
    }
};
