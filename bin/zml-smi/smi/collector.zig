const std = @import("std");
const model = @import("zml-smi/model");
const source = @import("zml-smi/source");
const DoubleBuffer = @import("double_buffer.zig").DoubleBuffer;

pub const Collector = struct {
    devices: std.ArrayList(model.Device) = .empty,
    two_readings: bool = false,
    sources: std.ArrayList(*Publication) = .empty,
    host: DoubleBuffer(Host),
    host_reader: source.HostReader,
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    options: Options,
    tasks: std.Io.Group = .init,
    stopping: std.atomic.Value(bool) = .init(false),

    pub const history_capacity = 1024;
    pub const Options = struct { poll_interval_ms: u16 = 500, history: bool = false };

    const Host = struct {
        data: model.HostData,
        processes: std.AutoArrayHashMapUnmanaged(u32, model.ProcessInfo) = .empty,
    };

    const Publication = struct {
        reader: source.Source,
        first: usize,
        scratch: std.heap.ArenaAllocator,
        readings: DoubleBuffer(source.Reading),
        history: [][history_capacity]model.HistoryPoint,
        next: usize = 0,
        count: usize = 0,
    };

    pub fn init(
        arena: std.mem.Allocator,
        gpa: std.mem.Allocator,
        io: std.Io,
        host_reader: source.HostReader,
        host: model.HostData,
        options: Options,
    ) Collector {
        return .{
            .host = .{ .values = @splat(.{ .value = .{ .data = host }, .arena = .init(gpa) }) },
            .host_reader = host_reader,
            .arena = arena,
            .gpa = gpa,
            .io = io,
            .options = options,
        };
    }

    pub fn deinit(self: *Collector) void {
        self.stopping.store(true, .release);
        self.tasks.cancel(self.io);
        for (self.sources.items) |publication| {
            publication.reader.close(publication.reader.context);
            publication.readings.deinit();
            publication.scratch.deinit();
        }
        self.host.deinit();
        self.host_reader.close(self.host_reader.context);
    }

    /// Takes ownership of an open source on success. Register before starting the workers.
    pub fn add(self: *Collector, reader: source.Source) !void {
        const count = reader.devices.len;
        if (count == 0) return reader.close(reader.context);
        const first = self.devices.items.len;
        if (count > std.math.maxInt(u16) - first) return error.TooManyDevices;

        const publication = try self.arena.create(Publication);
        publication.* = .{
            .reader = reader,
            .first = first,
            .scratch = .init(self.gpa),
            .readings = .{ .values = undefined },
            .history = if (self.options.history)
                try self.arena.alloc([history_capacity]model.HistoryPoint, count)
            else
                &.{},
        };
        for (&publication.readings.values) |*slot| {
            const samples = try self.arena.alloc(model.Sample, count);
            slot.arena = .init(self.gpa);
            slot.value = .{ .samples = samples, .allocator = slot.arena.allocator() };
            for (samples, reader.devices) |*sample, device| sample.* = .init(device.target);
        }
        try self.sources.ensureUnusedCapacity(self.arena, 1);
        try self.devices.appendSlice(self.arena, reader.devices);
        self.sources.appendAssumeCapacity(publication);
        self.two_readings = self.two_readings or reader.two_readings;
    }

    pub fn round(self: *Collector) void {
        for (self.sources.items) |publication| self.poll(publication);
        self.hostStep();
    }

    pub fn start(self: *Collector) !void {
        for (self.sources.items) |publication| {
            try self.tasks.concurrent(self.io, run, .{ self, publication });
        }
        try self.tasks.concurrent(self.io, run, .{ self, null });
    }

    fn run(self: *Collector, publication: ?*Publication) std.Io.Cancelable!void {
        const interval: std.Io.Duration = .fromMilliseconds(self.options.poll_interval_ms);
        while (!self.stopping.load(.acquire)) {
            const start_at: std.Io.Timestamp = .now(self.io, .awake);
            if (publication) |reader| self.poll(reader) else self.hostStep();
            const remaining = interval.nanoseconds - start_at.untilNow(self.io, .awake).nanoseconds;
            if (remaining > 0) try self.io.sleep(.fromNanoseconds(remaining), .awake);
        }
    }

    fn poll(self: *Collector, publication: *Publication) void {
        const slot = publication.readings.back();
        const back = &slot.value;
        const reader = publication.reader;
        const devices = reader.devices;
        defer _ = publication.scratch.reset(.retain_capacity);
        for (back.samples, devices) |*sample, device| sample.* = .init(device.target);
        _ = slot.arena.reset(.retain_capacity);
        back.* = .{ .samples = back.samples, .allocator = back.allocator };
        reader.poll(reader.context, publication.scratch.allocator(), back) catch
            back.processes.clearRetainingCapacity();
        for (back.samples, devices) |*sample, device| {
            if (std.meta.activeTag(sample.*) != device.target) sample.* = .init(device.target);
        }

        const completed: std.Io.Timestamp = .now(self.io, .awake);
        _ = publication.readings.lock(self.io);
        defer publication.readings.unlock(self.io);
        if (publication.history.len != 0) {
            for (publication.history, back.samples) |*ring, sample| {
                ring[publication.next] = .{ .at = completed, .common = sample.common() };
            }
            publication.next = (publication.next + 1) % history_capacity;
            publication.count = @min(publication.count + 1, history_capacity);
        }
        publication.readings.publishLocked();
    }

    fn hostStep(self: *Collector) void {
        const slot = self.host.back();
        const host = &slot.value;
        _ = slot.arena.reset(.retain_capacity);
        host.processes = .empty;
        const storage = slot.arena.allocator();
        for (self.sources.items) |publication| {
            const front = &publication.readings.lock(self.io).value;
            defer publication.readings.unlock(self.io);
            for (front.processes.items) |row| {
                if (row.remote or row.device_idx >= front.samples.len) continue;
                _ = host.processes.getOrPutValue(storage, row.pid, .{ .pid = row.pid }) catch continue;
            }
        }
        const reader = self.host_reader;
        reader.read(reader.context, storage, host.processes.values(), &host.data);
        _ = self.host.lock(self.io);
        self.host.publishLocked();
        self.host.unlock(self.io);
    }

    /// Copies each source's readings, process rows, and history under the same lock.
    /// OS process enrichment has its own cadence. The frame allocator owns the copies.
    pub fn capture(
        self: *Collector,
        arena: std.mem.Allocator,
        with_history: bool,
    ) std.mem.Allocator.Error!model.Frame {
        const samples = try arena.alloc(model.Sample, self.devices.items.len);
        const histories: [][]const model.HistoryPoint = if (with_history)
            try arena.alloc([]const model.HistoryPoint, samples.len)
        else
            &.{};
        @memset(histories, &.{});
        var processes: std.ArrayList(model.ProcessInfo) = .empty;
        var power: source.HostPower = .{};
        for (self.sources.items) |publication| {
            const front = &publication.readings.lock(self.io).value;
            defer publication.readings.unlock(self.io);
            @memcpy(samples[publication.first..][0..front.samples.len], front.samples);
            for (front.processes.items) |row| {
                if (row.device_idx >= front.samples.len) continue;
                var copy = row;
                copy.device_idx += @intCast(publication.first);
                copy.username = try arena.dupe(u8, row.username);
                copy.comm = try arena.dupe(u8, row.comm);
                try processes.append(arena, copy);
            }
            if (front.host_power.cpu_power_mw) |value| power.cpu_power_mw = value;
            if (front.host_power.system_power_mw) |value| power.system_power_mw = value;

            if (with_history and publication.count != 0) {
                const oldest = publication.next + history_capacity - publication.count;
                const first = oldest % history_capacity;
                const split = @min(publication.count, history_capacity - first);
                const device_histories = histories[publication.first..][0..front.samples.len];
                for (device_histories, publication.history) |*history, *ring| {
                    history.* = try std.mem.concat(arena, model.HistoryPoint, &.{
                        ring[first..][0..split], ring[0 .. publication.count - split],
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
            row.username = try arena.dupe(u8, details.username);
            row.comm = try arena.dupe(u8, details.comm);
            row.cpu_percent = details.cpu_percent;
            row.rss_kib = details.rss_kib;
        }
        return .{
            .devices = self.devices.items,
            .samples = samples,
            .processes = processes.items,
            .host = host,
            .history = if (with_history) .{
                .points = histories,
                .captured_at = .now(self.io, .awake),
                .interval_ms = @max(1, self.options.poll_interval_ms),
                .capacity = history_capacity,
            } else null,
        };
    }
};
