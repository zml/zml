const std = @import("std");
const pi = @import("zml-smi/info").process_info;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Collector = @import("zml-smi/collector").Collector;

pub const bdf_len = "0000:00:00.0".len;

pub const Target = struct {
    device_idx: u16,
    bdf: ?[bdf_len]u8,
};

pub fn init(collector: *Collector, list: *ProcessDoubleBuffer, targets: []const Target) !void {
    const state = try collector.arena.create(ProcessState);
    state.* = .{ .allocator = collector.arena };

    if (collector.poll_only) {
        defer state.previous.deinit(state.allocator);
        pollOnce(collector.gpa, collector.io, list, targets, state);
        collector.io.sleep(.fromMilliseconds(collector.worker.poll_interval_ms), .awake) catch {};
        pollOnce(collector.gpa, collector.io, list, targets, state);
    } else {
        try collector.spawnPoll(pollOnce, .{ collector.gpa, collector.io, list, targets, state });
    }
}

const ProcessState = struct {
    allocator: std.mem.Allocator,
    previous: std.AutoHashMapUnmanaged(u64, EngineSample) = .{},
};

pub const EngineSample = struct {
    engine_ns: u64,
    timestamp_ns: u64,
};

pub const DeviceSample = struct {
    mem_kib: u64 = 0,
    engine: ?EngineSample = null,
};

pub fn collectDeviceActivity(allocator: std.mem.Allocator, io: std.Io, targets: []const Target) !std.AutoHashMapUnmanaged(u16, EngineSample) {
    var usage = try collectDeviceUsage(allocator, io, targets);
    defer usage.deinit(allocator);

    var result: std.AutoHashMapUnmanaged(u16, EngineSample) = .{};
    errdefer result.deinit(allocator);

    var it = usage.iterator();
    while (it.next()) |entry| {
        const engine = entry.value_ptr.engine orelse continue;
        const gop = try result.getOrPut(allocator, entry.key_ptr.*);
        if (!gop.found_existing) {
            gop.value_ptr.* = .{ .engine_ns = 0, .timestamp_ns = engine.timestamp_ns };
        }
        gop.value_ptr.* = .{
            .engine_ns = gop.value_ptr.engine_ns + engine.engine_ns,
            .timestamp_ns = engine.timestamp_ns,
        };
    }

    return result;
}

pub fn collectDeviceUsage(allocator: std.mem.Allocator, io: std.Io, targets: []const Target) !std.AutoHashMapUnmanaged(u16, DeviceSample) {
    var fdinfo = try collectFdinfo(allocator, io, targets);
    defer fdinfo.deinit(allocator);

    var result: std.AutoHashMapUnmanaged(u16, DeviceSample) = .{};
    errdefer result.deinit(allocator);

    var it = fdinfo.iterator();
    while (it.next()) |entry| {
        const decoded = decodeProcessKey(entry.key_ptr.*);
        const gop = try result.getOrPut(allocator, decoded.device_idx);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        if (entry.value_ptr.mem_kib) |mem| gop.value_ptr.mem_kib += mem;
        if (entry.value_ptr.engine) |engine| {
            const current = gop.value_ptr.engine orelse EngineSample{ .engine_ns = 0, .timestamp_ns = engine.timestamp_ns };
            gop.value_ptr.engine = .{
                .engine_ns = current.engine_ns + engine.engine_ns,
                .timestamp_ns = engine.timestamp_ns,
            };
        }
    }

    return result;
}

fn pollOnce(allocator: std.mem.Allocator, io: std.Io, list: *ProcessDoubleBuffer, targets: []const Target, state: *ProcessState) void {
    const back = list.back();
    back.clearRetainingCapacity();

    var fdinfo = collectFdinfo(allocator, io, targets) catch {
        list.swap();
        return;
    };
    defer fdinfo.deinit(allocator);

    var it = fdinfo.iterator();
    while (it.next()) |entry| {
        const decoded = decodeProcessKey(entry.key_ptr.*);
        back.append(allocator, .{
            .pid = decoded.pid,
            .device_idx = decoded.device_idx,
            .dev_mem_kib = entry.value_ptr.mem_kib,
            .dev_util_percent = processUtil(state.previous.get(entry.key_ptr.*), entry.value_ptr.engine),
        }) catch break;
    }

    state.previous.clearRetainingCapacity();
    state.previous.ensureTotalCapacity(state.allocator, @intCast(fdinfo.count())) catch {};
    var sample_it = fdinfo.iterator();
    while (sample_it.next()) |entry| {
        if (entry.value_ptr.engine) |engine| {
            state.previous.putAssumeCapacity(entry.key_ptr.*, engine);
        }
    }

    list.swap();
}

const FdinfoSample = struct {
    mem_kib: ?u64 = null,
    engine: ?EngineSample = null,
};

fn collectFdinfo(allocator: std.mem.Allocator, io: std.Io, targets: []const Target) !std.AutoHashMapUnmanaged(u64, FdinfoSample) {
    var result: std.AutoHashMapUnmanaged(u64, FdinfoSample) = .{};
    errdefer result.deinit(allocator);

    var proc_dir = try std.Io.Dir.openDirAbsolute(io, "/proc", .{ .iterate = true });
    defer proc_dir.close(io);

    const now_ns: u64 = @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds);
    var proc_it = proc_dir.iterate();
    while (proc_it.next(io) catch null) |proc_entry| {
        const pid = std.fmt.parseInt(u32, proc_entry.name, 10) catch continue;

        var fdinfo_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const fdinfo_path = std.fmt.bufPrint(&fdinfo_path_buf, "/proc/{d}/fdinfo", .{pid}) catch continue;
        var fdinfo_dir = std.Io.Dir.openDirAbsolute(io, fdinfo_path, .{ .iterate = true }) catch continue;
        defer fdinfo_dir.close(io);

        var fd_it = fdinfo_dir.iterate();
        while (fd_it.next(io) catch null) |fd_entry| {
            var file_path_buf: [std.Io.Dir.max_path_bytes + 1]u8 = undefined;
            const file_path = std.fmt.bufPrintZ(&file_path_buf, "{s}/{s}", .{ fdinfo_path, fd_entry.name }) catch continue;
            const parsed = parseFdinfoFile(file_path, now_ns) catch continue;
            const dev_idx = matchTarget(targets, parsed.bdf orelse continue) orelse continue;
            const key = processKey(pid, dev_idx);
            const gop = try result.getOrPut(allocator, key);
            if (!gop.found_existing) gop.value_ptr.* = .{};
            mergeFdinfo(gop.value_ptr, parsed.sample);
        }
    }

    return result;
}

const ParsedFdinfo = struct {
    bdf: ?[bdf_len]u8 = null,
    sample: FdinfoSample = .{},
    resident_vram_kib: u64 = 0,
    total_vram_kib: u64 = 0,
    resident_memory_kib: u64 = 0,
    total_memory_kib: u64 = 0,
    cycle_counter: u64 = 0,
    total_cycle_counter: u64 = 0,
    engine_time_seen: bool = false,
};

fn parseFdinfoFile(path: [:0]const u8, timestamp_ns: u64) !ParsedFdinfo {
    const linux = std.os.linux;
    const open_rc = linux.open(path.ptr, .{
        .ACCMODE = .RDONLY,
        .CLOEXEC = true,
    }, 0);
    switch (linux.errno(open_rc)) {
        .SUCCESS => {},
        .ACCES, .NOENT, .NOTDIR, .PERM, .SRCH => return error.FileUnavailable,
        else => return error.FileUnavailable,
    }
    const fd: i32 = @intCast(open_rc);
    defer _ = linux.close(fd);

    var read_buf: [8192]u8 = undefined;
    var len: usize = 0;
    while (len < read_buf.len) {
        const read_rc = linux.read(fd, read_buf[len..].ptr, read_buf.len - len);
        switch (linux.errno(read_rc)) {
            .SUCCESS => {
                const n = read_rc;
                if (n == 0) break;
                len += n;
            },
            .INTR => continue,
            .ACCES, .BADF, .IO, .NOENT, .PERM, .SRCH => return error.FileUnavailable,
            else => return error.FileUnavailable,
        }
    }

    var parsed: ParsedFdinfo = .{};

    var start: usize = 0;
    while (std.mem.indexOfScalarPos(u8, read_buf[0..len], start, '\n')) |end| {
        parseFdinfoLine(&parsed, read_buf[start..end], timestamp_ns);
        start = end + 1;
    }
    if (start < len) {
        parseFdinfoLine(&parsed, read_buf[start..len], timestamp_ns);
    }
    finish(&parsed);

    return parsed;
}

fn finish(self: *ParsedFdinfo) void {
    if (self.resident_vram_kib > 0) {
        self.sample.mem_kib = self.resident_vram_kib;
    } else if (self.total_vram_kib > 0) {
        self.sample.mem_kib = self.total_vram_kib;
    } else if (self.resident_memory_kib > 0) {
        self.sample.mem_kib = self.resident_memory_kib;
    } else if (self.total_memory_kib > 0) {
        self.sample.mem_kib = self.total_memory_kib;
    }

    if (!self.engine_time_seen and self.total_cycle_counter > 0) {
        self.sample.engine = .{
            .engine_ns = self.cycle_counter,
            .timestamp_ns = self.total_cycle_counter,
        };
    }
}

pub fn parseFdinfoLine(parsed: *ParsedFdinfo, line: []const u8, timestamp_ns: u64) void {
    if (std.mem.startsWith(u8, line, "drm-pdev:")) {
        parsed.bdf = parseBdf(line["drm-pdev:".len..]);
        return;
    }

    if (std.mem.startsWith(u8, line, "vram mem:")) {
        parsed.total_vram_kib += memoryKiB(line["vram mem:".len..]);
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-cycles-")) {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return;
        parsed.cycle_counter += firstInt(line[colon + 1 ..]);
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-total-cycles-")) {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return;
        parsed.total_cycle_counter = @max(parsed.total_cycle_counter, firstInt(line[colon + 1 ..]));
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-engine-capacity-")) {
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-engine-")) {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return;
        const value = firstInt(line[colon + 1 ..]);
        parsed.engine_time_seen = true;
        const engine = parsed.sample.engine orelse EngineSample{ .engine_ns = 0, .timestamp_ns = timestamp_ns };
        parsed.sample.engine = .{
            .engine_ns = engine.engine_ns + value,
            .timestamp_ns = timestamp_ns,
        };
        return;
    }

    if (std.mem.startsWith(u8, line, "drm-")) {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return;
        const key = line[0..colon];
        const value = memoryKiB(line[colon + 1 ..]);
        if (value == 0) return;

        if (std.mem.startsWith(u8, key, "drm-resident-vram")) {
            parsed.resident_vram_kib += value;
            return;
        }
        if (std.mem.startsWith(u8, key, "drm-total-vram") or std.mem.startsWith(u8, key, "drm-memory-vram")) {
            parsed.total_vram_kib += value;
            return;
        }
        if (std.mem.startsWith(u8, key, "drm-resident-")) {
            parsed.resident_memory_kib += value;
            return;
        }
        if (std.mem.startsWith(u8, key, "drm-total-")) {
            parsed.total_memory_kib += value;
            return;
        }
        if (std.mem.eql(u8, key, "drm-resident-memory")) {
            parsed.resident_memory_kib += value;
            return;
        }
        if (std.mem.eql(u8, key, "drm-total-memory")) {
            parsed.total_memory_kib += value;
            return;
        }
    }
}

fn mergeFdinfo(dst: *FdinfoSample, src: FdinfoSample) void {
    if (src.mem_kib) |mem| dst.mem_kib = (dst.mem_kib orelse 0) + mem;
    if (src.engine) |engine| {
        const current = dst.engine orelse EngineSample{ .engine_ns = 0, .timestamp_ns = engine.timestamp_ns };
        dst.engine = .{
            .engine_ns = current.engine_ns + engine.engine_ns,
            .timestamp_ns = engine.timestamp_ns,
        };
    }
}

fn matchTarget(targets: []const Target, bdf: [bdf_len]u8) ?u16 {
    for (targets) |target| {
        if (target.bdf) |target_bdf| {
            if (std.mem.eql(u8, &target_bdf, &bdf)) return target.device_idx;
        }
    }
    return null;
}

pub fn processUtil(previous: ?EngineSample, current: ?EngineSample) ?u16 {
    const prev = previous orelse return null;
    const cur = current orelse return null;
    if (cur.timestamp_ns <= prev.timestamp_ns or cur.engine_ns < prev.engine_ns) return null;
    const engine_delta = cur.engine_ns - prev.engine_ns;
    const time_delta = cur.timestamp_ns - prev.timestamp_ns;
    if (time_delta == 0) return null;
    const pct = @min(engine_delta * 100 / time_delta, 100);
    return @intCast(if (pct == 0 and engine_delta > 0) 1 else pct);
}

fn processKey(pid: u32, device_idx: u16) u64 {
    return (@as(u64, device_idx) << 32) | pid;
}

fn decodeProcessKey(key: u64) struct { pid: u32, device_idx: u16 } {
    return .{
        .pid = @truncate(key),
        .device_idx = @truncate(key >> 32),
    };
}

pub const BdfParts = struct {
    domain: u32,
    bus: u32,
    device: u32,
    function: u32,
};

pub fn formatBdf(parts: BdfParts) [bdf_len]u8 {
    var buf: [bdf_len]u8 = undefined;
    _ = std.fmt.bufPrint(&buf, "{x:0>4}:{x:0>2}:{x:0>2}.{x}", .{
        parts.domain,
        parts.bus,
        parts.device,
        parts.function,
    }) catch unreachable;
    return buf;
}

fn parseBdf(raw: []const u8) ?[bdf_len]u8 {
    const trimmed = std.mem.trim(u8, raw, " \t\n");
    var out: [bdf_len]u8 = undefined;
    if (trimmed.len == bdf_len) {
        @memcpy(&out, trimmed[0..bdf_len]);
        return out;
    }
    if (trimmed.len == "00:00.0".len) {
        _ = std.fmt.bufPrint(&out, "0000:{s}", .{trimmed}) catch return null;
        return out;
    }
    return null;
}

fn firstInt(raw: []const u8) u64 {
    var iter = std.mem.tokenizeAny(u8, raw, " \t");
    return std.fmt.parseInt(u64, iter.next() orelse return 0, 10) catch 0;
}

fn memoryKiB(raw: []const u8) u64 {
    var iter = std.mem.tokenizeAny(u8, raw, " \t");
    const value = std.fmt.parseInt(u64, iter.next() orelse return 0, 10) catch return 0;
    const unit = iter.next() orelse return value;
    if (std.ascii.eqlIgnoreCase(unit, "KiB")) return value;
    if (std.ascii.eqlIgnoreCase(unit, "MiB")) return value * 1024;
    if (std.ascii.eqlIgnoreCase(unit, "GiB")) return value * 1024 * 1024;
    if (std.ascii.eqlIgnoreCase(unit, "B")) return value / 1024;
    return value;
}

test "oneAPI fdinfo parser" {
    var parsed: ParsedFdinfo = .{};
    parseFdinfoLine(&parsed, "drm-pdev:\t0000:03:00.0", 10);
    parseFdinfoLine(&parsed, "drm-total-memory:\t2048 KiB", 10);
    parseFdinfoLine(&parsed, "drm-engine-render:\t100 ns", 10);
    parseFdinfoLine(&parsed, "drm-engine-compute:\t50 ns", 10);
    finish(&parsed);

    try std.testing.expect(parsed.bdf != null);
    try std.testing.expectEqualSlices(u8, "0000:03:00.0", &parsed.bdf.?);
    try std.testing.expectEqual(@as(?u64, 2048), parsed.sample.mem_kib);
    try std.testing.expectEqual(@as(u64, 150), parsed.sample.engine.?.engine_ns);
}

test "oneAPI fdinfo parser cycle counters" {
    var parsed: ParsedFdinfo = .{};
    parseFdinfoLine(&parsed, "drm-pdev:\t0000:03:00.0", 10);
    parseFdinfoLine(&parsed, "drm-cycles-rcs:\t10", 10);
    parseFdinfoLine(&parsed, "drm-total-cycles-rcs:\t100", 10);
    parseFdinfoLine(&parsed, "drm-engine-capacity-vcs:\t2", 10);
    parseFdinfoLine(&parsed, "drm-cycles-ccs:\t30", 10);
    parseFdinfoLine(&parsed, "drm-total-cycles-ccs:\t100", 10);
    finish(&parsed);

    try std.testing.expectEqual(@as(u64, 40), parsed.sample.engine.?.engine_ns);
    try std.testing.expectEqual(@as(u64, 100), parsed.sample.engine.?.timestamp_ns);
}

test "oneAPI fdinfo parser xe vram regions" {
    var parsed: ParsedFdinfo = .{};
    parseFdinfoLine(&parsed, "drm-pdev:\t0000:03:00.0", 10);
    parseFdinfoLine(&parsed, "drm-total-system:\t184 MiB", 10);
    parseFdinfoLine(&parsed, "drm-total-vram0:\t4096 KiB", 10);
    parseFdinfoLine(&parsed, "drm-resident-vram0:\t3072 KiB", 10);
    finish(&parsed);

    try std.testing.expectEqual(@as(?u64, 3072), parsed.sample.mem_kib);
}

test "oneAPI fdinfo parser falls back to total vram" {
    var parsed: ParsedFdinfo = .{};
    parseFdinfoLine(&parsed, "drm-total-vram0:\t2 MiB", 10);
    finish(&parsed);

    try std.testing.expectEqual(@as(?u64, 2048), parsed.sample.mem_kib);
}

test "oneAPI process utilization delta" {
    try std.testing.expectEqual(@as(?u16, 50), processUtil(.{ .engine_ns = 100, .timestamp_ns = 1000 }, .{ .engine_ns = 600, .timestamp_ns = 2000 }));
    try std.testing.expectEqual(@as(?u16, 1), processUtil(.{ .engine_ns = 100, .timestamp_ns = 1000 }, .{ .engine_ns = 101, .timestamp_ns = 2000 }));
    try std.testing.expectEqual(@as(?u16, null), processUtil(null, .{ .engine_ns = 600, .timestamp_ns = 2000 }));
}
