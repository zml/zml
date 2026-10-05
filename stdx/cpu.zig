const std = @import("std");
const builtin = @import("builtin");

/// Restricts the calling thread and existing threads to the CPUs of a NUMA
/// node within the caller's current affinity, preserving each thread's narrower
/// affinity. Threads with no overlap are left alone. Returns the number of
/// threads whose affinity changed.
/// Call during startup, before threads can concurrently create other threads.
/// On failure, previously changed affinities remain in effect.
pub fn pinProcessToNumaNode(io: std.Io, node: u32) !usize {
    if (builtin.os.tag != .linux) return error.Unsupported;

    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "/sys/devices/system/node/node{d}/cpulist", .{node});

    var cpulist_buf: [4096]u8 = undefined;
    const cpulist = try std.Io.Dir.cwd().readFile(io, path, &cpulist_buf);

    var set: std.os.linux.cpu_set_t = @splat(0);
    try parseCpuList(cpulist, &set);

    return pinProcessToCpuSet(io, set);
}

/// CPUs the calling thread may run on, ordered for worker pools.
pub const AllowedCores = struct {
    /// Allowed CPU ids: one per physical core first, then the remaining SMT
    /// siblings, round by round.
    ids: []const usize,
    /// `ids[0..physical_count]` holds exactly one CPU per physical core. Equals
    /// `ids.len` when the topology is unknown.
    physical_count: usize,

    /// One CPU per physical core, skipping SMT siblings.
    pub fn physical(self: AllowedCores) []const usize {
        return self.ids[0..self.physical_count];
    }

    pub fn deinit(self: AllowedCores, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
    }
};

/// Returns the CPUs the calling thread may run on. Linux uses sysfs topology so
/// that a second logical CPU of a physical core comes only after every allowed
/// physical core got one.
pub fn allowedCores(allocator: std.mem.Allocator, io: std.Io) !AllowedCores {
    if (builtin.os.tag != .linux) return naturalOrder(allocator, try std.Thread.getCpuCount());

    const cpus = linuxAllowedCoreIds(allocator) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return naturalOrder(allocator, try std.Thread.getCpuCount()),
    };
    errdefer allocator.free(cpus);

    const physical_count = linuxOrderBySiblings(allocator, io, cpus) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => cpus.len,
    };
    return .{ .ids = cpus, .physical_count = physical_count };
}

fn pinProcessToCpuSet(io: std.Io, node_set: std.os.linux.cpu_set_t) !usize {
    const linux = std.os.linux;
    // Keep the caller's taskset/numactl choice as the process-wide limit, and
    // intersect each thread's own affinity below so private bindings stay narrow.
    const set = intersectCpuSets(node_set, try threadAffinity(0));
    if (std.mem.allEqual(usize, &set, 0)) return 0;

    var tasks = try std.Io.Dir.cwd().openDir(io, "/proc/self/task", .{ .iterate = true });
    defer tasks.close(io);

    // Pin the caller first so threads it creates inherit the target mask.
    var pinned: usize = @intFromBool(try pinThreadToCpuSet(0, set));
    var it = tasks.iterate();
    while (try it.next(io)) |entry| {
        const tid = std.fmt.parseInt(linux.pid_t, entry.name, 10) catch continue;
        // Threads may exit while the list is being walked.
        const changed = pinThreadToCpuSet(tid, set) catch |err| switch (err) {
            error.ThreadGone => continue,
            else => return err,
        };
        pinned += @intFromBool(changed);
    }

    return pinned;
}

fn intersectCpuSets(a: std.os.linux.cpu_set_t, b: std.os.linux.cpu_set_t) std.os.linux.cpu_set_t {
    var set = a;
    for (&set, b) |*word, other| word.* &= other;
    return set;
}

fn threadAffinity(tid: std.os.linux.pid_t) error{ ThreadGone, GetAffinityFailed }!std.os.linux.cpu_set_t {
    const linux = std.os.linux;
    var set: linux.cpu_set_t = @splat(0);
    switch (linux.errno(linux.sched_getaffinity(tid, @sizeOf(linux.cpu_set_t), &set))) {
        .SUCCESS => return set,
        .SRCH => return error.ThreadGone,
        else => return error.GetAffinityFailed,
    }
}

fn setThreadAffinity(tid: std.os.linux.pid_t, set: *const std.os.linux.cpu_set_t) error{ ThreadGone, SetAffinityFailed }!void {
    const linux = std.os.linux;
    const rc = linux.syscall3(.sched_setaffinity, @as(usize, @bitCast(@as(isize, tid))), @sizeOf(linux.cpu_set_t), @intFromPtr(set));
    switch (linux.errno(rc)) {
        .SUCCESS => {},
        .SRCH => return error.ThreadGone,
        else => return error.SetAffinityFailed,
    }
}

fn pinThreadToCpuSet(tid: std.os.linux.pid_t, target: std.os.linux.cpu_set_t) !bool {
    const current = try threadAffinity(tid);
    const set = intersectCpuSets(target, current);
    if (std.mem.allEqual(usize, &set, 0) or std.mem.eql(usize, &set, &current)) return false;
    try setThreadAffinity(tid, &set);
    return true;
}

/// Parses Linux CPU lists, including singleton CPUs and inclusive ranges.
fn parseCpuList(cpulist: []const u8, set: *std.os.linux.cpu_set_t) !void {
    const trimmed = std.mem.trim(u8, cpulist, " \t\r\n");
    if (trimmed.len == 0) return;

    const bits_per_word = @bitSizeOf(usize);

    var ranges = std.mem.splitScalar(u8, trimmed, ',');
    while (ranges.next()) |raw_range| {
        const range = std.mem.trim(u8, raw_range, " \t\r\n");
        var ends = std.mem.splitScalar(u8, range, '-');
        const first = std.fmt.parseInt(usize, ends.first(), 10) catch return error.InvalidCpuList;

        const last = if (ends.next()) |end|
            std.fmt.parseInt(usize, end, 10) catch return error.InvalidCpuList
        else
            first;

        if (last < first or ends.next() != null or last >= set.len * bits_per_word) return error.InvalidCpuList;
        for (first..last + 1) |cpu| {
            set[cpu / bits_per_word] |= @as(usize, 1) << @intCast(cpu % bits_per_word);
        }
    }
}

fn linuxAllowedCoreIds(allocator: std.mem.Allocator) ![]usize {
    const set = try threadAffinity(0);

    var core_ids: std.ArrayList(usize) = .empty;
    errdefer core_ids.deinit(allocator);

    var core_id: usize = 0;
    while (core_id < @bitSizeOf(std.os.linux.cpu_set_t)) : (core_id += 1) {
        if (cpuSetContains(set, core_id)) {
            try core_ids.append(allocator, core_id);
        }
    }

    if (core_ids.items.len == 0) return error.MissingCpuTopology;

    return core_ids.toOwnedSlice(allocator);
}

fn cpuSetContains(set: std.os.linux.cpu_set_t, core_id: usize) bool {
    const word_bit_count = @bitSizeOf(usize);
    const word_index = core_id / word_bit_count;
    const bit_index: std.math.Log2Int(usize) = @intCast(core_id % word_bit_count);
    return word_index < set.len and (set[word_index] & (@as(usize, 1) << bit_index)) != 0;
}

fn linuxOrderBySiblings(allocator: std.mem.Allocator, io: std.Io, cpus: []usize) !usize {
    const leaders = try allocator.alloc(usize, cpus.len);
    defer allocator.free(leaders);
    for (cpus, leaders) |cpu, *leader| leader.* = try linuxCoreLeader(io, cpu);
    return orderBySiblings(allocator, cpus, leaders);
}

/// Returns the lowest CPU id sharing a physical core with `cpu`.
fn linuxCoreLeader(io: std.Io, cpu: usize) !usize {
    var cpulist_buf: [4096]u8 = undefined;
    // `core_cpus_list` replaced `thread_siblings_list` in Linux 5.7.
    const cpulist = linuxReadTopology(io, cpu, "core_cpus_list", &cpulist_buf) catch |err| switch (err) {
        error.FileNotFound => try linuxReadTopology(io, cpu, "thread_siblings_list", &cpulist_buf),
        else => return err,
    };

    var set: std.os.linux.cpu_set_t = @splat(0);
    try parseCpuList(cpulist, &set);
    for (set, 0..) |word, word_index| {
        if (word != 0) return word_index * @bitSizeOf(usize) + @ctz(word);
    }
    return error.MissingCpuTopology;
}

fn linuxReadTopology(io: std.Io, cpu: usize, name: []const u8, buf: []u8) ![]const u8 {
    var path_buf: [96]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "/sys/devices/system/cpu/cpu{d}/topology/{s}", .{ cpu, name });
    return std.Io.Dir.cwd().readFile(io, path, buf);
}

/// Reorders `cpus` so that each round holds at most one CPU per physical core,
/// in ascending order, before the next round of SMT siblings. `leaders[i]`
/// identifies the physical core of `cpus[i]`. Returns the number of physical
/// cores, which is also the size of the first round.
fn orderBySiblings(allocator: std.mem.Allocator, cpus: []usize, leaders: []const usize) !usize {
    const Entry = struct {
        round: usize,
        cpu: usize,

        fn lessThan(_: void, lhs: @This(), rhs: @This()) bool {
            return if (lhs.round != rhs.round) lhs.round < rhs.round else lhs.cpu < rhs.cpu;
        }
    };

    const entries = try allocator.alloc(Entry, cpus.len);
    defer allocator.free(entries);

    var physical_count: usize = 0;
    for (entries, cpus, leaders, 0..) |*entry, cpu, leader, index| {
        var round: usize = 0;
        for (leaders[0..index]) |previous| round += @intFromBool(previous == leader);
        entry.* = .{ .round = round, .cpu = cpu };
        physical_count += @intFromBool(round == 0);
    }

    std.mem.sort(Entry, entries, {}, Entry.lessThan);
    for (cpus, entries) |*cpu, entry| cpu.* = entry.cpu;
    return physical_count;
}

fn naturalOrder(allocator: std.mem.Allocator, cpu_count: usize) !AllowedCores {
    const ids = try allocator.alloc(usize, cpu_count);
    for (ids, 0..) |*id, index| id.* = index;
    return .{ .ids = ids, .physical_count = ids.len };
}

test "CPU lists respect mask boundaries" {
    var set: std.os.linux.cpu_set_t = @splat(0);
    try parseCpuList("0-2,64,66-67\n", &set);
    try std.testing.expectEqual(@as(usize, 0b111), set[0]);
    try std.testing.expectEqual(@as(usize, 0b1101), set[1]);
    const capacity = @bitSizeOf(std.os.linux.cpu_set_t);
    var text: [32]u8 = undefined;
    try parseCpuList(try std.fmt.bufPrint(&text, "{d}", .{capacity - 1}), &set);
    try std.testing.expect((set[set.len - 1] & (@as(usize, 1) << (@bitSizeOf(usize) - 1))) != 0);
    try std.testing.expectError(error.InvalidCpuList, parseCpuList(try std.fmt.bufPrint(&text, "{d}", .{capacity}), &set));
    try std.testing.expectError(error.InvalidCpuList, parseCpuList("3-1", &set));
    try std.testing.expectError(error.InvalidCpuList, parseCpuList("1-2-3", &set));
}

const AffinityTest = struct {
    const CpuSet = std.os.linux.cpu_set_t;

    fn available(count: usize) !CpuSet {
        if (builtin.os.tag != .linux) return error.SkipZigTest;
        const current = try threadAffinity(0);
        var selected: CpuSet = @splat(0);
        var remaining = count;
        for (0..@bitSizeOf(CpuSet)) |core_id| {
            if (!cpuSetContains(current, core_id)) continue;
            selected[core_id / @bitSizeOf(usize)] |= @as(usize, 1) << @intCast(core_id % @bitSizeOf(usize));
            remaining -= 1;
            if (remaining == 0) break;
        }
        if (remaining != 0) return error.SkipZigTest;
        return selected;
    }

    fn check(caller: CpuSet, worker: CpuSet, node: CpuSet, expected: CpuSet) !void {
        const io = std.testing.io;
        const Worker = struct {
            io: std.Io,
            set: CpuSet,
            tid: std.os.linux.pid_t = undefined,
            ready: std.Io.Event = .unset,
            stop: std.Io.Event = .unset,
            failure: ?anyerror = null,

            fn run(self: *@This()) void {
                self.tid = @intCast(std.Thread.getCurrentId());
                setThreadAffinity(0, &self.set) catch |err| {
                    self.failure = err;
                };
                self.ready.set(self.io);
                self.stop.waitUncancelable(self.io);
            }
        };
        var state: Worker = .{ .io = io, .set = worker };
        const thread = try std.Thread.spawn(.{}, Worker.run, .{&state});
        defer {
            state.stop.set(io);
            thread.join();
        }
        try state.ready.wait(io);
        if (state.failure) |err| return err;

        // Restore every thread touched by the process-wide helper, including
        // any test runner IO workers, before the next test executes.
        var saved: std.ArrayList(struct { tid: std.os.linux.pid_t, set: CpuSet }) = .empty;
        defer {
            for (saved.items) |entry| setThreadAffinity(entry.tid, &entry.set) catch |err| switch (err) {
                error.ThreadGone => {},
                error.SetAffinityFailed => @panic("failed to restore test thread affinity"),
            };
            saved.deinit(std.testing.allocator);
        }
        var tasks = try std.Io.Dir.cwd().openDir(io, "/proc/self/task", .{ .iterate = true });
        defer tasks.close(io);
        var it = tasks.iterate();
        while (try it.next(io)) |entry| {
            const tid = std.fmt.parseInt(std.os.linux.pid_t, entry.name, 10) catch continue;
            const set = threadAffinity(tid) catch |err| switch (err) {
                error.ThreadGone => continue,
                else => return err,
            };
            try saved.append(std.testing.allocator, .{ .tid = tid, .set = set });
        }

        try setThreadAffinity(0, &caller);
        _ = try pinProcessToCpuSet(io, node);
        try std.testing.expectEqualSlices(usize, &expected, &(try threadAffinity(state.tid)));
        const expected_caller = intersectCpuSets(caller, node);
        try std.testing.expectEqualSlices(usize, &expected_caller, &(try threadAffinity(0)));
    }
};

test "process NUMA affinity preserves a narrower worker binding" {
    const caller = try AffinityTest.available(3);
    const node = try AffinityTest.available(2);
    const worker = try AffinityTest.available(1);
    try AffinityTest.check(caller, worker, node, worker);
}

test "process NUMA affinity visits workers when the caller is already pinned" {
    const caller = try AffinityTest.available(1);
    const worker = try AffinityTest.available(2);
    try AffinityTest.check(caller, worker, worker, caller);
}

test "process NUMA affinity leaves a worker with no overlapping CPUs alone" {
    const caller = try AffinityTest.available(2);
    const node = try AffinityTest.available(1);
    var worker = caller;
    for (&worker, node) |*word, excluded| word.* &= ~excluded;
    try AffinityTest.check(caller, worker, node, worker);
}

test "CPU lists accept whitespace and empty node CPU sets" {
    var set: std.os.linux.cpu_set_t = @splat(0);
    try parseCpuList(" \t\r\n", &set);
    try std.testing.expect(std.mem.allEqual(usize, &set, 0));
    try parseCpuList(" \t0-2, 64,66-67\r\n", &set);
    try std.testing.expectEqual(@as(usize, 0b111), set[0]);
    try std.testing.expectEqual(@as(usize, 0b1101), set[1]);
}

test "CPU lists reject malformed ranges" {
    for ([_][]const u8{ "3-1", "x", "-1", "1-", "1-2-3", "1,,2", "1,", ",1", "999999999999999999999999999999" }) |text| {
        var set: std.os.linux.cpu_set_t = @splat(0);
        try std.testing.expectError(error.InvalidCpuList, parseCpuList(text, &set));
    }
}

test "natural order treats every CPU as physical" {
    const cores = try naturalOrder(std.testing.allocator, 4);
    defer cores.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, cores.ids);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, cores.physical());
}

test "sibling order puts one CPU per physical core first" {
    const Case = struct { cpus: []const usize, leaders: []const usize, expected: []const usize, physical_count: usize };
    const cases = [_]Case{
        // Siblings numbered after all cores (typical Intel).
        .{ .cpus = &.{ 0, 1, 2, 3 }, .leaders = &.{ 0, 1, 0, 1 }, .expected = &.{ 0, 1, 2, 3 }, .physical_count = 2 },
        // Adjacent siblings (typical AMD, some ARM).
        .{ .cpus = &.{ 0, 1, 2, 3 }, .leaders = &.{ 0, 0, 2, 2 }, .expected = &.{ 0, 2, 1, 3 }, .physical_count = 2 },
        // Affinity allows only some siblings of cores {0,4}, {1,5}, {2,6}.
        .{ .cpus = &.{ 0, 1, 2, 5 }, .leaders = &.{ 0, 1, 2, 1 }, .expected = &.{ 0, 1, 2, 5 }, .physical_count = 3 },
        // Affinity allows only the second sibling of some cores.
        .{ .cpus = &.{ 1, 4, 5 }, .leaders = &.{ 1, 0, 1 }, .expected = &.{ 1, 4, 5 }, .physical_count = 2 },
        // No SMT.
        .{ .cpus = &.{ 0, 1, 2 }, .leaders = &.{ 0, 1, 2 }, .expected = &.{ 0, 1, 2 }, .physical_count = 3 },
    };

    for (cases) |case| {
        const cpus = try std.testing.allocator.dupe(usize, case.cpus);
        defer std.testing.allocator.free(cpus);

        const physical_count = try orderBySiblings(std.testing.allocator, cpus, case.leaders);
        try std.testing.expectEqualSlices(usize, case.expected, cpus);
        try std.testing.expectEqual(case.physical_count, physical_count);
    }
}

test "allowed cores cover the thread affinity" {
    const cores = try allowedCores(std.testing.allocator, std.testing.io);
    defer cores.deinit(std.testing.allocator);

    try std.testing.expect(cores.physical_count > 0 and cores.physical_count <= cores.ids.len);
    if (builtin.os.tag != .linux) return;

    const affinity = try threadAffinity(0);
    var expected_count: usize = 0;
    for (affinity) |word| expected_count += @popCount(word);
    try std.testing.expectEqual(expected_count, cores.ids.len);
    for (cores.ids) |cpu| try std.testing.expect(cpuSetContains(affinity, cpu));
}
