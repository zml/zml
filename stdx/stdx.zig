const std = @import("std");
const builtin = @import("builtin");

pub const BoundedArray = @import("bounded_array.zig").BoundedArray;
pub const BoundedArrayAligned = @import("bounded_array.zig").BoundedArrayAligned;
pub const crypto = @import("crypto.zig");
pub const debug = @import("debug.zig");
pub const flags = @import("flags.zig");
pub const fmt = @import("fmt.zig");
pub const Io = @import("Io.zig");
pub const json = @import("json.zig");
pub const math = @import("math.zig");
pub const meta = @import("meta.zig");
pub const process = @import("process.zig");
pub const queue = @import("queue.zig");
pub const SegmentedList = @import("segmented_list.zig").SegmentedList;
pub const time = @import("time.zig");
pub const unicode = @import("unicode.zig");

test {
    std.testing.refAllDecls(@This());
}

pub inline fn stackSlice(comptime max_len: usize, T: type, len: usize) []T {
    debug.assert(len <= max_len, "stackSlice can only create a slice of up to {} elements, got: {}", .{ max_len, len });
    var storage: [max_len]T = undefined;
    return storage[0..len];
}

pub const noalloc: std.mem.Allocator = if (builtin.mode == .ReleaseFast) undefined else std.testing.failing_allocator;

pub fn arenaWithCapacity(parent: std.mem.Allocator, initial_capacity: usize) std.mem.Allocator.Error!std.heap.ArenaAllocator {
    var a: std.heap.ArenaAllocator = .init(parent);

    _ = try a.allocator().alloc(u8, initial_capacity);
    std.debug.assert(a.state.used_list.?.end_index == initial_capacity);
    a.state.used_list.?.end_index = 0;
    return a;
}

pub fn pinToCore(core_id: usize) void {
    if (builtin.os.tag == .linux) {
        const CPUSet = std.bit_set.ArrayBitSet(usize, std.os.linux.CPU_SETSIZE * @sizeOf(usize));

        var set: CPUSet = .initEmpty();
        set.set(core_id);
        std.os.linux.sched_setaffinity(0, @ptrCast(&set.masks)) catch {};
    }
}

/// Restricts every thread of the current process to the CPUs of a NUMA node
/// that the calling thread may already run on. Threads created afterwards
/// inherit the affinity of their creator. Returns the number of threads that
/// were pinned: 0 when the current affinity is already within the node or
/// excludes all of it.
pub fn pinProcessToNumaNode(io: std.Io, node: u32) !usize {
    if (builtin.os.tag != .linux) return error.Unsupported;
    const linux = std.os.linux;

    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "/sys/devices/system/node/node{d}/cpulist", .{node});
    var cpulist_buf: [4096]u8 = undefined;
    const cpulist = try std.Io.Dir.cwd().readFile(io, path, &cpulist_buf);

    var set: linux.cpu_set_t = @splat(0);
    try parseCpuList(std.mem.trim(u8, cpulist, " \n"), &set);

    // Keep any narrower affinity chosen by the user (taskset, numactl).
    var current: linux.cpu_set_t = @splat(0);
    if (linux.errno(linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &current)) != .SUCCESS) {
        return error.GetAffinityFailed;
    }
    var changed = false;
    var empty = true;
    for (&set, current) |*word, current_word| {
        word.* &= current_word;
        changed = changed or word.* != current_word;
        empty = empty and word.* == 0;
    }
    if (empty or !changed) return 0;

    var tasks = try std.Io.Dir.cwd().openDir(io, "/proc/self/task", .{ .iterate = true });
    defer tasks.close(io);

    var pinned: usize = 0;
    var it = tasks.iterate();
    while (try it.next(io)) |entry| {
        const tid = std.fmt.parseInt(linux.pid_t, entry.name, 10) catch continue;
        const rc = linux.syscall3(.sched_setaffinity, @as(usize, @bitCast(@as(isize, tid))), @sizeOf(linux.cpu_set_t), @intFromPtr(&set));
        // Threads may exit while the list is being walked.
        switch (linux.errno(rc)) {
            .SUCCESS => pinned += 1,
            .SRCH => {},
            else => return error.SetAffinityFailed,
        }
    }
    return pinned;
}

/// Parses a Linux cpulist such as "0-71" or "0-3,8,10-11".
fn parseCpuList(cpulist: []const u8, set: *std.os.linux.cpu_set_t) !void {
    const bits_per_word = @bitSizeOf(usize);
    var ranges = std.mem.tokenizeScalar(u8, cpulist, ',');
    while (ranges.next()) |range| {
        const dash = std.mem.findScalar(u8, range, '-');
        const first = try std.fmt.parseInt(usize, range[0 .. dash orelse range.len], 10);
        const last = if (dash) |d| try std.fmt.parseInt(usize, range[d + 1 ..], 10) else first;
        if (last < first or last >= set.len * bits_per_word) return error.InvalidCpuList;
        for (first..last + 1) |cpu| {
            set[cpu / bits_per_word] |= @as(usize, 1) << @intCast(cpu % bits_per_word);
        }
    }
}

test parseCpuList {
    var set: std.os.linux.cpu_set_t = @splat(0);
    try parseCpuList("0-2,64,66-67", &set);
    try std.testing.expectEqual(@as(usize, 0b111), set[0]);
    try std.testing.expectEqual(@as(usize, 0b1101), set[1]);
    try std.testing.expectError(error.InvalidCpuList, parseCpuList("3-1", &set));
}

pub fn once(comptime f: anytype) once(f) {
    return .{};
}

/// An object that executes the function `f` just once.
/// It is undefined behavior if `f` re-enters the same Once instance.
pub fn Once(comptime f: anytype) type {
    const Args = std.meta.ArgsTuple(@TypeOf(f));
    return struct {
        done: bool = false,
        mutex: std.Io.Mutex = .init,

        /// Call the function `f`.
        /// If `call` is invoked multiple times `f` will be executed only the
        /// first time.
        /// The invocations are thread-safe.
        pub fn call(self: *@This(), io: std.Io, args: Args) void {
            if (@atomicLoad(bool, &self.done, .acquire))
                return;

            self.callSlow(io, args);
        }

        fn callSlow(self: *@This(), io: std.Io, args: Args) void {
            @branchHint(.cold);

            self.mutex.lock(io);
            defer self.mutex.unlock(io);

            // The first thread to acquire the mutex gets to run the initializer
            if (!self.done) {
                @call(.auto, f, args);
                @atomicStore(bool, &self.done, true, .release);
            }
        }
    };
}
