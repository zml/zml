const std = @import("std");
const builtin = @import("builtin");

pub const BoundedArray = @import("bounded_array.zig").BoundedArray;
pub const BoundedArrayAligned = @import("bounded_array.zig").BoundedArrayAligned;
pub const cpu = @import("cpu.zig");
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

pub const noalloc: std.mem.Allocator = if (builtin.mode == .fast) undefined else std.testing.failing_allocator;

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

        var set: CPUSet = .empty;
        set.set(core_id);
        std.os.linux.sched_setaffinity(0, @ptrCast(&set.masks)) catch {};
    }
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

/// Packed storage of sub-byte values
///
/// Note: before Zig 0.17, @Vector(u2, 4) was always one byte and could be used as a storage type,
/// But that's not true anymore.
pub fn Packed(T: type) type {
    return switch (@bitSizeOf(T)) {
        2 => packed struct(u8) {
            x: T,
            y: T,
            z: T,
            w: T,

            const P = @This();
            pub fn pack(x: T, y: T, z: T, w: T) P {
                return .{ .x = x, .y = y, .z = z, .w = w };
            }

            pub fn unpack(p: P) [4]T {
                return .{ p.x, p.y, p.z, p.w };
            }

            pub fn formatNumber(p: P, w: *std.Io.Writer, n: std.fmt.Number) std.Io.Writer.Error!void {
                try formatPackedNumber(T, &p.unpack(), w, n);
            }
        },
        4 => packed struct(u8) {
            x: T,
            y: T,

            const P = @This();
            pub fn pack(x: T, y: T) P {
                return .{ .x = x, .y = y };
            }

            pub fn unpack(p: P) [2]T {
                return .{ p.x, p.y };
            }

            pub fn formatNumber(p: P, w: *std.Io.Writer, n: std.fmt.Number) std.Io.Writer.Error!void {
                try formatPackedNumber(T, &p.unpack(), w, n);
            }
        },
        else => debug.compileError("stdx.Packed expects a bitsize of 2 or 4, got type {} with bitsize {d}", .{ T, @bitSizeOf(T) }),
    };
}

fn formatPackedNumber(T: type, values: []const T, w: *std.Io.Writer, n: std.fmt.Number) std.Io.Writer.Error!void {
    for (0.., values) |i, val| {
        if (i > 0) try w.writeByte(',');
        switch (@typeInfo(T)) {
            .@"struct" => try val.formatNumber(w, n),
            .int => try fmt.formatInt(val, n, w),
            else => try w.print("{}", .{val}),
        }
    }
}
