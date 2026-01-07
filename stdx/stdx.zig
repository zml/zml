const std = @import("std");
const builtin = @import("builtin");

pub const BoundedArray = @import("bounded_array.zig").BoundedArray;
pub const BoundedArrayAligned = @import("bounded_array.zig").BoundedArrayAligned;
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

test {
    std.testing.refAllDecls(@This());
}

pub inline fn stackSlice(comptime max_len: usize, T: type, len: usize) []T {
    debug.assert(len <= max_len, "stackSlice can only create a slice of up to {} elements, got: {}", .{ max_len, len });
    var storage: [max_len]T = undefined;
    return storage[0..len];
}

pub const noalloc: std.mem.Allocator = if (builtin.mode == .ReleaseFast) undefined else std.testing.failing_allocator;

pub fn pinToCore(core_id: usize) void {
    if (builtin.os.tag == .linux) {
        const CPUSet = std.bit_set.ArrayBitSet(usize, std.os.linux.CPU_SETSIZE * @sizeOf(usize));

        var set: CPUSet = .initEmpty();
        set.set(core_id);
        std.os.linux.sched_setaffinity(0, @ptrCast(&set.masks)) catch {};
    }
}
