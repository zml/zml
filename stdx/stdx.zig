pub const BoundedArray = @import("bounded_array.zig").BoundedArray;
pub const BoundedArrayAligned = @import("bounded_array.zig").BoundedArrayAligned;
pub const debug = @import("debug.zig");
pub const flags = @import("flags.zig");
pub const fmt = @import("fmt.zig");
pub const fs = @import("fs.zig");
pub const json = @import("json.zig");
pub const math = @import("math.zig");
pub const meta = @import("meta.zig");
pub const queue = @import("queue.zig");
pub const time = @import("time.zig");

const std = @import("std");
const builtin = @import("builtin");

test {
    std.testing.refAllDecls(@This());
}

pub inline fn stackSlice(comptime max_len: usize, T: type, len: usize) []T {
    debug.assert(len <= max_len, "stackSlice can only create a slice of up to {} elements, got: {}", .{ max_len, len });
    var storage: [max_len]T = undefined;
    return storage[0..len];
}

pub const noalloc: std.mem.Allocator = if (builtin.mode == .ReleaseFast) undefined else std.testing.failing_allocator;

pub const mem = struct {
    pub fn groupedAlloc(SliceTuple: type, allocator: std.mem.Allocator, len: [@typeInfo(SliceTuple).@"struct".fields.len]usize) error{OutOfMemory}!SliceTuple {
        var res: SliceTuple = undefined;
        var full_alloc_len: usize = 0;

        const SliceFields = @typeInfo(SliceTuple).@"struct".fields;
        inline for (SliceFields, 0.., &res) |field, i, *slice| {
            // needed because full_alloc_len starts at 0.
            @setRuntimeSafety(false);

            const T = std.meta.Child(field.type);
            full_alloc_len = std.mem.alignForward(usize, full_alloc_len, @alignOf(T));
            slice.ptr = @ptrFromInt(full_alloc_len);
            slice.len = len[i];
            full_alloc_len += @sizeOf(T) * len[i];
        }

        const bytes = try allocator.alignedAlloc(u8, @alignOf(SliceFields[0].type), full_alloc_len);
        inline for (&res) |*slice| {
            slice.ptr = @ptrFromInt(@intFromPtr(bytes.ptr) + @intFromPtr(slice.ptr));
        }

        return res;
    }
};
