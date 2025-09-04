const std = @import("std");
const builtin = @import("builtin");

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
    pub fn groupedAlloc(
        SliceTuple: type,
        allocator: std.mem.Allocator,
        len: [@typeInfo(SliceTuple).@"struct".fields.len]usize,
    ) error{OutOfMemory}!SliceTuple {
        var res: SliceTuple = undefined;
        var full_alloc_len: usize = 0;

        const SliceFields = @typeInfo(SliceTuple).@"struct".fields;
        inline for (SliceFields, len) |field, l| {
            // needed because full_alloc_len starts at 0.
            @setRuntimeSafety(false);

            const T = std.meta.Child(field.type);
            full_alloc_len = std.mem.alignForward(usize, full_alloc_len, @alignOf(T));
            @field(res, field.name).ptr = @ptrFromInt(full_alloc_len);
            @field(res, field.name).len = l;
            full_alloc_len += @sizeOf(T) * l;
        }

        const bytes = try allocator.alignedAlloc(u8, .of(SliceFields[0].type), full_alloc_len);
        // std.log.warn("groupedAlloc({s}) -> {*}[0..{d}] align({})", .{ @typeName(SliceTuple), bytes.ptr, bytes.len, alignment });

        inline for (SliceFields) |field| {
            @field(res, field.name).ptr = @ptrFromInt(@intFromPtr(bytes.ptr) + @intFromPtr(@field(res, field.name).ptr));
        }

        return res;
    }

    pub fn groupedFree(SliceTuple: type, allocator: std.mem.Allocator, slice_tuple: SliceTuple) void {
        const SliceFields = @typeInfo(SliceTuple).@"struct".fields;

        const slice_start: [*]u8 = @ptrCast(slice_tuple[0].ptr);
        const N = SliceFields.len;
        const FirstT = std.meta.Child(SliceFields[0].type);
        const LastT = std.meta.Child(SliceFields[N - 1].type);

        const last_slice = slice_tuple[N - 1];
        const slice_end: usize = @intFromPtr(last_slice.ptr) + @sizeOf(LastT) * last_slice.len;
        const slice_len = slice_end - @intFromPtr(slice_start);

        // std.log.warn("groupedFree({s}) -> {*}[0..{d}] align({})", .{ @typeName(SliceTuple), slice_start, slice_len, @alignOf(FirstT) });
        allocator.rawFree(slice_start[0..slice_len], .fromByteUnits(@alignOf(FirstT)), @returnAddress());
    }
};
