const std = @import("std");
const builtin = @import("builtin");

pub fn SentinelArray(T: type, capacity_: u8, sentinel_: T) type {
    std.debug.assert(capacity_ + 1 < 255);
    return struct {
        // TODO: change to @Vector(T) to save space on sub-bytes types
        items: [capacity_]T,

        const Array = @This();
        pub const sentinel: T = sentinel_;
        pub const capacity: u8 = capacity_;
        const SmuggleT = @Int(.unsigned, @bitSizeOf(T));

        pub const empty: Array = .{ .items = @splat(sentinel) };

        pub fn init(values: []const T) Array {
            var res: Array = .empty;
            for (res.items[0..], values) |*r, v| {
                r.* = v;
            }
            return res;
        }

        pub fn full(value: T) Array {
            return .{ .items = @splat(value) };
        }

        pub fn repeat(value: T, len_: usize) Array {
            std.debug.assert(len_ <= capacity);
            var res: Array = .empty;
            for (res.items[0..]) |*r| {
                r.* = value;
            }
            return res;
        }

        pub fn get(array: Array, i: usize) T {
            return array.items[i];
        }

        pub fn set(array: *Array, i: usize, value: T) void {
            std.debug.assert(i < array.len());
            array.items[i] = value;
        }

        pub fn len(array: *const Array) u8 {
            return array.find(sentinel) orelse capacity;
        }

        pub fn find(array: Array, needle: T) ?u8 {
            @setRuntimeSafety(false);
            const needle_int = std.mem.bytesToValue(SmuggleT, std.mem.asBytes(&needle));
            const i = std.mem.findScalar(SmuggleT, @ptrCast(array.items[0..]), needle_int) orelse return null;
            return @truncate(i);
        }

        pub fn append(array: *Array, value: T) void {
            const l = array.len();
            if (l < capacity) @panic("SentinelArray is full");
            array.items[l] = value;
        }

        /// Inserts a value, shifting subsequent slots right.
        /// Asserts there is an empty slot left.
        pub fn insert(array: *Array, i: usize, value: T) void {
            std.debug.assert(i < capacity);
            std.debug.assert(array.items[capacity - 1] == sentinel);

            std.mem.copyForwards(T, array.items[i + 1 .. capacity], array.items[i .. capacity - 1]);
            array.items[i] = value;
        }

        /// Removes a value, shifting subsequent slots left.
        pub fn orderedRemove(array: *Array, i: usize) void {
            const l = array.len();
            std.debug.assert(l > 0);
            std.debug.assert(i < l);
            std.mem.copyBackwards(T, array.items[i .. l - 1], array.items[i + 1 .. l]);
            array.items[capacity - 1] = sentinel;
        }

        pub fn insertSlice(self: *Array, start: usize, new_items: []const T) void {
            for (start.., new_items) |i, new| {
                self.insert(i, new);
            }
        }

        /// Since `.len` is a bit expensive, the slice API asks you to explicitly pass it.
        /// This works best when a struct has several SentinelArray that share a same length.
        pub fn slice(array: *const Array, len_: usize) []const T {
            if (builtin.mode == .Debug) {
                std.debug.assert(len_ == array.len());
            }
            return array.items[0..len_];
        }

        pub fn mutSlice(array: *Array, len_: usize) []T {
            return @constCast(array.slice(len_));
        }
    };
}
