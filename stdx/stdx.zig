pub const debug = @import("debug.zig");
pub const flags = @import("flags.zig");
pub const io = @import("io.zig");
pub const json = @import("json.zig");
pub const math = @import("math.zig");
pub const meta = @import("meta.zig");
pub const queue = @import("queue.zig");
pub const time = @import("time.zig");

pub inline fn stackSlice(comptime max_len: usize, T: type, len: usize) []T {
    debug.assert(len <= max_len, "stackSlice can only create a slice of up to {} elements, got: {}", .{ max_len, len });
    var storage: [max_len]T = undefined;
    return storage[0..len];
}
