const std = @import("std");
const c = @import("c");

pub const ZigAllocator = struct {
    pub inline fn from(allocator: std.mem.Allocator) c.zig_allocator {
        return .{
            .ctx = @ptrCast(@alignCast(&allocator)),
            .alloc = &alloc,
            .free = &free,
        };
    }

    pub fn alloc(ctx: ?*const anyopaque, elem: usize, nelems: usize, alignment: usize) callconv(.c) ?*anyopaque {
        const self: *const std.mem.Allocator = @ptrCast(@alignCast(ctx));
        const ret = self.rawAlloc(elem * nelems, std.math.log2_int(usize, alignment), @returnAddress()) orelse return null;
        return @ptrCast(ret);
    }

    pub fn free(ctx: ?*const anyopaque, ptr: ?*anyopaque, elem: usize, nelems: usize, alignment: usize) callconv(.c) void {
        const self: *const std.mem.Allocator = @ptrCast(@alignCast(ctx));
        const memory: [*c]u8 = @ptrCast(ptr);
        const size = elem * nelems;
        self.rawFree(memory[0..size], std.math.log2_int(usize, alignment), @returnAddress());
    }
};
