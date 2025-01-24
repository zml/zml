const std = @import("std");
const coro_base = @import("coro_base.zig");

pub const stack_size = 1 * 1024 * 1024;

pub const Stack = struct {
    pub const Data = struct {
        data: [stack_size]u8 align(coro_base.stack_alignment) = undefined,

        pub fn ptr(self: *const Data) [*]u8 {
            return &self.data;
        }
    };

    full: *Data,
    used: []u8,

    pub fn init(full: *Data) Stack {
        return .{
            .full = full,
            .used = full.data[full.data.len..],
        };
    }

    pub fn remaining(self: Stack) []align(coro_base.stack_alignment) u8 {
        return self.full.data[0 .. self.full.data.len - self.used.len];
    }

    pub fn push(self: *Stack, comptime T: type) !*T {
        const ptr_i = std.mem.alignBackward(
            usize,
            @intFromPtr(self.used.ptr - @sizeOf(T)),
            coro_base.stack_alignment,
        );
        if (ptr_i <= @intFromPtr(&self.full.data[0])) {
            return error.StackTooSmall;
        }
        self.used = self.full.data[ptr_i - @intFromPtr(&self.full.data[0]) ..];
        return @ptrFromInt(ptr_i);
    }
};

pub const PooledStackAllocator = struct {
    const Pool = std.heap.MemoryPoolAligned(Stack.Data, coro_base.stack_alignment);

    pool: Pool,

    pub fn init(allocator: std.mem.Allocator) PooledStackAllocator {
        return .{ .pool = Pool.init(allocator) };
    }

    pub fn deinit(self: *PooledStackAllocator) void {
        self.pool.deinit();
    }

    pub fn create(self: *PooledStackAllocator) !Stack {
        return Stack.init(try self.pool.create());
    }

    pub fn destroy(self: *PooledStackAllocator, stack: *Stack) void {
        self.pool.destroy(stack.full);
    }
};

pub const StackAllocator = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) StackAllocator {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *StackAllocator) void {
        _ = self; // autofix
    }

    pub fn create(self: *StackAllocator) !Stack {
        return Stack.init(try self.allocator.create(Stack.Data));
    }

    pub fn destroy(self: *StackAllocator, stack: *Stack) void {
        self.allocator.destroy(stack.full);
    }
};
