const std = @import("std");

pub const enable = true;
pub const enable_allocation = true;
pub const enable_callstack = true;
pub const callstack_depth = 10;

const ___tracy_c_zone_context = extern struct {
    id: u32,
    active: i32,

    pub inline fn end(self: @This()) void {
        ___tracy_emit_zone_end(self);
    }

    pub inline fn addText(self: @This(), text: []const u8) void {
        ___tracy_emit_zone_text(self, text.ptr, text.len);
    }

    pub inline fn setName(self: @This(), name: []const u8) void {
        ___tracy_emit_zone_name(self, name.ptr, name.len);
    }

    pub inline fn setColor(self: @This(), color: u32) void {
        ___tracy_emit_zone_color(self, color);
    }

    pub inline fn setValue(self: @This(), value: u64) void {
        ___tracy_emit_zone_value(self, value);
    }
};

pub const Ctx = if (enable) ___tracy_c_zone_context else struct {
    pub inline fn end(self: @This()) void {
        _ = self;
    }

    pub inline fn addText(self: @This(), text: []const u8) void {
        _ = self;
        _ = text;
    }

    pub inline fn setName(self: @This(), name: []const u8) void {
        _ = self;
        _ = name;
    }

    pub inline fn setColor(self: @This(), color: u32) void {
        _ = self;
        _ = color;
    }

    pub inline fn setValue(self: @This(), value: u64) void {
        _ = self;
        _ = value;
    }
};

const empty_ctx: Ctx = if (enable) .{ .id = 0, .active = 0 } else .{};

pub const Mode = enum {
    on,
    off,

    pub inline fn enabled(self: @This()) bool {
        return self == .on;
    }
};

pub const Scope = struct {
    enabled: bool = false,
    ctx: Ctx = empty_ctx,

    pub inline fn end(self: @This()) void {
        if (!self.enabled) return;
        self.ctx.end();
    }

    pub inline fn addText(self: @This(), text: []const u8) void {
        if (!self.enabled or text.len == 0) return;
        self.ctx.addText(text);
    }

    pub inline fn textFmt(self: @This(), comptime fmt: []const u8, args: anytype) void {
        if (!self.enabled) return;

        var buf: [160]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
        self.addText(text);
    }

    pub inline fn setName(self: @This(), name: []const u8) void {
        if (!self.enabled or name.len == 0) return;
        self.ctx.setName(name);
    }

    pub inline fn setColor(self: @This(), color: u32) void {
        if (!self.enabled) return;
        self.ctx.setColor(color);
    }

    pub inline fn setValue(self: @This(), value: u64) void {
        if (!self.enabled) return;
        self.ctx.setValue(value);
    }
};

pub inline fn trace(comptime src: std.builtin.SourceLocation) Ctx {
    if (!enable) return .{};

    const global = struct {
        const loc: ___tracy_source_location_data = .{
            .name = null,
            .function = src.fn_name.ptr,
            .file = src.file.ptr,
            .line = src.line,
            .color = 0,
        };
    };

    if (enable_callstack) {
        return ___tracy_emit_zone_begin_callstack(&global.loc, callstack_depth, 1);
    } else {
        return ___tracy_emit_zone_begin(&global.loc, 1);
    }
}

pub inline fn traceNamed(comptime src: std.builtin.SourceLocation, comptime name: [:0]const u8) Ctx {
    if (!enable) return .{};

    const global = struct {
        const loc: ___tracy_source_location_data = .{
            .name = name.ptr,
            .function = src.fn_name.ptr,
            .file = src.file.ptr,
            .line = src.line,
            .color = 0,
        };
    };

    if (enable_callstack) {
        return ___tracy_emit_zone_begin_callstack(&global.loc, callstack_depth, 1);
    } else {
        return ___tracy_emit_zone_begin(&global.loc, 1);
    }
}

pub inline fn scope(comptime src: std.builtin.SourceLocation, enabled_: bool) Scope {
    if (!enabled_) return .{};
    return .{
        .enabled = true,
        .ctx = trace(src),
    };
}

pub inline fn scopeNamed(comptime src: std.builtin.SourceLocation, comptime name: [:0]const u8, enabled_: bool) Scope {
    if (!enabled_) return .{};
    return .{
        .enabled = true,
        .ctx = traceNamed(src, name),
    };
}

pub inline fn scopeNamedOpt(comptime src: std.builtin.SourceLocation, comptime name: ?[:0]const u8, enabled_: bool) Scope {
    return if (name) |n| scopeNamed(src, n, enabled_) else .{};
}

pub inline fn setThreadName(name: [*:0]const u8) void {
    if (!enable) return;
    ___tracy_set_thread_name(name);
}

pub inline fn setThreadNameIf(name: [*:0]const u8, enabled_: bool) void {
    if (!enabled_) return;
    setThreadName(name);
}

pub inline fn isConnected() bool {
    if (!enable) return false;
    return ___tracy_connected() != 0;
}

pub inline fn fiberEnter(fiber: [*:0]const u8) void {
    if (!enable) return;
    ___tracy_fiber_enter(fiber);
}

pub inline fn fiberLeave() void {
    if (!enable) return;
    ___tracy_fiber_leave();
}

pub fn tracyAllocator(allocator: std.mem.Allocator) TracyAllocator(null) {
    return TracyAllocator(null).init(allocator);
}

pub fn TracyAllocator(comptime name: ?[:0]const u8) type {
    return struct {
        parent_allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(parent_allocator: std.mem.Allocator) Self {
            return .{
                .parent_allocator = parent_allocator,
            };
        }

        pub fn allocator(self: *Self) std.mem.Allocator {
            return .{
                .ptr = self,
                .vtable = &.{
                    .alloc = allocFn,
                    .resize = resizeFn,
                    .remap = remapFn,
                    .free = freeFn,
                },
            };
        }

        fn allocFn(ptr: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ptr));
            const result = self.parent_allocator.rawAlloc(len, alignment, ret_addr);
            if (result) |memory| {
                if (len != 0) {
                    if (name) |n| {
                        allocNamed(memory, len, n);
                    } else {
                        alloc(memory, len);
                    }
                }
            } else {
                messageColor("allocation failed", 0xFF0000);
            }
            return result;
        }

        fn resizeFn(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
            const self: *Self = @ptrCast(@alignCast(ptr));
            if (self.parent_allocator.rawResize(memory, alignment, new_len, ret_addr)) {
                if (name) |n| {
                    freeNamed(memory.ptr, n);
                    allocNamed(memory.ptr, new_len, n);
                } else {
                    free(memory.ptr);
                    alloc(memory.ptr, new_len);
                }

                return true;
            }

            // The compiler hits this path frequently during normal operation.
            return false;
        }

        fn remapFn(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ptr));
            if (self.parent_allocator.rawRemap(memory, alignment, new_len, ret_addr)) |new_memory| {
                if (name) |n| {
                    freeNamed(memory.ptr, n);
                    allocNamed(new_memory, new_len, n);
                } else {
                    free(memory.ptr);
                    alloc(new_memory, new_len);
                }
                return new_memory;
            } else {
                messageColor("reallocation failed", 0xFF0000);
                return null;
            }
        }

        fn freeFn(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            self.parent_allocator.rawFree(memory, alignment, ret_addr);
            if (memory.len != 0) {
                if (name) |n| {
                    freeNamed(memory.ptr, n);
                } else {
                    free(memory.ptr);
                }
            }
        }
    };
}

pub inline fn message(comptime msg: [:0]const u8) void {
    messageColor(msg, 0);
}

pub inline fn messageColor(comptime msg: [:0]const u8, color: u24) void {
    if (!enable) return;
    ___tracy_emit_logStringL(.Info, color, if (enable_callstack) callstack_depth else 0, msg.ptr);
}

pub inline fn messageCopy(msg: []const u8) void {
    messageColorCopy(msg, 0);
}

pub inline fn messageColorCopy(msg: []const u8, color: u24) void {
    if (!enable) return;
    ___tracy_emit_logString(.Info, color, if (enable_callstack) callstack_depth else 0, msg.len, msg.ptr);
}

pub inline fn frameMark() void {
    if (!enable) return;
    ___tracy_emit_frame_mark(null);
}

pub inline fn frameMarkIf(enabled_: bool) void {
    if (!enabled_) return;
    frameMark();
}

pub inline fn frameMarkNamed(comptime name: [:0]const u8) void {
    if (!enable) return;
    ___tracy_emit_frame_mark(name.ptr);
}

pub inline fn frameMarkNamedIf(comptime name: [:0]const u8, enabled_: bool) void {
    if (!enabled_) return;
    frameMarkNamed(name);
}

pub inline fn namedFrame(comptime name: [:0]const u8) Frame(name) {
    frameMarkStart(name);
    return .{};
}

pub fn Frame(comptime name: [:0]const u8) type {
    return struct {
        pub fn end(_: @This()) void {
            frameMarkEnd(name);
        }
    };
}

inline fn frameMarkStart(comptime name: [:0]const u8) void {
    if (!enable) return;
    ___tracy_emit_frame_mark_start(name.ptr);
}

inline fn frameMarkEnd(comptime name: [:0]const u8) void {
    if (!enable) return;
    ___tracy_emit_frame_mark_end(name.ptr);
}

extern fn ___tracy_emit_frame_mark_start(name: [*:0]const u8) void;
extern fn ___tracy_emit_frame_mark_end(name: [*:0]const u8) void;

inline fn alloc(ptr: [*]u8, len: usize) void {
    if (!enable) return;

    if (enable_callstack) {
        ___tracy_emit_memory_alloc_callstack(ptr, len, callstack_depth, 0);
    } else {
        ___tracy_emit_memory_alloc(ptr, len, 0);
    }
}

inline fn allocNamed(ptr: [*]u8, len: usize, comptime name: [:0]const u8) void {
    if (!enable) return;

    if (enable_callstack) {
        ___tracy_emit_memory_alloc_callstack_named(ptr, len, callstack_depth, 0, name.ptr);
    } else {
        ___tracy_emit_memory_alloc_named(ptr, len, 0, name.ptr);
    }
}

inline fn free(ptr: [*]u8) void {
    if (!enable) return;

    if (enable_callstack) {
        ___tracy_emit_memory_free_callstack(ptr, callstack_depth, 0);
    } else {
        ___tracy_emit_memory_free(ptr, 0);
    }
}

inline fn freeNamed(ptr: [*]u8, comptime name: [:0]const u8) void {
    if (!enable) return;

    if (enable_callstack) {
        ___tracy_emit_memory_free_callstack_named(ptr, callstack_depth, 0, name.ptr);
    } else {
        ___tracy_emit_memory_free_named(ptr, 0, name.ptr);
    }
}

pub const MessageSeverity = enum(i8) {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Fatal,
};

extern fn ___tracy_emit_zone_begin(srcloc: *const ___tracy_source_location_data, active: i32) ___tracy_c_zone_context;
extern fn ___tracy_emit_zone_begin_callstack(srcloc: *const ___tracy_source_location_data, depth: i32, active: i32) ___tracy_c_zone_context;
extern fn ___tracy_emit_zone_text(ctx: ___tracy_c_zone_context, txt: [*]const u8, size: usize) void;
extern fn ___tracy_emit_zone_name(ctx: ___tracy_c_zone_context, txt: [*]const u8, size: usize) void;
extern fn ___tracy_emit_zone_color(ctx: ___tracy_c_zone_context, color: u32) void;
extern fn ___tracy_emit_zone_value(ctx: ___tracy_c_zone_context, value: u64) void;
extern fn ___tracy_emit_zone_end(ctx: ___tracy_c_zone_context) void;
extern fn ___tracy_emit_memory_alloc(ptr: *const anyopaque, size: usize, secure: i32) void;
extern fn ___tracy_emit_memory_alloc_callstack(ptr: *const anyopaque, size: usize, depth: i32, secure: i32) void;
extern fn ___tracy_emit_memory_free(ptr: *const anyopaque, secure: i32) void;
extern fn ___tracy_emit_memory_free_callstack(ptr: *const anyopaque, depth: i32, secure: i32) void;
extern fn ___tracy_emit_memory_alloc_named(ptr: *const anyopaque, size: usize, secure: i32, name: [*:0]const u8) void;
extern fn ___tracy_emit_memory_alloc_callstack_named(ptr: *const anyopaque, size: usize, depth: i32, secure: i32, name: [*:0]const u8) void;
extern fn ___tracy_emit_memory_free_named(ptr: *const anyopaque, secure: i32, name: [*:0]const u8) void;
extern fn ___tracy_emit_memory_free_callstack_named(ptr: *const anyopaque, depth: i32, secure: i32, name: [*:0]const u8) void;
extern fn ___tracy_emit_logString(severity: MessageSeverity, color: i32, callstack_depth: i32, size: usize, txt: [*]const u8) void;
extern fn ___tracy_emit_logStringL(severity: MessageSeverity, color: i32, callstack_depth: i32, txt: [*:0]const u8) void;
extern fn ___tracy_emit_frame_mark(name: ?[*:0]const u8) void;
extern fn ___tracy_set_thread_name(name: [*:0]const u8) void;
extern fn ___tracy_connected() i32;
extern fn ___tracy_fiber_enter(fiber: [*:0]const u8) void;
extern fn ___tracy_fiber_leave() void;
pub extern fn ___tracy_begin_sampling_profiling() c_int;
pub extern fn ___tracy_end_sampling_profiling() void;

const ___tracy_source_location_data = extern struct {
    name: ?[*:0]const u8,
    function: [*:0]const u8,
    file: [*:0]const u8,
    line: u32,
    color: u32,
};
