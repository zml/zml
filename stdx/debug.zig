const std = @import("std");

pub inline fn guard(check: bool, src: std.builtin.SourceLocation) void {
    assert(check, "Invalid inputs {s}@{s}:{d}", .{ src.file, src.fn_name, src.line });
}

pub inline fn internalAssert(check: bool, comptime msg: []const u8, args: anytype) void {
    assert(check, "internal error: " ++ msg, args);
}

pub inline fn assert(check: bool, comptime msg: []const u8, args: anytype) void {
    if (!check) {
        panic(msg, args);
    }
}

pub inline fn panic(comptime format: []const u8, args: anytype) noreturn {
    std.debug.panic(format, args);
}

pub inline fn compileLog(comptime msg: []const u8, comptime args: anytype) void {
    @compileLog(std.fmt.comptimePrint(msg, args));
}

pub inline fn compileError(comptime msg: []const u8, comptime args: anytype) noreturn {
    @compileError(std.fmt.comptimePrint(msg, args));
}

pub inline fn assertComptime(comptime check: bool, comptime msg: []const u8, comptime args: anytype) void {
    if (check == false) {
        compileError(msg, args);
    }
}
