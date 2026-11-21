//! Keeps the message of the last PJRT error, which ZML only logs (scope
//! `pjrt`) before mapping it to an error code, so that a program can report a
//! plugin's refusal verbatim. Install `logFn` as `std_options.logFn`.

const std = @import("std");

var buf: [16 * 1024]u8 = undefined;
var len: usize = 0;
var mutex: std.atomic.Mutex = .unlocked;

pub fn reset() void {
    lock();
    defer mutex.unlock();
    len = 0;
}

/// The last message since `reset`, empty if none.
pub fn get() []const u8 {
    return buf[0..len];
}

fn set(comptime format: []const u8, args: anytype) void {
    lock();
    defer mutex.unlock();
    // A message longer than the buffer is kept truncated.
    len = if (std.fmt.bufPrint(&buf, format, args)) |message| message.len else |_| buf.len;
}

fn lock() void {
    while (!mutex.tryLock()) std.atomic.spinLoopHint();
}

pub fn logFn(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    if (scope == .pjrt and level == .warn) set(format, args);
    // A refused compile also prints the whole module, which the caller has or can dump.
    if (scope == .@"zml/Compiler" and comptime std.mem.eql(u8, format, "{s} ({}):\n{s}")) return;
    std.log.defaultLog(level, scope, format, args);
}
