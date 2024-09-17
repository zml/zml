const std = @import("std");
const c = @import("c");

pub fn madvise(ptr: [*]align(std.mem.page_size) u8, length: usize, advice: u32) std.posix.MadviseError!void {
    switch (std.posix.errno(c.madvise(ptr, @intCast(length), @intCast(advice)))) {
        .SUCCESS => return,
        .ACCES => return error.AccessDenied,
        .AGAIN => return error.SystemResources,
        .BADF => unreachable, // The map exists, but the area maps something that isn't a file.
        .INVAL => return error.InvalidSyscall,
        .IO => return error.WouldExceedMaximumResidentSetSize,
        .NOMEM => return error.OutOfMemory,
        .NOSYS => return error.MadviseUnavailable,
        else => |err| return std.posix.unexpectedErrno(err),
    }
}
