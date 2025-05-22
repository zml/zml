const std = @import("std");
const system = std.posix.system;
const builtin = @import("builtin");
const native_os = builtin.os.tag;

pub const RecvMsgError = std.posix.RecvFromError || error{
    // MSG_OOB is set, but no out-of-band data is available.
    OutOfBandEmpty,
};

/// https://github.com/ziglang/zig/issues/20660
pub fn recvmsg(fd: i32, msg: *std.posix.msghdr, flags: u32) RecvMsgError!usize {
    while (true) {
        const rc = system.recvmsg(fd, msg, flags);
        if (native_os == .windows) {
            @compileError("TODO: implement recvmsg for window");
        }
        const errno = std.posix.errno(rc);
        switch (errno) {
            .SUCCESS => return @intCast(rc),
            .AGAIN => return error.WouldBlock,
            .BADF => unreachable, // always a race condition
            .CONNREFUSED => return error.ConnectionRefused,
            .CONNRESET => return error.ConnectionResetByPeer,
            .FAULT => unreachable, // segfault
            .INTR => continue, // retry
            .INVAL => return error.OutOfBandEmpty,
            .NOBUFS => return error.SystemResources,
            .NOTCONN => return error.SocketNotConnected,
            .NOTSOCK => unreachable,
            .TIMEDOUT => return error.ConnectionTimedOut,
            .MSGSIZE => return error.MessageTooBig,
            .NOMEM => return error.SystemResources,

            else => return std.posix.unexpectedErrno(errno),
        }
    }
}

pub fn sendMessage(socket: std.posix.socket_t, addr: std.net.Address, data: []const []const u8, flags: u32) !void {
    // Note this @ptrCast abuses the fact that the layout of slices is the same as the layout of iovec.
    // If this ever change it will segfault right away.
    const msg: std.posix.msghdr_const = .{
        .name = &addr.any,
        .namelen = addr.getOsSockLen(),
        .iov = @ptrCast(data.ptr),
        .iovlen = @intCast(data.len),
        .control = null,
        .controllen = 0,
        .flags = 0,
    };
    _ = try std.posix.sendmsg(socket, &msg, flags);
}

pub fn recvMessage(socket: std.posix.socket_t, data: []const []u8, flags: u32) !usize {
    // Note this @ptrCast abuses the fact that the layout of slices is the same as the layout of iovec.
    // If this ever change it will segfault right away.
    var msg: std.posix.msghdr = .{
        .name = null,
        .namelen = 0,
        .iov = @constCast(@ptrCast(data.ptr)),
        .iovlen = @intCast(data.len),
        .control = null,
        .controllen = 0,
        .flags = 0,
    };
    return try recvmsg(socket, &msg, flags);
}
