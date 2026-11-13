const std = @import("std");

/// The open files of every process whose path starts with one of the prefixes.
/// One walk of /proc, whatever the number of prefixes. Skips zml-smi itself,
/// which may hold device nodes through a library it opened.
pub const FdHolders = struct {
    io: std.Io,
    prefixes: []const []const u8,
    proc: ?std.Io.Dir,
    pids: std.Io.Dir.Iterator = undefined,
    fds: ?std.Io.Dir = null,
    fd_names: std.Io.Dir.Iterator = undefined,
    pid: u32 = 0,
    own_pid: u32,
    link: [std.Io.Dir.max_path_bytes]u8 = undefined,

    pub const Holder = struct {
        pid: u32,
        /// The descriptor, for /proc/<pid>/fdinfo/<fd>.
        fd: u32,
        /// The path after the prefix, valid until the next call: "0" for /dev/accel0.
        rest: []const u8,
    };

    pub fn init(io: std.Io, prefixes: []const []const u8) FdHolders {
        return initAt(io, std.Io.Dir.cwd().openDir(io, "/proc", .{ .iterate = true }) catch null, prefixes);
    }

    fn initAt(io: std.Io, proc: ?std.Io.Dir, prefixes: []const []const u8) FdHolders {
        var self: FdHolders = .{ .io = io, .prefixes = prefixes, .proc = proc, .own_pid = @intCast(std.c.getpid()) };
        if (proc) |dir| self.pids = dir.iterate();
        return self;
    }

    pub fn next(self: *FdHolders) ?Holder {
        const proc = self.proc orelse return null;
        while (true) {
            if (self.fds) |fds| {
                while (self.fd_names.next(self.io) catch null) |entry| {
                    const fd = std.fmt.parseInt(u32, entry.name, 10) catch continue;
                    const len = fds.readLink(self.io, entry.name, &self.link) catch continue;
                    const target = self.link[0..len];
                    for (self.prefixes) |prefix| {
                        if (std.mem.startsWith(u8, target, prefix)) return .{ .pid = self.pid, .fd = fd, .rest = target[prefix.len..] };
                    }
                }
                fds.close(self.io);
                self.fds = null;
            }

            const entry = (self.pids.next(self.io) catch null) orelse return null;
            const pid = std.fmt.parseInt(u32, entry.name, 10) catch continue;
            if (pid == self.own_pid) continue;
            var path: [16]u8 = undefined;
            const fd_path = std.fmt.bufPrint(&path, "{d}/fd", .{pid}) catch continue;
            const fds = proc.openDir(self.io, fd_path, .{ .iterate = true }) catch continue;
            self.fds = fds;
            self.fd_names = fds.iterate();
            self.pid = pid;
        }
    }

    pub fn deinit(self: *FdHolders) void {
        if (self.fds) |fds| fds.close(self.io);
        if (self.proc) |proc| proc.close(self.io);
    }
};

test FdHolders {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "100/fd");
    try tmp.dir.createDirPath(io, "200/fd");
    try tmp.dir.symLink(io, "/dev/accel3", "100/fd/5", .{});
    try tmp.dir.symLink(io, "/dev/null", "100/fd/0", .{});
    try tmp.dir.symLink(io, "/dev/vfio/1", "200/fd/9", .{});

    var holders: FdHolders = .initAt(io, try tmp.dir.openDir(io, ".", .{ .iterate = true }), &.{ "/dev/accel", "/dev/vfio/" });
    defer holders.deinit();

    var count: usize = 0;
    while (holders.next()) |holder| : (count += 1) {
        switch (holder.pid) {
            100 => {
                try std.testing.expectEqual(5, holder.fd);
                try std.testing.expectEqualStrings("3", holder.rest);
            },
            200 => try std.testing.expectEqualStrings("1", holder.rest),
            else => return error.TestUnexpectedResult,
        }
    }
    try std.testing.expectEqual(2, count);
}
