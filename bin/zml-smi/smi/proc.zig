const std = @import("std");

/// The open files of every process whose path starts with one of the prefixes.
/// One walk of /proc, whatever the number of prefixes. Skips zml-smi itself,
/// which may hold device nodes through a library it opened.
pub const FdHolders = struct {
    io: std.Io,
    prefixes: []const []const u8,
    pids: ?std.Io.Dir.Iterator,
    fds: ?std.Io.Dir.Iterator = null,
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
        return .{
            .io = io,
            .prefixes = prefixes,
            .pids = if (proc) |dir| dir.iterate() else null,
            .own_pid = @intCast(std.c.getpid()),
        };
    }

    pub fn next(self: *FdHolders) ?Holder {
        const pids = if (self.pids) |*pids| pids else return null;
        while (true) {
            if (self.fds) |*fds| {
                const dir = fds.reader.dir;
                while (fds.next(self.io) catch null) |entry| {
                    const fd = std.fmt.parseInt(u32, entry.name, 10) catch continue;
                    const len = dir.readLink(self.io, entry.name, &self.link) catch continue;
                    for (self.prefixes) |prefix| {
                        const rest = std.mem.cutPrefix(u8, self.link[0..len], prefix) orelse continue;
                        return .{ .pid = self.pid, .fd = fd, .rest = rest };
                    }
                }
                dir.close(self.io);
                self.fds = null;
            }

            const entry = (pids.next(self.io) catch null) orelse return null;
            const pid = std.fmt.parseInt(u32, entry.name, 10) catch continue;
            if (pid == self.own_pid) continue;
            var path: [16]u8 = undefined;
            const fd_path = std.fmt.bufPrint(&path, "{d}/fd", .{pid}) catch continue;
            const fds = pids.reader.dir.openDir(self.io, fd_path, .{ .iterate = true }) catch continue;
            self.fds = fds.iterate();
            self.pid = pid;
        }
    }

    pub fn deinit(self: *FdHolders) void {
        if (self.fds) |*fds| fds.reader.dir.close(self.io);
        if (self.pids) |*pids| pids.reader.dir.close(self.io);
    }
};

pub const testing = if (@import("builtin").is_test) struct {
    pub const initAt = FdHolders.initAt;
} else struct {};
