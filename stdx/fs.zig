const std = @import("std");

extern fn dladdr(addr: *anyopaque, info: *Dl_info) c_int;

const Dl_info = extern struct {
    dli_fname: [*c]const u8,
    dli_fbase: *anyopaque,
    dli_sname: [*c]const u8,
    dli_saddr: *anyopaque,
};

fn selfSharedObjectPathImpl(addr: usize) []const u8 {
    var info: Dl_info = undefined;
    _ = dladdr(@ptrFromInt(addr), &info);
    return std.mem.span(info.dli_fname);
}

pub fn selfSharedObjectPath() []const u8 {
    return selfSharedObjectPathImpl(@returnAddress());
}

pub fn selfSharedObjectDirPath() []const u8 {
    return std.fs.path.dirname(selfSharedObjectPathImpl(@returnAddress())).?;
}

pub const path = struct {
    pub fn bufJoin(buf: []u8, paths: []const []const u8) ![]u8 {
        return try std.fmt.bufPrint(buf, "{f}", .{std.fs.path.fmtJoin(paths)});
    }

    pub fn bufJoinZ(buf: []u8, paths: []const []const u8) ![:0]u8 {
        return try std.fmt.bufPrintZ(buf, "{f}", .{std.fs.path.fmtJoin(paths)});
    }
};
