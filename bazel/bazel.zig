const std = @import("std");

const bazel_runfiles = @import("runfiles");
const stdx = @import("stdx");

var runfiles_once = stdx.onceWithArgs(struct {
    fn call(allocator: std.mem.Allocator, io: std.Io) !*bazel_runfiles.Runfiles {
        const r = try allocator.create(bazel_runfiles.Runfiles);
        r.* = try bazel_runfiles.Runfiles.create(.{ .allocator = allocator, .io = io }) orelse {
            stdx.debug.panic("Unable to find runfiles", .{});
        };
        return r;
    }
}.call);

pub fn runfiles(io: std.Io, source_repository: []const u8) !bazel_runfiles.Runfiles.WithSourceRepo {
    return (try runfiles_once.call(.{ std.heap.smp_allocator, io })).withSourceRepo(source_repository);
}
