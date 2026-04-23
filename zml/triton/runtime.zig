const std = @import("std");
const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");

pub const Runtime = struct {
    process: std.process.Child,
    process_mutex: *std.Io.Mutex,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Runtime {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();

        const runfiles = bazel.runfiles(bazel_builtin.current_repository) catch unreachable;
        const sandbox_path = runfiles.rlocationAlloc(arena.allocator(), "zml/zml/triton/generation/sandbox") catch unreachable;

        var map: std.process.Environ.Map = .init(arena.allocator());
        try map.put("LD_LIBRARY_PATH", std.Io.Dir.path.join(arena.allocator(), &.{ sandbox_path.?, "lib" }) catch unreachable);

        // Pointer on mutex is required to avoid copy and have a shared mutex
        const process_mutex = try allocator.create(std.Io.Mutex);
        errdefer allocator.destroy(process_mutex);
        process_mutex.* = .init;

        const process = try std.process.spawn(io, .{
            .argv = &.{std.Io.Dir.path.join(arena.allocator(), &.{ sandbox_path.?, "bin", "generate" }) catch unreachable},
            .stdin = .pipe,
            .stdout = .pipe,
            .environ_map = &map,
        });

        return .{
            .process = process,
            .process_mutex = process_mutex,
        };
    }

    pub fn deinit(self: *Runtime, allocator: std.mem.Allocator, io: std.Io) void {
        allocator.destroy(self.process_mutex);
        // NOTE(Corendos): This is not portable, but I couldn't find a better way with the current std.Io.
        _ = std.c.kill(self.process.id.?, .INT);
        _ = self.process.wait(io) catch unreachable;
    }
};
