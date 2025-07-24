const std = @import("std");
const runfiles = @import("runfiles");
const bazel_builtin = @import("bazel_builtin");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/runtimes/neuron");

pub export fn GetPjrtApi() *anyopaque {

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var r_ = runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) catch |err| { 
        stdx.debug.panic("Unable to find runfiles: {}", .{err});
    } orelse stdx.debug.panic("Runfiles not availeabwewefle", .{});

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sandbox_path = r.rlocation("libpjrt_neuron/sandbox", &path_buf) catch |err| {
        stdx.debug.panic("Failed to find sandbox path for NEURON runtime: {}", .{err});
    } orelse stdx.debug.panic("No NEURON sandbox path found", .{});

    var lib_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const library = stdx.fs.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libneuronpjrt.so" }) catch unreachable;

    var lib: std.DynLib = blk: {
        const handle = std.c.dlopen(library, .{ .LAZY = true, .GLOBAL = true, .NODELETE = true }) orelse {
            stdx.debug.panic("Unable to dlopen plugin: {s}", .{library});
        };
        break :blk .{ .inner = .{ .handle = handle } };
    };

    const sym = lib.lookup(*const fn () callconv(.C) *anyopaque, "GetPjrtApi") orelse {
         stdx.debug.panic("Unable to find symbol GetPjrtApi in plugin: {s}", .{library});
    };

    return sym();
}
