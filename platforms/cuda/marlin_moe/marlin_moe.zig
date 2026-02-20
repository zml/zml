const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
pub const Params = c.marlin_moe_wna16_params_t;
const runfiles = @import("runfiles");

const LaunchFn = *const fn (*const Params) callconv(.c) c.marlin_moe_status_t;
const LastErrorFn = *const fn () callconv(.c) [*c]const u8;

pub var marlin_moe_wna16_launch: LaunchFn = undefined;
pub var marlin_moe_last_error: LastErrorFn = undefined;

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    _ = allocator;
    const r = try bazel.runfiles(io, bazel_builtin.current_repository);

    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const so_path = (try r.rlocation("fused_moe/marlin_moe_c/build/libmarlin_moe_c.so", &buffer)) orelse return error.NotFound;

    var lib = std.DynLib.open(so_path) catch |err| {
        std.log.err("Failed to open libmarlin_moe_c.so: {any}", .{err});
        return err;
    };

    marlin_moe_wna16_launch = lib.lookup(LaunchFn, "marlin_moe_wna16_launch") orelse return error.NotFound;
    marlin_moe_last_error = lib.lookup(LastErrorFn, "marlin_moe_last_error") orelse return error.NotFound;
}
