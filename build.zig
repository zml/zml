const std = @import("std");

pub fn build(b: *std.Build) !void {
    @import("bazel/zls_bridge.zig").build(b) catch |err| {
        std.debug.print("Editor metadata failed ({t}). Start ZLS with tools/zls.sh; build and test ZML with Bazel.\n", .{err});
        return err;
    };
}
