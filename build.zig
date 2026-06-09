const std = @import("std");

/// !!! This build.zig only exposes stdx, not ZML !!!
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_step = b.step("test", "Run all tests across ZML and deps");

    // stdx
    const stdx = b.addModule("stdx", .{
        .root_source_file = b.path("stdx/stdx.zig"),
        .target = target,
        .optimize = optimize,
    });

    const stdx_test = b.addTest(.{ .root_module = stdx });
    const run_stdx_tests = b.addRunArtifact(stdx_test);
    test_step.dependOn(&run_stdx_tests.step);
}
