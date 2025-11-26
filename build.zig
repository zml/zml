const std = @import("std");

/// Partial copy of:
/// https://github.com/zml/zml/pull/262
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

    const run_stdx_tests = b.addRunArtifact(b.addTest(.{ .root_module = stdx }));
    test_step.dependOn(&run_stdx_tests.step);
}
