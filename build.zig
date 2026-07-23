const std = @import("std");

/// !!! This build.zig only exposes some utilities, not ZML the ML framework !!!
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

    // vfs
    const vfs = b.addModule("vfs", .{
        .root_source_file = b.path("vfs/vfs.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "stdx", .module = stdx }},
    });

    const vfs_test = b.addTest(.{ .root_module = vfs });
    const run_vfs_tests = b.addRunArtifact(vfs_test);
    test_step.dependOn(&run_vfs_tests.step);

    const vfs_example = b.addExecutable(.{
        .name = "vfs_example",
        .root_module = b.addModule("vfs_example", .{
            .root_source_file = b.path("vfs/example.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vfs", .module = vfs },
            },
        }),
    });

    const run_vfs_example = b.addRunArtifact(vfs_example);
    if (b.args) |args| {
        run_vfs_example.addArgs(args);
    }
    const step_vfs_example = b.step("run_vfs_example", "Run VFS example");
    step_vfs_example.dependOn(&run_vfs_example.step);
}
