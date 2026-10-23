//! Workload sweeps through the default public MoE API.
const std = @import("std");
const zml = @import("zml");
const bench = @import("moe_bench");

pub fn main(init: std.process.Init) !void {
    const args = zml.stdx.flags.parse(init.minimal.args, bench.cases.Args);
    args.validate() catch |err| zml.stdx.flags.fatal("invalid benchmark arguments: {s}; use --help", .{@errorName(err)});
    const platform = if (args.platform) |target| try zml.Platform.init(init.gpa, init.io, target, .{ .cpu = .{ .device_count = args.cpu_devices } }) else try zml.Platform.auto(init.gpa, init.io, .{ .cpu = .{ .device_count = args.cpu_devices } });
    defer platform.deinit(init.gpa, init.io);
    const experts_mesh = try platform.registerMesh("experts", .mesh(.{ .experts = .high_bandwidth }));
    try bench.suite.run(init, platform, experts_mesh, args, null, bench.suite.DefaultCompiler{});
}
