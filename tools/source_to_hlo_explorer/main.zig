const std = @import("std");

const zml = @import("zml");
const model = @import("zml_explorer_model");
const stdx = zml.stdx;

const artifact = @import("artifact.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main(init: std.process.Init) !void {
    const CliArgs = struct {
        pub const help =
            \\zml-source-to-hlo-poc --output=PATH
            \\
            \\Compile the bundled two-operation model on CPU and write an explorer
            \\artifact bundle to PATH (default: zml-explorer-artifacts).
        ;

        output: []const u8 = "zml-explorer-artifacts",
    };

    const allocator = init.gpa;
    const io = init.io;
    const args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    try std.Io.Dir.cwd().createDirPath(io, args.output);
    const run_id = std.Io.Timestamp.now(io, .real).nanoseconds;
    const xla_dir_name = try std.fmt.allocPrint(allocator, "xla-{d}", .{run_id});
    defer allocator.free(xla_dir_name);
    const xla_dump_dir = try std.Io.Dir.path.join(allocator, &.{ args.output, xla_dir_name });
    defer allocator.free(xla_dump_dir);
    try std.Io.Dir.cwd().createDirPath(io, xla_dump_dir);

    try artifact.writeFile(io, args.output, "source.zig", @embedFile("source.zig"));

    const platform: *zml.Platform = try .init(allocator, io, .cpu, .{});
    defer platform.deinit(allocator, io);

    const shape = zml.Shape.init(.{ .feature = 4 }, .f32);
    const x: zml.Tensor = .fromShape(shape);
    const y: zml.Tensor = .fromShape(shape);

    var exe = try platform.compileFn(allocator, io, model.forward, .{ x, y }, .{
        .program_name = "zml_source_to_hlo_poc",
        .xla_dump_to = xla_dump_dir,
        .xla_dump_hlo_as_text = true,
        .explorer_dump_to = args.output,
    });
    defer exe.deinit();

    try artifact.finalizeWithSourceMap(allocator, io, args.output, xla_dump_dir, model.zml_source_map_json);

    std.log.info("Explorer artifacts written to {s}", .{args.output});
}
