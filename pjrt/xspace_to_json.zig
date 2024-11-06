const flags = @import("tigerbeetle/flags");
const std = @import("std");

const TraceConverter = @import("convert/trace_converter.zig").TraceConverter;

const CliArgs = struct {
    pub const help =
        \\ llama --path=path_to_profiling_data
    ;
    path: []const u8,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);

    var converter = try TraceConverter.init(allocator, cli_args.path);
    defer converter.deinit();

    const output = try converter.toJson(allocator);
    defer allocator.free(output);

    std.debug.print("{s}\n", .{output});
}
