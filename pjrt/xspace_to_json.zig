const std = @import("std");
const stdx = @import("stdx");
const flags = stdx.flags;

const TraceContainer = @import("convert/trace_container.zig").TraceContainer;

const CliArgs = struct {
    pub const help =
        \\ llama --path=path_to_profiling_data
    ;
    path: []const u8,
    max_events: ?usize = null,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);

    var fd = try std.fs.openFileAbsolute(cli_args.path, .{});
    defer fd.close();

    const pb_buffer = try fd.readToEndAlloc(allocator, (try fd.stat()).size);
    defer allocator.free(pb_buffer);
    if (pb_buffer.len == 0) return error.EmptyBuffer;

    var converter = TraceContainer.init(allocator);
    defer converter.deinit();
    try converter.parseXSpaceBytes(pb_buffer, cli_args.max_events);

    var path_buffer: [1028]u8 = undefined;

    const output_path = try std.fmt.bufPrint(&path_buffer, "{s}/{s}.json", .{
        std.fs.path.dirname(cli_args.path) orelse "",
        std.fs.path.stem(cli_args.path),
    });

    var output_file = try std.fs.createFileAbsolute(output_path, .{});
    defer output_file.close();

    try converter.toJson(output_file.writer().any());

    std.debug.print("Wrote JSON to {s}\n", .{output_path});
}
