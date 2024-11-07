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

    var fd = try std.fs.openFileAbsolute(cli_args.path, .{});
    defer fd.close();

    const pb_buffer = try fd.readToEndAlloc(allocator, (try fd.stat()).size);
    defer allocator.free(pb_buffer);
    if (pb_buffer.len == 0) return error.EmptyBuffer;

    var converter = try TraceConverter.init(allocator, pb_buffer);
    defer converter.deinit();

    const output = try converter.toJson(allocator);
    defer allocator.free(output);

    var path_buffer: [1028]u8 = undefined;
    var output_path = std.ArrayListUnmanaged(u8).initBuffer(&path_buffer);
    output_path.appendSliceAssumeCapacity(cli_args.path[0..std.mem.lastIndexOf(u8, cli_args.path, std.fs.path.extension(cli_args.path)).?]);
    output_path.appendSliceAssumeCapacity(".json");

    var output_file = try std.fs.createFileAbsolute(output_path.items, .{});
    defer output_file.close();

    try output_file.writeAll(output);
    std.debug.print("Wrote JSON to {s}\n", .{output_path.items});
}
