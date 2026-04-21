const std = @import("std");

const stdx = @import("stdx");
const xspace_to_perfetto = @import("tools/xspace_to_perfetto");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var upb_arena = std.heap.ArenaAllocator.init(allocator);
    defer upb_arena.deinit();

    const args = stdx.flags.parseProcessArgs(init.minimal, xspace_to_perfetto.Args);
    const input_path = args.positional.input;
    const output_path = args.positional.output orelse try std.fmt.allocPrint(allocator, "{s}.trace.json", .{input_path});
    defer if (args.positional.output == null) allocator.free(output_path);

    const stat = try std.Io.Dir.statFile(.cwd(), io, input_path, .{});

    var file = try std.Io.Dir.openFile(.cwd(), io, input_path, .{ .mode = .read_only });
    defer file.close(io);

    const xspace = try allocator.alloc(u8, stat.size);
    defer allocator.free(xspace);
    _ = try file.readPositionalAll(io, xspace, 0);

    try xspace_to_perfetto.dumpXSpaceProtoToTraceJsonFile(allocator, io, xspace, output_path);

    var stdout = std.Io.File.stdout().writer(io, &.{});
    try stdout.interface.print("{s}\n", .{output_path});
}
