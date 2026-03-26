const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);

    if (args.len != 4) {
        try std.Io.File.stderr().writeStreamingAll(io, "Usage: merge_amdgpu_ids <primary> <secondary> <output>\n");
        std.process.exit(1);
    }

    var entries: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer entries.deinit(arena);

    const cwd = std.Io.Dir.cwd();

    try parseFile(io, cwd, arena, args[2], &entries);
    try parseFile(io, cwd, arena, args[1], &entries);

    const out = try cwd.createFile(io, args[3], .{});
    defer out.close(io);

    var buf: [4096]u8 = undefined;
    var writer = out.writer(io, &buf);

    try writer.interface.print("# List of AMDGPU IDs\n" ++
        "#\n" ++
        "# Syntax:\n" ++
        "# device_id,\trevision_id,\tproduct_name        <-- single tab after comma\n" ++
        "\n" ++
        "1.0.0\n", .{});

    var it = entries.valueIterator();
    while (it.next()) |v| {
        try writer.interface.print("{s}\n", .{v.*});
    }

    try writer.flush();
}

fn parseFile(
    io: std.Io,
    cwd: std.Io.Dir,
    arena: std.mem.Allocator,
    path: []const u8,
    entries: *std.StringHashMapUnmanaged([]const u8),
) !void {
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    var buf: [4096]u8 = undefined;
    var reader = file.reader(io, &buf);

    while (true) {
        const raw = try reader.interface.takeDelimiter('\n') orelse break;

        const line = std.mem.trimEnd(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;

        var iter = std.mem.splitSequence(u8, line, ",");

        const device = std.mem.trim(u8, iter.next() orelse continue, " \t");
        const revision = std.mem.trim(u8, iter.next() orelse continue, " \t");
        const product = std.mem.trim(u8, iter.rest(), " \t");

        const key = try std.fmt.allocPrint(arena, "{s},{s}", .{ device, revision });
        const value = try std.fmt.allocPrint(arena, "{s},\t{s},\t{s}", .{ device, revision, product });

        try entries.put(arena, key, value);
    }
}
