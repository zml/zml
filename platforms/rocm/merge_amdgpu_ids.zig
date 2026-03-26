const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);

    if (args.len != 4) {
        try std.Io.File.stderr().writeStreamingAll(io, "Usage: merge_amdgpu_ids <primary> <secondary> <output>\n");
        std.process.exit(1);
    }

    var entries = std.StringHashMap([]const u8).init(gpa);
    defer entries.deinit();

    const cwd = std.Io.Dir.cwd();

    try parseFile(io, cwd, arena, args[2], &entries);
    try parseFile(io, cwd, arena, args[1], &entries);

    const out = try cwd.createFile(io, args[3], .{});
    defer out.close(io);

    var buf: [4096]u8 = undefined;
    var writer = out.writer(io, &buf);

    try writer.interface.print("# List of AMDGPU IDs\n#\n# Syntax:\n# device_id,\trevision_id,\tproduct_name\n\n1.0.0\n", .{});

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
    entries: *std.StringHashMap([]const u8),
) !void {
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    var buf: [4096]u8 = undefined;
    var reader = file.reader(io, &buf);

    while (true) {
        const raw = try reader.interface.takeDelimiter('\n') orelse break;

        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;

        const first_comma = std.mem.indexOfScalar(u8, line, ',') orelse continue;
        const rest = line[first_comma + 1 ..];
        const second_comma = std.mem.indexOfScalar(u8, rest, ',') orelse continue;

        const device_id = std.mem.trim(u8, line[0..first_comma], " \t");
        const revision_id = std.mem.trim(u8, rest[0..second_comma], " \t");
        const product_name = std.mem.trim(u8, rest[second_comma + 1 ..], " \t");

        const key = try std.fmt.allocPrint(arena, "{s},{s}", .{ device_id, revision_id });
        const value = try std.fmt.allocPrint(arena, "{s},\t{s},\t{s}", .{ device_id, revision_id, product_name });
        try entries.put(key, value);
    }
}
