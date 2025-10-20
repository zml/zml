const std = @import("std");
const builtin = @import("builtin");

pub fn main() !void {
    const gpa = std.heap.page_allocator;

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    const file_path = args[1];
    var file = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
    defer file.close();

    if (builtin.zig_version.major == 0 and builtin.zig_version.minor >= 15) {
        var write_buffer: [512]u8 = undefined;
        var writer = std.fs.File.stdout().writer(&write_buffer);
        var stdout = &writer.interface;

        var reader = file.reader(&.{});
        _ = try reader.interface.streamRemaining(stdout);
        try stdout.flush();
    } else {
        var stdout = std.io.getStdOut().writer();
        var buffer: [4096]u8 = undefined;
        while (true) {
            const n = try file.read(&buffer);
            if (n == 0) break; // EOF
            try stdout.writeAll(buffer[0..n]);
        }
    }
}
