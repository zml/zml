const std = @import("std");
const builtin = @import("builtin");

const PATTERN = "@@__BUILD_WORKSPACE_DIRECTORY__@@";

pub const StringReplaceReader = struct {
    interface: std.Io.Reader = .{
        .buffer = &.{},
        .vtable = &.{
            .stream = stream,
        },
        .seek = 0,
        .end = 0,
    },
    in: *std.Io.Reader,
    pattern: []const u8,
    replace_by: []const u8,

    pub fn stream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *StringReplaceReader = @fieldParentPtr("interface", r);
        if (std.mem.eql(u8, try self.in.peek(self.pattern.len), self.pattern)) {
            self.in.toss(self.pattern.len);
            try w.writeAll(self.replace_by);
            return self.replace_by.len;
        }
        return self.in.streamDelimiterLimit(w, self.pattern[0], limit) catch |err| switch (err) {
            std.Io.Reader.StreamDelimiterLimitError.StreamTooLong => return 0,
            else => |e| return e,
        };
    }
};

pub fn main_pre_015() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const args = try std.process.argsAlloc(arena.allocator());
    defer std.process.argsFree(arena.allocator(), args);

    const file_path = args[1];
    const content = try std.fs.cwd().readFileAlloc(arena.allocator(), file_path, std.math.maxInt(usize));
    const build_workspace_directory = try std.process.getEnvVarOwned(arena.allocator(), "BUILD_WORKSPACE_DIRECTORY");

    const need = std.mem.replacementSize(u8, content, PATTERN, build_workspace_directory);
    const replaced = try arena.allocator().alloc(u8, need);
    _ = std.mem.replace(u8, content, PATTERN, build_workspace_directory, replaced);
    try std.io.getStdOut().writeAll(replaced);
}

pub fn main_015() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const args = try std.process.argsAlloc(arena.allocator());
    defer std.process.argsFree(arena.allocator(), args);

    const file_path = args[1];
    var file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var read_buffer: [8192]u8 = undefined;
    var reader = file.reader(&read_buffer);

    const build_workspace_directory = try std.process.getEnvVarOwned(arena.allocator(), "BUILD_WORKSPACE_DIRECTORY");
    var replacer: StringReplaceReader = .{
        .in = &reader.interface,
        .pattern = "@@__BUILD_WORKSPACE_DIRECTORY__@@",
        .replace_by = build_workspace_directory,
    };

    var write_buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&write_buffer);

    _ = try replacer.interface.streamRemaining(&stdout_writer.interface);
    try stdout_writer.interface.flush();
}

pub fn main_016() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();

    var threaded: std.Io.Threaded = .init(arena.allocator(), .{});
    defer threaded.deinit();
    const io = threaded.io();

    const path = try std.Io.Dir.realPathFileAlloc(.cwd(), io, ".", arena.allocator());

    const execrootidx = std.mem.find(u8, path, "execroot/_main").?;
    const workspace_dir = path[0 .. execrootidx + "execroot/_main".len];

    const args = try std.process.argsAlloc(arena.allocator());
    defer std.process.argsFree(arena.allocator(), args);

    const file_path = args[1];
    var file = try std.Io.Dir.cwd().openFile(io, file_path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buffer: [8192]u8 = undefined;
    var reader = file.reader(io, &read_buffer);

    var replacer: StringReplaceReader = .{
        .in = &reader.interface,
        .pattern = "@@__BUILD_WORKSPACE_DIRECTORY__@@",
        .replace_by = workspace_dir,
    };

    var write_buffer: [8192]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &write_buffer);
    _ = try replacer.interface.streamRemaining(&writer.interface);
    try writer.interface.flush();
}

pub const main = if (builtin.zig_version.major == 0 and builtin.zig_version.minor >= 16)
    main_016
else if (builtin.zig_version.major == 0 and builtin.zig_version.minor == 15)
    main_015
else
    main_pre_015;
