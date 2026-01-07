const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.vfs);

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const Command = union(enum) {
    cat,
    tree,
    stat,
};

pub const Args = struct {
    command: ?Command = null,
    path: ?[]const u8 = null,
    reader_buffer_size_mb: u32 = 32,
    async_limit: ?usize = null,
};

// -- cat https://iprs.fly.dev
pub fn main() !void {
    log.info("VFS playground was compiled with {}", .{@import("builtin").mode});

    var debug_allocator: ?std.heap.DebugAllocator(.{}) = null;
    const allocator = if (@import("builtin").mode == .Debug) blk: {
        debug_allocator = .init;
        break :blk debug_allocator.?.allocator();
    } else std.heap.c_allocator;
    defer if (debug_allocator) |*da| std.debug.assert(da.deinit() == .ok);

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const args: Args = blk: {
        var ret: Args = .{};
        var it = std.process.args();
        defer it.deinit();
        var index: usize = 0;
        while (it.next()) |arg| {
            if (std.mem.startsWith(u8, arg, "--reader-buffer-size-mb=")) {
                ret.reader_buffer_size_mb = try std.fmt.parseUnsigned(u32, arg["--reader-buffer-size-mb=".len..], 10);
            } else if (std.mem.startsWith(u8, arg, "--async-limit=")) {
                ret.async_limit = try std.fmt.parseUnsigned(usize, arg["--async-limit=".len..], 10);
            } else {
                if (index == 0) {
                    // skip program name
                } else if (index == 1) {
                    if (std.mem.eql(u8, arg, "cat")) {
                        ret.command = .cat;
                    } else if (std.mem.eql(u8, arg, "stat")) {
                        ret.command = .stat;
                    } else if (std.mem.eql(u8, arg, "tree")) {
                        ret.command = .tree;
                    } else {
                        log.err("Unknown command: {s}", .{arg});
                        return;
                    }
                } else if (index == 2) {
                    ret.path = arg;
                } else {
                    log.err("Unknown argument: {s}", .{arg});
                    return;
                }
                index += 1;
            }
        }

        if (ret.command == null) {
            log.err("Missing command as argument", .{});
            return;
        }

        if (ret.path == null) {
            log.err("Missing path as argument", .{});
            return;
        }

        break :blk ret;
    };

    if (args.async_limit) |limit| threaded.setAsyncLimit(.limited(limit));

    log.info("Running with threaded io async limit set to {?d}", .{threaded.async_limit.toInt()});

    var http_client: std.http.Client = .{
        .allocator = allocator,
        .io = threaded.io(),
        .connection_pool = .{
            .free_size = threaded.async_limit.toInt() orelse 16,
        },
    };

    try http_client.initDefaultProxies(allocator);
    defer http_client.deinit();

    var vfs_file: zml.io.VFS.File = .init(
        allocator,
        threaded.io(),
        .{
            .direct_io = true,
            .direct_io_alignment = .fromByteUnits(4 * 1024),
        },
    );
    defer vfs_file.deinit();

    var vfs_https: zml.io.VFS.HTTP = try .init(allocator, threaded.io(), .{
        .http_client = &http_client,
        .protocol = .https,
    });
    defer vfs_https.deinit();

    var hf_auth: zml.io.VFS.HF.Auth = try .auto(allocator, threaded.io());
    defer hf_auth.deinit(allocator);

    var hf_vfs: zml.io.VFS.HF = try .init(
        allocator,
        threaded.io(),
        .{
            .http_client = &http_client,
            .auth = hf_auth,
        },
    );
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, threaded.io());
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("https", vfs_https.io());
    try vfs.register("hf", hf_vfs.io());

    const io = vfs.io();

    const path = args.path.?;

    const buffer = try allocator.alloc(u8, args.reader_buffer_size_mb * 1024 * 1024);
    defer allocator.free(buffer);

    var stdout_writer = std.Io.File.stdout().writer(io, buffer);
    defer stdout_writer.interface.flush() catch {};

    switch (args.command.?) {
        .cat => {
            var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
            defer file.close(io);

            var reader = file.reader(io, &.{});
            const read = try reader.interface.streamRemaining(&stdout_writer.interface);
            _ = try stdout_writer.interface.write("\n");

            log.info("Wrote {d} bytes to stdout  from {s}", .{ read, path });
        },
        .tree => {
            var dir = try std.Io.Dir.openDir(.cwd(), io, path, .{ .iterate = true });
            defer dir.close(io);

            try printTree(io, &stdout_writer.interface, dir, 0, 10);
        },
        .stat => {
            const stat = std.Io.Dir.statFile(.cwd(), io, path, .{}) catch |err| blk: {
                if (err == error.IsDir) {
                    var dir = try std.Io.Dir.openDir(.cwd(), io, path, .{});
                    defer dir.close(io);

                    break :blk try dir.stat(io);
                }
                return err;
            };

            log.info("Stat for {s}: size={d} bytes, kind={s}", .{ path, stat.size, @tagName(stat.kind) });
        },
    }
}

fn printTree(io: std.Io, writer: *std.Io.Writer, dir: std.Io.Dir, depth: usize, max_depth: usize) !void {
    if (depth > max_depth) return;

    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        for (0..depth) |_| {
            try writer.writeAll("  ");
        }

        const size = if (entry.kind == .file) blk: {
            const stat = try dir.statFile(io, entry.name, .{ .follow_symlinks = true });
            break :blk stat.size;
        } else if (entry.kind == .directory) blk: {
            const stat = try dir.stat(io);
            break :blk stat.size;
        } else {
            return error.UnexpectedEntryKind;
        };

        var buf: [256]u8 = undefined;
        const slice = try std.fmt.bufPrint(&buf, "- {s} kind={s} size={d}\n", .{ entry.name, @tagName(entry.kind), size });
        try writer.writeAll(slice);

        if (entry.kind == .directory) {
            var sub_dir = try dir.openDir(io, entry.name, .{ .iterate = true });
            defer sub_dir.close(io);

            try printTree(io, writer, sub_dir, depth + 1, max_depth);
        }
    }
}
