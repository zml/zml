const std = @import("std");

const VFS = @import("vfs");

const log = std.log.scoped(.@"vfs/example");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Command = enum { cat, ls, cp, stat, realpath };

const help =
    \\ zml/VFS demo: run cat, ls, cp, stat or realpath over s3, gs, hf, http, or local filesystem.
    \\
    \\ Examples:
    \\ ls hf://Qwen/Qwen3-235B-A22B-Instruct-2507
    \\ ls s3://noaa-goes19/ABI-Flood-Day-Shapefiles/2025/08
    \\ ls gs://gcp-public-data-landsat/
    \\ cat https://iprs.fly.dev
    \\
;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var it = init.minimal.args.iterate();
    _ = it.next(); // skip program name

    const cmd = it.next() orelse {
        std.debug.print("{s}", .{help});
        return error.MissingCommand;
    };
    const command: Command = std.meta.stringToEnum(Command, cmd) orelse {
        std.debug.print("{s}", .{help});
        return error.CommandInvalid;
    };
    const path = it.next() orelse return error.MissingPath;

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };

    try http_client.initDefaultProxies(allocator, init.environ_map);
    defer http_client.deinit();

    var vfs_file: VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var vfs_https: VFS.HTTP = try .init(allocator, init.io, &http_client, .https);
    defer vfs_https.deinit();

    var hf_vfs: VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var s3_vfs: VFS.S3 = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer s3_vfs.deinit();

    var gcs_vfs: VFS.GCS = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer gcs_vfs.deinit();

    var vfs: VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("https", vfs_https.io());
    try vfs.register("hf", hf_vfs.io());
    try vfs.register("s3", s3_vfs.io());
    try vfs.register("gs", gcs_vfs.io());

    const buffer = try allocator.alignedAlloc(u8, .fromByteUnits(4 * 1024), 16 * 1024 * 1024);
    defer allocator.free(buffer);

    var stdout_writer = std.Io.File.stdout().writer(init.io, buffer);
    defer stdout_writer.interface.flush() catch {};

    const io: std.Io = vfs.io();
    runCmd(io, allocator, &stdout_writer.interface, command, path, &it) catch |err| {
        std.debug.print("Failed cmd: {t} {s} -> {}", .{ command, path, err });
        return err;
    };
}

pub fn runCmd(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, command: Command, path: []const u8, it: *std.process.Args.Iterator) !void {
    switch (command) {
        .cat => {
            var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
            defer file.close(io);

            var reader: std.Io.File.Reader = .initStreaming(file, io, &.{});

            const read = try reader.interface.streamRemaining(stdout);
            _ = try stdout.write("\n");

            try stdout.print("Wrote {B:.2} to stdout from {s}\n", .{ read, path });
        },
        .ls => {
            var dir = try std.Io.Dir.openDir(.cwd(), io, path, .{ .iterate = true });
            defer dir.close(io);

            const dir_stat = try dir.stat(io);
            try stdout.print("{s} - {B:.2}\n", .{ path, dir_stat.size });

            var counts: TreeCounts = .{};
            try printTree(io, allocator, stdout, dir, "", 10, &counts);
            try stdout.print("\n{d} directories, {d} files\n", .{ counts.dirs, counts.files });
        },
        .cp => {
            const destination_path = it.next() orelse {
                try stdout.print("Usage: cp <source> <destination>\n", .{});
                return error.InvalidArgument;
            };

            const source = try std.Io.Dir.openFile(.cwd(), io, path, .{});
            defer source.close(io);

            const destination = try std.Io.Dir.createFile(.cwd(), io, destination_path, .{});
            defer destination.close(io);

            var reader: std.Io.File.Reader = .initStreaming(source, io, &.{});
            // Reuse stdout buffer for the file copy
            var writer = destination.writer(io, stdout.buffer);

            const read = try reader.interface.streamRemaining(&writer.interface);
            try writer.interface.flush();

            try stdout.print("Copied {B:.2} from {s} to {s}\n", .{ read, path, destination_path });
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

            try stdout.print("{s}: {B:.2} ({s})\n", .{ path, stat.size, @tagName(stat.kind) });
        },
        .realpath => {
            var dir = try std.Io.Dir.openDir(.cwd(), io, path, .{});
            defer dir.close(io);

            var real_path_buf: [256]u8 = undefined;
            const len = try dir.realPath(io, &real_path_buf);

            try stdout.print("{s}\n", .{real_path_buf[0..len]});
        },
    }
}

const TreeCounts = struct { dirs: u64 = 0, files: u64 = 0 };

fn printTree(io: std.Io, allocator: std.mem.Allocator, writer: *std.Io.Writer, dir: std.Io.Dir, prefix: []const u8, max_depth: usize, counts: *TreeCounts) !void {
    if (max_depth == 0) return;

    var entries: std.ArrayList(std.Io.Dir.Entry) = .empty;
    try entries.ensureTotalCapacity(allocator, 128);
    defer entries.deinit(allocator);

    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        try entries.append(allocator, entry);
    }

    for (entries.items, 0..) |entry, idx| {
        const is_last = (idx == entries.items.len - 1);
        const connector = if (is_last) "└── " else "├── ";
        const extension = if (is_last) "    " else "│   ";

        const size: u64 = switch (entry.kind) {
            .file => blk: {
                const stat = try dir.statFile(io, entry.name, .{});
                break :blk stat.size;
            },
            .directory => blk: {
                var sub_dir = dir.openDir(io, entry.name, .{}) catch break :blk 0;
                defer sub_dir.close(io);

                const stat = try sub_dir.stat(io);
                break :blk stat.size;
            },
            else => 0,
        };

        try writer.print("{s}{s}{s} - {B:.2}\n", .{ prefix, connector, entry.name, size });

        if (entry.kind == .directory) {
            counts.dirs += 1;
            var sub_dir = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
            defer sub_dir.close(io);

            var new_prefix_buf: [4096]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&new_prefix_buf, "{s}{s}", .{ prefix, extension }) catch continue;
            try printTree(io, allocator, writer, sub_dir, new_prefix, max_depth - 1, counts);
        } else {
            counts.files += 1;
        }
    }
}
