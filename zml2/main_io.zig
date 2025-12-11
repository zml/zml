const builtin = @import("builtin");
const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.@"zml/main_io");

pub fn main() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer std.debug.assert(debug_allocator.deinit() == .ok);

    const allocator = debug_allocator.allocator();

    var threaded: std.Io.Threaded = .init(allocator);
    defer threaded.deinit();

    if (try std.process.hasEnvVar(allocator, "ASYNC_LIMIT")) {
        const limit = try std.process.getEnvVarOwned(allocator, "ASYNC_LIMIT");
        defer allocator.free(limit);

        threaded.async_limit = .limited(try std.fmt.parseInt(usize, limit, 10));
    }

    log.debug("Running with threaded async limit: {?d}", .{threaded.async_limit.toInt()});

    var http_client: std.http.Client = .{
        .allocator = allocator,
        .io = threaded.io(),
        .connection_pool = .{
            .free_size = threaded.async_limit.toInt() orelse 16,
        },
    };

    try http_client.initDefaultProxies(allocator);
    defer http_client.deinit();

    var vfs_file: zml.io.VFS.File = .init(threaded.io());

    var vfs_http: zml.io.VFS.HTTP = try .init(allocator, threaded.io(), &http_client, .{});
    defer vfs_http.deinit();

    var hf_auth: zml.io.VFS.HF.Auth = try .auto(allocator, threaded.io());
    defer hf_auth.deinit(allocator);

    var hf_vfs: zml.io.VFS.HF = try .init(.{
        .allocator = allocator,
        .io = threaded.io(),
        .http_client = &http_client,
        .config = .{
            .auth = hf_auth,
        },
    });
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = .init(allocator, threaded.io());
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("http", vfs_http.io());
    try vfs.register("https", vfs_http.io());
    try vfs.register("hf", hf_vfs.io());

    const io = vfs.io();

    const repo_uri = "hf://Qwen/Qwen3-8B";
    // const repo_uri = "file:///Users/hugo/Developer/Llama-3.1-8B-Instruct";
    // const repo_uri = "https://storage.googleapis.com/zig-vfs/Llama-3.1-8B-Instruct/";
    // const repo_uri = "http://9960x-5090x2:8003/";

    {
        const root_dir = try vfs.openAbsoluteDir(io, "https://storage.googleapis.com", .{});
        defer root_dir.close(io);

        const buckets_dir = try root_dir.openDir(io, "zig-vfs", .{});
        defer buckets_dir.close(io);

        const model_dir = try buckets_dir.openDir(io, "Llama-3.1-8B-Instruct", .{});
        defer model_dir.close(io);

        const file = try model_dir.openFile(io, "model.safetensors.index.json", .{});
        defer file.close(io);

        const stat = try file.stat(io);
        log.info("Opened remote file with size: {d} bytes by navigating dirs", .{stat.size});
    }

    {
        const path = try std.fs.path.join(allocator, &.{ repo_uri, "model-00004-of-00005.safetensors" });
        defer allocator.free(path);

        const file = try vfs.openAbsoluteFile(io, path, .{});
        defer file.close(io);

        const stat = try file.stat(io);

        var group: std.Io.Group = .init;
        defer group.cancel(io);

        const chunk_size: usize = 32 * 1024 * 1024;
        const num_chunks = (stat.size + chunk_size - 1) / chunk_size;

        for (0..num_chunks) |i| {
            const offset = i * chunk_size;
            const size = if (offset + chunk_size > stat.size) stat.size - offset else chunk_size;

            group.async(threaded.io(), readData, .{ allocator, io, &vfs, path, offset, size, i });
        }

        group.wait(io);
    }

    {
        const path = try std.fs.path.join(allocator, &.{ repo_uri, "model.safetensors.index.json" });
        defer allocator.free(path);

        const file = try vfs.openAbsoluteFile(io, path, .{});
        defer file.close(io);

        const stat = try file.stat(io);
        log.info("Opened remote file with size: {d} bytes with openAbsoluteFile", .{stat.size});
    }

    {
        const model_dir = try vfs.openAbsoluteDir(io, repo_uri, .{});

        const safetensors_index = try model_dir.openFile(io, "model.safetensors.index.json", .{});
        defer safetensors_index.close(io);

        const stat = try safetensors_index.stat(io);
        log.info("Opened remote index file with size: {d} bytes with openAbsoluteDir + openFile", .{stat.size});
    }

    {
        const path = try std.fs.path.join(allocator, &.{ repo_uri, "model-00004-of-00005.safetensors" });
        defer allocator.free(path);

        const file = try vfs.openAbsoluteFile(io, path, .{});
        defer file.close(io);

        var reader = file.reader(io, &.{});

        const read_size = 16 * 1024 * 1024;

        var allocating_writer: std.Io.Writer.Allocating = try .initCapacity(allocator, read_size);
        defer allocating_writer.deinit();

        const read = try reader.interface.stream(&allocating_writer.writer, .limited(read_size));
        std.debug.assert(read == read_size);
    }

    {
        const path = try std.fs.path.join(allocator, &.{ repo_uri, "model-00004-of-00005.safetensors" }); // SHA256: 92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b
        defer allocator.free(path);

        const file = try vfs.openAbsoluteFile(io, path, .{});
        defer file.close(io);

        const stat = try file.stat(io);

        const buf_size = 64 * 1024 * 1024;
        const buf = try allocator.alloc(u8, buf_size);
        defer allocator.free(buf);

        var sha256: std.crypto.hash.sha2.Sha256 = .init(.{});
        const compute_sha = true;

        // Read directly without extra buffering layers
        var total_read: usize = 0;
        var bufs = [_][]u8{buf};

        while (true) {
            const n = try io.vtable.fileReadStreaming(io.userdata, file, &bufs);
            if (n == 0) break;
            if (compute_sha) sha256.update(buf[0..n]);
            total_read += n;
        }

        if (compute_sha) {
            var hash: [32]u8 = undefined;
            sha256.final(&hash);
            log.info("SHA256: {x}", .{hash});
        }

        std.debug.assert(total_read == stat.size);
    }
}

fn readData(
    allocator: std.mem.Allocator,
    io: std.Io,
    vfs: *zml.io.VFS,
    uri: []const u8,
    offset: usize,
    size: usize,
    chunk_index: usize,
) void {
    const buffer = allocator.alloc(u8, size) catch {
        log.err("Failed to allocate buffer of size {d} bytes for chunk {d}", .{ size, chunk_index });
        return;
    };
    defer allocator.free(buffer);

    var file = vfs.openAbsoluteFile(io, uri, .{}) catch {
        log.err("Failed to open file at {s}", .{uri});
        return;
    };

    var reader = file.reader(io, &.{});

    reader.seekTo(@intCast(offset)) catch {
        log.err("Failed to seek to offset {d} for chunk {d}", .{ offset, chunk_index });
        return;
    };

    var total_read: usize = 0;
    while (total_read < size) {
        const n = reader.interface.readSliceShort(buffer[total_read..]) catch {
            log.err("Failed to read data for chunk {d} at offset {d}", .{ chunk_index, offset + total_read });
            return;
        };
        if (n == 0) break;
        total_read += n;
    }

    std.debug.assert(total_read == size);
    log.info("Read chunk {d} of size {d} bytes at offset {d}", .{ chunk_index, size, offset });
}
