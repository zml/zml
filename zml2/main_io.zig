const builtin = @import("builtin");
const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.@"zml/main_io");

pub const std_options: std.Options = .{
    .log_level = .debug,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .@"zml/safetensors", .level = .debug },
        .{ .scope = .@"zml/io/vfs/http", .level = .info },
    },
};

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

    const default_uri = "hf://Qwen/Qwen3-8B";
    // const default_uri = "file:///Users/hugo/Developer/Llama-3.1-8B-Instruct";
    // const default_uri = "https://storage.googleapis.com/zig-vfs/Llama-3.1-8B-Instruct/";
    // const default_uri = "http://9960x-5090x2:8003/";

    var args = std.process.args();
    _ = args.next(); // skip program name

    const uri = if (args.next()) |uri| blk: {
        log.info("Using repo URI from args: {s}", .{uri});
        break :blk uri;
    } else blk: {
        log.info("Using default repo URI: {s}", .{default_uri});
        break :blk default_uri;
    };

    {
        var timer: std.time.Timer = try .start();
        defer {
            const elapsed = timer.read();
            log.info("Completed in {d} ms", .{elapsed / std.time.ns_per_ms});
        }

        var registry = try zml.safetensors.parseFromPath(allocator, io, &vfs, uri);
        defer registry.deinit();

        // var it = registry.iterator();
        // while (it.next()) |entry| {
        //     const tensor = entry.value_ptr.*;
        //     log.info("{f}", .{tensor});
        // }

        log.info("Parsed {d} tensors", .{registry.tensors.count()});

        {
            const tensor_name = "model.layers.31.mlp.down_proj.weight";
            // const tensor_name = "model.layers.31.input_layernorm.weight";
            const tensor = registry.tensors.get(tensor_name) orelse return error.TensorNotFound;
            log.info("Reading {f}...", .{tensor});

            const tensor_reader_buf = try allocator.alloc(u8, tensor.byteSize());
            defer allocator.free(tensor_reader_buf);

            const writer_buffer = try allocator.alloc(u8, tensor.byteSize());
            defer allocator.free(writer_buffer);

            var tensor_reader = try registry.reader(io, &vfs, tensor_name, tensor_reader_buf);
            defer tensor_reader.deinit();

            var writer: std.Io.Writer = .fixed(writer_buffer);
            const read = try tensor_reader.interface.streamRemaining(&writer);

            log.info("Read tensor data: {d} bytes", .{read});
        }

        {
            var err: ?anyerror = null;

            var pool: MemoryPool = try .init(allocator, threaded.async_limit.toInt() orelse 16, 256 * 1024 * 1024);
            defer pool.deinit(io);

            var group: std.Io.Group = .init;
            defer group.cancel(io);

            var timer_async_read: std.time.Timer = try .start();
            var it = registry.iterator();
            while (it.next()) |entry| {
                const tensor_name = entry.key_ptr.*;
                group.async(threaded.io(), readTensor, .{ io, &vfs, &pool, &registry, tensor_name, &err });
            }

            group.wait(io);

            if (err) |e| {
                log.err("Error reading tensor: {any}", .{e});
                return e;
            }

            const elapsed = timer_async_read.read();
            const read_mb = @as(f64, @floatFromInt(registry.totalBytes())) / (1024.0 * 1024.0);
            const read_time_s = @as(f64, @floatFromInt(elapsed)) / @as(f64, std.time.ns_per_s);
            log.info("Throughput: {d:.2} MB in {d:.2} s = {d:.2} MB/s", .{ read_mb, read_time_s, read_mb / read_time_s });
        }
    }

    // {
    //     const root_dir = try vfs.openAbsoluteDir(io, "https://storage.googleapis.com", .{});
    //     defer root_dir.close(io);

    //     const buckets_dir = try root_dir.openDir(io, "zig-vfs", .{});
    //     defer buckets_dir.close(io);

    //     const model_dir = try buckets_dir.openDir(io, "Llama-3.1-8B-Instruct", .{});
    //     defer model_dir.close(io);

    //     const file = try model_dir.openFile(io, "model.safetensors.index.json", .{});
    //     defer file.close(io);

    //     const stat = try file.stat(io);
    //     log.info("Opened remote file with size: {d} bytes by navigating dirs", .{stat.size});
    // }

    // {
    //     const path = try std.fs.path.join(allocator, &.{ default_uri, "model-00004-of-00005.safetensors" });
    //     defer allocator.free(path);

    //     const file = try vfs.openAbsoluteFile(io, path, .{});
    //     defer file.close(io);

    //     const stat = try file.stat(io);

    //     var group: std.Io.Group = .init;
    //     defer group.cancel(io);

    //     const chunk_size: usize = 32 * 1024 * 1024;
    //     const num_chunks = (stat.size + chunk_size - 1) / chunk_size;

    //     for (0..num_chunks) |i| {
    //         const offset = i * chunk_size;
    //         const size = if (offset + chunk_size > stat.size) stat.size - offset else chunk_size;

    //         group.async(threaded.io(), readData, .{ allocator, io, &vfs, path, offset, size, i });
    //     }

    //     group.wait(io);
    // }

    // {
    //     const path = try std.fs.path.join(allocator, &.{ repo_uri, "model.safetensors.index.json" });
    //     defer allocator.free(path);

    //     const file = try vfs.openAbsoluteFile(io, path, .{});
    //     defer file.close(io);

    //     const stat = try file.stat(io);
    //     log.info("Opened remote file with size: {d} bytes with openAbsoluteFile", .{stat.size});
    // }

    // {
    //     const model_dir = try vfs.openAbsoluteDir(io, repo_uri, .{});

    //     const safetensors_index = try model_dir.openFile(io, "model.safetensors.index.json", .{});
    //     defer safetensors_index.close(io);

    //     const stat = try safetensors_index.stat(io);
    //     log.info("Opened remote index file with size: {d} bytes with openAbsoluteDir + openFile", .{stat.size});
    // }

    // {
    //     const path = try std.fs.path.join(allocator, &.{ repo_uri, "model-00004-of-00005.safetensors" });
    //     defer allocator.free(path);

    //     const file = try vfs.openAbsoluteFile(io, path, .{});
    //     defer file.close(io);

    //     var reader = file.reader(io, &.{});

    //     const read_size = 16 * 1024 * 1024;

    //     var allocating_writer: std.Io.Writer.Allocating = try .initCapacity(allocator, read_size);
    //     defer allocating_writer.deinit();

    //     const read = try reader.interface.stream(&allocating_writer.writer, .limited(read_size));
    //     std.debug.assert(read == read_size);
    // }

    // {
    //     const path = try std.fs.path.join(allocator, &.{ default_uri, "model-00004-of-00005.safetensors" }); // SHA256: 92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b
    //     defer allocator.free(path);

    //     const file = try vfs.openAbsoluteFile(io, path, .{});
    //     defer file.close(io);

    //     const stat = try file.stat(io);

    //     const buf_size = 64 * 1024 * 1024;
    //     const buf = try allocator.alloc(u8, buf_size);
    //     defer allocator.free(buf);

    //     var sha256: std.crypto.hash.sha2.Sha256 = .init(.{});
    //     const compute_sha = true;

    //     // Read directly without extra buffering layers
    //     var total_read: usize = 0;
    //     var bufs = [_][]u8{buf};

    //     while (true) {
    //         const n = try io.vtable.fileReadStreaming(io.userdata, file, &bufs);
    //         if (n == 0) break;
    //         if (compute_sha) sha256.update(buf[0..n]);
    //         total_read += n;
    //     }

    //     if (compute_sha) {
    //         var hash: [32]u8 = undefined;
    //         sha256.final(&hash);
    //         log.info("SHA256: {x}", .{hash});
    //     }

    //     std.debug.assert(total_read == stat.size);
    // }
}

fn readTensor(
    io: std.Io,
    vfs: *zml.io.VFS,
    pool: *MemoryPool,
    registry: *zml.safetensors.TensorRegistry,
    tensor_name: []const u8,
    err_ptr: *?anyerror,
) void {
    readTensorImpl(io, vfs, pool, registry, tensor_name) catch |err| {
        err_ptr.* = err;
        log.err("Failed to read tensor {s}: {any}", .{ tensor_name, err });
    };
}

pub fn readTensorImpl(
    io: std.Io,
    vfs: *zml.io.VFS,
    pool: *MemoryPool,
    registry: *zml.safetensors.TensorRegistry,
    tensor_name: []const u8,
) !void {
    var total_timer = try std.time.Timer.start();

    const tensor = registry.tensors.get(tensor_name) orelse return error.TensorNotFound;

    const handle = try pool.alloc(io, tensor.byteSize() * 2);
    defer pool.free(io, handle.index);

    var tensor_reader = try registry.reader(io, vfs, tensor_name, handle.data[0..tensor.byteSize()]);
    defer tensor_reader.deinit();

    var writer: std.Io.Writer = .fixed(handle.data[tensor.byteSize()..]);

    var read_timer = try std.time.Timer.start();
    const read = try tensor_reader.interface.streamRemaining(&writer);

    const read_time = read_timer.read();
    const total_time = total_timer.read();
    const read_mb = @as(f64, @floatFromInt(read)) / (1024.0 * 1024.0);
    const read_time_s = @as(f64, @floatFromInt(read_time)) / @as(f64, std.time.ns_per_s);
    const throughput = if (read_time_s > 0) read_mb / read_time_s else 0;

    log.info("Tensor {s} read {d}/{d} bytes at offset {d} | read={d}ms total={d}ms | {d:.2} MB/s", .{
        tensor.name,
        read,
        tensor.byteSize(),
        tensor.offset,
        read_time / std.time.ns_per_ms,
        total_time / std.time.ns_per_ms,
        throughput,
    });
}

const MemoryPool = struct {
    const Slots = std.ArrayList(Slot);
    const Slot = struct {
        buf: []u8,
        capacity: usize,
        in_use: bool,
    };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex,

    slots: Slots,
    default_capacity: usize,

    pub fn init(allocator: std.mem.Allocator, initial_count: usize, slot_size: usize) !MemoryPool {
        var slots: Slots = .{};

        for (0..initial_count) |_| {
            const buf = try allocator.alloc(u8, slot_size);
            try slots.append(allocator, .{ .buf = buf, .capacity = slot_size, .in_use = false });
        }

        return .{
            .allocator = allocator,
            .slots = slots,
            .default_capacity = slot_size,
            .mutex = .init,
        };
    }

    pub fn deinit(self: *MemoryPool, io: std.Io) void {
        self.mutex.lockUncancelable(io);

        for (self.slots.items) |slot| {
            self.allocator.free(slot.buf);
        }

        self.mutex.unlock(io);
        self.slots.deinit(self.allocator);
    }

    pub fn alloc(self: *MemoryPool, io: std.Io, n: usize) !struct { data: []u8, index: usize } {
        self.mutex.lockUncancelable(io);

        var idx: usize = 0;
        while (idx < self.slots.items.len) : (idx += 1) {
            if (!self.slots.items[idx].in_use and self.slots.items[idx].capacity >= n) {
                self.slots.items[idx].in_use = true;
                const data = self.slots.items[idx].buf[0..n];
                self.mutex.unlock(io);
                return .{ .data = data, .index = idx };
            }
        }

        const capacity = if (n > self.default_capacity) n else self.default_capacity;

        log.warn("MemoryPool: growing pool, allocating new slot of size {d} bytes", .{capacity});

        const buf = try self.allocator.alloc(u8, capacity);
        try self.slots.append(self.allocator, .{ .buf = buf, .capacity = capacity, .in_use = true });

        const new_index = self.slots.items.len - 1;
        const data = buf[0..n];

        self.mutex.unlock(io);

        return .{ .data = data, .index = new_index };
    }

    pub fn free(self: *MemoryPool, io: std.Io, index: usize) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        if (index < self.slots.items.len) {
            self.slots.items[index].in_use = false;
        }
    }
};

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
