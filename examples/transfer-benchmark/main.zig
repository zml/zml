const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const Args = struct {
    pub const help =
        \\ benchmark --model=./model
    ;
    model: []const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };

    try http_client.initDefaultProxies(allocator, init.environ_map);
    defer http_client.deinit();

    var vfs_https: zml.io.VFS.HTTP = try .init(allocator, init.io, &http_client, .https);
    defer vfs_https.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var s3_vfs: zml.io.VFS.S3 = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer s3_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("https", vfs_https.io());
    try vfs.register("hf", hf_vfs.io());
    try vfs.register("s3", s3_vfs.io());

    const io = vfs.io();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const args = stdx.flags.parseProcessArgs(init.minimal, Args);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.model);
    defer registry.deinit();

    var store = zml.io.TensorStore.fromRegistry(allocator, &registry);
    defer store.deinit();

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();
    var transfer_progress = progress.start("Transferring tensors (MiB)...", @intCast(registry.totalBytes() / zml.MiB));
    defer transfer_progress.end();

    log.info("⏱️ Running benchmark...", .{});

    const now: std.Io.Timestamp = .now(io, .awake);

    var total_bytes: usize = 0;
    const tensor_count = try transferAll(
        allocator,
        io,
        platform,
        .{
            .parallelism = 16,
            .store = &store,
            .dma_chunks = 4,
            .dma_chunk_size = 32 * zml.MiB,
            .progress = &transfer_progress,
            .total_bytes = &total_bytes,
        },
    );

    const elapsed = now.untilNow(io, .awake);
    const elapsed_ns = elapsed.toNanoseconds();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;

    log.info("✅ Benchmark done!", .{});

    const transfer_mib = @as(f64, @floatFromInt(total_bytes)) / @as(f64, zml.MiB);
    const mib_per_s = transfer_mib / elapsed_s;
    log.info("Transferred {d} tensors ({d} MiB) - Elapsed: {D} - {d:.3} MiB/s", .{
        tensor_count,
        total_bytes >> 20,
        stdx.fmt.fmtDuration(elapsed),
        mib_per_s,
    });
}

fn transferAll(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    opts: zml.io.LoadOpts,
) !usize {
    const first_device = platform.devices[0]; // Temporary until sharding is re-exposed
    const dma_allocator: zml.mem.DmaAllocator = .init(allocator, &first_device);
    var buffer_pool: zml.mem.DynamicBufferPool = .init(opts.dma_chunks, opts.dma_chunk_size);
    defer buffer_pool.deinit(dma_allocator.allocator());

    const keys = opts.store.registry.tensors.keys();
    const buffers = try allocator.alloc(zml.Buffer, keys.len);
    const buffers_initialized = try allocator.alloc(bool, keys.len);
    @memset(buffers_initialized, false);
    defer {
        for (buffers, buffers_initialized) |*buffer, is_initialized| {
            if (is_initialized) buffer.deinit();
        }
        allocator.free(buffers_initialized);
        allocator.free(buffers);
    }

    var ctx: TransferCtx = .{
        .dma_allocator = dma_allocator.allocator(),
        .pinned_buffer_pool = &buffer_pool,
        .io = io,
        .memory = first_device.memory(.default),
        .group = .init(opts.parallelism),
        .progress = opts.progress,
        .store = opts.store,
        .buffers = buffers,
        .buffers_initialized = buffers_initialized,
    };

    for (keys, 0..) |key, i| {
        ctx.group.concurrent(io, transferOne, .{ i, key, &ctx }) catch unreachable;
    }

    ctx.group.await(io) catch unreachable;

    if (opts.total_bytes) |total_bytes_ptr| {
        total_bytes_ptr.* = ctx.total.load(.monotonic);
    }

    return keys.len;
}

const TransferCtx = struct {
    dma_allocator: std.mem.Allocator,
    pinned_buffer_pool: *zml.mem.DynamicBufferPool,
    io: std.Io,
    memory: *const zml.Memory,
    group: stdx.Io.LimitedGroup,
    total: std.atomic.Value(usize) = .init(0),
    progress: ?*std.Progress.Node,
    store: *const zml.io.TensorStore,
    buffers: []zml.Buffer,
    buffers_initialized: []bool,
};

fn transferOne(i: usize, key: []const u8, ctx: *TransferCtx) !void {
    var reader = ctx.store.getReader(key, ctx.io, &.{}) catch unreachable;
    defer reader.deinit();

    var memory_writer = zml.io.MemoryWriter.init(
        ctx.dma_allocator,
        ctx.io,
        ctx.memory,
        ctx.pinned_buffer_pool,
        reader.tensor.shape,
        &ctx.buffers[i],
    ) catch unreachable;
    defer memory_writer.deinit(ctx.dma_allocator);
    ctx.buffers_initialized[i] = true;

    const scale = zml.MiB;
    const total = if (ctx.progress) |progress| blk: {
        const child_total = @max(@as(u64, 1), reader.tensor.shape.byteSize() / scale);
        var child = progress.start(reader.tensor.name, child_total);
        defer child.end();

        var progress_writer: zml.io.ProgressWriter = .init(memory_writer.interface(), &child, .{ .scale = scale });
        const written = reader.interface.streamRemaining(&progress_writer.interface) catch unreachable;
        progress_writer.interface.flush() catch unreachable;
        break :blk written;
    } else blk: {
        const written = reader.interface.streamRemaining(memory_writer.interface()) catch unreachable;
        memory_writer.interface().flush() catch unreachable;
        break :blk written;
    };

    const previous = ctx.total.fetchAdd(total, .monotonic);
    if (ctx.progress) |progress| {
        progress.setCompletedItems((previous + total) / scale);
    }
}
