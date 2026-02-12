const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k);
}

const Args = struct {
    pub const help =
        \\ benchmark --model=./model --tensor="model.embed_tokens.weight"
    ;
    model: []const u8,
    tensor: []const u8 = "model.embed_tokens.weight",
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    std.debug.print("{f}", .{platform.fmtVerbose()});
    const args = stdx.flags.parseProcessArgs(init.minimal, Args);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.model);
    defer registry.deinit();

    var reader = try registry.reader(io, args.tensor, &.{});

    var buffer: zml.Buffer = undefined;
    defer buffer.deinit();

    log.info("⏱️ Running benchmark...", .{});

    const now: std.Io.Timestamp = .now(io, .awake);

    try transfer(allocator, io, platform, reader.tensor.shape, &reader.interface, &buffer, 4, 32 * zml.MiB);

    const elapsed = now.untilNow(io, .awake);
    const elapsed_ns = elapsed.toNanoseconds();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;

    log.info("✅ Benchmark done!", .{});

    const size = reader.tensor.shape.byteSize();
    const flops = @as(f64, @floatFromInt(size >> 20)) / elapsed_s;
    log.info("Transfer {d} MiB - Elapsed: {D} - {d:.3} MiB/s", .{
        size >> 20,
        stdx.fmt.fmtDuration(elapsed),
        flops,
    });
}

pub fn transfer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    reader: *std.Io.Reader,
    buffer: *zml.Buffer,
    dma_chunks: usize,
    dma_chunk_size: usize,
) !void {
    const first_device = platform.devices[0]; // Temporary until sharding is re-exposed
    const dma_allocator: zml.mem.DmaAllocator = .init(allocator, &first_device);
    var buffer_pool: zml.mem.DynamicBufferPool = .init(dma_chunks, dma_chunk_size);
    defer buffer_pool.deinit(dma_allocator.allocator());

    var memory_writer = zml.io.MemoryWriter.init(
        dma_allocator.allocator(),
        io,
        first_device.memory(.default),
        &buffer_pool,
        shape,
        buffer,
    ) catch unreachable;
    defer memory_writer.deinit(dma_allocator.allocator());

    _ = reader.streamRemaining(memory_writer.interface()) catch unreachable;
    memory_writer.interface().flush() catch unreachable;
}
