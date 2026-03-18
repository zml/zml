const std = @import("std");
const zml = @import("zml");

pub fn sliceTokenPrefix(
    io: std.Io,
    platform: *zml.Platform,
    src: zml.Buffer,
    sharding: zml.sharding.Sharding,
    token_limit: usize,
) !zml.Buffer {
    const shape = src.shape();
    if (shape.rank() != 3) return error.UnexpectedRank;

    const dims = shape.dims();
    const b: usize = @intCast(dims[0]);
    const t: usize = @intCast(dims[1]);
    const d: usize = @intCast(dims[2]);
    const out_t = @min(token_limit, t);
    const elem_size: usize = src.shape().dtype().sizeOf();

    const out_shape = zml.Shape.init(.{ dims[0], @as(i64, @intCast(out_t)), dims[2] }, src.shape().dtype());
    var src_slice = try src.toSliceAlloc(std.heap.smp_allocator, io);
    defer src_slice.free(std.heap.smp_allocator);

    var out_slice = try zml.Slice.alloc(std.heap.smp_allocator, out_shape);
    defer out_slice.free(std.heap.smp_allocator);

    const src_bytes = src_slice.constData();
    const out_bytes = out_slice.data();
    const src_batch_stride = t * d * elem_size;
    const dst_batch_stride = out_t * d * elem_size;
    const per_batch_copy = out_t * d * elem_size;

    var bi: usize = 0;
    while (bi < b) : (bi += 1) {
        const src_batch_off = bi * src_batch_stride;
        const dst_batch_off = bi * dst_batch_stride;
        std.mem.copyForwards(u8, out_bytes[dst_batch_off .. dst_batch_off + per_batch_copy], src_bytes[src_batch_off .. src_batch_off + per_batch_copy]);
    }

    return zml.Buffer.fromSlice(io, platform, out_slice, sharding);
}

pub fn loadBufferFromStore(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const shape = store.view().getShape(key) orelse {
        std.log.err("Tensor not found in fixture: {s}", .{key});
        return error.NotFound;
    };

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();

    _ = try reader.interface.readSliceAll(host_bytes);
    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}
