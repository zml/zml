const std = @import("std");
const asynk = @import("async");

const stdx = @import("stdx");
const zml = @import("zml");
const Shape = zml.Shape;

pub const ShardWriter = struct {
    tensor_shape: Shape,
    shard_shape: Shape,
    interface: std.Io.Writer,
    out: *std.Io.Writer,

    fn init(out: *std.Io.Writer, buffer: []u8, tensor_shape: Shape, shard_shape: Shape) ShardWriter {
        return .{
            .tensor_shape = tensor_shape,
            .shard_shape = shard_shape,
            .interface = .{
                .buffer = buffer,
                .vtable = &.{ .drain = drain },
            },
            .out = out,
        };
    }

    pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *ShardWriter = @alignCast(@fieldParentPtr("interface", w));
        return self.out.writeSplat(data, splat);
    }
};

fn computeCoordinates(tensor_shape: Shape, shard_shape: Shape, offset: usize) [Shape.MAX_RANK]usize {
    var coordinates: [Shape.MAX_RANK]usize = @splat(0);
    var offset_in_elem = offset / tensor_shape.dtype().sizeOf();
    const strides = tensor_shape.computeStrides();
    for (0..tensor_shape.rank()) |dim| {
        const offset_in_dim = offset_in_elem / @as(usize, @intCast(strides.get(dim)));
        const coordinate_in_dim = offset_in_dim / @as(usize, @intCast(shard_shape.dim(dim)));
        offset_in_elem -= offset_in_dim * @as(usize, @intCast(strides.get(dim)));
        coordinates[dim] = coordinate_in_dim;
    }

    return coordinates;
}

test computeCoordinates {
    {
        const tensor_shape: Shape = .init(.{ 1024, 1024 }, .u8);
        const shard_shape: Shape = .init(.{ 512, 512 }, .u8);
        const rank = tensor_shape.rank();

        try std.testing.expectEqualSlices(usize, &[2]usize{ 0, 0 }, computeCoordinates(tensor_shape, shard_shape, 0)[0..rank]);
        try std.testing.expectEqualSlices(usize, &[2]usize{ 0, 0 }, computeCoordinates(tensor_shape, shard_shape, 511)[0..rank]);
        try std.testing.expectEqualSlices(usize, &[2]usize{ 0, 1 }, computeCoordinates(tensor_shape, shard_shape, 512)[0..rank]);
        try std.testing.expectEqualSlices(usize, &[2]usize{ 1, 0 }, computeCoordinates(tensor_shape, shard_shape, 1024 * 512)[0..rank]);
        try std.testing.expectEqualSlices(usize, &[2]usize{ 1, 1 }, computeCoordinates(tensor_shape, shard_shape, 1024 * 512 + 512)[0..rank]);
    }

    {
        const tensor_shape: Shape = .init(.{ 1024, 1024, 1024 }, .u8);
        const shard_shape: Shape = .init(.{ 512, 512, 512 }, .u8);
        const rank = tensor_shape.rank();

        try std.testing.expectEqualSlices(usize, &[3]usize{ 0, 0, 0 }, computeCoordinates(tensor_shape, shard_shape, 0)[0..rank]);
        try std.testing.expectEqualSlices(usize, &[3]usize{ 0, 0, 0 }, computeCoordinates(tensor_shape, shard_shape, 511)[0..rank]);
        try std.testing.expectEqualSlices(usize, &[3]usize{ 0, 0, 1 }, computeCoordinates(tensor_shape, shard_shape, 512)[0..rank]);
        try std.testing.expectEqualSlices(usize, &[3]usize{ 0, 1, 0 }, computeCoordinates(tensor_shape, shard_shape, 1024 * 512)[0..rank]);
        try std.testing.expectEqualSlices(usize, &[3]usize{ 0, 1, 1 }, computeCoordinates(tensor_shape, shard_shape, 1024 * 512 + 512)[0..rank]);
        try std.testing.expectEqualSlices(usize, &[3]usize{ 1, 0, 0 }, computeCoordinates(tensor_shape, shard_shape, 1024 * 1024 * 512)[0..rank]);
    }
}

pub const TensorWriter = struct {
    shard_writers: []ShardWriter,
    temp_writers: []std.Io.Writer.Allocating,
    interface: std.Io.Writer,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, tensor_shape: Shape, buffer: []u8, platform: zml.Platform) !TensorWriter {
        const device_count = platform.getDevices().len;
        var dims: stdx.BoundedArray(i64, Shape.MAX_RANK) = try .init(tensor_shape.rank());
        for (0..tensor_shape.rank()) |dim| {
            if (tensor_shape._sharding_info[dim]) {
                dims.slice()[dim] = @divExact(tensor_shape.dim(dim), @as(i64, @intCast(device_count)));
            } else {
                dims.slice()[dim] = tensor_shape.dim(dim);
            }
        }
        const shard_shape = Shape.init(dims.slice(), tensor_shape.dtype());

        const shard_writers = try allocator.alloc(ShardWriter, device_count);
        errdefer allocator.free(shard_writers);

        const temp_writers = try allocator.alloc(std.Io.Writer.Allocating, device_count);
        errdefer allocator.free(temp_writers);

        for (0..device_count) |i| {
            temp_writers[i] = std.Io.Writer.Allocating.init(allocator);
            shard_writers[i] = ShardWriter.init(&temp_writers[i].writer, &.{}, tensor_shape, shard_shape);
        }

        return .{
            .shard_writers = shard_writers,
            .temp_writers = temp_writers,
            .interface = .{
                .buffer = buffer,
                .vtable = &.{
                    .drain = drain,
                },
            },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: TensorWriter) void {
        self.allocator.free(self.shard_writers);
        for (self.temp_writers) |*temp_writer| temp_writer.deinit();
        self.allocator.free(self.temp_writers);
    }

    pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *TensorWriter = @alignCast(@fieldParentPtr("interface", w));
        var maybe_written: ?usize = null;
        for (self.shard_writers) |*shard_writer| {
            const shard_written = try shard_writer.interface.writeSplat(data, splat);
            if (maybe_written) |written| {
                std.debug.assert(written == shard_written);
            } else {
                maybe_written = shard_written;
            }
        }

        return maybe_written.?;
    }
};

pub const RandomReader = struct {
    random: std.Random,
    interface: std.Io.Reader,

    pub fn init(random: std.Random, buffer: []u8) RandomReader {
        return .{
            .random = random,
            .interface = .{
                .buffer = buffer,
                .vtable = &.{
                    .stream = stream,
                },
                .seek = 0,
                .end = 0,
            },
        };
    }

    pub fn stream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        //std.log.info("Calling stream with limit: {any}", .{limit});
        const self: *RandomReader = @alignCast(@fieldParentPtr("interface", r));
        const bytes = w.unusedCapacitySlice();
        const to_write = @min(bytes.len, limit.toInt() orelse bytes.len);
        self.random.bytes(bytes[0..to_write]);
        //@memset(bytes[0..to_write], 0xca);

        return to_write;
    }
};

test "random reader" {
    try asynk.AsyncThread.main(std.heap.c_allocator, testRandomReader);
}

fn testRandomReader() !void {
    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});

    var generator = std.Random.DefaultPrng.init(0);
    const random = generator.random();

    var random_reader: RandomReader = .init(random, &.{});
    _ = &random_reader; // autofix

    const tensor_shape: Shape = .init(.{ 16, 16 }, .u8);
    var tensor_writer: TensorWriter = try .init(std.testing.allocator, tensor_shape, &.{}, platform);
    defer tensor_writer.deinit();

    _ = try random_reader.interface.stream(&tensor_writer.interface, .limited(16 * 16));
    try tensor_writer.interface.flush();

    for (tensor_writer.temp_writers, 0..) |*temp_writer, i| {
        std.debug.print("shard: {d} - written: {}\n", .{ i, temp_writer.written().len });
    }
}
