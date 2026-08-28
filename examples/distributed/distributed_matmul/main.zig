const std = @import("std");

const distributed_example = @import("distributed_example");
const zml = @import("zml");

const rows_per_partition = 4;
const feature_count = 8;
const output_count = 4;

const MatmulOutput = struct {
    product: zml.Tensor,
    loss: zml.Tensor,
};

const Matmul = struct {
    pub fn forward(
        input: zml.Tensor,
        weights: zml.Tensor,
    ) MatmulOutput {
        const product = input.dot(weights, .feature)
            .withPartitioning(.{
            .batch = .data,
            .output = .replicated,
        });
        const loss = product.mean(.batch)
            .mean(.output)
            .withPartitioning(.{
                .batch = .replicated,
                .output = .replicated,
            })
            .reshape(.{});
        return .{ .product = product, .loss = loss };
    }
};

fn expectShardCounts(
    buffer: *const zml.Buffer,
    global: usize,
    local: usize,
) !void {
    if (buffer.numGlobalShards() != @as(u32, @intCast(global)) or
        buffer.numShards() != @as(u32, @intCast(local)))
    {
        return error.UnexpectedShardCount;
    }
}

fn expectReplicated(buffer: *const zml.Buffer) !void {
    const shape = buffer.shape();
    var shards = buffer.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        if (slices.len != shape.rank()) return error.UnexpectedReplication;
        for (slices.constSlice(), shape.dims()) |slice, dim| {
            if (slice.start != 0 or slice.size != dim) {
                return error.UnexpectedReplication;
            }
        }
    }
}

fn expectClose(
    rank: usize,
    device_id: u32,
    index: usize,
    expected: f32,
    actual: f32,
) !void {
    if (@abs(expected - actual) <= 0.00001) return;
    std.log.err(
        "matmul mismatch: rank={d} device={d} index={d} " ++
            "expected={d} actual={d}",
        .{ rank, device_id, index, expected, actual },
    );
    return error.UnexpectedMatmulValue;
}

fn verifyProduct(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    rank: usize,
    output_size: usize,
    reference: []const f32,
    product: *const zml.Buffer,
) !void {
    var batch_starts: [zml.Platform.MAX_NUM_DEVICES]i64 = undefined;
    var shard_count: usize = 0;
    var shards = product.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        const batch_slice = slices.get(0);
        const output_slice = slices.get(1);
        var addressable = false;
        for (platform.addressableDevices()) |device| {
            addressable = addressable or
                device.id() == shard.globalDeviceId();
        }
        if (!addressable or output_slice.start != 0 or
            output_slice.size != @as(i64, @intCast(output_size)))
        {
            return error.UnexpectedProductPlacement;
        }
        for (batch_starts[0..shard_count]) |start| {
            if (start == batch_slice.start) {
                return error.DuplicateProductShard;
            }
        }
        batch_starts[shard_count] = batch_slice.start;
        shard_count += 1;

        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);

        const local_output_size: usize = @intCast(output_slice.size);
        for (local.constItems(f32), 0..) |actual, index| {
            const local_row = index / local_output_size;
            const local_output = index % local_output_size;
            const global_row: usize = @intCast(
                batch_slice.start + @as(i64, @intCast(local_row)),
            );
            const global_output: usize = @intCast(
                output_slice.start + @as(i64, @intCast(local_output)),
            );
            const global_index = global_row * output_size + global_output;
            try expectClose(
                rank,
                shard.globalDeviceId(),
                global_index,
                reference[global_index],
                actual,
            );
        }

        const preview = local.constItems(f32);
        std.debug.print(
            "rank={d} device={d} batch={d}..{d} values={any}\n",
            .{
                rank,
                shard.globalDeviceId(),
                batch_slice.start,
                batch_slice.start + batch_slice.size,
                preview[0..@min(preview.len, 4)],
            },
        );
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try distributed_example.Job.parse(init);

    var platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);

    const global_device_count = platform.globalDevices().len;
    const local_device_count = platform.addressableDevices().len;
    const batch_size = global_device_count * rows_per_partition;
    if (global_device_count != 4 or local_device_count != 2) {
        return error.UnexpectedTopology;
    }

    const data_sharding = try distributed_example.dataSharding(platform);
    const input_shape = zml.Shape.init(.{
        .batch = batch_size,
        .feature = feature_count,
    }, .f32).withPartitioning(.{
        .batch = .data,
        .feature = .replicated,
    });
    const weight_shape = zml.Shape.init(.{
        .feature = feature_count,
        .output = output_count,
    }, .f32).withPartitioning(.{
        .feature = .replicated,
        .output = .replicated,
    });
    const product_shape = zml.Shape.init(.{
        .batch = batch_size,
        .output = output_count,
    }, .f32).withPartitioning(.{
        .batch = .data,
        .output = .replicated,
    });

    const host_input = try zml.Slice.alloc(allocator, input_shape);
    defer host_input.free(allocator);
    for (host_input.items(f32), 0..) |*value, index| {
        const row = index / feature_count;
        const feature = index % feature_count;
        value.* = @floatFromInt(row + feature + 1);
    }

    const host_weights = try zml.Slice.alloc(allocator, weight_shape);
    defer host_weights.free(allocator);
    for (host_weights.items(f32), 0..) |*value, index| {
        const feature = index / output_count;
        const output = index % output_count;
        value.* = @floatFromInt((feature % 3 + 1) * (output + 1));
    }

    const reference = try zml.Slice.alloc(allocator, product_shape);
    defer reference.free(allocator);
    for (0..batch_size) |batch| {
        for (0..output_count) |output| {
            var value: f32 = 0;
            for (0..feature_count) |feature| {
                value += host_input.constItems(f32)[
                    batch * feature_count + feature
                ] * host_weights.constItems(f32)[
                    feature * output_count + output
                ];
            }
            reference.items(f32)[batch * output_count + output] = value;
        }
    }
    var reference_loss: f32 = 0;
    for (reference.constItems(f32)) |value| reference_loss += value;
    reference_loss /= @floatFromInt(reference.constItems(f32).len);

    var input = try zml.Buffer.fromSlice(
        io,
        platform,
        host_input,
        data_sharding,
    );
    defer input.deinit();
    var weights = try zml.Buffer.fromSlice(
        io,
        platform,
        host_weights,
        zml.Sharding.replicated,
    );
    defer weights.deinit();
    try expectShardCounts(
        &input,
        global_device_count,
        local_device_count,
    );
    try expectShardCounts(
        &weights,
        global_device_count,
        local_device_count,
    );
    try expectReplicated(&weights);

    var executable = try platform.compileFn(
        allocator,
        io,
        Matmul.forward,
        .{
            zml.Tensor.fromShape(input_shape),
            zml.Tensor.fromShape(weight_shape),
        },
        .{
            .shardings = &.{data_sharding},
            .program_name = "distributed-matmul",
        },
    );
    defer executable.deinit();
    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);

    arguments.set(.{ input, weights });
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var output = results.get(zml.Bufferized(MatmulOutput));
    defer output.product.deinit();
    defer output.loss.deinit();
    try expectShardCounts(
        &output.product,
        global_device_count,
        local_device_count,
    );
    try expectShardCounts(
        &output.loss,
        global_device_count,
        local_device_count,
    );
    try expectReplicated(&output.loss);
    try verifyProduct(
        allocator,
        io,
        platform,
        job.process_index,
        output_count,
        reference.constItems(f32),
        &output.product,
    );

    var loss_shards = output.loss.shards();
    while (loss_shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        try expectClose(
            job.process_index,
            shard.globalDeviceId(),
            0,
            reference_loss,
            local.constItems(f32)[0],
        );
    }
    try expectClose(
        job.process_index,
        platform.addressableDevices()[0].id(),
        0,
        reference_loss,
        try output.loss.getValue(f32, io),
    );
    std.debug.print(
        "rank={d} global_devices={d} local_devices={d} loss={d}\n",
        .{
            job.process_index,
            global_device_count,
            local_device_count,
            reference_loss,
        },
    );
    try platform.barrier("distributed-matmul-before-shutdown");
}
