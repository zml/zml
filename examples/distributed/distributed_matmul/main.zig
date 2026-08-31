const std = @import("std");

const zml = @import("zml");

const CliArgs = struct {
    pub const help =
        \\Usage: distributed_matmul COORDINATOR RANK PROCESS_COUNT NAMESPACE
    ;

    positional: struct {
        coordinator: []const u8,
        rank: usize,
        processCount: usize,
        namespace: []const u8,
    },
};

const feature_count = 8;
const output_count = 4;

const Output = struct {
    product: zml.Tensor,
    loss: zml.Tensor,
};

const Matmul = struct {
    pub fn forward(input: zml.Tensor, weights: zml.Tensor) Output {
        const product = input.dot(weights, .feature).withPartitioning(.{
            .batch = .data,
            .output = .replicated,
        });
        return .{
            .product = product,
            .loss = product.mean(.batch).mean(.output).reshape(.{}),
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, CliArgs).positional;
    if (args.processCount == 0 or
        args.rank >= args.processCount or
        args.namespace.len == 0)
    {
        return error.InvalidDistributedJob;
    }
    var platform = try zml.Platform.init(allocator, io, .cuda, .{
        .distributed = .{
            .coordinator_address = try .parseLiteral(args.coordinator),
            .process_index = args.rank,
            .process_count = args.processCount,
            .namespace = args.namespace,
            .local_device_ids = &.{ 0, 1 },
        },
        .xla_gpu = .{
            .allocator = .{ .bfc = .{ .preallocate = false } },
        },
    });
    defer platform.deinit(allocator, io);

    if (platform.globalDevices().len != 4 or
        platform.addressableDevices().len != 2)
    {
        return error.UnexpectedTopology;
    }
    const sharding = try platform.registerShardingWithStrategy(
        "data",
        .mesh(.{ .data = .low_bandwidth }),
        .parseBindings(.{ .data = .{ .network, .link } }),
    );
    const input_shape = zml.Shape.init(.{
        .batch = platform.globalDevices().len * 4,
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

    const host_input = try zml.Slice.alloc(allocator, input_shape);
    defer host_input.free(allocator);
    @memset(host_input.items(f32), 1);
    const host_weights = try zml.Slice.alloc(allocator, weight_shape);
    defer host_weights.free(allocator);
    @memset(host_weights.items(f32), 1);
    var input = try zml.Buffer.fromSlice(
        io,
        platform,
        host_input,
        sharding,
    );
    defer input.deinit();
    var weights = try zml.Buffer.fromSlice(
        io,
        platform,
        host_weights,
        platform.replicated_sharding,
    );
    defer weights.deinit();

    var executable = try platform.compileFn(
        allocator,
        io,
        Matmul.forward,
        .{
            zml.Tensor.fromShape(input_shape),
            zml.Tensor.fromShape(weight_shape),
        },
        .{
            .shardings = &.{sharding},
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

    var output = results.get(zml.Bufferized(Output));
    defer output.product.deinit();
    defer output.loss.deinit();
    var shards = output.product.shards();
    while (shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        const values = local.constItems(f32);
        for (values) |value| {
            if (value != feature_count) return error.UnexpectedProduct;
        }
        std.debug.print(
            "sharded_product: device={d} slices={any} values={any}\n",
            .{
                shard.globalDeviceId(),
                shard.globalSlices().constSlice(),
                values[0..@min(values.len, 8)],
            },
        );
    }
    const loss = try output.loss.getValue(f32, io);
    if (loss != feature_count) {
        return error.UnexpectedLoss;
    }
    std.debug.print("replicated_loss={d}\n", .{loss});
    try platform.barrier("distributed-matmul-before-shutdown");
}
