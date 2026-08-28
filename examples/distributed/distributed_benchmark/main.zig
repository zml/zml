const std = @import("std");
const log = std.log;

const distributed_example = @import("distributed_example");
const zml = @import("zml");
const stdx = zml.stdx;

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k).withPartitioning(.{
        .m = .m,
        .n = .replicated,
    });
}

pub fn main(init: std.process.Init) !void {
    const CliArgs = struct {
        pub const help =
            \\Usage:
            \\  distributed_benchmark [--size=4096] [--dtype=f16]
            \\    COORDINATOR RANK PROCESS_COUNT NAMESPACE
        ;

        size: usize = 4096,
        dtype: zml.DataType = .f16,
        positional: struct {
            coordinator: []const u8,
            rank: usize,
            processCount: usize,
            namespace: []const u8,
        },
    };

    const allocator = init.gpa;
    const io = init.io;
    const cli_args = stdx.flags.parse(init.minimal.args, CliArgs);
    const args = cli_args.positional;
    if (args.processCount == 0 or
        args.rank >= args.processCount or
        args.namespace.len == 0)
    {
        return error.InvalidDistributedJob;
    }
    const job: distributed_example.Job = .{
        .coordinator_address = try .parseLiteral(args.coordinator),
        .process_index = args.rank,
        .process_count = args.processCount,
        .namespace = args.namespace,
    };

    var platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    try distributed_example.expectTopology(platform, 4, 2);
    if (cli_args.size == 0 or
        @mod(cli_args.size, job.process_count) != 0 or
        @mod(cli_args.size, platform.addressableDevices().len) != 0)
    {
        return error.InvalidMatrixSize;
    }

    if (job.process_index == 0) log.info("\n{f}", .{platform.fmtVerbose()});
    log.info(
        "rank={d} global_devices={d} local_devices={d}",
        .{
            job.process_index,
            platform.globalDevices().len,
            platform.addressableDevices().len,
        },
    );

    const benchmark_sharding =
        try platform.registerShardingWithStrategy(
            "distributed-benchmark",
            .mesh(.{
                .m = .low_bandwidth,
                .n = .high_bandwidth,
            }),
            .parseBindings(.{
                .m = .network,
                .n = .link,
            }),
        );
    const a_shape = zml.Shape.init(.{
        .m = cli_args.size,
        .k = cli_args.size,
    }, cli_args.dtype).withPartitioning(.{
        .m = .m,
        .k = .replicated,
    });
    const b_shape = zml.Shape.init(.{
        .k = cli_args.size,
        .n = cli_args.size,
    }, cli_args.dtype).withPartitioning(.{
        .k = .replicated,
        .n = .n,
    });

    var executable = blk: {
        log.info("rank={d} compiling benchmark", .{job.process_index});
        const now: std.Io.Timestamp = .now(io, .awake);
        defer log.info(
            "rank={d} compiled benchmark [{f}]",
            .{ job.process_index, now.untilNow(io, .awake) },
        );
        break :blk try platform.compileFn(
            allocator,
            io,
            benchmark,
            .{
                zml.Tensor.fromShape(a_shape),
                zml.Tensor.fromShape(b_shape),
            },
            .{
                .shardings = &.{benchmark_sharding},
                .program_name = "distributed-benchmark",
            },
        );
    };
    defer executable.deinit();

    // Matching seeds keep host-replicated shards identical on every process.
    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();
    var a_buffer = try createRandomBuffer(
        allocator,
        io,
        platform,
        a_shape,
        benchmark_sharding,
        random,
    );
    defer a_buffer.deinit();
    var b_buffer = try createRandomBuffer(
        allocator,
        io,
        platform,
        b_shape,
        benchmark_sharding,
        random,
    );
    defer b_buffer.deinit();
    try distributed_example.expectShardCounts(&a_buffer, 4, 2);
    try distributed_example.expectShardCounts(&b_buffer, 4, 2);
    try distributed_example.expectAddressable(platform, &a_buffer);
    try distributed_example.expectAddressable(platform, &b_buffer);

    var executable_arguments = try executable.args(allocator);
    defer executable_arguments.deinit(allocator);
    var executable_results = try executable.results(allocator);
    defer executable_results.deinit(allocator);
    executable_arguments.set(.{ a_buffer, b_buffer });

    {
        executable.callOpts(
            io,
            executable_arguments,
            &executable_results,
            .{ .wait = true },
        );
        var warmup = executable_results.get(zml.Buffer);
        defer warmup.deinit();
    }

    try platform.barrier("distributed-benchmark-ready");
    const run_start: std.Io.Timestamp = .now(io, .awake);
    executable.callOpts(
        io,
        executable_arguments,
        &executable_results,
        .{ .wait = true },
    );
    var result = executable_results.get(zml.Buffer);
    defer result.deinit();
    try platform.barrier("distributed-benchmark-finished");
    const elapsed = run_start.untilNow(io, .awake);

    try distributed_example.expectShardCounts(&result, 4, 2);
    try distributed_example.expectAddressable(platform, &result);
    if (job.process_index == 0) {
        const elapsed_s = @as(
            f64,
            @floatFromInt(elapsed.toNanoseconds()),
        ) / std.time.ns_per_s;
        const size: f64 = @floatFromInt(cli_args.size);
        const flops = 2.0 * size * size * size / elapsed_s;
        log.info(
            "Dot product size: {d}x{d} - Datatype: {s} - " ++
                "Elapsed: {f} - {d:.3} GFLOP/s",
            .{
                cli_args.size,
                cli_args.size,
                @tagName(cli_args.dtype),
                elapsed,
                flops / 1_000_000_000,
            },
        );
    }
    try platform.barrier("distributed-benchmark-before-shutdown");
}

fn createRandomBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
    random: std.Random,
) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    switch (shape.dtype()) {
        inline else => |value_type| {
            const ZigType = value_type.toZigType();
            switch (comptime value_type.class()) {
                .bool, .complex => unreachable,
                .integer => {
                    for (slice.items(ZigType)) |*value| {
                        value.* = random.int(ZigType);
                    }
                },
                .float => {
                    const value = random.float(f32);
                    for (slice.items(ZigType)) |*element| {
                        element.* = switch (ZigType) {
                            f64, f32 => value,
                            f16 => @floatCast(value),
                            inline else => |T| if (@hasDecl(T, "fromF32"))
                                T.fromF32(value)
                            else
                                unreachable,
                        };
                    }
                },
            }
        },
    }

    return .fromSlice(io, platform, slice, sharding);
}
