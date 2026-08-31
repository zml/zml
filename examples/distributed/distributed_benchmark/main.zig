const std = @import("std");
const log = std.log;

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
    if (cli_args.size == 0 or
        @mod(cli_args.size, args.processCount) != 0 or
        @mod(cli_args.size, platform.addressableDevices().len) != 0)
    {
        return error.InvalidMatrixSize;
    }

    if (args.rank == 0) log.info("\n{f}", .{platform.fmtVerbose()});
    log.info(
        "rank={d} global_devices={d} local_devices={d}",
        .{
            args.rank,
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
        log.info("rank={d} compiling benchmark", .{args.rank});
        const now: std.Io.Timestamp = .now(io, .awake);
        defer log.info(
            "rank={d} compiled benchmark [{f}]",
            .{ args.rank, now.untilNow(io, .awake) },
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

    const host_a = try zml.Slice.alloc(allocator, a_shape);
    defer host_a.free(allocator);
    @memset(host_a.data(), 0);
    const host_b = try zml.Slice.alloc(allocator, b_shape);
    defer host_b.free(allocator);
    @memset(host_b.data(), 0);
    var a_buffer = try zml.Buffer.fromSlice(
        io,
        platform,
        host_a,
        benchmark_sharding,
    );
    defer a_buffer.deinit();
    var b_buffer = try zml.Buffer.fromSlice(
        io,
        platform,
        host_b,
        benchmark_sharding,
    );
    defer b_buffer.deinit();

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

    if (args.rank == 0) {
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
