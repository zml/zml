const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k).withPartitioning(.{ .m = .m, .n = .replicated });
}

pub fn main(init: std.process.Init) !void {
    const CliArgs = struct {
        pub const help =
            \\ benchmark --size=4096 --dtype=f16
        ;
        size: usize = 4096,
        dtype: zml.DataType = .f16,
    };

    const allocator = init.gpa;
    const io = init.io;

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const logical_mesh: zml.sharding.LogicalMesh = .init("benchmark_mesh", .{
        .m = .low_bandwidth,
        .n = .high_bandwidth,
    });
    const strategy: zml.sharding.Strategy = try .suggest(logical_mesh, platform.physical_mesh);
    const benchmark_sharding: zml.sharding.Sharding = try .initFromStrategy(platform, logical_mesh, strategy);

    const cli_args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    const a_shape = zml.Shape.init(.{ .m = cli_args.size, .k = cli_args.size }, cli_args.dtype)
        .withPartitioning(.{ .m = .m, .k = .replicated });
    const b_shape = zml.Shape.init(.{ .k = cli_args.size, .n = cli_args.size }, cli_args.dtype)
        .withPartitioning(.{ .k = .replicated, .n = .n });

    const a: zml.Tensor = .fromShape(a_shape);
    const b: zml.Tensor = .fromShape(b_shape);

    var exe = blk: {
        log.info("⏱️ Compiling benchmark...", .{});
        const now: std.Io.Timestamp = .now(io, .awake);
        defer log.info("✅ Compiled benchmark [{f}]", .{now.untilNow(io, .awake)});
        break :blk try platform.compileFn(allocator, io, benchmark, .{ a, b }, .{ .shardings = &.{benchmark_sharding} });
    };
    defer exe.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var a_buffer = try createRandomBuffer(allocator, io, platform, a.shape(), benchmark_sharding, random);
    defer a_buffer.deinit();
    var b_buffer = try createRandomBuffer(allocator, io, platform, b.shape(), benchmark_sharding, random);
    defer b_buffer.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ a_buffer, b_buffer });

    log.info("⏱️ Running benchmark...", .{});

    // Ignore first run
    {
        exe.call(exe_args, &exe_results);
        var result = exe_results.get(zml.Buffer);
        defer result.deinit();
    }

    // call our executable module
    const run_start: std.Io.Timestamp = .now(io, .awake);
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    _ = try result.await(io);
    defer result.deinit();
    const elapsed = run_start.untilNow(io, .awake);
    const elapsed_ns = elapsed.toNanoseconds();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;

    log.info("✅ Benchmark done!", .{});

    const floating_op_count = 2 * cli_args.size * cli_args.size * cli_args.size;
    const flops = @as(f64, @floatFromInt(floating_op_count)) / elapsed_s;
    log.info("Dot product size: {d}x{d} - Datatype: {s} - Elapsed: {f} - {d:.3} GFLOP/s", .{
        cli_args.size,
        cli_args.size,
        @tagName(cli_args.dtype),
        elapsed,
        flops / 1_000_000_000,
    });
}

fn createRandomBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, sharding: zml.sharding.Sharding, random: std.Random) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    for (slice.items(ZigType)) |*e| e.* = random.int(ZigType);
                },
                .float => {
                    const value = random.float(f32);
                    for (slice.items(ZigType)) |*e| e.* = switch (ZigType) {
                        f64, f32 => value,
                        f16 => @floatCast(value),
                        inline else => |T| if (@hasDecl(T, "fromF32")) T.fromF32(value) else unreachable,
                    };
                },
                .complex => unreachable,
            }
        },
    }

    return .fromSlice(io, platform, slice, sharding);
}
