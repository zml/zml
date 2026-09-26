const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k).withPartitioning(.{ .m = .m, .n = .replicated });
}

pub fn addNegate(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.add(b).negate();
}

pub fn main(init: std.process.Init) !void {
    const CliArgs = struct {
        pub const help =
            \\ benchmark --size=4096 --dtype=f16
        ;
        size: usize = 4096,
        dtype: zml.DataType = .f16,
        operation: enum { matmul, add_negate } = .matmul,
        iterations: usize = 1,
    };

    const allocator = init.gpa;
    const io = init.io;

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    log.info("\n{f}", .{platform.fmtVerbose()});

    const benchmark_sharding: zml.Sharding = try platform.registerSharding("benchmark_mesh", .mesh(
        .{ .m = .low_bandwidth, .n = .high_bandwidth },
    ));

    const cli_args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);
    if (cli_args.iterations == 0) return error.InvalidIterations;
    if (cli_args.operation == .add_negate and cli_args.dtype != .f32) return error.AddNegateRequiresF32;

    const a_shape = switch (cli_args.operation) {
        .matmul => zml.Shape.init(.{ .m = cli_args.size, .k = cli_args.size }, cli_args.dtype)
            .withPartitioning(.{ .m = .m, .k = .replicated }),
        .add_negate => zml.Shape.init(.{ .m = cli_args.size }, cli_args.dtype)
            .withPartitioning(.{ .m = .m }),
    };
    const b_shape = switch (cli_args.operation) {
        .matmul => zml.Shape.init(.{ .k = cli_args.size, .n = cli_args.size }, cli_args.dtype)
            .withPartitioning(.{ .k = .replicated, .n = .n }),
        .add_negate => a_shape,
    };

    const a: zml.Tensor = .fromShape(a_shape);
    const b: zml.Tensor = .fromShape(b_shape);

    var exe = blk: {
        log.info("⏱️ Compiling benchmark...", .{});
        const now: std.Io.Timestamp = .now(io, .awake);
        defer log.info("✅ Compiled benchmark [{f}]", .{now.untilNow(io, .awake)});
        break :blk switch (cli_args.operation) {
            .matmul => try platform.compileFn(allocator, io, benchmark, .{ a, b }, .{ .shardings = &.{benchmark_sharding} }),
            .add_negate => try platform.compileFn(allocator, io, addNegate, .{ a, b }, .{ .shardings = &.{benchmark_sharding} }),
        };
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
        try result.await(io);
    }

    // call our executable module
    const run_start: std.Io.Timestamp = .now(io, .awake);
    var result: ?zml.Buffer = null;
    defer if (result) |*buffer| buffer.deinit();
    for (0..cli_args.iterations) |_| {
        if (result) |*buffer| buffer.deinit();
        exe.call(exe_args, &exe_results);
        result = exe_results.get(zml.Buffer);
        try result.?.await(io);
    }
    const elapsed = run_start.untilNow(io, .awake);
    const elapsed_ns = elapsed.toNanoseconds();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;

    log.info("✅ Benchmark done!", .{});

    if (cli_args.operation == .add_negate) {
        const lhs = try a_buffer.toSliceAlloc(allocator, io);
        defer lhs.free(allocator);
        const rhs = try b_buffer.toSliceAlloc(allocator, io);
        defer rhs.free(allocator);
        const actual = try result.?.toSliceAlloc(allocator, io);
        defer actual.free(allocator);
        for (lhs.items(f32), rhs.items(f32), actual.items(f32)) |a_value, b_value, value| {
            if (value != -(a_value + b_value)) return error.IncorrectAddNegate;
        }
        log.info("Verified every add/negate result against host inputs", .{});
    }
    const floating_op_count = switch (cli_args.operation) {
        .matmul => 2 * cli_args.size * cli_args.size * cli_args.size,
        .add_negate => 2 * cli_args.size,
    };
    const flops = @as(f64, @floatFromInt(floating_op_count * cli_args.iterations)) / elapsed_s;
    log.info("Operation: {s} - Size: {d} - Datatype: {s} - Iterations: {d} - Total: {f} - Mean: {d:.3} us - {d:.3} GFLOP/s", .{
        @tagName(cli_args.operation), cli_args.size,
        @tagName(cli_args.dtype),     cli_args.iterations,
        elapsed,                      elapsed_s * 1_000_000 / @as(f64, @floatFromInt(cli_args.iterations)),
        flops / 1_000_000_000,
    });
}

fn createRandomBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, sharding: zml.Sharding, random: std.Random) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toPackedZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    for (slice.items(ZigType)) |*e| e.* = @bitCast(random.int(@Int(.unsigned, @bitSizeOf(ZigType))));
                },
                .float => {
                    const value = random.float(f32);
                    for (slice.items(ZigType)) |*e| e.* = switch (ZigType) {
                        f64, f32 => value,
                        f16 => @floatCast(value),
                        zml.floats.Float4E2M1.Packed => .fromF32(value, -value),
                        inline else => |T| if (@hasDecl(T, "fromF32")) .fromF32(value) else unreachable,
                    };
                },
                .complex => unreachable,
            }
        },
    }

    return .fromSlice(io, platform, slice, sharding);
}
