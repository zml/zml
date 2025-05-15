const std = @import("std");
const zml = @import("zml");
const stdx = @import("stdx");
const asynk = @import("async");
const flags = stdx.flags;

// set log level to debug to print the generated IR
pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.withSharding(.{.k}).dot(b.withSharding(.{.k}), .{.k}).withSharding(.{.m});
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const CliArgs = struct {
        pub const help =
            \\ benchmark --size=4096 --dtype=f16
        ;
        size: usize = 4096,
        dtype: zml.DataType = .f16,
    };

    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    // Auto-select platform
    const platform = context.autoPlatform(.{}).withCompilationOptions(.{
        .sharding_enabled = false,
    });
    context.printAvailablePlatforms(platform);

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);

    const a_shape = zml.Shape.init(.{ cli_args.size, cli_args.size }, cli_args.dtype).withTags(.{ .m, .k }).withSharding(.{.k});
    const b_shape = a_shape.withTags(.{ .k, .n }).withSharding(.{.k});
    var timer: stdx.time.Timer = try .start();

    std.debug.print("\nCompiling model to MLIR....\n", .{});
    std.debug.print("-" ** 160 ++ "\n", .{});
    // Start compiling.
    // The shape of the input tensor, we have to pass in manually.
    timer.reset();
    var compilation = try asynk.asyncc(zml.compileFn, .{ allocator, benchmark, .{ a_shape, b_shape }, platform });

    // Wait for compilation to finish
    const executable = try compilation.awaitt();
    defer executable.deinit();
    const compilation_elapsed = timer.lap();
    std.debug.print("-" ** 160 ++ "\n\n", .{});
    std.debug.print("✅ Compiled Benchmark model in {}! \n", .{compilation_elapsed});

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var a_buffer = try createRandomBuffer(allocator, platform, a_shape, random);
    // _ = try a_buffer.awaitt();
    defer a_buffer.deinit();
    std.debug.print("2 \n", .{});
    var b_buffer = try createRandomBuffer(allocator, platform, b_shape, random);
    // _ = try b_buffer.awaitt();
    defer b_buffer.deinit();

    std.debug.print("\nRunning benchmark....\n", .{});

    // Ignore first run
    {
        var result: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
        // var pouet = try result.toMemory(.host_pinned);
        // _ = try pouet.awaitt();
        defer result.deinit();
        // defer pouet.deinit();
    }

    try asynk.sleep(3000);

    // call our executable module
    timer.reset();
    var result = a_buffer;
    for (0..10) |_| {
        result = executable.call(.{ result, b_buffer });
        _ = try result.awaitt();
        // var pouet = try result.toMemory(.host_pinned);
        // pouet = pouet;
    }

    // defer result.deinit();
    const elapsed = timer.lap();

    std.debug.print("\n✅ Benchmark done!\n\n", .{});

    const floating_op_count = 2 * cli_args.size * cli_args.size * cli_args.size;
    const flops: f64 = @floatFromInt(elapsed.hz() * floating_op_count);
    std.debug.print("Dot product size: {d}x{d} - Datatype: {s} - Elapsed: {} - {d:.3} GFLOP/s\n\n", .{ cli_args.size, cli_args.size, @tagName(cli_args.dtype), elapsed, flops / 1e9 });
}

fn createRandomBuffer(allocator: std.mem.Allocator, platform: zml.Platform, shape: zml.Shape, random: std.Random) !zml.Buffer {
    const data = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(data);

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    for (std.mem.bytesAsSlice(ZigType, data)) |*e| e.* = random.int(ZigType);
                },
                .float => {
                    const value = random.float(f64);
                    for (std.mem.bytesAsSlice(ZigType, data)) |*e| e.* = if (ZigType == f64)
                        value
                    else if (ZigType == f32)
                        @floatCast(value)
                    else if (ZigType == f16)
                        @floatCast(value)
                    else
                        @bitCast(random.int(std.meta.Int(.unsigned, @bitSizeOf(ZigType))));
                },
                .complex => unreachable,
            }
        },
    }

    var host_buffer = zml.HostBuffer.fromBytes(shape, data);
    errdefer host_buffer.deinit(allocator);
    // var b = zml.Buffer.asViewOfHostBuffer(platform, host_buffer);
    // return try b.toMemory(.device);
    return zml.Buffer.from(platform, host_buffer);
}
