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

const mesh: zml.Mesh = .init(.{ .y = 4 });

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .{.k}); //.withSharding(mesh, .{ .y = .m }).
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

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);

    // Auto-select platform
    const platform = context.autoPlatform(.{ .cpu = .{ .cpu_device_count = mesh.numRequiredDevices() } }).withCompilationOptions(.{
        .sharding_enabled = true,
    });
    context.printAvailablePlatforms(platform);

    const shape: zml.Shape = .init(.{ cli_args.size, cli_args.size }, cli_args.dtype);
    const a_shape = shape.withTags(.{ .m, .k }).withPartitionning(.{ .y = .k });
    const b_shape = shape.withTags(.{ .k, .n }).withPartitionning(.{ .y = .k });

    const sharding: zml.Sharding = .init(mesh, a_shape);
    const a_small = sharding.shard();
    std.debug.print("{}\n", .{a_small});

    var timer = try std.time.Timer.start();

    std.debug.print("\nCompiling model to MLIR....\n", .{});
    std.debug.print("-" ** 160 ++ "\n", .{});
    // Start compiling.
    // The shape of the input tensor, we have to pass in manually.
    timer.reset();
    var compilation = try asynk.asyncc(zml.compileFn, .{ allocator, benchmark, .{ a_shape, b_shape }, mesh, platform });

    // Wait for compilation to finish
    const executable = try compilation.awaitt();
    defer executable.deinit();
    const compilation_elapsed = timer.lap() / std.time.ns_per_ms;
    std.debug.print("-" ** 160 ++ "\n\n", .{});
    std.debug.print("✅ Compiled Benchmark model in {d} milliseconds! \n", .{compilation_elapsed});

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var a_buffer = try createRandomBuffer(allocator, platform, a_shape, random);
    defer a_buffer.deinit();
    var b_buffer = try createRandomBuffer(allocator, platform, b_shape, random);
    defer b_buffer.deinit();

    std.debug.print("\nRunning benchmark....\n", .{});

    // Ignore first run
    {
        var result: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
        defer result.deinit();
    }

    // call our executable module
    timer.reset();
    var result: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
    defer result.deinit();
    const elapsed_ns = timer.lap();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;

    std.debug.print("\n✅ Benchmark done!\n\n", .{});

    const floating_op_count = 2 * cli_args.size * cli_args.size * cli_args.size;
    const flops = @as(f64, @floatFromInt(floating_op_count)) / elapsed_s;
    std.debug.print("Dot product size: {d}x{d} - Datatype: {s} - Elapsed: {d:.3}ms - {d:.3} GFLOP/s\n\n", .{ cli_args.size, cli_args.size, @tagName(cli_args.dtype), elapsed_ms, flops / 1_000_000_000 });
}

fn createRandomBuffer(allocator: std.mem.Allocator, platform: zml.Platform, shape: zml.Shape, random: std.Random) !zml.Buffer {
    const data = try allocator.alloc(u8, shape.byteSize());
    std.debug.print("Creating random buffer of size {d} ptr {*} bytes for shape: {s}\n", .{ data.len, data.ptr, shape });
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
    return zml.Buffer.from(platform, mesh, host_buffer);
}
