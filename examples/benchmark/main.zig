const std = @import("std");

const asynk = @import("async");
const stdx = @import("stdx");
const flags = stdx.flags;
const zml = @import("zml");

// set log level to debug to print the generated IR
pub const std_options: std.Options = .{
    .log_level = .warn,
    .logFn = asynk.logFn(std.log.defaultLog),
};

pub fn benchmark(x: zml.Tensor, q_w: zml.Tensor, q_scale: zml.Tensor) zml.Tensor {
    return x.dot(dequantize(q_w, q_scale), .{.d});
}

fn dequantize(quantized_weight: zml.Tensor, q_scale: zml.Tensor) zml.Tensor {
    var fp4_encoding_buf: [16]zml.floats.BFloat16 = undefined;
    for (0..16) |i| {
        const i_u4: u4 = @truncate(i);
        const f4: zml.floats.Float4E2M1 = @bitCast(i_u4);
        fp4_encoding_buf[i] = .fromF32(f4.toF32());
    }
    std.log.warn("fp4: {d}", .{fp4_encoding_buf});
    const fp4_encoding: zml.Tensor = .constantTensor(.fromSlice(.{ .fp4 = 16 }, &fp4_encoding_buf));

    const scale: zml.Tensor = .pow(.constant(q_scale.shape(), .init(.bf16, 2)), q_scale.convert(.bf16));
    const actual_weight = fp4_encoding.gatherValues(.fp4, quantized_weight, .{});
    return actual_weight.mul(scale.broad(actual_weight.shape())).merge(.{ .d = .{ .block, .d } });
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.smp_allocator, asyncMain);
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
        .sharding_enabled = true,
        .xla_dump_to = "/tmp/zml/benchmark",
        .xla_dump_fusion_visualization = true,
    });
    context.printAvailablePlatforms(platform);

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);

    const x_shape: zml.Shape = .init(.{ .b = 16, .d = cli_args.size }, .bf16);
    const w_shape: zml.Shape = .init(.{ .d_out = cli_args.size, .block = @divExact(cli_args.size, 32), .d = 32 }, .i4);
    const s_shape: zml.Shape = .init(.{ .d_out = cli_args.size, .block = @divExact(cli_args.size, 32) }, .i8);

    var timer = try std.time.Timer.start();
    std.debug.print("\nCompiling model to MLIR....\n", .{});
    std.debug.print("-" ** 160 ++ "\n", .{});
    // Start compiling.
    // The shape of the input tensor, we have to pass in manually.
    timer.reset();
    var compilation = try asynk.asyncc(zml.compileFn, .{ allocator, benchmark, .{ x_shape, w_shape, s_shape }, platform });

    // Wait for compilation to finish
    const executable = try compilation.awaitt();
    defer executable.deinit();
    const compilation_elapsed = timer.lap() / std.time.ns_per_ms;
    std.debug.print("-" ** 160 ++ "\n\n", .{});
    std.debug.print("✅ Compiled Benchmark model in {d} milliseconds! \n", .{compilation_elapsed});

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var x = try createRandomBuffer(allocator, platform, x_shape, random);
    defer x.deinit();
    var w = try createRandomBuffer(allocator, platform, w_shape, random);
    defer w.deinit();
    var s = try createRandomBuffer(allocator, platform, s_shape, random);
    defer s.deinit();

    std.debug.print("\nRunning benchmark....\n", .{});

    // Ignore first run
    {
        var result: zml.Buffer = executable.call(.{ x, w, s });
        defer result.deinit();
    }

    // call our executable module
    timer.reset();
    var result: zml.Buffer = executable.call(.{ x, w, s });
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
    return zml.Buffer.from(platform, host_buffer, .{});
}
