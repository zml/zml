const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");
const flags = @import("tigerbeetle/flags");

const async_ = asynk.async_;

/// Model definition
const Benchmark = struct {
    pub fn forward(self: Benchmark, a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        _ = self;
        return a.dot(b, .{.k});
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try asynk.AsyncThread.main(gpa.allocator(), asyncMain, .{});
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

    // Arena allocator for BufferStore etc.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    // Auto-select platform
    const platform = context.autoPlatform();
    {
        // List available targets
        std.debug.print("Available Platforms:\n", .{});
        const selected_prefix = "✅";
        const not_selected_prefix = "• ";
        const selected_postfix = "(AUTO-SELECTED)\n";
        const not_selected_postfix = "\n";
        for (zml.platform.available_targets) |target| {
            std.debug.print("  {s} {s} {s}", .{
                if (target == platform.target) selected_prefix else not_selected_prefix,
                @tagName(target),
                if (target == platform.target) selected_postfix else not_selected_postfix,
            });

            // now the platform's devices
            if (context.platforms.get(target)) |pfm| {
                for (pfm.getDevices(), 0..) |device, index| {
                    const deviceKind = device.getDescription(platform.pjrt_api).getKind(platform.pjrt_api);
                    std.debug.print("       ◦ #{d}: {s}\n", .{
                        index,
                        deviceKind,
                    });
                    // we only list 1 CPU device
                    if (target == .cpu) break;
                }
            }
        }
        std.debug.print("\n", .{});
    }

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);

    const input_shape = zml.Shape.init(.{ cli_args.size, cli_args.size }, cli_args.dtype);

    var timer = try std.time.Timer.start();

    std.debug.print("\nCompiling model to MLIR....\n", .{});
    std.debug.print("-" ** 160 ++ "\n", .{});
    // Start compiling.
    // The shape of the input tensor, we have to pass in manually.
    timer.reset();
    var compilation = try async_(zml.module.compileModel, .{ allocator, Benchmark{}, .forward, .{ input_shape.withTags(.{ .m, .k }), input_shape.withTags(.{ .k, .n }) }, platform });

    // Wait for compilation to finish
    const compiled = try compilation.await_();
    const compilation_elapsed = timer.lap() / std.time.ns_per_ms;
    std.debug.print("-" ** 160 ++ "\n\n", .{});
    std.debug.print("✅ Compiled Benchmark model in {d} milliseconds! \n", .{compilation_elapsed});

    // pass the model weights to the compiled module to create an executable module
    var executable = try compiled.prepare(arena, .{});
    defer executable.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var a_buffer = try createRandomBuffer(allocator, platform, input_shape, random);
    defer a_buffer.deinit();
    var b_buffer = try createRandomBuffer(allocator, platform, input_shape, random);
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
    return zml.Buffer.from(platform, host_buffer);
}
