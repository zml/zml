const std = @import("std");
const zml = @import("zml");
const stdx = @import("stdx");
const flags = stdx.flags;

// set log level to debug to print the generated IR
pub const std_options: std.Options = .{
    .log_level = .warn,
};

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k);
}

pub fn main() !void {
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

    var threaded: std.Io.Threaded = .init(allocator);
    defer threaded.deinit();

    const io = threaded.io();

    zml.init();
    defer zml.deinit();

    // Auto-select platform
    const platform: zml.Platform = try .auto(io, .{});

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);

    const a = zml.Tensor.init(.{ cli_args.size, cli_args.size }, cli_args.dtype).withTags(.{ .m, .k });
    const b = zml.Tensor.init(.{ cli_args.size, cli_args.size }, cli_args.dtype).withTags(.{ .k, .n });
    var timer = try std.time.Timer.start();

    std.debug.print("\nCompiling model to MLIR....\n", .{});
    std.debug.print("-" ** 160 ++ "\n", .{});

    timer.reset();

    var exe = try platform.compile(allocator, io, benchmark, .{ a, b });
    defer exe.deinit();

    const compilation_elapsed = timer.lap() / std.time.ns_per_ms;
    std.debug.print("-" ** 160 ++ "\n\n", .{});
    std.debug.print("✅ Compiled Benchmark model in {d} milliseconds! \n", .{compilation_elapsed});

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var a_buffer = try createRandomBuffer(allocator, io, platform, a.shape(), random);
    defer a_buffer.deinit();
    var b_buffer = try createRandomBuffer(allocator, io, platform, b.shape(), random);
    defer b_buffer.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ a_buffer, b_buffer });

    std.debug.print("\nRunning benchmark....\n", .{});

    // Ignore first run
    {
        exe.call(exe_args, &exe_results);
        var result = exe_results.get(zml.Buffer);
        defer result.deinit();
    }

    // call our executable module
    timer.reset();
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    const elapsed_ns = timer.lap();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;

    std.debug.print("\n✅ Benchmark done!\n\n", .{});

    const floating_op_count = 2 * cli_args.size * cli_args.size * cli_args.size;
    const flops = @as(f64, @floatFromInt(floating_op_count)) / elapsed_s;
    std.debug.print("Dot product size: {d}x{d} - Datatype: {s} - Elapsed: {d:.3}ms - {d:.3} GFLOP/s\n\n", .{ cli_args.size, cli_args.size, @tagName(cli_args.dtype), elapsed_ms, flops / 1_000_000_000 });
}

fn createRandomBuffer(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, shape: zml.Shape, random: std.Random) !zml.Buffer {
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

    return .fromSlice(io, platform, slice);
}
