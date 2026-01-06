const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

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

    const allocator = std.heap.smp_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    zml.init();
    defer zml.deinit();

    // Auto-select platform
    const platform: zml.Platform = try .auto(io, .{});

    const cli_args: CliArgs = blk: {
        var ret: CliArgs = .{};
        var it = std.process.args();
        defer it.deinit();
        while (it.next()) |arg| {
            if (std.mem.startsWith(u8, arg, "--dtype=")) {
                ret.dtype = std.meta.stringToEnum(zml.DataType, arg["--dtype=".len..]) orelse unreachable;
            } else if (std.mem.startsWith(u8, arg, "--size=")) {
                ret.size = try std.fmt.parseUnsigned(usize, arg["--size=".len..], 10);
            }
        }
        break :blk ret;
    };

    const a: zml.Tensor = .init(.{ .m = cli_args.size, .k = cli_args.size }, cli_args.dtype);
    const b: zml.Tensor = .init(.{ .k = cli_args.size, .n = cli_args.size }, cli_args.dtype);

    var exe = blk: {
        log.info("⏱️ Compiling benchmark...", .{});
        var timer = try std.time.Timer.start();
        defer log.info("✅ Compiled benchmark [{D}]", .{timer.read()});
        break :blk try platform.compileFn(allocator, io, benchmark, .{ a, b });
    };
    defer exe.deinit();

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

    log.info("⏱️ Running benchmark...", .{});

    // Ignore first run
    {
        exe.call(exe_args, &exe_results);
        var result = exe_results.get(zml.Buffer);
        defer result.deinit();
    }

    // call our executable module
    var timer = try std.time.Timer.start();
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    _ = try result.await(io);
    defer result.deinit();
    const elapsed_ns = timer.read();
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;

    log.info("✅ Benchmark done!", .{});

    const floating_op_count = 2 * cli_args.size * cli_args.size * cli_args.size;
    const flops = @as(f64, @floatFromInt(floating_op_count)) / elapsed_s;
    log.info("Dot product size: {d}x{d} - Datatype: {s} - Elapsed: {D} - {d:.3} GFLOP/s", .{
        cli_args.size,
        cli_args.size,
        @tagName(cli_args.dtype),
        elapsed_ns,
        flops / 1_000_000_000,
    });
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
