const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k);
}

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    zml.init();
    defer zml.deinit();

    // Auto-select platform
    const platform: zml.Platform = try .auto(io, .{});

    const a: zml.Tensor = .init(.{ .m = 128, .k = 128 }, .f16);
    const b: zml.Tensor = .init(.{ .k = 128, .n = 128 }, .f16);

    var exe = blk: {
        log.info("⏱️ Compiling benchmark...", .{});
        var timer = try std.time.Timer.start();
        defer log.info("✅ Compiled benchmark [{D}]", .{timer.read()});
        break :blk try platform.compileFn(allocator, io, benchmark, .{ a, b });
    };
    defer exe.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();
    _ = random; // autofix

    var a_data: [128 * 128]f16 = undefined;
    for (a_data[0..], 0..) |*d, i| d.* = @floatCast(@as(f64, @floatFromInt(i)) / a_data.len);
    const a_slice: zml.ConstSlice = .init(a.shape(), std.mem.sliceAsBytes(&a_data));
    const a_buffer: zml.Buffer = try .fromSlice(io, platform, a_slice);
    defer a_buffer.deinit();

    var b_buffer: zml.Buffer = b: {
        var transfer: zml.io.Transfer = try .init(allocator, &.{b.shape()}, platform);
        defer transfer.deinit(platform);

        const buffer = try allocator.alloc(u8, 128 * 128 * 2);
        defer allocator.free(buffer);

        var writer = try transfer.getWriter(io, 0, buffer);
        var reader: std.Io.Reader = .fixed(std.mem.sliceAsBytes(&a_data));

        _ = try reader.streamRemaining(&writer.interface);
        try writer.interface.flush();

        break :b try transfer.getBuffer(0);
    };
    defer b_buffer.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ a_buffer, a_buffer });

    // call our executable module
    exe.call(exe_args, &exe_results);
    var result1 = exe_results.get(zml.Buffer);
    _ = try result1.await(io);
    defer result1.deinit();

    const result1_slice = try result1.toSliceAlloc(allocator, io);
    defer result1_slice.free(allocator);

    std.log.info("result1: {d}", .{result1_slice});

    exe_args.set(.{ b_buffer, b_buffer });

    // call our executable module
    exe.call(exe_args, &exe_results);
    var result2 = exe_results.get(zml.Buffer);
    _ = try result2.await(io);
    defer result2.deinit();

    const result2_slice = try result2.toSliceAlloc(allocator, io);
    defer result2_slice.free(allocator);
    std.log.info("result2: {d}", .{result2_slice});
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
