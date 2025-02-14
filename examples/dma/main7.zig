const std = @import("std");
const os = std.os;
const zml = @import("zml");
const asynk = @import("async");
const stdx = @import("stdx");

// set log level to debug to print the generated IR
pub const std_options = .{
    .log_level = .debug,
};

pub const Runtime = struct {
    cudaProfilerStart: CudaProfilerStart,
    cudaProfilerStop: CudaProfilerStop,

    const CudaProfilerStart = *const fn () callconv(.C) c_int;
    const CudaProfilerStop = *const fn () callconv(.C) c_int;

    pub fn init() !Runtime {
        var cudart = try std.DynLib.open("libcudart.so.12");
        defer cudart.close();

        return .{
            .cudaProfilerStart = cudart.lookup(Runtime.CudaProfilerStart, "cudaProfilerStart") orelse return error.NotFound,
            .cudaProfilerStop = cudart.lookup(Runtime.CudaProfilerStop, "cudaProfilerStop") orelse return error.NotFound,
        };
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

const Input = struct {
    a: zml.Buffer,
    b: zml.Buffer,
};

const Execute = asynk.Channel(Input, 16);
const PopResult = asynk.Channel(zml.Buffer, 1);

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    var output = a;
    for (0..10) |_| {
        output = output.add(a.dot(b, .{.k}));
    }
    return output; // .toMemory(.pinned_host)
    // var output = a;
    // for (0..10) |_| {
    //     output = a.mul(b);
    // }
    // return output.toMemory(.pinned_host);
    // return a.mul(b);
}

fn inference_loop(executable: anytype, platform: zml.Platform, popResultChannel: *PopResult, input: Input) !void {
    var current = input.a;
    while (true) {
        const result = executable.call(.{ current, input.b });
        const buffer_pinned = try result.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
        popResultChannel.send(buffer_pinned);
        current.deinit();
        current = result;
    }
}

fn results_loop(platform: zml.Platform, runtime_: Runtime, popResultChannel: *PopResult) !void {
    var i: usize = 0;
    while (popResultChannel.recv()) |buffer| {
        var buffer_pinned = buffer;
        defer buffer_pinned.deinit();
        _ = try buffer_pinned.awaitt();
        std.debug.print("--------------------------------\n", .{});
        std.debug.print("result _memory: {s}\n", .{buffer_pinned._memory.debugString(platform.pjrt_api)});
        // const data = try buffer_pinned.dataInMemory();
        // std.debug.print("result: {any}\n", .{data[0..100]});
        // std.debug.print("i: {d}\n", .{i});
        i += 1;

        if (i == 100) {
            _ = runtime_.cudaProfilerStop();
        }
    }
}

pub fn asyncMain() !void {
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

    var runtime = Runtime.init() catch unreachable;

    const size = 2048;
    const dtype: zml.DataType = .bf16;

    const a_shape = zml.Shape.init(.{ size, size }, dtype).withTags(.{ .m, .k }).withSharding(.{.k});
    const b_shape = a_shape.withTags(.{ .k, .n }).withSharding(.{.k});

    // Wait for compilation to finish
    const executable = try zml.compileFn(allocator, benchmark, .{ a_shape, b_shape }, platform);
    defer executable.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    const random_buffer_a = try createRandomBuffer(allocator, a_shape, random);
    defer allocator.free(random_buffer_a);
    var buffer_a = try zml.Buffer.fromToMemory(platform, random_buffer_a, a_shape, .device);
    defer buffer_a.deinit();
    _ = try buffer_a.awaitt();

    const random_buffer_b = try createRandomBuffer(allocator, a_shape, random);
    defer allocator.free(random_buffer_b);
    var buffer_b = try zml.Buffer.fromToMemory(platform, random_buffer_b, a_shape, .device);
    defer buffer_b.deinit();
    _ = try buffer_b.awaitt();

    std.debug.print("buffer_a _memory: {s}\n", .{buffer_a.getMemory().debugString(platform.pjrt_api)});
    std.debug.print("buffer_b _memory: {s}\n", .{buffer_b.getMemory().debugString(platform.pjrt_api)});

    var result: zml.Buffer = executable.call(.{ buffer_a, buffer_b });
    std.debug.print("result _memory: {s}\n", .{buffer_a.getMemory().debugString(platform.pjrt_api)});
    var hb = try result.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    _ = try hb.awaitt();
    hb.deinit();
    result.deinit();

    std.debug.print("Warmup finished\n", .{});

    // var popResultChannel = PopResult.init();
    // _ = popResultChannel; // autofix

    std.debug.print("buffer_a _memory: {s}\n", .{buffer_a.getMemory().debugString(platform.pjrt_api)});
    std.debug.print("buffer_b _memory: {s}\n", .{buffer_b.getMemory().debugString(platform.pjrt_api)});

    _ = runtime.cudaProfilerStart();

    var result2: zml.Buffer = executable.call(.{ buffer_a, buffer_b });
    var result_2_to_device = try result2.copyToMemory(platform, .pinned_host);
    var result3: zml.Buffer = executable.call(.{ result2, buffer_b });
    _ = try result3.awaitt();
    _ = try result_2_to_device.awaitt();
    // var result2_pinned = try result2.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    // _ = try result2.awaitt();
    // std.debug.print("result _memory: {s}\n", .{result2.getMemory().debugString(platform.pjrt_api)});
    // std.debug.print("result _memory kindId: {any}\n", .{result2.getMemory().id(platform.pjrt_api)});
    // const data = try result2.dataInMemory();

    // std.debug.print("result _memory value: {any}\n", .{std.mem.bytesAsSlice(f16, data)[0..100]});
    defer result2.deinit();
    defer result3.deinit();

    _ = runtime.cudaProfilerStop();

    try asynk.sleep(100);

    // var frame1 = try asynk.asyncc(inference_loop, .{ &executable, platform, &popResultChannel, .{ .a = buffer_a, .b = buffer_b } });
    // var frame2 = try asynk.asyncc(results_loop, .{ platform, runtime, &popResultChannel });
    // _ = frame2.awaitt() catch unreachable;
    // _ = frame1.awaitt() catch unreachable;
}

fn createRandomBuffer(allocator: std.mem.Allocator, shape: zml.Shape, random: std.Random) ![]u8 {
    const data = try allocator.alloc(u8, shape.byteSize());

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    for (std.mem.bytesAsSlice(ZigType, data)) |*e| e.* = random.int(ZigType);
                },
                .float => {
                    const value = 1;
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

    return data;
}
