const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");
const stdx = @import("stdx");

// set log level to debug to print the generated IR
pub const std_options = .{
    .log_level = .info,
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

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    var output = a;
    // for (0..10) |_| {
    //     output = a.mul(b);
    // }
    // return output.toMemory(.pinned_host);
    for (0..10) |_| {
        output = output.add(a.dot(b, .{.k}));
    }
    return output.toMemory(.pinned_host);
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
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

    // const profile: ?[]const u8 = "/home/zml/monorepo-hugo/profile-hugo.json";
    // var profiler = if (profile != null) platform.getProfiler(null) else undefined;
    // defer profiler.deinit();

    const runtime = Runtime.init() catch unreachable;

    const size = 8192;
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
    std.debug.print("buffer_a: {any}\n", .{buffer_a});

    const random_buffer_b = try createRandomBuffer(allocator, a_shape, random);
    defer allocator.free(random_buffer_b);
    var buffer_b = try zml.Buffer.fromToMemory(platform, random_buffer_a, a_shape, .device);
    defer buffer_b.deinit();
    _ = try buffer_b.awaitt();
    std.debug.print("buffer_b: {any}\n", .{buffer_b});

    const result_host_buffer_data = try allocator.alloc(u8, a_shape.byteSize());
    defer allocator.free(result_host_buffer_data);
    _ = runtime.cudaProfilerStart();
    var result: zml.Buffer = executable.call(.{ buffer_a, buffer_b });
    std.debug.print("result _memory: {s}\n", .{result.getMemory().debugString(platform.pjrt_api)});
    const hb = try result.copyToMemory(platform, zml.Buffer.MemoryKind.device);
    var result2: zml.Buffer = executable.call(.{ hb, buffer_b });
    std.debug.print("result2 _memory: {s}\n", .{result2.getMemory().debugString(platform.pjrt_api)});
    // _ = try hb.awaitt();
    defer result.deinit();
    _ = runtime.cudaProfilerStop();
    std.debug.print("Warmup finished\n", .{});

    // try asynk.sleep(10_000);

    // {
    //     const result_host_buffer1_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer1_data);
    //     const result_host_buffer2_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer2_data);
    //     const result_host_buffer3_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer3_data);

    //     if (profile != null) profiler.start();

    //     var result1: zml.Buffer = executable.call(.{ buffer_a, buffer_b });
    //     defer result1.deinit();

    //     std.debug.print("result1 _memory: {s}\n", .{result1._memory.debugString(platform.pjrt_api)});
    //     var hb1 = try result1.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    //     _ = try hb1.awaitt();
    //     std.debug.print("hb1 _memory: {s}\n", .{hb1._memory.debugString(platform.pjrt_api)});

    //     if (profile) |profile_file| {
    //         profiler.stop();
    //         try profiler.dumpAsJsonTo(allocator, std.fs.cwd(), profile_file);
    //     }
    // }

    // {
    //     const result_host_buffer1_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer1_data);
    //     const result_host_buffer2_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer2_data);
    //     const result_host_buffer3_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer3_data);

    //     // if (profile != null) profiler.start();
    //     _ = runtime.cudaProfilerStart();

    //     var result1: zml.Buffer = executable.call(.{ buffer_a, buffer_b });
    //     defer result1.deinit();
    //     var result2: zml.Buffer = executable.call(.{ result1, buffer_b });
    //     defer result2.deinit();
    //     var result3: zml.Buffer = executable.call(.{ buffer_a, buffer_b });
    //     defer result3.deinit();
    //     var result4: zml.Buffer = executable.call(.{ result1, result2 });
    //     defer result4.deinit();

    //     var hb1 = try result1.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    //     var hb2 = try result2.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    //     var hb3 = try result3.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    //     var hb4 = try result4.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);

    //     _ = try hb1.awaitt();
    //     _ = try hb2.awaitt();
    //     _ = try hb3.awaitt();
    //     _ = try hb4.awaitt();

    //     _ = runtime.cudaProfilerStop();

    //     // if (profile) |profile_file| {
    //     //     profiler.stop();
    //     //     try profiler.dumpAsJsonTo(allocator, std.fs.cwd(), profile_file);
    //     // }
    // }

    // {
    //     const result_host_buffer1_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer1_data);
    //     const result_host_buffer2_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer2_data);
    //     const result_host_buffer3_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer3_data);

    //     // if (profile != null) profiler.start();
    //     _ = runtime.cudaProfilerStart();

    //     var result1: zml.Buffer = executable.call(.{ buffer_a, buffer_b });
    //     defer result1.deinit();

    //     var hb1 = try result1.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);

    //     var result2: zml.Buffer = executable.call(.{ result1, buffer_b });
    //     defer result2.deinit();

    //     var hb2 = try result2.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);

    //     var result3: zml.Buffer = executable.call(.{ buffer_a, buffer_b });
    //     defer result3.deinit();

    //     _ = try hb1.awaitt();
    //     _ = try hb2.awaitt();

    //     var result4: zml.Buffer = executable.call(.{ result1, result2 });
    //     defer result4.deinit();

    //     var hb3 = try result3.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    //     var hb4 = try result4.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);

    //     _ = try hb3.awaitt();
    //     _ = try hb4.awaitt();

    //     _ = runtime.cudaProfilerStop();

    //     // if (profile) |profile_file| {
    //     //     profiler.stop();
    //     //     try profiler.dumpAsJsonTo(allocator, std.fs.cwd(), profile_file);
    //     // }
    // }

    // {
    //     const result_host_buffer1_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer1_data);
    //     const result_host_buffer2_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer2_data);
    //     const result_host_buffer3_data = try allocator.alloc(u8, a_shape.byteSize());
    //     defer allocator.free(result_host_buffer3_data);

    //     if (profile != null) profiler.start();

    //     var result1: zml.Buffer = executable.call(.{ buffer_a, buffer_b });
    //     defer result1.deinit();

    //     var hb1 = try result1.toHostEx(result_host_buffer1_data, .{});

    //     var result2: zml.Buffer = executable.call(.{ result1, buffer_b });
    //     defer result2.deinit();

    //     var result3: zml.Buffer = executable.call(.{ buffer_a, buffer_b });
    //     defer result3.deinit();

    //     var hb2 = try result2.toHostEx(result_host_buffer2_data, .{});

    //     _ = try hb1.awaitt();
    //     _ = try hb2.awaitt();

    //     var result4: zml.Buffer = executable.call(.{ result1, result2 });
    //     defer result4.deinit();

    //     var hb3 = try result3.toHostEx(result_host_buffer3_data, .{});
    //     var hb4 = try result4.toHostEx(result_host_buffer3_data, .{});

    //     _ = try hb3.awaitt();
    //     _ = try hb4.awaitt();

    //     if (profile) |profile_file| {
    //         profiler.stop();
    //         try profiler.dumpAsJsonTo(allocator, std.fs.cwd(), profile_file);
    //     }
    // }
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
