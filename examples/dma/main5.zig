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

extern fn cudaHostRegister(ptr: **anyopaque, size: usize, flags: c_int) c_int;
extern fn cudaMallocHost(ptr: **anyopaque, size: usize) c_int;
extern fn cudaMemcpy(dst: ?*anyopaque, src: ?*anyopaque, count: usize, kind: usize) c_int;
extern fn cudaFreeHost(ptr: *anyopaque) c_int;

pub fn cudaMallocHostZig(size: usize) []u8 {
    var res: []u8 = undefined;
    _ = cudaMallocHost(@ptrCast(&(res.ptr)), size);
    return res[0..size];
}

pub fn cudaFreeHostZig(ptr: *anyopaque) void {
    _ = cudaFreeHost(ptr);
}

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    // return a.add(b);
    return a.withSharding(.{.k}).dot(b.withSharding(.{.k}), .{.k}).withSharding(.{.m});
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const HAllocator = @import("halloc.zig");
    const halloc: std.mem.Allocator = .{
        .ptr = undefined,
        .vtable = &HAllocator.vtable,
    };

    var context = try zml.Context.init();
    defer context.deinit();

    // Auto-select platform
    const platform = context.autoPlatform(.{}).withCompilationOptions(.{
        .sharding_enabled = false,
    });
    context.printAvailablePlatforms(platform);

    const size = 8192;
    const dtype: zml.DataType = .f16;

    const a_shape = zml.Shape.init(.{ size, size }, dtype).withTags(.{ .m, .k }).withSharding(.{.k});
    const b_shape = a_shape.withTags(.{ .k, .n }).withSharding(.{.k});

    var compilation = try asynk.asyncc(zml.compileFn, .{ allocator, halloc, benchmark, .{ a_shape, b_shape }, platform });

    // Wait for compilation to finish
    const executable = try compilation.awaitt();
    defer executable.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    // bazel run -c opt --@zml//runtimes:cuda=true --@zml//runtimes:cpu=false --run_under="sudo /opt/nvidia/nsight-systems-cli/2025.1.1/bin/nsys profile -t cuda,nvtx,cublas,cublas-verbose,cusparse,cusparse-verbose,cudnn --gpu-metrics-devices='cuda-visible' --cuda-memory-usage true --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop" //dma
    const runtime = Runtime.init() catch unreachable;

    _ = runtime.cudaProfilerStart();

    var a_host_buffer = try createRandomBuffer(allocator, a_shape, random);
    defer a_host_buffer.deinit();

    var b_host_buffer = try createRandomBuffer(allocator, platform, b_shape, random);
    defer b_host_buffer.deinit();

    var a_buffer = try zml.Buffer.from(platform, a_host_buffer);
    defer a_buffer.deinit();
    var b_buffer = try zml.Buffer.from(platform, a_host_buffer);
    defer b_buffer.deinit();

    var result: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
    _ = try result.awaitt();
    _ = runtime.cudaProfilerStop();

    const result_host_buffer_data = cudaMallocHostZig(a_shape.byteSize());
    var hb = try result.toHost2(result_host_buffer_data);
    _ = try hb.awaitt();
    defer result.deinit();

    {
        const result_host_buffer1_data = cudaMallocHostZig(a_shape.byteSize());
        const result_host_buffer2_data = cudaMallocHostZig(a_shape.byteSize());
        // const result_host_buffer2_data = try halloc.alloc(u8, a_shape.byteSize());
        // const result_host_buffer3_data = cudaMallocHostZig(a_shape.byteSize());

        // var result_host_buffer1_data = try allocator.alloc(u8, a_shape.byteSize());
        // defer allocator.free(result_host_buffer1_data);
        // const result_host_buffer2_data = try allocator.alloc(u8, a_shape.byteSize());
        // defer allocator.free(result_host_buffer2_data);
        // const result_host_buffer3_data = try allocator.alloc(u8, a_shape.byteSize());
        // defer allocator.free(result_host_buffer3_data);

        // _ = runtime.cudaProfilerStart();
        // const profile: ?[]const u8 = "/home/zml/monorepo-hugo/profileXX.json";
        // var profiler = if (profile != null) platform.getProfiler(null) else undefined;
        // defer profiler.deinit();

        // if (profile != null) profiler.start();

        // _ = cudaHostRegister(@ptrCast(&(result_host_buffer1_data.ptr)), result_host_buffer1_data.len, 2);

        var result1: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
        var hb1 = try result1.toHost2(result_host_buffer1_data);
        var result2: zml.Buffer = executable.call(.{ result1, b_buffer });
        var hb2 = try result2.toHost2(result_host_buffer2_data);
        // var result3: zml.Buffer = executable.call(.{ result, result2 });
        // var hb3 = try result3.toHost2(result_host_buffer3_data);
        // var result4: zml.Buffer = executable.call(.{ a_buffer, result1 });
        _ = try hb1.awaitt();
        _ = try hb2.awaitt();
        // _ = try hb3.awaitt();
        // var result3: zml.Buffer = executable.call(.{ result, result2 });
        // var hb4 = try result4.toHost2(result_host_buffer3_data);
        // _ = try hb4.awaitt();
        defer result1.deinit();
        // defer result2.deinit();
        // defer result3.deinit();
        // defer result4.deinit();
        // if (profile) |profile_file| {
        //     profiler.stop();
        //     try profiler.dumpAsJsonTo(allocator, std.fs.cwd(), profile_file);
        // }
    }
}

fn createRandomBuffer(allocator: std.mem.Allocator, shape: zml.Shape, random: std.Random) !zml.HostBuffer {
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

    // std.debug.print("data: {any}\n", .{std.mem.bytesAsSlice(f16, data)[0..100]});
    var host_buffer = zml.HostBuffer.fromBytes(shape, data);
    errdefer host_buffer.deinit(allocator);
    return host_buffer;
}
