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

// __host__​cudaError_t cudaMallocHost ( void** ptr, size_t size )

extern fn cudaMallocHost(ptr: **anyopaque, size: usize) c_int;
extern fn cudaMemcpy(dst: ?*anyopaque, src: ?*anyopaque, count: usize, kind: usize) c_int;

pub fn cudaMallocHostZig(size: usize) []u8 {
    var res: []u8 = undefined;
    _ = cudaMallocHost(@ptrCast(&(res.ptr)), size);
    return res[0..size];
}

pub fn toHost(buf: zml.Buffer, dst: []u8) !void {
    const shard = buf._shards.get(0);
    const pjrt_buf = shard.buffer;
    const odp = pjrt_buf.getOpaqueDeviceMemoryDataPointer(shard.api) catch unreachable;
    // const src = @ptrCast(odp);
    _ = cudaMemcpy(dst.ptr, odp, dst.len, 2);
}

pub fn benchmark(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
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

    var context = try zml.Context.init();
    defer context.deinit();

    // Auto-select platform
    const platform = context.autoPlatform(.{}).withCompilationOptions(.{
        .sharding_enabled = false,
    });
    context.printAvailablePlatforms(platform);

    const size = 8192;
    const dtype: zml.DataType = .bf16;

    // const profile: ?[]const u8 = "/home/zml/monorepo-hugo/profile.json";
    // var profiler = if (profile != null) platform.getProfiler(null) else undefined;
    // defer profiler.deinit();

    // if (profile != null) profiler.start();

    const a_shape = zml.Shape.init(.{ size, size }, dtype).withTags(.{ .m, .k }).withSharding(.{.k});
    const b_shape = a_shape.withTags(.{ .k, .n }).withSharding(.{.k});

    var compilation = try asynk.asyncc(zml.compileFn, .{ allocator, benchmark, .{ a_shape, b_shape }, platform });

    // Wait for compilation to finish
    const executable = try compilation.awaitt();
    defer executable.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var a_buffer = try createRandomBuffer(allocator, platform, a_shape, random);
    defer a_buffer.deinit();
    var b_buffer = try createRandomBuffer(allocator, platform, b_shape, random);
    defer b_buffer.deinit();

    const result_host_buffer_data = cudaMallocHostZig(a_shape.byteSize());

    // const result_host_buffer_data = try allocator.alloc(u8, a_shape.byteSize());
    // defer allocator.free(result_host_buffer_data);

    const result_host_buffer2_data = try allocator.alloc(u8, a_shape.byteSize());
    defer allocator.free(result_host_buffer2_data);
    const result_host_buffer3_data = try allocator.alloc(u8, a_shape.byteSize());
    defer allocator.free(result_host_buffer3_data);

    // Ignore first run
    {
        var result: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
        defer result.deinit();
        try asynk.sleep(1000); // to see more quickly stuffs in the profiling ui
    }

    // for (0..10) |i| {
    //     _ = i; // autofix
    // bazel run -c opt --@zml//runtimes:cuda=true --@zml//runtimes:cpu=false --run_under="sudo /opt/nvidia/nsight-systems-cli/2025.1.1/bin/nsys profile -t cuda,nvtx,cublas,cublas-verbose,cusparse,cusparse-verbose,cudnn --gpu-metrics-devices='cuda-visible' --cuda-memory-usage true --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop" //dma
    const runtime = Runtime.init() catch unreachable;

    {
        var result: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
        // var result2: zml.Buffer = executable.call(.{ result, b_buffer });
        // var result3: zml.Buffer = executable.call(.{ result, result2 });
        // var timer = try stdx.time.Timer.start();
        _ = runtime.cudaProfilerStart();
        // var hb = try result.toHost(result_host_buffer_data);
        try toHost(result, result_host_buffer_data);
        // std.log.info("Time result.toHost: {}\n", .{timer.read()});
        // var hb2 = try result2.toHost(result_host_buffer2_data);
        // std.log.info("Time result2.toHost: {}\n", .{timer.read()});
        // var hb3 = try result3.toHost(result_host_buffer3_data);
        // std.log.info("Time result3.toHost: {}\n", .{timer.read()});
        // _ = try hb.awaitt();
        // std.log.info("Time hb.awaitt: {}\n", .{timer.read()});
        // _ = try hb2.awaitt();
        // std.log.info("Time hb2.awaitt: {}\n", .{timer.read()});
        // _ = try hb3.awaitt();
        // std.log.info("Time hb3.awaitt: {}\n", .{timer.read()});
        _ = runtime.cudaProfilerStop();
        defer result.deinit();
        // defer result2.deinit();
        // defer result3.deinit();
    }
    // }
    std.log.info("Time hb3.awaitt: \n", .{});

    try asynk.sleep(1000); // to see more quickly stuffs in the profiling ui

    // if (profile) |profile_file| {
    //     profiler.stop();
    //     try profiler.dumpAsJsonTo(allocator, std.fs.cwd(), profile_file);
    // }
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
