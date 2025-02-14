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
    return a.withSharding(.{.k}).dot(b.withSharding(.{.k}), .{.k}).withSharding(.{.m});
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const HAllocator = @import("halloc.zig");
    const halloc: std.mem.Allocator = .{
        .ptr = undefined,
        .vtable = &HAllocator.vtable,
    };

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

    const size = 4096;
    const dtype: zml.DataType = .f32;

    const a_shape = zml.Shape.init(.{ size, size }, dtype).withTags(.{ .m, .k }).withSharding(.{.k});
    const b_shape = a_shape.withTags(.{ .k, .n }).withSharding(.{.k});

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    {
        const data_a = try allocator.alloc(u8, a_shape.byteSize());
        const data_b = try allocator.alloc(u8, a_shape.byteSize());

        for (std.mem.bytesAsSlice(f32, data_a)) |*e| e.* = random.float(f32);
        for (std.mem.bytesAsSlice(f32, data_b)) |*e| e.* = random.float(f32);

        var host_buffer_a = zml.HostBuffer.fromBytes(a_shape, data_a);
        defer host_buffer_a.deinit(allocator);
        var buffer_a = try zml.Buffer.from(platform, host_buffer_a);
        defer buffer_a.deinit();
        _ = try buffer_a.awaitt();
        std.debug.print("buffer_a: {any}\n", .{buffer_a});

        var host_buffer_b = zml.HostBuffer.fromBytes(a_shape, data_a);
        defer host_buffer_b.deinit(allocator);
        var buffer_b = try zml.Buffer.from(platform, host_buffer_b);
        _ = try buffer_b.awaitt();
        std.debug.print("buffer_b: {any}\n", .{buffer_b});
        buffer_b.deinit();

        const result_host_buffer_data = cudaMallocHostZig(a_shape.byteSize());
        // defer result_host_buffer_data.deinit();
        // _ = cudaHostRegister(@ptrCast(&(result_host_buffer_data.ptr)), result_host_buffer_data.len, 2);

        var executable = try zml.compileFn(allocator, halloc, benchmark, .{ a_shape, b_shape }, platform);
        // Wait for compilation to finish
        // const executable = try compilation.awaitt();
        defer executable.deinit();

        var result: zml.Buffer = executable.call(.{ buffer_a, buffer_b });

        var result_host_buffer = try result.toHostEx(result_host_buffer_data, .{});
        _ = try result_host_buffer.awaitt();
        std.debug.print("result_host_buffer: {any}\n", .{result_host_buffer});
    }

    std.debug.print("end\n", .{});
}
