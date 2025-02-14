const std = @import("std");
const os = std.os;
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

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

const Input = struct {
    slice: zml.Buffer,
    index: ?zml.Buffer,
    token: ?zml.Buffer,
};

const OutputTensors = struct {
    slice: zml.Tensor,
    index: zml.Tensor,
    token: zml.Tensor,
};

const Buffers = struct {
    slice: zml.Buffer,
    index: zml.Buffer,
    token: zml.Buffer,
};

const Execute = asynk.Channel(Input, 16);
const PopResult = asynk.Channel(zml.Buffer, 1);

pub fn program(input: zml.Tensor, index: ?zml.Tensor, token: ?zml.Tensor) OutputTensors {
    const idx: zml.Tensor = index orelse zml.Tensor.scalar(0, .i32);
    const tok: zml.Tensor = token orelse zml.Tensor.scalar(0, .i32);

    var updated_slice = input.dynamicUpdateSlice(.{ .s = idx, .m = idx }, tok.convert(.bf16));

    const new_tok = tok.convert(.bf16).broadcastLeft(updated_slice.shape());
    for (0..10) |_| {
        updated_slice = updated_slice.matmul(new_tok);
    }

    return .{
        .slice = updated_slice.reuseBuffer(input),
        .index = idx.addConstant(1),
        .token = tok.addConstant(2),
    };
}

fn inference_loop(executable: anytype, platform: zml.Platform, popResultChannel: *PopResult, input: Buffers) !void {
    const slice = input.slice;
    var index = input.index;
    var token = input.token;

    while (true) {
        const result = executable.call(.{ slice, index, token });
        const buffer_pinned = try result.token.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
        try asynk.sleep(0);
        index = result.index;
        token = result.token;
        popResultChannel.send(buffer_pinned);
    }
}

fn results_loop(platform: zml.Platform, runtime_: Runtime, popResultChannel: *PopResult) !void {
    var i: usize = 0;
    while (popResultChannel.recv()) |buffer| {
        var buffer_pinned = buffer;
        defer buffer_pinned.deinit();
        _ = try buffer_pinned.awaitt();
        std.debug.print("result _memory: {s}\n", .{buffer_pinned.getMemory().debugString(platform.pjrt_api)});
        // const inMem = try buffer_pinned.getValueFromDataInMemory(i32);
        // std.debug.print("result: {any}\n", .{inMem});
        i += 1;

        if (i == 128) {
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

    const runtime = Runtime.init() catch unreachable;

    const size = 2048;
    const dtype: zml.DataType = .bf16;
    const i_dtype: zml.DataType = .i32;

    const slice_shape = zml.Shape.init(.{ .s = size, .m = size }, dtype);
    const index_shape = zml.Shape.init(.{}, i_dtype);
    const token_shape = zml.Shape.init(.{ .s = size, .m = 1 }, i_dtype);

    // Wait for compilation to finish
    const executable = try zml.compileFn(allocator, program, .{ slice_shape, index_shape, token_shape }, platform);
    defer executable.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    const slice_random = try createRandomBuffer(allocator, slice_shape, random, 1);
    defer allocator.free(slice_random);
    var slice_buffer = try zml.Buffer.fromToMemory(platform, slice_random, slice_shape, .device);
    defer slice_buffer.deinit();
    _ = try slice_buffer.awaitt();

    const index_random = try createRandomBuffer(allocator, index_shape, random, 0);
    defer allocator.free(index_random);
    var index_buffer = try zml.Buffer.fromToMemory(platform, index_random, index_shape, .device);
    defer index_buffer.deinit();
    _ = try index_buffer.awaitt();

    const token_random = try createRandomBuffer(allocator, token_shape, random, 0);
    defer allocator.free(token_random);
    var token_buffer = try zml.Buffer.fromToMemory(platform, token_random, token_shape, .device);
    defer token_buffer.deinit();
    _ = try token_buffer.awaitt();

    const input: Buffers = .{ .slice = slice_buffer, .index = index_buffer, .token = token_buffer };

    var result = executable.call(.{ input.slice, input.index, input.token });
    var result_slice_in_pinned_memory = try result.slice.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    _ = try result_slice_in_pinned_memory.awaitt();
    defer {
        result.slice.deinit();
        result.index.deinit();
        result.token.deinit();
    }

    std.debug.print("Warmup finished\n", .{});

    // var slice_buffer_2 = try slice_buffer.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    // _ = try slice_buffer_2.awaitt();

    // var index_buffer2 = try index_buffer.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    // _ = try index_buffer2.awaitt();

    // var token_buffer2 = try index_buffer.copyToMemory(platform, zml.Buffer.MemoryKind.pinned_host);
    // _ = try token_buffer2.awaitt();

    // _ = runtime.cudaProfilerStart();

    // const input2: Buffers = .{ .slice = slice_buffer_2, .index = index_buffer2, .token = token_buffer2 };
    // _ = input2; // autofix

    // std.debug.print("slice: {s}\n", .{slice_buffer_2._memory.debugString(platform.pjrt_api)});
    // std.debug.print("index: {s}\n", .{index_buffer2._memory.debugString(platform.pjrt_api)});
    // std.debug.print("token: {s}\n", .{token_buffer2._memory.debugString(platform.pjrt_api)});

    var popResultChannel = PopResult.init();

    _ = runtime.cudaProfilerStart();

    var frame1 = try asynk.asyncc(inference_loop, .{ &executable, platform, &popResultChannel, input });
    var frame2 = try asynk.asyncc(results_loop, .{ platform, runtime, &popResultChannel });
    _ = frame2.awaitt() catch unreachable;
    _ = frame1.awaitt() catch unreachable;
}

fn createRandomBuffer(allocator: std.mem.Allocator, shape: zml.Shape, random: std.Random, given_value: f32) ![]u8 {
    const data = try allocator.alloc(u8, shape.byteSize());

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    const value = given_value;
                    for (std.mem.bytesAsSlice(ZigType, data)) |*e| e.* = @intFromFloat(value);
                },
                .float => {
                    const value = given_value;
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
