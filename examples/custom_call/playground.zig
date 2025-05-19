const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");

const cuda = zml.context.cuda;
const ffi = zml.ffi;

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

const log = std.log.scoped(.@"examples/custom_call");

const CuLaunchHostFunc = *const fn (
    hStream: *anyopaque,
    fn_: *const fn (user_data: *const anyopaque) callconv(.c) void,
    userData: *anyopaque,
) callconv(.c) c_int;

const UserData = struct {
    value: *const ffi.FFIBuffer,
    platform: zml.Platform,
};

fn host_func(data: *const anyopaque) callconv(.c) void {
    const user_data = @as(*const UserData, @alignCast(@ptrCast(data)));
    const tracer = user_data.platform.tracer;
    const frame = tracer.frameStart("HostFunc");
    defer tracer.frameEnd(frame, "HostFunc");
    log.warn("HostFunc data: {any}", .{user_data.value});
    std.time.sleep(10 * std.time.ns_per_ms);
}

pub const AddOp = struct {
    const Self = @This();

    platform: zml.Platform,
    allocator: std.mem.Allocator,
    buffer: zml.Buffer,
    cuLaunchHostFunc: CuLaunchHostFunc,
    user_data: UserData = undefined,

    buffers: []*const ffi.FFIBuffer = undefined,
    stream: *ffi.FFIStream = undefined,

    pub fn call(self: *Self, value: *const ffi.FFIBuffer) !void {
        const tracer = self.platform.tracer;
        _ = tracer; // autofix

        self.user_data = .{ .value = value, .platform = self.platform };
        log.warn("user_data: {*}", .{&self.user_data});

        const res = self.cuLaunchHostFunc(
            self.stream,
            @ptrCast(&host_func),
            @ptrCast(&self.user_data),
        );

        log.warn("cuLaunchHostFunc: {any}", .{res});

        // std.time.sleep(1 * std.time.ns_per_ms);

        // const frame_as_view_of = tracer.frameStart("AddOp as view of");
        // const value_pinned = zml.Buffer.asViewOfDeviceBuffer(self.platform, ffi.getShape(value), null, value.data.asPtr());
        // log.warn("{s}@{*} value: {any}", .{ @typeName(Self), self, (try value_pinned.dataInMemory())[0..100] });
        // tracer.frameEnd(frame_as_view_of, "AddOp as view of");
        // const ptr = value.data.asPtr();
        // log.warn("{*} -> {*}", .{ ptr, self.buffers[0].data.asPtr() });
        // const frame_copy_to_memory = tracer.frameStart("AddOp copyToMemory");
        // const value_pinned = value_device.copyToMemory(self.platform, .host_pinned) catch unreachable;
        // tracer.frameEnd(frame_copy_to_memory, "AddOp copyToMemory");

        // const frame_await = tracer.frameStart("AddOp await");
        // _ = value_pinned.awaitt() catch unreachable;
        // tracer.frameEnd(frame_await, "AddOp await");

        // const frame_memcpy = tracer.frameStart("AddOp memcpy");
        // cuda.memcpyToDeviceAsync(self.buffers[0].data, value_pinned.asPinnedHostBuffer().data, self.stream);
        // tracer.frameEnd(frame_memcpy, "AddOp memcpy");
    }
};

pub const AddOpAlreadyPinned = struct {
    const Self = @This();

    platform: zml.Platform,
    buffer: zml.Buffer,

    buffers: []*const ffi.FFIBuffer = undefined,
    stream: *ffi.FFIStream = undefined,

    pub fn call(self: *Self) !void {
        const tracer = self.platform.tracer;

        const frame_await = tracer.frameStart("AddOpAlreadyPinned sleep 1 ms");
        std.time.sleep(1 * std.time.ns_per_ms);
        tracer.frameEnd(frame_await, "AddOpAlreadyPinned 1 ms");

        const frame_memcpy = tracer.frameStart("AddOpAlreadyPinned memcpy");
        cuda.memcpyToDeviceAsync(self.buffers[0].data, self.buffer.asPinnedHostBuffer().data, self.stream);
        tracer.frameEnd(frame_memcpy, "AddOpAlreadyPinned memcpy");
    }
};

const Layer = struct {
    pub fn forward(_: Layer, a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        _ = b; // autofix
        // const a_ = a.toMemory(.host_pinned);
        const custom_call_result = zml.custom_call(AddOp, .{a}, &[_]zml.Shape{a.shape()}, a.getContext(), &.{0});
        // const temp = a.withTags(.{ .x, .y }).dynamicUpdateSlice(.{ .x = zml.Tensor.scalar(0, .i32), .y = zml.Tensor.scalar(0, .i32) }, a);
        // var res = a.negate();
        // res = zml.Tensor.optimizationBarrier(&.{res})[0];

        var res = custom_call_result[0];

        for (0..10) |_| {
            res = res.matmul(res);
        }
        res = res.matmul(res);
        // res = zml.Tensor.optimizationBarrier(&.{res})[0];

        // res = res.negate();

        // const res2 = a.withTags(.{ .x, .y }).dynamicUpdateSlice(.{ .x = zml.Tensor.scalar(0, .i32), .y = zml.Tensor.scalar(0, .i32) }, res);

        // return res2.reuseBuffer(a);

        // const custom_call_result = zml.custom_call(AddOp, .{a}, &[_]zml.Shape{a.shape()}, a.getContext(), &.{});

        return res.reuseBuffer(a);

        // // const custom_call_result = zml.custom_call(AddOpAlreadyPinned, .{}, &[_]zml.Shape{a.shape()}, a.getContext(), &.{0});
        // return b.matmul(custom_call_result[0]).toMemory(.host_pinned);
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    const shape = zml.Shape.init(.{ 4096, 4096 }, .bf16);

    const buffers: zml.aio.BufferStore.Buffers = .{};

    const buffer_store: zml.aio.BufferStore = .{
        .arena = arena_state,
        .buffers = buffers,
    };

    const model_shapes = try zml.aio.populateModel(Layer, allocator, buffer_store);

    var compilation = try asynk.asyncc(zml.compileModel, .{ allocator, Layer.forward, model_shapes, .{ shape, shape }, platform });

    var model_weights = try zml.aio.loadModelBuffers(Layer, model_shapes, buffer_store, arena, platform);
    defer zml.aio.unloadBuffers(&model_weights); // for good practice

    const compiled = try compilation.awaitt();

    var executable = compiled.prepare(model_weights).withExecutionContext();
    defer executable.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var custom_call_buffer = try createRandomBuffer(allocator, platform, shape, random);
    _ = try custom_call_buffer.awaitt();
    defer custom_call_buffer.deinit();

    var libpjrt_cuda = std.DynLib.open("libpjrt_cuda.so") catch unreachable;
    defer libpjrt_cuda.close();

    const cuLaunchHostFunc = libpjrt_cuda.lookup(CuLaunchHostFunc, "cuLaunchHostFunc");

    const add_op_ctx = try allocator.create(AddOp);
    add_op_ctx.* = .{ .platform = platform, .allocator = allocator, .buffer = custom_call_buffer, .cuLaunchHostFunc = cuLaunchHostFunc.? };
    try executable.inner.attach(add_op_ctx);

    const add_op_already_pinned_ctx = try allocator.create(AddOpAlreadyPinned);
    add_op_already_pinned_ctx.* = .{ .platform = platform, .buffer = custom_call_buffer };
    try executable.inner.attach(add_op_already_pinned_ctx);

    var input_buffer_a = try createRandomBuffer(allocator, platform, shape, random);
    _ = try input_buffer_a.awaitt();
    defer input_buffer_a.deinit();
    var input_buffer_b = try createRandomBuffer(allocator, platform, shape, random);
    _ = try input_buffer_b.awaitt();
    defer input_buffer_b.deinit();

    var input_buffer_a_pinned = try input_buffer_a.copyToMemory(platform, .host_pinned);
    defer input_buffer_a_pinned.deinit();
    _ = try input_buffer_a_pinned.awaitt();

    var result: zml.Buffer = input_buffer_a_pinned;
    defer result.deinit();

    for (0..5) |i| {
        log.warn("Iteration {d}", .{i});
        result = executable.call(.{ result, input_buffer_b });
    }

    log.warn("Result: {any}", .{result});
}

fn createRandomBuffer(allocator: std.mem.Allocator, platform: zml.Platform, shape: zml.Shape, random: std.Random) !zml.Buffer {
    const data = try allocator.alloc(u8, shape.byteSize());
    errdefer allocator.free(data);

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
