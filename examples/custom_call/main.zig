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

pub const AddOp = struct {
    const Self = @This();

    platform: zml.Platform,

    buffers: []*const ffi.FFIBuffer = undefined,
    stream: *ffi.FFIStream = undefined,

    pub fn call(self: *Self, a_: *const ffi.FFIBuffer, b_: *const ffi.FFIBuffer) !void {
        const a_device = zml.Buffer.asViewOfDeviceBuffer(self.platform, ffi.getShape(a_), null, a_.data.asPtr());
        const a = a_device.copyToMemory(self.platform, .host_pinned) catch unreachable;

        const b_device = zml.Buffer.asViewOfDeviceBuffer(self.platform, ffi.getShape(b_), null, b_.data.asPtr());
        const b = b_device.copyToMemory(self.platform, .host_pinned) catch unreachable;

        _ = a.awaitt() catch unreachable;
        _ = b.awaitt() catch unreachable;

        const a_value = try a.getValue(f32);
        const b_value = try b.getValue(f32);

        const result: f32 = a_value + b_value;
        const result_hb = zml.HostBuffer.fromBytes(a_device.shape(), std.mem.asBytes(&result));

        cuda.memcpyToDeviceAsync(self.buffers[0].data, result_hb.data, self.stream);
    }
};

pub const LogResultOp = struct {
    const Self = @This();

    platform: zml.Platform,

    buffers: []*const ffi.FFIBuffer = undefined,
    stream: *ffi.FFIStream = undefined,

    pub fn call(self: *Self, value: *const ffi.FFIBuffer) !void {
        const value_device = zml.Buffer.asViewOfDeviceBuffer(self.platform, ffi.getShape(value), null, value.data.asPtr());
        const value_pinned = value_device.copyToMemory(self.platform, .host_pinned) catch unreachable;

        log.warn("{s}@{*} value: {d}", .{ @typeName(Self), self, try value_pinned.getValue(f32) });

        const data = try value_pinned.dataInMemory();
        cuda.memcpyToDeviceAsync(self.buffers[0].data, data, self.stream);
    }
};

pub const LogValuesVoidOp = struct {
    const Self = @This();

    _platform: *const zml.Platform,

    pub fn call(self: *Self, result: zml.HostBuffer, a: zml.HostBuffer, b: zml.HostBuffer) !void {
        _ = self; // autofix
        log.info("LogValuesVoidOp mem result: {any}", .{result.items(f32)});
        log.info("LogValuesVoidOp mem a: {any}", .{a.items(f32)});
        log.info("LogValuesVoidOp mem b: {any}", .{b.items(f32)});
    }

    fn getPlatform(self: *Self) zml.Platform {
        return self._platform.*;
    }
};

/// Model definition
const Layer = struct {
    // a = 40x128 bf16 ou f32
    pub fn forward(_: Layer, a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        const a_ = zml.custom_call(LogResultOp, .{a}, &[_]zml.Shape{a.shape()}, a.getContext(), &.{0});
        const results = zml.custom_call(AddOp, .{ a_[0], b }, &[_]zml.Shape{a.shape()}, a.getContext(), &.{0});
        return results[0];
        // return zml.custom_call(LogResultOp, .{results[2]}, &[_]zml.Shape{a.shape()});
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Arena allocator for BufferStore etc.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    // const shape = zml.Shape.init(.{ 40, 128 }, .bf16);
    const shape = zml.Shape.init(.{}, .f32);

    // We manually produce a BufferStore. You would not normally do that.
    // A BufferStore is usually created by loading model data from a file.
    const buffers: zml.aio.BufferStore.Buffers = .{};

    // the actual BufferStore
    const buffer_store: zml.aio.BufferStore = .{
        .arena = arena_state,
        .buffers = buffers,
    };

    // A clone of our model, consisting of shapes. We only need shapes for compiling.
    // We use the BufferStore to infer the shapes.
    const model_shapes = try zml.aio.populateModel(Layer, allocator, buffer_store);

    // Start compiling. This uses the inferred shapes from the BufferStore.
    // The shape of the input tensor, we have to pass in manually.
    var compilation = try asynk.asyncc(zml.compileModel, .{ allocator, Layer.forward, model_shapes, .{ shape, shape }, platform });

    // Produce a bufferized weights struct from the fake BufferStore.
    // This is like the inferred shapes, but with actual values.
    // We will need to send those to the computation device later.
    var model_weights = try zml.aio.loadModelBuffers(Layer, model_shapes, buffer_store, arena, platform);
    defer zml.aio.unloadBuffers(&model_weights); // for good practice

    // Wait for compilation to finish
    const compiled = try compilation.awaitt();

    // pass the model weights to the compiled module to create an executable module
    var executable = compiled.prepare(model_weights).withExecutionContext();
    defer executable.deinit();

    const add_op_ctx: AddOp = .{ .platform = platform };
    try executable.inner.attach(&add_op_ctx);
    const log_result_op_ctx: LogResultOp = .{ .platform = platform };
    try executable.inner.attach(&log_result_op_ctx);
    // const log_values_op_ctx: LogValuesVoidOp = .{ ._platform = &platform };
    // try executable.attach(&log_values_op_ctx);

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();
    _ = random; // autofix

    for (0..2) |i| {
        log.warn("Iteration {d}", .{i});

        // prepare input buffers
        // var input_buffer_a = try createRandomBuffer(allocator, platform, shape, random);
        // _ = try input_buffer_a.awaitt();
        // defer input_buffer_a.deinit();
        // var input_buffer_b = try createRandomBuffer(allocator, platform, shape, random);
        // _ = try input_buffer_b.awaitt();
        // defer input_buffer_b.deinit();

        var input_a = [1]f32{1.0};
        var input_buffer_a = try zml.Buffer.from(platform, zml.HostBuffer.fromSlice(shape, &input_a));
        _ = try input_buffer_a.awaitt();
        defer input_buffer_a.deinit();
        var input_b = [1]f32{1.0};
        var input_buffer_b = try zml.Buffer.from(platform, zml.HostBuffer.fromSlice(shape, &input_b));
        _ = try input_buffer_b.awaitt();
        defer input_buffer_b.deinit();

        var result: zml.Buffer = executable.call(.{ input_buffer_a, input_buffer_b });
        defer result.deinit();

        // fetch the result to CPU memory
        // const cpu_result = try result.toHostAlloc(arena);
        // log.warn(
        //     "\nThe result of {d} + {d} = {d}\n",
        //     .{ &input_a, &input_b, cpu_result.items(f32) },
        // );

        var cpu_result = try result.toHostAlloc(arena);
        _ = try cpu_result.awaitt();
        log.warn(
            "\nThe result is {d}\n",
            .{cpu_result.items(f32)},
        );

        std.time.sleep(1 * std.time.ns_per_s);
    }
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
