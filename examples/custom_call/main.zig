const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");

pub const std_options: std.Options = .{
    .log_level = .warn,
    .logFn = asynk.logFn(std.log.defaultLog),
};

const log = std.log.scoped(.@"examples/custom_call");

pub const AddOp = struct {
    const Self = @This();

    _platform: *const zml.Platform,

    pub fn beforeCustomCall(a: zml.Tensor, b: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        return .{ a, b, zml.Tensor.scalar(0, .f32) };
    }

    pub fn call(self: *Self, a: zml.Buffer, b: zml.Buffer, result: zml.Buffer) !struct { zml.Buffer, zml.Buffer, zml.Buffer } {
        _ = self; // autofix
        // const a_ = try a.getValue(f32);
        // const b_ = try b.getValue(f32);

        // log.warn("{s}@{*} a value: {d}", .{ @typeName(Self), self, a_ });
        // log.warn("{s}@{*} b value: {d}", .{ @typeName(Self), self, b_ });

        // const add_result = a_ + b_;
        // try result.setValue(f32, add_result);
        // log.warn("{s}@{*} result value: {d}", .{ @typeName(Self), self, try result.getValue(f32) });

        return .{ a, b, result };
    }

    fn getPlatform(self: *Self) zml.Platform {
        return self._platform.*;
    }
};

pub const LogResultOp = struct {
    const Self = @This();

    _platform: *const zml.Platform,

    pub fn beforeCustomCall(value: zml.Tensor) zml.Tensor {
        return value;
    }

    pub fn call(self: *Self, value: zml.Buffer) !zml.Buffer {
        _ = self; // autofix
        // log.warn("{s}@{*} value: {d}", .{ @typeName(Self), self, try value.getValue(f32) });
        return value;
    }

    fn getPlatform(self: *Self) zml.Platform {
        return self._platform.*;
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
        var result = a.add(b);
        // const results = zml.custom_call(AddOp, .{ a, b });
        // result = results[0].mul(results[1]);
        result = zml.custom_call(LogResultOp, .{result});
        return result.add(a);
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

    const shape = zml.Shape.init(.{ 40, 128 }, .bf16);

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

    const add_op_ctx: AddOp = .{ ._platform = &platform };
    try executable.attach(&add_op_ctx);
    const log_result_op_ctx: LogResultOp = .{ ._platform = &platform };
    try executable.attach(&log_result_op_ctx);
    // const log_values_op_ctx: LogValuesVoidOp = .{ ._platform = &platform };
    // try executable.attach(&log_values_op_ctx);

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    for (0..2) |i| {
        log.warn("Iteration {d}", .{i});

        // prepare input buffers
        var input_buffer_a = try createRandomBuffer(allocator, platform, shape, random);
        defer input_buffer_a.deinit();
        var input_buffer_b = try createRandomBuffer(allocator, platform, shape, random);
        defer input_buffer_b.deinit();

        var result: zml.Buffer = executable.call(.{ input_buffer_a, input_buffer_b });
        defer result.deinit();

        // fetch the result to CPU memory
        // const cpu_result = try result.toHostAlloc(arena);
        // log.warn(
        //     "\nThe result of {d} + {d} = {d}\n",
        //     .{ &input_a, &input_b, cpu_result.items(f32) },
        // );

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
