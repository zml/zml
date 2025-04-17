const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

const log = std.log.scoped(.@"examples/custom_call");

pub const AddOp = struct {
    const Self = @This();

    _platform: *const zml.Platform,

    pub fn beforeCustomCall(a: zml.Tensor, b: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        return .{ a, b };
    }

    pub fn call(self: *Self, a: zml.Buffer, _: zml.Buffer) !zml.Buffer {
        _ = self; // autofix
        log.info("mem a: {any}", .{a.items(f32)});
        return a;
    }

    fn getPlatform(self: *Self) zml.Platform {
        return self._platform.*;
    }
};

pub const LogResultOp = struct {
    const Self = @This();

    _platform: *const zml.Platform,

    // pub fn beforeCustomCall(result: zml.Tensor) zml.Tensor {
    //     return result;
    // }

    pub fn call(self: *Self, result: zml.Buffer) !zml.Buffer {
        log.info("LogResultOp result: {*}", .{try result._shards.get(0).getOpaqueDeviceMemoryDataPointer(self._platform.pjrt_api)});
        return result;
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
    pub fn forward(_: Layer, a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        // const result = zml.custom_call(AddOp, .{ a.toMemory(.host_pinned), b.toMemory(.host_pinned) });
        // const a_ = a.toMemory(.host_pinned);
        // const b_ = b.toMemory(.host_pinned);
        const a_ = a.toMemory(.host_pinned);
        const b_ = b.toMemory(.host_pinned);
        const c = a_.clamp(b_, a_);
        const d = c.toMemory(.device);
        const on_device = a.add(d).toMemory(.host_pinned);
        // var result = zml.custom_call(LogResultOp, .{on_device});
        // const result = on_device.toMemory(.device).add(b);
        // zml.custom_call(LogValuesVoidOp, .{ result, a, b });
        // var result = a.add(b).toMemory(.host_pinned);
        var result = zml.custom_call(LogResultOp, .{on_device});
        return result.add(zml.Tensor.scalar(5, .f32));
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

    const shape = zml.Shape.init(.{1}, .f32);

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

    // prepare input buffers
    var input_a = [1]f32{1.0};
    var input_buffer_a = try zml.Buffer.from(platform, zml.HostBuffer.fromSlice(shape, &input_a));
    defer input_buffer_a.deinit();
    var input_b = [1]f32{1.0};
    var input_buffer_b = try zml.Buffer.from(platform, zml.HostBuffer.fromSlice(shape, &input_b));
    defer input_buffer_b.deinit();

    // call our executable module
    var result: zml.Buffer = executable.call(.{ input_buffer_a, input_buffer_b });
    const result_mem = result.getMemory();
    std.debug.print("<<< Result memory: {any}\n", .{result_mem.kind(platform.pjrt_api)});
    defer result.deinit();

    // fetch the result to CPU memory
    const cpu_result = try result.toHostAlloc(arena);
    std.debug.print(
        "\nThe result of {d} + {d} = {d}\n",
        .{ &input_a, &input_b, cpu_result.items(f32) },
    );
}
