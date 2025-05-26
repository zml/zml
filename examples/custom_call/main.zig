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

pub const AddOpHostFuncCtx = struct {
    a: zml.Buffer,
    b: zml.Buffer,
    result: zml.Buffer,
};

fn add_op_host_func(data: *const anyopaque) callconv(.c) void {
    const ctx = @as(*const AddOpHostFuncCtx, @alignCast(@ptrCast(data)));

    const a_ptr = std.mem.bytesAsValue(f32, ctx.a.asPinnedHostBuffer().bytes());
    const b_ptr = std.mem.bytesAsValue(f32, ctx.b.asPinnedHostBuffer().bytes());
    const result_ptr = std.mem.bytesAsValue(f32, ctx.result.asPinnedHostBuffer().mutBytes());
    result_ptr.* = a_ptr.* + b_ptr.*;
}

pub const AddOp = struct {
    pub var type_id: i64 = undefined;
    const Self = @This();

    a: zml.Buffer,
    b: zml.Buffer,
    result: zml.Buffer,

    host_func_ctx: AddOpHostFuncCtx = undefined,

    platform: zml.Platform,
    results: []*const ffi.FFIBuffer = undefined,
    stream: *ffi.FFIStream = undefined,

    pub fn init(
        platform: zml.Platform,
        host_buffer: zml.HostBuffer,
    ) !AddOp {
        const a = try zml.Buffer.fromEx(platform, host_buffer, .{ .memory = .host_pinned });
        _ = try a.awaitt();

        const b = try zml.Buffer.fromEx(platform, host_buffer, .{ .memory = .host_pinned });
        _ = try b.awaitt();

        const result = try zml.Buffer.fromEx(platform, host_buffer, .{ .memory = .host_pinned });
        _ = try result.awaitt();

        return .{
            .a = a,
            .b = b,
            .result = result,
            .platform = platform,
        };
    }

    pub fn call(self: *Self, a: *const ffi.FFIBuffer, b: *const ffi.FFIBuffer) !void {
        cuda.memcpyToHostAsync(self.a.asPinnedHostBuffer().mutBytes(), a.data, self.stream);
        cuda.memcpyToHostAsync(self.b.asPinnedHostBuffer().mutBytes(), b.data, self.stream);

        self.host_func_ctx = .{
            .a = self.a,
            .b = self.b,
            .result = self.result,
        };

        _ = cuda.cuLaunchHostFunc(self.stream, @ptrCast(&add_op_host_func), @ptrCast(&self.host_func_ctx));

        cuda.memcpyToDeviceAsync(self.results[0].data, self.result.asPinnedHostBuffer().bytes(), self.stream);
    }
};

/// Model definition
const Layer = struct {
    pub fn forward(_: Layer, a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        const result = zml.custom_call(AddOp, .{ a, b }, &[_]zml.Shape{a.shape()});
        return result[0];
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

    try executable.registerInContext(AddOp);

    const result_hb = try zml.HostBuffer.empty(allocator, shape);
    defer result_hb.deinit(allocator);

    var add_op: AddOp = try .init(platform, result_hb);
    try executable.bindToContext(AddOp, &add_op);

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
    var cpu_result = try result.toHostAlloc(arena);
    _ = try cpu_result.awaitt();

    log.warn(
        "\nThe result of {d} + {d} = {d}\n",
        .{ &input_a, &input_b, cpu_result.items(f32) },
    );
}
