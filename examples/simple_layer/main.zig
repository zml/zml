const std = @import("std");

const asynk = @import("async");
const zml = @import("zml");

/// Model definition
const Layer = struct {
    bias: ?zml.Tensor = null,
    weight: zml.Tensor,

    pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor {
        var y = self.weight.mul(x);
        if (self.bias) |bias| {
            y = y.add(bias);
        }
        return y;
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

    // Our weights and bias to use
    var weights = [4]f32{ 2.0, 2.0, 2.0, 2.0 };
    var bias = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input_shape = zml.Shape.init(.{4}, .f32);

    // We manually produce a BufferStore. You would not normally do that.
    // A BufferStore is usually created by loading model data from a file.
    var store: zml.aio.BufferStore = .init(allocator);
    defer store.deinit();
    try store.buffers.put(store.arena.allocator(), "weight", zml.HostBuffer.fromArray(&weights));
    try store.buffers.put(store.arena.allocator(), "bias", zml.HostBuffer.fromArray(&bias));

    // A clone of our model, consisting of shapes. We only need shapes for compiling.
    // We use the BufferStore to infer the shapes.
    var model_shapes = try zml.aio.populateModel(Layer, allocator, store);
    model_shapes.weight = model_shapes.weight.withSharding(.{-1});
    model_shapes.bias = model_shapes.bias.?.withSharding(.{-1});

    // Start compiling. This uses the inferred shapes from the BufferStore.
    // The shape of the input tensor, we have to pass in manually.
    var compilation = try asynk.asyncc(zml.compileModel, .{ allocator, Layer.forward, model_shapes, .{input_shape}, platform });

    // Produce a bufferized weights struct from the fake BufferStore.
    // This is like the inferred shapes, but with actual values.
    // We will need to send those to the computation device later.
    var model_weights = try zml.aio.loadModelBuffers(Layer, model_shapes, store, arena, platform);
    defer zml.aio.unloadBuffers(&model_weights); // for good practice

    // Wait for compilation to finish
    const compiled = try compilation.await();

    // pass the model weights to the compiled module to create an executable module
    var executable = compiled.prepare(model_weights);
    defer executable.deinit();

    // prepare an input buffer
    // Here, we use zml.HostBuffer.fromSlice to show how you would create a HostBuffer
    // with a specific shape from an array.
    // For situations where e.g. you have an [4]f32 array but need a .{2, 2} input shape.
    var input = [4]f32{ 5.0, 5.0, 5.0, 5.0 };
    var input_buffer = try zml.Buffer.from(platform, zml.HostBuffer.fromSlice(input_shape, &input), .{});
    defer input_buffer.deinit();

    // call our executable module
    var result: zml.Buffer = executable.call(.{input_buffer});
    defer result.deinit();

    // fetch the result to CPU memory
    const cpu_result = try result.toHostAlloc(arena);
    std.debug.print(
        "\nThe result of {any} * {any} + {any} = {any}\n",
        .{ &weights, &input, &bias, cpu_result.items(f32) },
    );
}
