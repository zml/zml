//! This examples show how to specify a model,
//! compile it, and execute it with specific weights.
//!
//! This code is very explicit and doesn't use zml.aio helper functions
//! in order to show what is happening behind the scene.
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

pub fn asyncMain() !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    // Start ZML, detect available devices and chose a platform.
    var context = try zml.Context.init();
    defer context.deinit();
    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    // Create a skeleton for our model with only the shapes.
    // The `buffer_id` are typically not set manually
    // but instead set by a zml.BufferStore
    // to map abstract Tensors to actual part of the model weight file.
    const model_shapes: Layer = .{
        .bias = zml.Tensor{
            ._shape = zml.Shape.init(.{4}, .f16).withSharding(.{-1}),
            ._id = .{ .buffer_id = 0 },
        },
        .weight = zml.Tensor{
            ._shape = zml.Shape.init(.{4}, .f16).withSharding(.{-1}),
            ._id = .{ .buffer_id = 1 },
        },
    };

    // Start compilation in a different thread.
    // We already specified the shape of the model weights,
    // but we still need to specifiy the shape of the input tensor.
    const input_shape = zml.Shape.init(.{4}, .f16);
    var compilation = try asynk.asyncc(
        zml.compileModel,
        .{ std.heap.page_allocator, Layer.forward, model_shapes, .{input_shape}, platform },
    );

    // Now we need to create a model instance with actual weights.
    const weights = [4]f16{ 2.0, -2.0, 1.0, -1.0 };
    const bias = [4]f16{ 1.0, 2.0, 3.0, 4.0 };

    var model_weights: zml.Bufferized(Layer) = .{
        .bias = try zml.Buffer.fromSlice(platform, .{4}, &bias),
        .weight = try zml.Buffer.fromSlice(platform, .{4}, &weights),
    };
    defer zml.aio.unloadBuffers(&model_weights);

    // Wait for compilation to finish
    const compiled = try compilation.awaitt();
    defer compiled.deinit();

    // pass the model weights to the compiled module to create an executable module.
    // This is where the shapes of the buffers will be compared to the
    // shape expected by the executable.
    // Note: we don't call `executable.deinit()` since it uses the same memory than `compiled`.
    const executable = compiled.prepare(model_weights);

    // prepare the input buffer
    const input = [4]f16{ 5.0, 5.0, 5.0, 5.0 };
    var input_buffer = try zml.Buffer.fromSlice(platform, input_shape, &input);
    defer input_buffer.deinit();

    // call our executable module, the result is still on the device.
    var result: zml.Buffer = executable.call(.{input_buffer});
    defer result.deinit();

    // copy the result to CPU memory
    const cpu_result: zml.HostBuffer = try result.toHostAlloc(gpa);
    defer cpu_result.deinit(gpa);
    std.debug.print(
        "\nThe result of {any} * {any} + {any} = {any}\n",
        .{ &weights, &input, &bias, cpu_result.items(f32) },
    );
}

const log = std.log.scoped(.simple_layer);

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.smp_allocator, asyncMain);
}
