const std = @import("std");

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
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Io runtime
    var threaded: std.Io.Threaded = .init(allocator);
    defer threaded.deinit();

    const io = threaded.io();

    // Initialize ZML globally
    zml.init();
    defer zml.deinit();

    // Auto-select platform
    const platform: zml.Platform = try .auto(io, .{});

    // Initialize model instance and input
    const model: Layer = .{
        .weight = zml.Tensor.init(.{4}, .f32),
        .bias = zml.Tensor.init(.{4}, .f32),
    };
    const input: zml.Tensor = .init(.{4}, .f32);

    // Compile our executable
    var exe = try platform.compileModel(allocator, io, Layer.forward, model, .{input});
    defer exe.deinit();

    // Our weights and bias to use
    const weights_data = [4]f32{ 2.0, 2.0, 2.0, 2.0 };
    const weights_slice: zml.ConstSlice = .init(model.weight.shape(), std.mem.sliceAsBytes(&weights_data));
    var weights_buffer: zml.Buffer = try .fromSlice(io, platform, weights_slice);

    const bias_data = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    const bias_slice: zml.ConstSlice = .init(model.bias.?.shape(), std.mem.sliceAsBytes(&bias_data));
    var bias_buffer: zml.Buffer = try .fromSlice(io, platform, bias_slice);

    // Don't forget to deinit buffers
    defer weights_buffer.deinit();
    defer bias_buffer.deinit();

    // Init the bufferized version of the model
    const model_buffers: zml.Bufferized(Layer) = .{
        .weight = weights_buffer,
        .bias = bias_buffer,
    };

    // Prepare an input buffer
    var input_data = [4]f32{ 5.0, 5.0, 5.0, 5.0 };
    const input_slice: zml.Slice = .init(input.shape(), std.mem.sliceAsBytes(&input_data));
    var input_buffer: zml.Buffer = try .fromSlice(io, platform, input_slice);
    defer input_buffer.deinit();

    // Create the Arguments and Results struct to interact with the executable
    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    // Set the arguments from the bufferized version of the model and the input buffer
    args.set(.{ model_buffers, input_buffer });

    // Call our executable module
    exe.call(args, &results, io);

    // Retrieve the output
    var result = results.get(zml.Buffer);
    defer result.deinit();

    // fetch the result to CPU memory
    const cpu_result = try result.toSliceAlloc(allocator, io);
    defer cpu_result.free(allocator);
    std.debug.print(
        "\nThe result of {d} * {d} + {d} = {d}\n",
        .{ weights_slice, input_slice, bias_slice, cpu_result },
    );
}
