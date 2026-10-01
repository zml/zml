const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.llm);
const prompt: []const u8 = "Photo foodporn avec un reflet de lumière d'un chef cuisiner qui tient son fier pot au feu";

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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const layer: Layer = .{
        .weight = zml.Tensor.init(.{3}, .f16),
        .bias = zml.Tensor.init(.{3}, .f16),
    };

    // Our computation require an input tensor
    const input: zml.Tensor = .init(.{3}, .f16);

    var executable = try platform.compile(allocator, io, layer, .forward, .{input}, .{});
    defer executable.deinit();

    const weight_slice: zml.Slice = .init(layer.weight.shape(), std.mem.sliceAsBytes(&[3]f16{ 1.0, 2.0, 3.0 }));
    const bias_slice: zml.Slice = .init(layer.bias.?.shape(), std.mem.sliceAsBytes(&[3]f16{ 1.0, 1.0, 1.0 }));
    var layer_buffers: zml.Bufferized(Layer) = .{
        .weight = try zml.Buffer.fromSlice(io, platform, weight_slice, .replicated),
        .bias = try zml.Buffer.fromSlice(io, platform, bias_slice, .replicated),
    };
    defer layer_buffers.weight.deinit();
    defer layer_buffers.bias.?.deinit();

    // create the input buffer
    const input_slice: zml.Slice = .init(input.shape(), std.mem.sliceAsBytes(&[3]f16{ 5.0, 5.0, 5.0 }));
    var input_buffer: zml.Buffer = try .fromSlice(io, platform, input_slice, .replicated);
    defer input_buffer.deinit();

    // create the Args and Results structs
    var args = try executable.args(allocator);
    defer args.deinit(allocator);

    var results = try executable.results(allocator);
    defer results.deinit(allocator);

    // fill the Args
    args.set(.{ layer_buffers, input_buffer });

    // call our executable
    executable.call(args, &results);

    // Retrieve the resulting buffer
    var result = results.get(zml.Buffer);
    defer result.deinit();

    // fetch the result buffer to CPU memory
    const result_slice = try result.toSliceAlloc(allocator, io);
    defer result_slice.free(allocator);

    std.debug.print(
        "\n\nThe result of {d} * {d} + {d} = {d}\n",
        .{ weight_slice, input_slice, bias_slice, result_slice },
    );
}
