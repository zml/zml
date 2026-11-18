const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const model = @import("model.zig");

const Args = struct {
    model: []const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.model);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const forecaster: model.Model = .init(store.view().withPrefix("representation.patch_proj"));

    // Dummy input
    const input: zml.Tensor = .init(.{ .b = 1024, .d = 64 }, .f32);

    var executable = try platform.compile(allocator, io, forecaster, .forward, .{input}, .{});
    defer executable.deinit();

    var weights = try zml.mem.bufferize(allocator, model.Model, &forecaster);
    defer weights.hidden_layer.weight.deinit();
    defer if (weights.hidden_layer.bias) |*b| b.deinit();
    defer weights.output_layer.weight.deinit();
    defer if (weights.output_layer.bias) |*b| b.deinit();
    defer weights.residual_layer.weight.deinit();
    defer if (weights.residual_layer.bias) |*b| b.deinit();

    var loader: zml.io.Loader = try .init(allocator, platform, .default);
    defer loader.deinit();

    try loader.load(io, model.Model, &forecaster, &weights, &store, .{});
    try loader.await(io);

    // Dummy data
    const buf: []f32 = try allocator.alloc(f32, 1024 * 64);
    defer allocator.free(buf);
    @memset(buf, 1.0);

    const input_slice: zml.Slice = .init(input.shape(), std.mem.sliceAsBytes(buf));
    var input_buffer: zml.Buffer = try .fromSlice(io, platform, input_slice);
    defer input_buffer.deinit();

    var exe_args = try executable.args(allocator);
    defer exe_args.deinit(allocator);

    var results = try executable.results(allocator);
    defer results.deinit(allocator);

    exe_args.set(.{ weights, input_buffer });

    executable.call(io, exe_args, &results, .{});
    var result = results.get(zml.Buffer);
    defer result.deinit();

    const result_slice = try result.toSliceAlloc(allocator, io);
    defer result_slice.free(allocator);

    std.debug.print(
        "\n\nThe result is {d}\n",
        .{ result_slice },
    );
}
