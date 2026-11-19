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

    const forecaster: model.Model = .{
        .residuals = .init(store.view().withPrefix("representation.patch_proj")),
    };

    // Dummy input
    const input: zml.Tensor = .init(.{ .b = 1, .v = 1, .t = 2000 }, .f32);

    var executable = try platform.compile(allocator, io, forecaster, .forward, .{input}, .{});
    defer executable.deinit();

    var weights = try zml.mem.bufferize(allocator, model.Model, &forecaster);
    defer weights.residuals.hidden_layer.weight.deinit();
    defer if (weights.residuals.hidden_layer.bias) |*b| b.deinit();
    defer weights.residuals.output_layer.weight.deinit();
    defer if (weights.residuals.output_layer.bias) |*b| b.deinit();
    defer weights.residuals.residual_layer.weight.deinit();
    defer if (weights.residuals.residual_layer.bias) |*b| b.deinit();

    var loader: zml.io.Loader = try .init(allocator, platform, .default);
    defer loader.deinit();

    try loader.load(io, model.Model, &forecaster, &weights, &store, .{});
    try loader.await(io);

    // Dummy data
    const buf: []f32 = try allocator.alloc(f32, 3200);
    defer allocator.free(buf);
    for (buf, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) - 30.0;
    @memset(buf[0..3], std.math.nan(f32)); // leading missing data: the series starts late
    for ([_]usize{ 10, 1210, 1640 }) |i| buf[i] = std.math.inf(f32); // isolated +inf values
    buf[1500] = -std.math.inf(f32); // and one -inf
    @memset(buf[1800..1820], std.math.nan(f32)); // a gap inside the context

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
        .{result_slice},
    );
}
