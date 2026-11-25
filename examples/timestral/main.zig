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
    var model_dir = try std.Io.Dir.cwd().openDir(io, args.model, .{});
    defer model_dir.close(io);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const config = try parseConfig(model.Config, allocator, io, model_dir);
    defer config.deinit();

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.model);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const forecaster: model.Model = .init(store.view(), config.value);

    // Dummy input
    const input: zml.Tensor = .init(.{ .batch = 1, .variate = 1, .time = 320 }, .f32);

    var executable = try platform.compile(allocator, io, forecaster, .forward, .{input}, .{});
    defer executable.deinit();

    var weights = try zml.mem.bufferize(allocator, model.Model, &forecaster);
    defer weights.representation.patch_proj.hidden_layer.weight.deinit();
    defer if (weights.representation.patch_proj.hidden_layer.bias) |*b| b.deinit();
    defer weights.representation.patch_proj.output_layer.weight.deinit();
    defer if (weights.representation.patch_proj.output_layer.bias) |*b| b.deinit();
    defer weights.representation.patch_proj.residual_layer.weight.deinit();
    defer if (weights.representation.patch_proj.residual_layer.bias) |*b| b.deinit();
    defer weights.representation.forecast_query.deinit();

    var loader: zml.io.Loader = try .init(allocator, platform, .default);
    defer loader.deinit();

    try loader.load(io, model.Model, &forecaster, &weights, &store, .{});
    try loader.await(io);

    // Dummy data
    const buf: []f32 = try allocator.alloc(f32, 320);
    defer allocator.free(buf);
    for (buf, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) - 30.0;
    @memset(buf[0..3], std.math.nan(f32)); // leading missing data: the series starts late
    for ([_]usize{ 10, 121, 164 }) |i| buf[i] = std.math.inf(f32); // isolated +inf values
    buf[150] = -std.math.inf(f32); // and one -inf
    @memset(buf[180..182], std.math.nan(f32)); // a gap inside the context

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

pub fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}
