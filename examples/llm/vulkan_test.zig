const std = @import("std");
const zml = @import("zml");

fn identity(x: zml.Tensor) zml.Tensor {
    return x;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .init(allocator, io, .vulkan, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false } } },
    });
    defer platform.deinit(allocator, io);

    const input_shape: zml.Tensor = .init(.{4}, .f32);
    var exe = try platform.compileFn(allocator, io, identity, .{input_shape}, .{});
    defer exe.deinit();

    const host_input = [_]f32{ -1, 2, -3, 4 };
    var input = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(input_shape.shape(), std.mem.sliceAsBytes(host_input[0..])),
        .replicated,
    );
    defer input.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{input});

    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    exe.call(args, &results);

    var output: zml.Buffer = results.get(zml.Buffer);
    defer output.deinit();

    var host_output: [4]f32 = undefined;
    try output.toSlice(
        io,
        zml.Slice.init(output.shape(), std.mem.sliceAsBytes(host_output[0..])),
    );
    std.debug.print("Vulkan output: {any}\n", .{host_output});
    try std.testing.expectEqualSlices(f32, &.{ -1, 2, -3, 4 }, &host_output);
}
