const std = @import("std");
const zml = @import("zml");

fn identity(x: zml.Tensor) zml.Tensor {
    return x;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .init(allocator, io, .vulkan, .{});
    defer platform.deinit(allocator, io);

    const input_shape: zml.Tensor = .init(.{4}, .f32);
    var exe = try platform.compileFn(allocator, io, identity, .{input_shape}, .{});
    defer exe.deinit();

    const host = [_]f32{ 1, 2, 3, 4 };
    var in_buf: zml.Buffer = try .fromSlice(io, platform, zml.Slice.init(input_shape.shape(), std.mem.sliceAsBytes(host[0..])), .replicated);
    defer in_buf.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{in_buf});
    exe.call(args, &results);

    var out: zml.Buffer = results.get(zml.Buffer);
    defer out.deinit();

    var host_out: [4]f32 = undefined;
    try out.toSlice(io, zml.Slice.init(out.shape(), std.mem.sliceAsBytes(host_out[0..])));
    try std.testing.expectEqualSlices(f32, &host, &host_out);
}
