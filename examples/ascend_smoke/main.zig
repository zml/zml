const std = @import("std");
const zml = @import("zml");

const Model = struct {
    pub fn forward(x: zml.Tensor) zml.Tensor {
        return x.mul(x).add(x);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform = try zml.Platform.auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const shape = zml.Shape.init(.{4}, .f32);
    var exe = try platform.compileFn(allocator, io, Model.forward, .{zml.Tensor.init(.{4}, .f32)}, .{});
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    const data = [_]f32{ 1, 2, 3, 4 };
    var input: zml.Buffer = try .fromSlice(io, platform, zml.Slice.init(shape, std.mem.sliceAsBytes(&data)));
    defer input.deinit();

    args.set(.{input});
    exe.call(io, args, &results, .{});
    var output: zml.Buffer = results.get(zml.Buffer);
    defer output.deinit();

    var host: [4]f32 = undefined;
    try output.toSlice(io, zml.Slice.init(shape, std.mem.sliceAsBytes(&host)));
    std.debug.print("ascend result: {d} {d} {d} {d}\n", .{ host[0], host[1], host[2], host[3] });
}
