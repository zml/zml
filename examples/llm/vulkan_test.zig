const std = @import("std");
const zml = @import("zml");

fn abs(x: zml.Tensor) zml.Tensor {
    return x.abs();
}

fn flatten(x: zml.Tensor) zml.Tensor {
    return x.flatten();
}

fn convertU8ToF32(x: zml.Tensor) zml.Tensor {
    return x.convert(.f32);
}

// fn convertI32ToU8(x: zml.Tensor) zml.Tensor {
//     return x.convert(.u8);
// }

fn dot(weight: zml.Tensor, input: zml.Tensor) zml.Tensor {
    return weight.dot(input, .d);
}

fn add(input: zml.Tensor, bias: zml.Tensor) zml.Tensor {
    return input.add(bias);
}

fn relu(x: zml.Tensor) zml.Tensor {
    return x.relu();
}

fn argMax(x: zml.Tensor) zml.Tensor {
    return x.argMax(.d).indices;
}

fn runUnary(
    comptime Input: type,
    comptime Output: type,
    comptime output_len: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime operation_name: []const u8,
    comptime dump_dir: []const u8,
    comptime function: anytype,
    input_shape: zml.Tensor,
    host_input: []const Input,
    expected: []const Output,
) !void {
    std.debug.print("Vulkan operation: {s}\n", .{operation_name});

    var exe = try platform.compileFn(allocator, io, function, .{input_shape}, .{
        .program_name = operation_name,
        .xla_dump_to = dump_dir,
        .xla_dump_emitter_re = "mlir-fusion",
    });
    defer exe.deinit();

    var input = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(input_shape.shape(), std.mem.sliceAsBytes(host_input)),
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

    var host_output: [output_len]Output = undefined;
    try output.toSlice(
        io,
        zml.Slice.init(output.shape(), std.mem.sliceAsBytes(host_output[0..])),
    );
    std.debug.print("  {s} output: {any}\n", .{ operation_name, host_output });
    try std.testing.expectEqualSlices(Output, expected, &host_output);
}

fn runBinary(
    comptime Input: type,
    comptime Output: type,
    comptime output_len: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime operation_name: []const u8,
    comptime dump_dir: []const u8,
    comptime function: anytype,
    lhs_shape: zml.Tensor,
    lhs_host: []const Input,
    rhs_shape: zml.Tensor,
    rhs_host: []const Input,
    expected: []const Output,
) !void {
    std.debug.print("Vulkan operation: {s}\n", .{operation_name});

    var exe = try platform.compileFn(allocator, io, function, .{ lhs_shape, rhs_shape }, .{
        .program_name = operation_name,
        .xla_dump_to = dump_dir,
        .xla_dump_emitter_re = "mlir-fusion",
    });
    defer exe.deinit();

    var lhs = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(lhs_shape.shape(), std.mem.sliceAsBytes(lhs_host)),
        .replicated,
    );
    defer lhs.deinit();
    var rhs = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(rhs_shape.shape(), std.mem.sliceAsBytes(rhs_host)),
        .replicated,
    );
    defer rhs.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ lhs, rhs });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    exe.call(args, &results);

    var output: zml.Buffer = results.get(zml.Buffer);
    defer output.deinit();

    var host_output: [output_len]Output = undefined;
    try output.toSlice(
        io,
        zml.Slice.init(output.shape(), std.mem.sliceAsBytes(host_output[0..])),
    );
    std.debug.print("  {s} output: {any}\n", .{ operation_name, host_output });
    try std.testing.expectEqualSlices(Output, expected, &host_output);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .init(allocator, io, .vulkan, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false } } },
    });
    defer platform.deinit(allocator, io);

    try runUnary(f32, f32, 4, allocator, io, platform, "abs", "/tmp/xla-vulkan/abs", abs, .init(.{4}, .f32), &.{ -1, 2, -3, 4 }, &.{ 1, 2, 3, 4 });
    try runUnary(f32, f32, 4, allocator, io, platform, "flatten", "/tmp/xla-vulkan/flatten", flatten, .init(.{ 2, 2 }, .f32), &.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 4 });
    try runUnary(u8, f32, 4, allocator, io, platform, "convert_u8_to_f32", "/tmp/xla-vulkan/convert_u8_to_f32", convertU8ToF32, .init(.{4}, .u8), &.{ 0, 1, 127, 255 }, &.{ 0, 1, 127, 255 });
    // try runUnary(i32, u8, 4, allocator, io, platform, "convert_i32_to_u8", "/tmp/xla-vulkan/convert_i32_to_u8", convertI32ToU8, .init(.{4}, .i32), &.{ 0, 1, 127, 255 }, &.{ 0, 1, 127, 255 });
    try runBinary(f32, f32, 2, allocator, io, platform, "dot", "/tmp/xla-vulkan/dot", dot, .init(.{ .d_out = 2, .d = 3 }, .f32), &.{ 1, 2, 3, 4, 5, 6 }, .init(.{ .d = 3 }, .f32), &.{ 2, 1, -1 }, &.{ 1, 7 });
    try runBinary(f32, f32, 4, allocator, io, platform, "add", "/tmp/xla-vulkan/add", add, .init(.{ .d = 4 }, .f32), &.{ 1, 2, 3, 4 }, .init(.{ .d = 4 }, .f32), &.{ 10, -2, 0, 0.5 }, &.{ 11, 0, 3, 4.5 });
    try runUnary(f32, f32, 5, allocator, io, platform, "relu", "/tmp/xla-vulkan/relu", relu, .init(.{5}, .f32), &.{ -3, -0.5, 0, 2, 7 }, &.{ 0, 0, 0, 2, 7 });
    try runUnary(f32, i32, 1, allocator, io, platform, "argMax", "/tmp/xla-vulkan/argMax", argMax, .init(.{ .d = 4 }, .f32), &.{ 1, 7, 7, 3 }, &.{1});
}
