const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const zml = @import("zml");

pub const std_options: std.Options = .{
    .log_level = .info,
};

var kernel_source_global: []const u8 = &.{};

fn createSequenceBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.sharding.Sharding,
    offset: f32,
) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    for (slice.items(f32), 0..) |*elem, i| {
        elem.* = offset + @as(f32, @floatFromInt(i));
    }

    return zml.Buffer.fromSlice(io, platform, slice, sharding);
}

fn loadKernelSource(allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
    const r = try zml.bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const kernel_path = try r.rlocation("zml/examples/neuron_nki/add_one.py", &path_buf) orelse return error.NotFound;
    return try zml.stdx.Io.Dir.readFileAlloc(.cwd(), io, kernel_path, allocator, .unlimited);
}

fn addOne(x: zml.Tensor) zml.Tensor {
    return zml.ops.neuronNki(.{x}, .{x.shape()}, .{
        .name = "add_one",
        .entrypoint = "add_one",
        .source = kernel_source_global,
    })[0];
}

fn createExpectedSlice(allocator: std.mem.Allocator, shape: zml.Shape, offset: f32) !zml.Slice {
    const slice = try zml.Slice.alloc(allocator, shape);
    for (slice.items(f32), 0..) |*elem, i| {
        elem.* = offset + @as(f32, @floatFromInt(i)) + 1;
    }
    return slice;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    kernel_source_global = try loadKernelSource(allocator, io);
    defer allocator.free(kernel_source_global);

    const platform = try zml.Platform.init(allocator, io, .neuron, .{});
    defer platform.deinit(allocator, io);

    const replicated = try zml.sharding.replicatedSharding(platform);

    const x: zml.Tensor = .init(.{ 128, 512 }, .f32);
    var exe = try platform.compileFn(allocator, io, addOne, .{x}, .{ .shardings = &.{replicated} });
    defer exe.deinit();

    var input = try createSequenceBuffer(allocator, io, platform, x.shape(), replicated, 0);
    defer input.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, addOne, .{input});
    defer output.deinit();

    var expected = try createExpectedSlice(allocator, x.shape(), 0);
    defer expected.free(allocator);

    try zml.testing.expectClose(io, output, expected, .{});

    std.log.info("NKI add_one example succeeded on {s}", .{@tagName(platform.target)});
}
