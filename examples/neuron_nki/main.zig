const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const zml = @import("zml");

const log = std.log.scoped(.neuron_nki);

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/module", .level = .info },
    },
};

const Kernel = struct {
    source: []const u8,

    fn addOne(self: Kernel, x: zml.Tensor) zml.Tensor {
        return zml.ops.neuronNki(.{x}, .{x.shape()}, .{
            .name = "add_one",
            .entrypoint = "add_one",
            .source = self.source,
        })[0];
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const runfiles = try zml.bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const kernel_path = try runfiles.rlocation("zml/examples/neuron_nki/add_one.py", &path_buf);
    const kernel_source = try std.Io.Dir.readFileAlloc(.cwd(), io, kernel_path.?, allocator, .unlimited);
    defer allocator.free(kernel_source);
    const kernel: Kernel = .{ .source = kernel_source };

    const platform: *zml.Platform = try .init(allocator, io, .neuron, .{});
    defer platform.deinit(allocator, io);

    const sharding = try zml.sharding.replicatedSharding(platform);

    const x: zml.Tensor = .init(.{ 128, 512 }, .f32);

    var exe = try platform.compileFn(allocator, io, Kernel.addOne, .{ kernel, x }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    const input_slice = try zml.Slice.alloc(allocator, x.shape());
    defer input_slice.free(allocator);

    for (input_slice.items(f32), 0..) |*elem, i| {
        elem.* = @as(f32, @floatFromInt(i));
    }

    var input_buffer: zml.Buffer = try .fromSlice(io, platform, input_slice, sharding);
    defer input_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, Kernel.addOne, .{input_buffer});
    defer output.deinit();

    const expected: zml.Slice = try .alloc(allocator, x.shape());
    defer expected.free(allocator);

    for (expected.items(f32), 0..) |*elem, i| {
        elem.* = @as(f32, @floatFromInt(i)) + 1;
    }

    try zml.testing.expectClose(io, output, expected, .{});

    log.info("NKI add_one example succeeded on {s}", .{@tagName(platform.target)});
}
