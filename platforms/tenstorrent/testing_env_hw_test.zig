//! Opens the cards the way every ZML package test does with the Tenstorrent
//! flag: through `zml.testing.env()`, whose platform is never deinit'ed. The
//! platform layer must close the client when the process exits, or the cards
//! stay unusable until they are reset; a hardware test run after this one
//! shows they are not (Bazel tag tt_hardware, run under the card lock).

const std = @import("std");

const zml = @import("zml");

const smoke = @import("smoke.zig");

fn scaleShift(x: zml.Tensor) zml.Tensor {
    return x.scale(2).addConstant(1).withPartitioning(.model, .{ .b = .model, .d = .replicated });
}

test "zml.testing.env() runs y = x * 2 + 1 on the cards and leaves them open for the exit release" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const platform = zml.testing.env();
    try std.testing.expect(platform.target == .tenstorrent);
    const mesh = zml.testing.tpMesh();

    const x = try smoke.input(allocator, platform.devices.len);
    defer allocator.free(x);
    const shape = zml.Shape.init(.{ .b = @as(i64, @intCast(x.len / smoke.cols)), .d = smoke.cols }, .f32)
        .withPartitioning(mesh, .{ .b = .model });

    var exe = try platform.compileFn(allocator, io, scaleShift, .{zml.Tensor.fromShape(shape)}, .{
        .meshes = &.{mesh},
        .program_name = "scale_shift",
    });
    defer exe.deinit();

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, shape, std.mem.sliceAsBytes(x));
    defer x_buffer.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var y_buffer: zml.Buffer = undefined;
    runner.run(io, .{x_buffer}, .{&y_buffer}, .{ .wait = true });
    defer y_buffer.deinit();

    const y = try y_buffer.toSliceAlloc(allocator, io);
    defer y.free(allocator);
    for (x, y.items(f32)) |xi, yi| try std.testing.expectEqual(xi * 2 + 1, yi);
}
