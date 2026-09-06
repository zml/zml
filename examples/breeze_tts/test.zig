const std = @import("std");
test {
    std.testing.refAllDecls(@import("config.zig"));
    std.testing.refAllDecls(@import("audio.zig"));
    std.testing.refAllDecls(@import("sampling.zig"));
}
const zml = @import("zml");
const T = zml.Tensor;
fn evaluate(comptime f: anytype, expected: []const f32) !void {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const p = zml.testing.env();
    var zero = try zml.Buffer.scalar(io, p, 0, .f32);
    defer zero.deinit();
    var exe = try p.compileFn(a, io, f, .{T.init(.{}, .f32)}, .{});
    defer exe.deinit();
    var runner = try exe.runner(a);
    defer runner.deinit(a);
    var result: zml.Buffer = undefined;
    runner.run(.{zero}, .{&result});
    defer result.deinit();
    const s = try result.toSliceAlloc(a, io);
    defer s.free(a);
    try std.testing.expectEqual(expected.len, s.items(f32).len);
    for (s.items(f32), expected) |got, want| {
        if (std.math.isInf(want)) try std.testing.expectEqual(want, got) else try std.testing.expectApproxEqAbs(want, got, 1e-5);
    }
}
test "causal transposed convolution reverses taps and trims the right edge" {
    const Local = struct {
        fn forward(zero: T) T {
            const taps = [_]f32{ 1, 2, 3, 4 };
            const values = [_]f32{ 1, 2, 3 };
            const conv: @import("codec.zig").Conv = .{ .weight = T.constantTensor(.init(.{ .cout = 1, .cin = 1, .kernel = 4 }, .f32), std.mem.asBytes(&taps)), .bias = null, .stride = 2, .dilation = 1, .transpose = true, .groups = 1 };
            return conv.forward(T.constantTensor(.init(.{ .s = 3, .d = 1 }, .f32), std.mem.asBytes(&values)).add(zero));
        }
    };
    try evaluate(Local.forward, &.{ 1, 2, 5, 8, 9, 14 });
}
test "causal encoder convolution pads incomplete final strides" {
    const Local = struct {
        fn forward(zero: T) T {
            const values = [_]f32{ 1, 2, 3, 4, 5 };
            const conv: @import("codec.zig").Conv = .{ .weight = T.scalar(1, .f32).broad(.init(.{ .cout = 1, .cin = 1, .kernel = 4 }, .f32)), .bias = null, .stride = 2, .dilation = 1, .transpose = false, .groups = 1 };
            return conv.forward(T.constantTensor(.init(.{ .s = 5, .d = 1 }, .f32), std.mem.asBytes(&values)).add(zero));
        }
    };
    try evaluate(Local.forward, &.{ 3, 10, 12 });
}
test "text encoder local attention includes both directions with asymmetric edges" {
    const Local = struct {
        fn forward(zero: T) T {
            return @import("layers.zig").mask(4, 4, T.scalar(0, .u32), 2, true, .f32).add(zero);
        }
    };
    const n = -std.math.inf(f32);
    try evaluate(Local.forward, &.{ 0, 0, n, n, n, 0, 0, n, n, n, 0, 0, n, n, n, 0 });
}
test "Mimi final downsampling replicates both boundary samples" {
    const Local = struct {
        fn forward(zero: T) T {
            const values = [_]f32{ 1, 2, 3, 4, 5 };
            const conv: @import("codec.zig").Conv = .{ .weight = T.scalar(1, .f32).broad(.init(.{ .cout = 1, .cin = 1, .kernel = 4 }, .f32)), .bias = null, .stride = 2, .dilation = 1, .transpose = false, .groups = 1, .replicate = true };
            return conv.forward(T.constantTensor(.init(.{ .s = 5, .d = 1 }, .f32), std.mem.asBytes(&values)).add(zero));
        }
    };
    try evaluate(Local.forward, &.{ 5, 10, 17 });
}
