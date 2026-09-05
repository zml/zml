const std = @import("std");
const zml = @import("zml");
const Tensor = zml.Tensor;
const codec = @import("codec.zig");
test {
    std.testing.refAllDecls(@import("tokenizer.zig"));
    std.testing.refAllDecls(@import("audio_io.zig"));
}

fn evaluate(comptime function: anytype, expected: []const f32) !void {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var zero = try zml.Buffer.scalar(io, platform, 0, .f32);
    defer zero.deinit();
    var exe = try platform.compileFn(a, io, function, .{Tensor.init(.{}, .f32)}, .{});
    defer exe.deinit();
    var runner = try exe.runner(a);
    defer runner.deinit(a);
    var output: zml.Buffer = undefined;
    runner.run(.{zero}, .{&output});
    defer output.deinit();
    const actual = try output.toSliceAlloc(a, io);
    defer actual.free(a);
    try std.testing.expectEqual(expected.len, actual.items(f32).len);
    for (actual.items(f32), expected) |value, target| {
        if (std.math.isInf(target)) {
            try std.testing.expectEqual(target, value);
        } else try std.testing.expectApproxEqAbs(target, value, 1e-5);
    }
}

test "codec transposed convolution reverses taps and trims only the right" {
    const Local = struct {
        fn forward(zero: Tensor) Tensor {
            const taps = [_]f32{ 1, 2, 3, 4 };
            const conv: codec.Conv = .{
                .direction = Tensor.constantTensor(.init(.{ .cout = 1, .cin = 1, .kernel = 4 }, .f32), std.mem.asBytes(&taps)),
                .magnitude = Tensor.scalar(@sqrt(@as(f32, 30)), .f32).broad(.init(.{ .cout = 1, .cin = 1, .kernel = 1 }, .f32)),
            };
            const input = [_]f32{ 1, 2, 3 };
            return conv.forward(Tensor.constantTensor(.init(.{ .s = 3, .d = 1 }, .f32), std.mem.asBytes(&input)).add(zero), true, false);
        }
    };
    try evaluate(Local.forward, &.{ 1, 2, 5, 8, 9, 14 });
}

test "codec reflection padding does not repeat the first sample" {
    const Local = struct {
        fn forward(zero: Tensor) Tensor {
            const conv: codec.Conv = .{
                .direction = Tensor.scalar(1, .f32).broad(.init(.{ .cout = 1, .cin = 1, .kernel = 3 }, .f32)),
                .magnitude = Tensor.scalar(@sqrt(@as(f32, 3)), .f32).broad(.init(.{ .cout = 1, .cin = 1, .kernel = 1 }, .f32)),
            };
            const input = [_]f32{ 1, 2, 3, 4 };
            return conv.forward(Tensor.constantTensor(.init(.{ .s = 4, .d = 1 }, .f32), std.mem.asBytes(&input)).add(zero), false, true);
        }
    };
    try evaluate(Local.forward, &.{ 6, 5, 6, 9 });
}

test "transposed convolution maps input and output channels correctly" {
    const Local = struct {
        fn forward(zero: Tensor) Tensor {
            const taps = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
            const magnitudes = [_]f32{ @sqrt(@as(f32, 650)), @sqrt(@as(f32, 4250)) };
            const conv: codec.Conv = .{
                .direction = Tensor.constantTensor(.init(.{ .cout = 2, .cin = 3, .kernel = 4 }, .f32), std.mem.asBytes(&taps)),
                .magnitude = Tensor.constantTensor(.init(.{ .cout = 2, .cin = 1, .kernel = 1 }, .f32), std.mem.asBytes(&magnitudes)),
            };
            const input = [_]f32{ 1, 10, 2, 20 };
            return conv.forward(Tensor.constantTensor(.init(.{ .s = 2, .d = 2 }, .f32), std.mem.asBytes(&input)).add(zero), true, false);
        }
    };
    try evaluate(Local.forward, &.{ 131, 175, 219, 142, 186, 230, 415, 547, 679, 448, 580, 712 });
}

test "codec ALiBi mask has causal inclusive left windows" {
    const Local = struct {
        fn forward(zero: Tensor) Tensor {
            return codec.alibiMask(4, 2).slice(.h, .single(0)).convert(.f32).add(zero);
        }
    };
    const neg = -std.math.inf(f32);
    try evaluate(Local.forward, &.{ 0, neg, neg, neg, -1, 0, neg, neg, -2, -1, 0, neg, neg, -2, -1, 0 });
}

test "acoustic quantizer clips and rounds half to even on device" {
    const Local = struct {
        fn forward(x: Tensor) Tensor {
            return @import("model.zig").quantize(x.convert(.bf16));
        }
    };
    const a = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const input = [_]f32{ -2, -1, -0.75, -0.25, 0.25, 0.75, 1, 2 };
    const shape = zml.Shape.init(.{ .cb = input.len }, .f32);
    var buffer = try zml.Buffer.fromSlice(io, platform, .init(shape, std.mem.asBytes(&input)), .replicated);
    defer buffer.deinit();
    var exe = try platform.compileFn(a, io, Local.forward, .{Tensor.init(shape, .f32)}, .{});
    defer exe.deinit();
    var runner = try exe.runner(a);
    defer runner.deinit(a);
    var output: zml.Buffer = undefined;
    runner.run(.{buffer}, .{&output});
    defer output.deinit();
    const actual = try output.toSliceAlloc(a, io);
    defer actual.free(a);
    try std.testing.expectEqualSlices(u32, &.{ 2, 2, 4, 10, 14, 20, 22, 22 }, actual.items(u32));
}
