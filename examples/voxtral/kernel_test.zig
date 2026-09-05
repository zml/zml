const std = @import("std");
const zml = @import("zml");
const Tensor = zml.Tensor;
const mel = @import("mel_spectrogram.zig");
const common = @import("common.zig");

test "400-point DFT power: impulse, DC, and sinusoid" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var input: [3][400]f32 = @splat(@splat(0));
    input[0][0] = 1;
    @memset(&input[1], 1);
    for (&input[2], 0..) |*sample, i| {
        sample.* = @floatCast(@sin(2.0 * std.math.pi * 17.0 * @as(f64, @floatFromInt(i)) / 400.0));
    }
    var buffer = try zml.Buffer.fromSlice(io, platform, .init(.init(.{ 3, 400 }, .f32), std.mem.asBytes(&input)), .replicated);
    defer buffer.deinit();
    var exe = try platform.compileFn(allocator, io, mel.dftPower, .{Tensor.init(.{ 3, 400 }, .f32)}, .{});
    defer exe.deinit();
    try zml.testing.expectEqualShapes(.init(.{ .frames = 3, .freq_bins = 201 }, .f32), exe.output_shapes[0]);
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var output: zml.Buffer = undefined;
    runner.run(.{buffer}, .{&output});
    defer output.deinit();
    const actual = try output.toSliceAlloc(allocator, io);
    defer actual.free(allocator);
    for (actual.items(f32), 0..) |value, i| {
        const frame = i / 201;
        const bin = i % 201;
        const expected: f32 = switch (frame) {
            0 => 1,
            1 => if (bin == 0) 160000 else 0,
            2 => if (bin == 17) 40000 else 0,
            else => unreachable,
        };
        try std.testing.expectApproxEqAbs(expected, value, @max(1e-4, expected * 1e-5));
    }
}

test "streaming mel frontend preserves shape and Hann-window power" {
    const Local = struct {
        fn forward() Tensor {
            const frontend: mel.LogMelSpectrogram = .{
                .window = .hann,
                .n_fft = 400,
                .hop_len = 160,
                .global_log_mel_max = 1.5,
                .mel_filters = Tensor.scalar(1, .f32).broad(zml.Shape.init(.{ .freq_bins = 201, .mel = 128 }, .f32)),
            };
            return frontend.melStep(Tensor.scalar(1, .f32).broad(zml.Shape.init(.{ .samples = 1520 }, .f32)));
        }
    };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var exe = try platform.compileFn(allocator, io, Local.forward, .{}, .{});
    defer exe.deinit();
    try zml.testing.expectEqualShapes(.init(.{ .channels = 128, .time = 8 }, .f32), exe.output_shapes[0]);
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var output: zml.Buffer = undefined;
    runner.run(.{}, .{&output});
    defer output.deinit();
    const actual = try output.toSliceAlloc(allocator, io);
    defer actual.free(allocator);
    for (actual.items(f32)) |value| {
        try std.testing.expectApproxEqAbs((@log10(@as(f32, 50000)) + 4) / 4, value, 1e-4);
    }
}

test "chunked attention applies a per-query sliding window" {
    const Local = struct {
        fn forward() Tensor {
            const q = Tensor.scalar(0, .f32).broad(zml.Shape.init(.{ .q = 4, .h = 1, .hd = 1 }, .f32));
            const k = Tensor.scalar(0, .f32).broad(zml.Shape.init(.{ .k = 7, .h = 1, .hd = 1 }, .f32));
            const v = Tensor.arange(.{ .end = 7 }, .f32).withTags(.{.k}).broad(k.shape());
            return common.attention(q, k, v, Tensor.scalar(3, .u32), .{ .vanilla = {} }, .{ .vanilla = {} }, 4);
        }
    };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var exe = try platform.compileFn(allocator, io, Local.forward, .{}, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var output: zml.Buffer = undefined;
    runner.run(.{}, .{&output});
    defer output.deinit();
    const actual = try output.toSliceAlloc(allocator, io);
    defer actual.free(allocator);
    for (actual.items(f32), [_]f32{ 1.5, 2.5, 3.5, 4.5 }) |value, expected| {
        try std.testing.expectApproxEqAbs(expected, value, 1e-5);
    }
}
