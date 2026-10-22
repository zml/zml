const zml = @import("zml");

pub const QwenImage21DupUp3D = @This();

out_channels: i64,
factor_t: i64,
factor_s: i64 = 1,

pub fn forward(
    self: QwenImage21DupUp3D,
    x: zml.Tensor,
    // first_chunk: bool, // Always true
) zml.Tensor {
    const ft = self.factor_t;
    const fs = self.factor_s;
    const repeats = @divExact(self.out_channels * ft * fs * fs, x.dim(.c));

    const interleaved = x.stutter1d(@intCast(x.axis(.c)), @intCast(repeats));
    const expanded = interleaved.splitAxis(.c, .{ .c = self.out_channels, .sa = ft, .sb = fs, .sd = fs });
    const permuted = expanded.transpose(.{ .bs, .c, .t, .sa, .kx, .sb, .ky, .sd });
    const merged = permuted.merge(.{ .t = .{ .t, .sa }, .kx = .{ .kx, .sb }, .ky = .{ .ky, .sd } });

    // if (first_chunk) {
    return merged.slice(.t, .{ .start = ft - 1 });
    // }
    // return merged;
}
