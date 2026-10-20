const zml = @import("zml");

const QwenImage21Rope = @This();

theta: f32 = 10000,
axes_dim: [3]i64 = .{ 16, 56, 56 },

pub fn init() QwenImage21Rope {
    return QwenImage21Rope{};
}

pub fn forward(self: QwenImage21Rope, text_len: i64, height: i64, width: i64) zml.Tensor {
    const text_pos = zml.Tensor.arange(.{ .end = text_len }, .f32).withTags(.{.seq_len});

    const grid: zml.Shape = .init(.{ .h = height, .w = width }, .f32);
    const image_frame = zml.Tensor.scalar(text_len, .f32).broad(.init(.{ .seq_len = height * width }, .f32));
    const image_h = zml.Tensor.iota(grid, .h).convert(.f32)
        .addConstant(-(height - @divFloor(height, 2)))
        .merge(.{ .seq_len = .{ .h, .w } });
    const image_w = zml.Tensor.iota(grid, .w).convert(.f32)
        .addConstant(-(width - @divFloor(width, 2)))
        .merge(.{ .seq_len = .{ .h, .w } });

    const positions = [3]zml.Tensor{
        zml.Tensor.concatenate(&.{ text_pos, image_frame }, .seq_len),
        zml.Tensor.concatenate(&.{ text_pos, image_h }, .seq_len),
        zml.Tensor.concatenate(&.{ text_pos, image_w }, .seq_len),
    };

    var angles: [3]zml.Tensor = undefined;
    for (positions, self.axes_dim, 0..) |pos, dim, i| {
        angles[i] = pos.outer(self.compute_inv_freq(dim));
    }
    const angle = zml.Tensor.concatenate(&angles, .hd);
    return zml.Tensor.complex(angle.cos(), angle.sin());
}

fn compute_inv_freq(self: QwenImage21Rope, dim: i64) zml.Tensor {
    const exponent = zml.Tensor.arange(.{ .end = dim, .step = 2 }, .f32).scale(1.0 / @as(f32, @floatFromInt(dim)));
    const freq = zml.Tensor.scalar(self.theta, .f32).pow(exponent);
    return zml.Tensor.scalar(1, .f32).div(freq).withTags(.{.hd});
}
