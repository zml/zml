const zml = @import("zml");

const QwenImage21TemporalTimesteps = @This();

timestep_dim: i64 = 256,
max_period: f32 = 10000.0,
time_factor: f32 = 1000.0,

pub fn init() QwenImage21TemporalTimesteps {
    return QwenImage21TemporalTimesteps{};
}

pub fn forward(self: QwenImage21TemporalTimesteps, timestep: zml.Tensor) zml.Tensor {
    const t = timestep.convert(.f32).scale(self.time_factor);
    const args = t.outer(self.computeFreqs());
    const embedding = zml.Tensor.concatenate(&.{ args.cos(), args.sin() }, .d);
    if (@mod(self.timestep_dim, 2) == 1) {
        return embedding.pad(0, .{ .d = zml.Tensor.Pad{ .high = 1 } });
    }
    return embedding;
}

fn computeFreqs(self: QwenImage21TemporalTimesteps) zml.Tensor {
    const half = @divFloor(self.timestep_dim, 2);
    return zml.Tensor.arange(.{ .end = half }, .f32)
        .scale(-@log(self.max_period) / @as(f32, @floatFromInt(half)))
        .exp()
        .withTags(.{.d});
}
