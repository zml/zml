const zml = @import("zml");

// TimestepEmbedding(
//   (linear_1): Linear(in_features=256, out_features=4096, bias=False)
//   (act): SiLU()
//   (linear_2): Linear(in_features=4096, out_features=4096, bias=False)
// )

pub const TimestepEmbedding = @This();

linear_1: zml.nn.Linear,
linear_2: zml.nn.Linear,

pub fn unloadBuffers(self: *zml.Bufferized(TimestepEmbedding)) void {
    zml.nn.Linear.unloadBuffers(&self.linear_1);
    zml.nn.Linear.unloadBuffers(&self.linear_2);
}
