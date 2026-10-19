const zml = @import("zml");

const QwenImage21TemporalTimesteps = @import("./QwenImage21TemporalTimesteps.zig");
const TimestepEmbedding = @import("./TimestepEmbedding.zig");

// QwenImage21TimestepProjEmbeddings(
//   (time_proj): QwenImage21TemporalTimesteps()
//   (timestep_embedder): TimestepEmbedding(
//     (linear_1): Linear(in_features=256, out_features=4096, bias=False)
//     (act): SiLU()
//     (linear_2): Linear(in_features=4096, out_features=4096, bias=False)
//   )
// )

const QwenImage21TimestepProjEmbeddings = @This();

time_proj: QwenImage21TemporalTimesteps,
timestep_embedder: TimestepEmbedding,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21TimestepProjEmbeddings)) void {
    TimestepEmbedding.unloadBuffers(&self.timestep_embedder);
}
