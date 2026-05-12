const zml = @import("../../zml.zig");
const ragged_paged = @import("platforms/tpu/ragged_paged");

pub const Kernel = zml.kernel.mosaic_tpu.Kernel(ragged_paged.Cfg, .{
    .name = "ragged_paged_attention_kernel",
    .inputs = &.{ "kv_lens", "page_indices", "cu_q_lens", "seq_buf_idx", "num_seqs", "q", "kv_pages" },
    .outputs = &.{"o"},
    .run = ragged_paged.run,
});
