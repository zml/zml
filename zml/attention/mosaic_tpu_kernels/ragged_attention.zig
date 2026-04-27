//! Declarative `zml.kernel.mosaic_tpu.Kernel` wrapper around the
//! Mosaic-TPU ragged paged attention IR built by
//! `platforms/tpu/ragged_paged.zig`. Production callers reach the kernel
//! through `Kernel.call(inputs, outputs, opts)`; the IR string is reachable
//! offline via `Kernel.emit(allocator, ctx, cfg)`.

const zml = @import("../../zml.zig");
const ragged_paged = @import("platforms/tpu/ragged_paged");

pub const Cfg = ragged_paged.Cfg;

pub const Kernel = zml.kernel.mosaic_tpu.Kernel(Cfg, .{
    .name = "ragged_paged_attention_kernel",
    .inputs = &.{ "kv_lens", "page_indices", "cu_q_lens", "seq_buf_idx", "num_seqs", "q", "kv_pages" },
    .outputs = &.{"o"},
    .run = ragged_paged.run,
});
