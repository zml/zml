const std = @import("std");

const benchmark = @import("benchmark.zig");
const cases = @import("cases.zig");

pub fn main(init: std.process.Init) !void {
    try benchmark.run(init, &.{
        cases.Case{ .name = "qwen_decode", .head_dim = 256, .batch = 16, .kv_heads = 1, .group = 12, .kv_len = 4096, .mask = .indices, .selection_count = 2051, .random_indices = true, .shuffle_pages = true },
        cases.Case{ .name = "qwen_decode_single", .head_dim = 256, .batch = 1, .kv_heads = 1, .group = 12, .kv_len = 4096, .mask = .indices, .selection_count = 2051, .random_indices = true, .shuffle_pages = true },
        cases.Case{ .name = "qwen_prefill", .head_dim = 256, .batch = 2, .kv_heads = 1, .group = 12, .prefill = 2, .chunk = 64, .kv_len = 4096, .mask = .indices, .selection_count = 2051, .random_indices = true, .shuffle_pages = true },
        cases.Case{ .name = "qwen_mixed", .head_dim = 256, .batch = 16, .kv_heads = 1, .group = 12, .prefill = 4, .chunk = 64, .kv_len = 4096, .mask = .indices, .selection_count = 2051, .random_indices = true, .shuffle_pages = true },
        cases.Case{ .name = "qwen_prefill_short", .head_dim = 256, .batch = 2, .kv_heads = 1, .group = 12, .prefill = 2, .chunk = 64, .kv_len = 320, .mask = .indices, .selection_count = 2051, .padded_selection = true },
    }, 32);
}
