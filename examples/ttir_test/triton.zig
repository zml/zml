const std = @import("std");
const zml = @import("zml");

pub fn main() !void {
    // Dummy tensor dimensions (adjust as needed)
    const token_count = 8;
    const num_heads = 4;
    const head_size = 16;

    // Dummy tensors (replace with actual ZML tensor creation as needed)
    var query = zml.Tensor.initDummy([token_count, num_heads, head_size]);
    var key_cache = zml.Tensor.initDummy([token_count, num_heads, head_size]);
    var value_cache = zml.Tensor.initDummy([token_count, num_heads, head_size]);
    var o = zml.Tensor.initDummy([token_count, num_heads, head_size]);
    var start_loc = zml.Tensor.initDummy([token_count]);
    var max_input_len: i32 = 128;
    var context_seq_lens = zml.Tensor.initDummy([token_count]);
    var max_seq_len: i32 = 128;
    var scale: f32 = 1.0;
    var block_tables = zml.Tensor.initDummy([token_count]);
    var k_scale: ?f32 = null;
    var v_scale: ?f32 = null;

    const args = .{
        query,
        key_cache,
        value_cache,
        o,
        start_loc,
        max_input_len,
        context_seq_lens,
        max_seq_len,
        scale,
        true, // causal
        .{-1, -1}, // window_size
        block_tables,
        0.0, // softcap
        null, // q_descale
        k_scale,
        v_scale,
    };

    const grid = .{1, 1, 1}; // Dummy grid, adjust as needed
    const result = zml.ops.triton(
        args,
        .{query.shape()}, // Output shape
        .{
            .name = "2d_unified_attention",
            .ir = @embedFile("2d_unified_attention.ttir"),
            .grid = grid,
            .num_stages = 1,
            .num_warps = 4,
            .debug = true,
            .output_operand_aliases = &.{3}, // Index of output tensor in args
        }
    )[0];

    std.debug.print("Result shape: {}\n", .{result.shape()});
}
