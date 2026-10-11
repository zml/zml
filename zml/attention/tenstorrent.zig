const std = @import("std");

const zml = @import("../zml.zig");

/// tt-metal's paged KV cache operations.
pub const paged_cache = @import("tt_kernels/paged_cache.zig");
/// tt-metal's flash-decode attention, over a paged cache or not.
pub const sdpa_decode = @import("tt_kernels/sdpa_decode.zig");
/// tt-metal's flash-attention prefill, plain or chunked over a paged cache.
pub const sdpa = @import("tt_kernels/sdpa.zig");

/// Cores the attention kernels run on.
const grid: [2]u32 = .{ 8, 8 };

/// Causal attention with tt-metal's kernels (`tt_kernels/`), scaled by
/// `1/sqrt(hd)`.
///
/// Decode (`q.dim(.q) == 1`) runs flash decode with `token_index` as each
/// user's position. Prefill runs flash attention over the first `q.dim(.q)`
/// keys: the prompt starts at position 0.
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const is_decode = q.dim(.q) == 1;
    const had_b = q.shape().hasTag(.b) != null;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(q.dim(.hd))));

    // Keys and values as `[b, h, k, hd]`.
    const k4 = if (had_b) k.transpose(.{ .b, .h, .k, .hd }) else k.insertAxes(0, .{.b}).transpose(.{ .b, .h, .k, .hd });
    const v4 = if (had_b) v.transpose(.{ .b, .h, .k, .hd }) else v.insertAxes(0, .{.b}).transpose(.{ .b, .h, .k, .hd });

    if (is_decode) {
        // `[1, b, h, hd]` queries, one i32 position per user.
        const q4 = if (had_b) q.transpose(.{ .q, .b, .h, .hd }) else q.insertAxes(.h, .{.b});
        const positions = if (had_b and token_index.shape().hasTag(.b) != null)
            token_index
        else if (had_b)
            token_index.insertAxes(.last, .{.b}).broad(zml.Shape.init(.{ .b = q.dim(.b) }, token_index.dtype()))
        else
            token_index.insertAxes(.last, .{.b});
        const out = decodeHeadGroups(q4, k4, v4, positions.convert(.i32), .{ .scale = scale, .grid = grid });
        return if (had_b) out.transpose(.{ .b, .q, .h, .hd }) else out.squeeze(.b);
    }

    const q4 = if (had_b) q.transpose(.{ .b, .h, .q, .hd }) else q.insertAxes(0, .{.b}).transpose(.{ .b, .h, .q, .hd });
    const len = q.dim(.q);
    const out = sdpa.sdpa(q4, k4.slice(.k, .{ .end = len }), v4.slice(.k, .{ .end = len }), .{ .scale = scale, .grid = grid });
    return if (had_b) out.transpose(.{ .b, .q, .h, .hd }) else out.transpose(.{ .q, .b, .h, .hd }).squeeze(.b);
}

/// tt-metal's causal flash decode handles up to 32 query heads (one tile of
/// rows); like tt-metal's models, more heads are split in groups of 32, each
/// with its key-value heads.
pub fn decodeHeadGroups(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, positions: zml.Tensor, opts: sdpa_decode.Options) zml.Tensor {
    const heads = q.dim(.h);
    if (heads <= 32) return sdpa_decode.sdpaDecode(q, k, v, positions, opts);
    const kv_heads = k.dim(.h);
    const ratio = @divExact(heads, kv_heads);
    std.debug.assert(@mod(heads, 32) == 0 and (@mod(32, ratio) == 0 or @mod(ratio, 32) == 0));
    var groups: [64]zml.Tensor = undefined;
    const n: usize = @intCast(@divExact(heads, 32));
    for (groups[0..n], 0..) |*g, i| {
        const start: i64 = @intCast(32 * i);
        const kv_start = @divFloor(start, ratio);
        const kv_end = @divFloor(start + 32 + ratio - 1, ratio);
        g.* = sdpa_decode.sdpaDecode(
            q.slice(.h, .{ .start = start, .end = start + 32 }),
            k.slice(.h, .{ .start = kv_start, .end = kv_end }),
            v.slice(.h, .{ .start = kv_start, .end = kv_end }),
            positions,
            opts,
        );
    }
    return zml.Tensor.concatenate(groups[0..n], .h);
}
