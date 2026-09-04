const std = @import("std");

// =============================================================================
// draft/rope.zig — text-encoder mRoPE on host
// =============================================================================

/// Text-encoder mRoPE: one position per token, broadcast across T/H/W.
pub fn fillArangePositions(out: []f32, seq: u32) void {
    var i: u32 = 0;
    while (i < seq) : (i += 1) {
        const v: f32 = @floatFromInt(i);
        out[i * 3 + 0] = v;
        out[i * 3 + 1] = v;
        out[i * 3 + 2] = v;
    }
}

/// Interleaved MiniMax mRoPE on the host. `section` is the official (t,h,w) split.
pub fn hostInterleavedMrope(
    pos: []const f32,
    seq: u32,
    head_dim: u32,
    theta: f32,
    section: [3]i64,
    cos: []f32,
    sin: []f32,
) void {
    const half = head_dim / 2;
    std.debug.assert(pos.len >= seq * 3);
    std.debug.assert(cos.len >= seq * head_dim);
    var i: u32 = 0;
    while (i < seq) : (i += 1) {
        const pt = pos[i * 3 + 0];
        const ph = pos[i * 3 + 1];
        const pw = pos[i * 3 + 2];
        var f: u32 = 0;
        while (f < half) : (f += 1) {
            var p = pt;
            const h_end = @as(u32, @intCast(section[1] * 3));
            const w_end = @as(u32, @intCast(section[2] * 3));
            if (f < h_end and f % 3 == 1) p = ph;
            if (f < w_end and f % 3 == 2) p = pw;
            const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(half)));
            const ang = p * freq;
            const c = @cos(ang);
            const s = @sin(ang);
            cos[i * head_dim + f] = c;
            cos[i * head_dim + half + f] = c;
            sin[i * head_dim + f] = s;
            sin[i * head_dim + half + f] = s;
        }
    }
}
