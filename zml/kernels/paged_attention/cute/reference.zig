const std = @import("std");
const zml = @import("zml");
const cases = @import("cases.zig");
const fixtures = @import("fixtures.zig");

/// One query/head, independent of GPU tensor operations. Duplicate indices are
/// separate softmax entries; invalid indices are excluded.
pub fn row(c: cases.Case, f: fixtures.Fixture, seq: usize, token: usize, head: usize, scores: []f32, out: []f32) void {
    const len: usize = @intCast(f.lengths[seq]);
    const start: usize = @intCast(f.starts[seq]);
    const qlen: usize = @intCast(f.starts[seq + 1] - f.starts[seq]);
    const position = len - qlen + token - start;
    const count = if (c.mask == .indices) cases.topk else len;
    const scale = c.scale orelse 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    const q = f.q[(token * c.heads() + head) * c.head_dim ..][0..c.head_dim];
    var maximum: f32 = if (c.sink) f.sinks[head] else -std.math.inf(f32);
    for (0..count) |j| {
        const key = keyAt(c, f, token, j);
        const valid = key >= 0 and key < len and switch (c.mask) {
            .indices, .non_causal => true,
            .causal => key <= position,
            .window => key <= position and @as(i64, @intCast(key)) > @as(i64, @intCast(position)) - c.window,
        };
        scores[j] = -std.math.inf(f32);
        if (!valid) continue;
        const offset = f.kvOffset(c, seq, @intCast(key), head / c.group);
        var dot: f32 = 0;
        for (q, f.k[offset..][0..c.head_dim]) |qv, kv| dot += qv.toF32() * kv.toF32();
        scores[j] = dot * scale;
        maximum = @max(maximum, scores[j]);
    }
    @memset(out, 0);
    if (maximum == -std.math.inf(f32)) return;
    var denominator: f32 = if (c.sink) @exp(f.sinks[head] - maximum) else 0;
    for (0..count) |j| {
        if (scores[j] == -std.math.inf(f32)) continue;
        const weight = @exp(scores[j] - maximum);
        denominator += weight;
        const offset = f.kvOffset(c, seq, @intCast(keyAt(c, f, token, j)), head / c.group);
        for (out, f.v[offset..][0..c.head_dim]) |*o, v| o.* += weight * v.toF32();
    }
    for (out) |*v| v.* /= denominator;
}

fn keyAt(c: cases.Case, f: fixtures.Fixture, token: usize, j: usize) i64 {
    return if (c.mask == .indices) f.indices[token * cases.topk + j] else @intCast(j);
}

pub fn check(a: std.mem.Allocator, c: cases.Case, f: fixtures.Fixture, actual: []const zml.floats.BFloat16) !f32 {
    const active_tokens: usize = @intCast(f.starts[c.batch]);
    for (actual[0 .. active_tokens * c.heads() * c.head_dim]) |v| {
        if (!std.math.isFinite(v.toF32())) return error.NonFiniteOutput;
    }
    const scores = try a.alloc(f32, @max(c.kv_len, cases.topk));
    defer a.free(scores);
    const expected = try a.alloc(f32, c.head_dim);
    defer a.free(expected);
    var max_error: f32 = 0;
    var failures: usize = 0;
    for (0..c.batch) |s| {
        for (@intCast(f.starts[s])..@intCast(f.starts[s + 1])) |t| {
            for (0..c.heads()) |h| {
                row(c, f, s, t, h, scores, expected);
                const base = (t * c.heads() + h) * c.head_dim;
                for (expected, actual[base..][0..c.head_dim], 0..) |want, got_bf16, d| {
                    const got = got_bf16.toF32();
                    const err = @abs(want - got);
                    max_error = @max(max_error, err);
                    // Fixed BF16 error budget; never adapted to observed failures.
                    if (!std.math.isFinite(want) or err > 0.003 + 0.02 * @abs(want)) {
                        if (failures == 0) std.debug.print("Mismatch seq={d} token={d} head={d} dim={d}: expected={d} actual={d}\n", .{ s, t, h, d, want, got });
                        failures += 1;
                    }
                }
            }
        }
    }
    if (failures > 0) return error.ReferenceMismatch;
    return max_error;
}

test "reference single key, uniform causal scores, and sink" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c: cases.Case = .{ .head_dim = 64, .batch = 1, .kv_heads = 1, .group = 1, .prefill = 1, .chunk = 2, .kv_len = 2 };
    const f = try fixtures.Fixture.init(arena.allocator(), c);
    @memset(f.q, .fromF32(0));
    @memset(f.v[0..64], .fromF32(2));
    @memset(f.v[64..128], .fromF32(4));
    var scores: [2]f32 = undefined;
    var out: [64]f32 = undefined;
    row(c, f, 0, 0, 0, &scores, &out);
    for (out) |v| try std.testing.expectEqual(@as(f32, 2), v);
    row(c, f, 0, 1, 0, &scores, &out);
    for (out) |v| try std.testing.expectEqual(@as(f32, 3), v);
    c.sink = true;
    f.sinks[0] = 0;
    row(c, f, 0, 1, 0, &scores, &out);
    for (out) |v| try std.testing.expectEqual(@as(f32, 2), v);
}

test "reference windows, noncausal, duplicate and invalid indices" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c: cases.Case = .{ .head_dim = 64, .batch = 1, .kv_heads = 1, .group = 1, .prefill = 1, .chunk = 2, .kv_len = 2 };
    const f = try fixtures.Fixture.init(arena.allocator(), c);
    @memset(f.q, .fromF32(0));
    @memset(f.v[0..64], .fromF32(2));
    @memset(f.v[64..128], .fromF32(4));
    var scores: [cases.topk]f32 = undefined;
    var out: [64]f32 = undefined;
    c.mask = .non_causal;
    row(c, f, 0, 0, 0, &scores, &out);
    for (out) |v| try std.testing.expectEqual(@as(f32, 3), v);
    c.mask = .window;
    c.window = 1;
    row(c, f, 0, 1, 0, &scores, &out);
    for (out) |v| try std.testing.expectEqual(@as(f32, 4), v);
    c.mask = .indices;
    @memset(f.indices, -1);
    @memcpy(f.indices[0..4], &[_]i32{ 0, 0, 1, 2 });
    row(c, f, 0, 0, 0, &scores, &out);
    for (out) |v| try std.testing.expectApproxEqAbs(@as(f32, 8.0 / 3.0), v, 1e-6);
    row(c, f, 0, 1, 0, &scores, &out);
    for (out) |v| try std.testing.expectEqual(@as(f32, 0), v);
}
