const std = @import("std");
pub fn sample(logits: []const f32, random: std.Random, temperature: f32, top_k: usize, history: []const u32, penalty: f32, eos: bool) !u32 {
    var values: [2052]f32 = undefined;
    var ids: [2052]u32 = undefined;
    if (logits.len > values.len or logits.len < 2048) return error.InvalidLogits;
    if (top_k == 0 or top_k > 2048 or !std.math.isFinite(temperature) or temperature < 0 or !std.math.isFinite(penalty) or penalty <= 0) return error.InvalidSamplingOptions;
    const limit = if (temperature == 0) 1 else top_k;
    var count: usize = 0;
    for (logits, 0..) |v, i| {
        if (i >= 2048 and !(eos and i == 2051)) continue;
        if (!std.math.isFinite(v)) return error.NonFiniteLogits;
        var score = v;
        if (std.mem.indexOfScalar(u32, history, @intCast(i)) != null) score = if (score < 0) score * penalty else score / penalty;
        if (count == limit and score <= values[count - 1]) continue;
        var j = @min(count, limit - 1);
        while (j > 0 and values[j - 1] < score) : (j -= 1) {
            values[j] = values[j - 1];
            ids[j] = ids[j - 1];
        }
        values[j] = score;
        ids[j] = @intCast(i);
        count = @min(count + 1, limit);
    }
    if (temperature == 0) return ids[0];
    count = @min(count, top_k);
    const max = values[0];
    var total: f32 = 0;
    for (values[0..count]) |*v| {
        v.* = @exp((v.* - max) / temperature);
        total += v.*;
    }
    var pick = random.float(f32) * total;
    for (values[0..count], ids[0..count]) |v, id| {
        pick -= v;
        if (pick <= 0) return id;
    }
    return ids[count - 1];
}
test "reserved codec ids are suppressed but backbone EOS is allowed" {
    var logits: [2052]f32 = @splat(0);
    logits[2048] = 100;
    logits[2051] = 50;
    logits[7] = 10;
    var rng: std.Random.DefaultPrng = .init(1);
    try std.testing.expectEqual(@as(u32, 7), try sample(&logits, rng.random(), 0, 50, &.{}, 1.1, false));
    try std.testing.expectEqual(@as(u32, 2051), try sample(&logits, rng.random(), 0, 50, &.{}, 1.1, true));
}
