// AVX-512 32-bit float vector type.  We may want to
// generalize this later to other types (e.g. u16 for bfloat16).
const VF = @Vector(16, f32);
const std = @import("std");

/// Options controlling generation. The default values correspond to greedy decoding.
const SamplingStrategy = struct {
    topk: u32 = 1,
    topp: ?f32 = null, 
    temperature: f32 = 1.0
};

// for now, rng is anytype.  We'll do better later.

/// Selects a token given activations and sampling options.  The output is the
/// index of the selected token.
pub fn sample(activations: []const VF, opts: SamplingStrategy, rng: anytype) usize {
    _ = rng;
    if (opts.topk == 1) return greedy_sample(activations);
    @panic("not implemented yet");
}

const minfloat = -std.math.inf(f32);

fn greedy_sample(activations: []const VF) usize {
    var max_index: usize = 0;
    var max_value = minfloat;
    for (activations, 0..) | chunk, chunk_index | {
        const this_max = @reduce(.Max, chunk);
        if (this_max > max_value) {
            max_value = this_max;
            const max_chunk: VF = @splat(max_value);
            const map = max_chunk == chunk;
            // take the first index that matches
            const sub_index = @ctz(@as(u16, @bitCast(map)));
            max_index = chunk_index * 16 + sub_index;
        }
    }
    return max_index;
}

test "greedy sample works" {
    const activations = [_]VF{
        .{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
        .{16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
    };

    try std.testing.expectEqual(15, greedy_sample(activations[0..]));
}
