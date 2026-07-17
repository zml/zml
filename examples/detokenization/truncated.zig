const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");
const algebra = @import("algebra.zig");
const tokens = @import("tokens.zig");
const sampling = @import("sampling.zig");

const LmHeadMatrix = algebra.LmHeadMatrix;
const Sampler = sampling.Sampler;
const Logit = sampling.Logit;
const SamplingResult = sampling.SamplingResult;

const sampling_top_k = sampling.sampling_top_k;
const sampling_truncate_coords = 256;
const sampling_truncate_dense = 256;

pub const QueryCoord = struct {
    coord: usize = 0,
    value: f32 = -std.math.floatMax(f32),
};

pub const TruncateSampler = struct {
    zml_handler: *main.Zml_handler,
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    vocab_size: usize,
    d: usize,
    logits: []f32,
    candidates: [sampling_truncate_dense]Logit,
    
    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix) !TruncateSampler {
        return .{
            .zml_handler = zml_handler,
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .vocab_size = lm_head.n,
            .d = lm_head.d,
            .logits = try zml_handler.allocator.alloc(f32, lm_head.n),
            .candidates = [_]Logit{.{}} ** sampling_truncate_dense,
        };
    }

    pub fn deinit(self: *TruncateSampler) void {
        self.allocator.free(self.logits);
    }

    pub fn sample(self: *TruncateSampler, query: []const f32) SamplingResult {
        // find the top sampling_truncate_coords coordinates of query of max abs value
        var prefix: [sampling_truncate_coords]QueryCoord = [_]QueryCoord{.{}} ** sampling_truncate_coords;
        for (0..self.d) |coord| {
            const v = @abs(query[coord]);
            var insert_pos: usize = sampling_truncate_coords - 1;
            if (v < prefix[insert_pos].value) continue;
            while (insert_pos > 0 and v > prefix[insert_pos - 1].value) {
                prefix[insert_pos] = prefix[insert_pos - 1];
                insert_pos -= 1;
            }
            prefix[insert_pos] = .{ .coord = coord, .value = v };
        }

        // compute partial logits using only the prefix coord
        @memset(self.logits, 0.0);
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);
        for (0..sampling_truncate_coords) |i| {
            const coord = prefix[i].coord;
            const value = query[coord];
            const query_vec: Vec = @splat(value);
            const rows_coord_values = self.lm_head.data_t[coord * self.vocab_size ..][0..self.vocab_size];

            var row_i: usize = 0;
            while (row_i + simd_len <= self.vocab_size) : (row_i += simd_len) {
                const row_vec: Vec = rows_coord_values[row_i..][0..simd_len].*;
                var logits_vec: Vec = self.logits[row_i..][0..simd_len].*;
                logits_vec = @mulAdd(Vec, row_vec, query_vec, logits_vec);
                // is this necessary ?
                self.logits[row_i..][0..simd_len].* = logits_vec;
            }
            while (row_i < self.vocab_size) : (row_i += 1) {
                self.logits[row_i] += rows_coord_values[row_i] * value;
            }
        }

        // find top sampling_truncate_dense partial logits
        for (0..sampling_truncate_dense) |i| {
            self.candidates[i].logit = -std.math.floatMax(f32);
        }
        for (0..self.vocab_size) |token| {
            const logit = self.logits[token];
            var insert_pos: usize = sampling_truncate_dense - 1;
            if (logit <= self.candidates[insert_pos].logit) continue;
            while (insert_pos > 0 and logit > self.candidates[insert_pos - 1].logit) {
                self.candidates[insert_pos] = self.candidates[insert_pos - 1];
                insert_pos -= 1;
            }
            self.candidates[insert_pos] = .{ .row = token, .logit = logit };
        }

        // dense score them and return top k
        for (0..sampling_truncate_dense) |i| {
            self.candidates[i].logit = Sampler.computeLogit(self.lm_head, self.candidates[i].row, query);
        }
        std.mem.sort(Logit, &self.candidates, {}, Logit.decreasingOrder);
        return .{ .candidates = self.candidates[0..sampling_top_k].*, .nb = 0 };
    }

};
