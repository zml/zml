const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");

// This class will be used to implement a cascading style sampler based on the
// orthogonal basis obtained from the SVD decomposition of the lm_head matrix,
// which we call the SVD basis.
// 
// We note d = hidden_size
// We note v = vocab_size
// 
// Given a lm_head matrix H of dimensions [v, d], the SVD decomposition
// of H is H = U * S * V^T, where U and V are rotation matrices of sizes
// [d, d] and [v, v] respectively, and S is diagonal.
// 
// By forming G = H^T * H, which is a [d, d] symmetric positive definite matrix,
// the eigendecomposition of G is G = U * S^2 * U^T.
// The singular values are then the square roots of the eigen values,
// and the SVD basis is U.
// 
// We note M = H * U the representation of the lm_head in the SVD basis.
// 
// Given an embedding x in R^d, the logits we need for sampling are H * x
// the i-th logit is logit(i) = Hi * i where Hi is the i-th row of H
// using U*U^T = Id, we can write:
//     logit(i) = Hi * (U * U^T) * x
//              = (Hi * U) * (U^T * x)
//              = Mi * y,
// where y is the expression of x in the SVD basis: y = U^T * x.
// this means that the logit computation can be performed naively in the SVD basis.
// 
// The diagonal matrix S has the property that singular values appear
// in descending order along the diagonal. This means in the SVD basis,
// the coordinates are of decreasing contribution to the logit. This property
// is what we will exploit to devise the token sampling strategy.
// 
// We define a set of checkpoints C = { 1, 8, 64, 256 }.
// Given an already rotated embedding vector y to sample from,
// we have that logit(i) = sum_j Mi_j * y_j
// For any k in C, we have that logit(i) = sum_{j <= k} Mi_j * y_j + sum_{j > k} Mi_j * y_j
// For any k, this defines an approximation of the logits:
//     logit(i) ≈ sum_{j <= k} Mi_j * y_j = logit_approx(i, k)
// With Cauchy-Schwarz inequality, we can bound the error of the approximation:
//     |logit(i) - logit_approx(i, k)| = |sum_{j > k} Mi_j * y_j|
//                                     <= ||Mi||_{>k} * ||y||_{>k} = Err(i, k)
// Where ||.||_{>k} is the L2 norm of the vector when ignoring the first k coordinates.
// This gives an upper bound on the real logit for each token i:
//     logit(i) <= logit_approx(i, k) + Err(i, k)
// If i_max is the index of the token with the largest approximate logit, then:
//     logit(i_max) >= logit_approx(i_max, k) - Err(i_max, k)
// This means that for any k, we have a lower bound on the larget final logit.
// This allows us to prune tokens based on a logit threshold value.
// If logit(i) <= logit(j) + a then exp(logit(i)) <= exp(a) * exp(logit(j)),
// meaning that token i has a probability lower than exp(a) times that of token j.
// If we decide to use exp(a) = 0,5 as a pruning threshold, we can prune tokens
// with logit(i) <= logit(j) - ln(2) = logit(j) - 0,693 for any i and j.
// We generalize by havind a variable threshold = -ln(2) = -0,693 by default but might change later.
// The sampling algorithm is then:
//     Compute the ||y||_{>k} values for all k
//     For k in C:
//         Compute approximate logits for k
//         Compute best lower bound on max logit: L = max_i logit_approx(i, k) - Err(i, k)
//         For each i:
//             If logit_approx(i, k) + Err(i, k) <= L -  then prune token i
//     Compute real logits for all remaining tokens
//     Sample with softmax using the real logits
// 
// The ||Mi||_{>k} values for all checkpoints can be precomputed as they remain constant.
// 
// This first version is the "safe" version, as the CS inequality is deterministic.
// In real world scenarios, the tail of >k coordinates never exactly align with the
// worst case of CS, making the inequality quite pessimistic. By treating the data
// as random gaussian variables, we can tighten the bound by orders of magnitude.
// In this case we use the following Z-score error bound:
// |logit(i) - logit_approx(i, k)| <= Z * Err(i, k) / sqrt(d - k)
// With Z = 5, the success probability of that bound is ≈ 99.99997%

pub const SvdSampler = struct {
    pub const default_checkpoints = [_]usize{ 1, 8, 64, 256 };

    const BoundMode = enum {
        safe,
        unsafe,
    };

    allocator: std.mem.Allocator,
    m: zml.Slice,
    m_t: zml.Slice,
    vocab_size: usize,
    d: usize,
    checkpoints: []usize,
    row_tail_norms: []f32,
    approx_logits: []f32,
    exact_logits: []f32,
    active: []bool,
    prune_threshold_logit: f32 = -0.693,
    unsafe_z_score: f32 = 5.0,

    pub fn init(allocator: std.mem.Allocator, m: zml.Slice, m_t: zml.Slice) !SvdSampler {
        if (m.dtype() != .f32 or m_t.dtype() != .f32) return error.UnsupportedDType;
        if (m.shape.rank() != 2 or m_t.shape.rank() != 2) return error.InvalidShape;

        const m_dims = m.shape.dims();
        const m_t_dims = m_t.shape.dims();
        const vocab_size: usize = @intCast(m_dims[0]);
        const d: usize = @intCast(m_dims[1]);
        if (vocab_size == 0 or d == 0) return error.InvalidShape;
        if (@as(usize, @intCast(m_t_dims[0])) != d or @as(usize, @intCast(m_t_dims[1])) != vocab_size) return error.InvalidShape;

        const checkpoints = try allocator.dupe(usize, default_checkpoints[0..default_checkpoints.len]);
        errdefer allocator.free(checkpoints);

        const row_tail_norms = try allocator.alloc(f32, checkpoints.len * vocab_size);
        errdefer allocator.free(row_tail_norms);

        const approx_logits = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(approx_logits);

        const exact_logits = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(exact_logits);

        const active = try allocator.alloc(bool, vocab_size);
        errdefer allocator.free(active);

        var sampler: SvdSampler = .{
            .allocator = allocator,
            .m = m,
            .m_t = m_t,
            .vocab_size = vocab_size,
            .d = d,
            .checkpoints = checkpoints,
            .row_tail_norms = row_tail_norms,
            .approx_logits = approx_logits,
            .exact_logits = exact_logits,
            .active = active,
        };
        sampler.precomputeRowTailNorms();
        return sampler;
    }

    pub fn deinit(self: *SvdSampler) void {
        self.allocator.free(self.checkpoints);
        self.allocator.free(self.row_tail_norms);
        self.allocator.free(self.approx_logits);
        self.allocator.free(self.exact_logits);
        self.allocator.free(self.active);
    }

    pub fn sampleSafe(self: *SvdSampler, y: []const f32, random: std.Random) !u32 {
        return self.sample(y, random, .safe);
    }

    pub fn sampleUnsafe(self: *SvdSampler, y: []const f32, random: std.Random) !u32 {
        return self.sample(y, random, .unsafe);
    }


    fn precomputeRowTailNorms(self: *SvdSampler) void {
        const m_items = self.m.constItems(f32);
        for (self.checkpoints, 0..) |checkpoint, checkpoint_index| {
            const checkpoint_tail_norms = self.rowTailNorms(checkpoint_index);
            for (0..self.vocab_size) |token| {
                const row = m_items[token * self.d ..][0..self.d];
                var sum_sq: f64 = 0.0;
                for (row[checkpoint..]) |value| {
                    const value_f64: f64 = @floatCast(value);
                    sum_sq += value_f64 * value_f64;
                }
                checkpoint_tail_norms[token] = @floatCast(@sqrt(sum_sq));
            }
        }
    }

    fn rowTailNorms(self: *SvdSampler, checkpoint_index: usize) []f32 {
        const start = checkpoint_index * self.vocab_size;
        return self.row_tail_norms[start..][0..self.vocab_size];
    }

    
    fn sample(self: *SvdSampler, y: []const f32, random: std.Random, mode: BoundMode) !u32 {
        if (y.len != self.d) return error.InvalidEmbeddingShape;

        @memset(self.approx_logits, 0.0);
        @memset(self.active, true);
        var active_count = self.vocab_size;
        var previous_checkpoint: usize = 0;

        const m_t_items = self.m_t.constItems(f32);
        for (self.checkpoints, 0..) |checkpoint, checkpoint_index| {
            std.log.info("    checkpoint: {d}, active_count: {d}", .{checkpoint_index, active_count});
            // increment logit computation from previous checkpoint to current checkpoint
            for (previous_checkpoint..checkpoint) |coord| {
                const coord_values = m_t_items[coord * self.vocab_size ..][0..self.vocab_size];
                for (0..self.vocab_size) |token| {
                    if (!self.active[token]) continue;
                    self.approx_logits[token] += coord_values[token] * y[coord];
                }
            }
            previous_checkpoint = checkpoint;

            // we can improve this by computing it in one pass at start
            const y_tail_norm = tailNorm(y, checkpoint);
            
            const row_tail_norms = self.rowTailNorms(checkpoint_index);
            const best_lower_bound = self.bestLowerBound(row_tail_norms, y_tail_norm, checkpoint, mode);
            const prune_limit = best_lower_bound + self.prune_threshold_logit;

            for (0..self.vocab_size) |token| {
                if (!self.active[token]) continue;
                const err = self.errorBound(row_tail_norms[token], y_tail_norm, checkpoint, mode);
                const upper_bound = self.approx_logits[token] + err;
                if (upper_bound <= prune_limit) {
                    self.active[token] = false;
                    active_count -= 1;
                }
            }

            if (active_count == 1) break;
        }

        std.log.info("    full sampling, active_count: {d}", .{active_count});
        return self.sampleExact(y, random);
    }

    fn bestLowerBound(self: *SvdSampler, row_tail_norms: []const f32, y_tail_norm: f32, checkpoint: usize, mode: BoundMode) f32 {
        var best = -std.math.inf(f32);
        for (0..self.vocab_size) |token| {
            if (!self.active[token]) continue;
            const err = self.errorBound(row_tail_norms[token], y_tail_norm, checkpoint, mode);
            best = @max(best, self.approx_logits[token] - err);
        }
        return best;
    }

    fn errorBound(self: *SvdSampler, row_tail_norm: f32, y_tail_norm: f32, checkpoint: usize, mode: BoundMode) f32 {
        // TODO: compute this once for each checkpoint to avoid many square roots
        var cs_err = row_tail_norm * y_tail_norm;
        if (mode == .unsafe) {
            const tail_dim: f32 = @floatFromInt(self.d - checkpoint);
            cs_err *= self.unsafe_z_score / @sqrt(tail_dim);
        }
        return cs_err;
    }

    fn sampleExact(self: *SvdSampler, y: []const f32, random: std.Random) u32 {
        var max_logit = -std.math.inf(f32);
        var best_token: usize = 0;
        var best_logit = -std.math.inf(f32);

        for (0..self.vocab_size) |token| {
            if (!self.active[token]) continue;
            const logit = self.exactLogit(token, y);
            self.exact_logits[token] = logit;
            if (logit > max_logit) max_logit = logit;
            if (logit > best_logit) {
                best_logit = logit;
                best_token = token;
            }
        }

        var total: f32 = 0.0;
        for (0..self.vocab_size) |token| {
            if (!self.active[token]) continue;
            total += @exp(self.exact_logits[token] - max_logit);
        }
        if (!std.math.isFinite(total) or total <= 0.0) return @intCast(best_token);

        const threshold = random.float(f32) * total;
        var cumulative: f32 = 0.0;
        for (0..self.vocab_size) |token| {
            if (!self.active[token]) continue;
            cumulative += @exp(self.exact_logits[token] - max_logit);
            if (cumulative >= threshold) return @intCast(token);
        }
        return @intCast(best_token);
    }

    fn exactLogit(self: *SvdSampler, token: usize, y: []const f32) f32 {
        // TODO: we already computed the dot product for biggest checkpoint, can start from there
        const m_items = self.m.constItems(f32);
        const row = m_items[token * self.d ..][0..self.d];
        var logit: f32 = 0.0;
        for (row, y) |weight, value| {
            logit += weight * value;
        }
        return logit;
    }

    fn tailNorm(y: []const f32, checkpoint: usize) f32 {
        var sum_sq: f32 = 0.0;
        for (y[checkpoint..]) |value| {
            sum_sq += value * value;
        }
        return @sqrt(sum_sq);
    }
};
