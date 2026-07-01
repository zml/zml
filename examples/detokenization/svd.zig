const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");
const tokens = @import("tokens.zig");
const Tokenizer = zml.tokenizer.Tokenizer;

// This class will be used to implement a cascading style sampler based on the
// orthogonal basis obtained from the SVD decomposition of the lm_head matrix,
// which we call the SVD basis.
// 
// We note d = hidden_size
// We note v = vocab_size
// 
// Given a lm_head matrix H of dimensions [v, d], the SVD decomposition
// of H is H = V * S * U^T, where U and V are rotation matrices of sizes
// [d, d] and [v, v] respectively, and S is diagonal.
// 
// By forming G = H^T * H, a [d, d] symmetric positive definite matrix,
// and computing its eigendecomposition, the singular values are then
// square roots of the eigen values, and the SVD basis the disgonalization
// matrix. This works because:
//      G = H^T * H = (U * S * V^T) * (V * S * U^T) = U * S^2 * U^T.
// 
// We note M = H * U the representation of the lm_head in the SVD basis.
// In this basis, we should see that the magnitude of cooddinates decreases.
// 
// Given an embedding x in R^d, the logits we need for sampling are H * x
// the i-th logit is logit(i) = Hi * i where Hi is the i-th row of H
// using U*U^T = Id, we can write:
//     logit(i) = Hi * (U * U^T) * x
//              = (Hi * U) * (U^T * x)
//              = Mi * y,
// where y is the expression of x in the SVD basis: y = U^T * x.
// this means that the logit computation can be performed in the SVD basis.
// 
// The diagonal matrix S has the property that singular values appear
// in descending order along the diagonal. This means in the SVD basis,
// the coordinates are of decreasing magnitude. This property
// is what we will exploit to devise the token sampling strategy.
// 
// We define a set of checkpoints C = { 1, 8, 64, 256 }.
// Given an already rotated embedding vector y to sample from,
// we have that logit(i) = sum_j Mi_j * y_j
// For any k in C, logit(i) = sum_{j <= k} Mi_j * y_j + sum_{j > k} Mi_j * y_j
// For any k, this defines an approximation of the logits:
//     logit(i) ≈ sum_{j <= k} Mi_j * y_j = logit_approx(i, k)
// 
// With Cauchy-Schwarz inequality, we can bound the error of the approximation:
//     |logit(i) - logit_approx(i, k)| = |sum_{j > k} Mi_j * y_j|
//                                     <= ||Mi||_{>k} * ||y||_{>k} = Err(i, k)
// Where ||.||_{>k} is the L2 norm when ignoring the first k coordinates.
// This gives an upper bound on the real logit for each token i:
//     logit(i) <= logit_approx(i, k) + Err(i, k)
// If i_max is the index of the token with the largest approximate logit, then:
//     logit(i_max) >= logit_approx(i_max, k) - Err(i_max, k)
// This means that for any k, we have a lower bound on the larget final logit.
// 
// This allows us to prune tokens based on a logit threshold value.
// If logit(i) <= logit(j) + a then exp(logit(i)) <= exp(a) * exp(logit(j)),
// meaning that token i has a probability lower than exp(a) times that of token j.
// If we decide to use exp(a) = 0,5 as a pruning threshold, we can prune tokens
// with logit(i) <= logit(j) - ln(2) = logit(j) - 0,693 for any i and j.
// We generalize by havind a variable threshold (= -ln(2) = -0,693 by default).
// 
// The sampling algorithm is then:
//     Compute the ||y||_{>k} values for all k
//     For k in C:
//         Compute approximate logits for k
//             logit_approx(i, k) = sum_{j <= k} Mi_j * y_j
//         Compute best lower bound on max logit:
//             L = max_i { logit_approx(i, k) - Err(i, k) }
//         For each i:
//             If logit_approx(i, k) + Err(i, k) <= L - threshold, then prune token i
//     Compute real logits for all remaining tokens
//     Sample with softmax using the real logits
// 
// The ||Mi||_{>k} values for all checkpoints can be precomputed as they are constant.
// 
// This first version is the "safe" version, as the CS inequality is deterministic.
// In real world scenarios, the tail of >k coordinates never exactly align with the
// worst case of CS, making the inequality quite pessimistic. By treating the data
// as random gaussian variables, we can tighten the bound by orders of magnitude.
// In this case we use the following Z-score error bound:
// |logit(i) - logit_approx(i, k)| <= Z * Err(i, k) / sqrt(d - k)
// With Z = 5, the success probability of that bound is ≈ 99.99997%

pub const SvdSampler = struct {
    pub const default_checkpoints = [_]usize{ 1, 8, 64, 256, 1024 };

    const BoundMode = enum {
        safe,
        unsafe,
    };

    const SamplingCandidate = struct {
        token: usize,
        logit: f32,
        weight: f32,

        fn beforeThan(_: void, lhs: SamplingCandidate, rhs: SamplingCandidate) bool {
            return lhs.logit > rhs.logit;
        }

        fn absDecreasing(_: void, lhs: SamplingCandidate, rhs: SamplingCandidate) bool {
            return @abs(lhs.logit) > @abs(rhs.logit);
        }
    };

    const IntSamplingCandidate = struct {
        token: usize,
        logit: i32,

        fn beforeThan(_: void, lhs: IntSamplingCandidate, rhs: IntSamplingCandidate) bool {
            return lhs.logit > rhs.logit;
        }

        fn absDecreasing(_: void, lhs: IntSamplingCandidate, rhs: IntSamplingCandidate) bool {
            return @abs(lhs.logit) > @abs(rhs.logit);
        }
    };

    zml_handler: *main.Zml_handler,
    allocator: std.mem.Allocator,
    m: zml.Slice,
    m_t: zml.Slice,
    vocab_size: usize,
    d: usize,
    checkpoints: []usize,
    row_tail_norms: []f32,
    row_norms: []f32,
    approx_logits: []f32,
    checkpoint_approx_logits: []f32,
    exact_logits: []f32,
    sampling_candidates: []SamplingCandidate,
    top256_candidates: []SamplingCandidate,
    active: []bool,
    top_k: usize = 20,
    temp: f32 = 0.6,
    prune_threshold_logit: f32 = -0.693,
    unsafe_z_score: f32 = 5.0,
    tokenizer: Tokenizer,
    use_svd: bool = true,
    order: []usize,

    pub fn init(zml_handler: *main.Zml_handler, m: zml.Slice, m_t: zml.Slice, tokenizer: Tokenizer, use_svd: bool) !SvdSampler {
        const allocator = zml_handler.allocator;
        
        const m_dims = m.shape.dims();
        const vocab_size: usize = @intCast(m_dims[0]);
        const d: usize = @intCast(m_dims[1]);

        const checkpoints = try allocator.dupe(usize, default_checkpoints[0..default_checkpoints.len]);
        errdefer allocator.free(checkpoints);

        const row_norms = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(row_norms);

        const row_tail_norms = try allocator.alloc(f32, checkpoints.len * vocab_size);
        errdefer allocator.free(row_tail_norms);
        
        const approx_logits = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(approx_logits);

        const checkpoint_approx_logits = try allocator.alloc(f32, checkpoints.len * vocab_size);
        errdefer allocator.free(checkpoint_approx_logits);

        const exact_logits = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(exact_logits);

        const sampling_candidates = try allocator.alloc(SamplingCandidate, vocab_size);
        errdefer allocator.free(sampling_candidates);

        const top256_candidates = try allocator.alloc(SamplingCandidate, 256);
        errdefer allocator.free(top256_candidates);

        const active = try allocator.alloc(bool, vocab_size);
        errdefer allocator.free(active);

        const order = try allocator.alloc(usize, d);
        errdefer allocator.free(order);

        var sampler: SvdSampler = .{
            .zml_handler = zml_handler,
            .allocator = allocator,
            .m = m,
            .m_t = m_t,
            .vocab_size = vocab_size,
            .d = d,
            .checkpoints = checkpoints,
            .row_norms = row_norms,
            .row_tail_norms = row_tail_norms,
            .approx_logits = approx_logits,
            .checkpoint_approx_logits = checkpoint_approx_logits,
            .exact_logits = exact_logits,
            .sampling_candidates = sampling_candidates,
            .top256_candidates = top256_candidates,
            .active = active,
            .tokenizer = tokenizer,
            .use_svd = use_svd,
            .order = order,
        };
        sampler.computeRowNorms();
        sampler.precomputeRowTailNorms();
        return sampler;
    }

    pub fn deinit(self: *SvdSampler) void {
        self.allocator.free(self.checkpoints);
        self.allocator.free(self.row_tail_norms);
        self.allocator.free(self.approx_logits);
        self.allocator.free(self.checkpoint_approx_logits);
        self.allocator.free(self.exact_logits);
        self.allocator.free(self.sampling_candidates);
        self.allocator.free(self.top256_candidates);
        self.allocator.free(self.active);
        self.allocator.free(self.row_norms);
        self.allocator.free(self.order);
    }

    pub fn setSamplingParams(self: *SvdSampler, temperature: f32, top_k: usize, top_p: f32) void {
        self.temperature = temperature;
        self.top_k = top_k;
        self.top_p = top_p;
    }


    pub fn sampleFast(self: *SvdSampler, y: []const f32) !void {
        for (y, 0..) |value, coord| {
            self.sampling_candidates[coord] = .{ .token = coord, .logit = value, .weight = 0.0 };
        }
        std.mem.sort(SamplingCandidate, self.sampling_candidates[0..self.d], {}, SamplingCandidate.absDecreasing);
        
        // compute approximate logits using the 256 largest magnitude coordinates of the query
        self.zml_handler.tic(&self.zml_handler.timers.svd_sparse_256);
        @memset(self.approx_logits, 0.0);
        const m_t_items = self.m_t.constItems(f32);
        const simd_len = 32;
        const Vec = @Vector(simd_len, f32);
        for (0..256) |i| {
            const coord = self.sampling_candidates[i].token;
            const y_value = y[coord];
            const coord_values = m_t_items[coord * self.vocab_size ..][0..self.vocab_size];
            const y_vec: Vec = @splat(y_value);

            var token: usize = 0;
            while (token + simd_len <= self.vocab_size) : (token += simd_len) {
                const coord_vec: Vec = coord_values[token..][0..simd_len].*;
                var logits_vec: Vec = self.approx_logits[token..][0..simd_len].*;
                logits_vec = @mulAdd(Vec, coord_vec, y_vec, logits_vec);
                self.approx_logits[token..][0..simd_len].* = logits_vec;
            }
            while (token < self.vocab_size) : (token += 1) {
                self.approx_logits[token] += coord_values[token] * y_value;
            }
        }
        self.zml_handler.toc(&self.zml_handler.timers.svd_sparse_256);

        // find the 256 highest approx logits candidates
        self.zml_handler.tic(&self.zml_handler.timers.svd_top_256);
        for (0..256) |tok| {
            self.sampling_candidates[tok] = .{ .token = tok, .logit = self.approx_logits[tok], .weight = 0.0 };
        }
        const top_approx = self.sampling_candidates[0..256];
        std.mem.sort(SamplingCandidate, top_approx, {}, SamplingCandidate.beforeThan);
        for (256..self.vocab_size) |tok| {
            const logit = self.approx_logits[tok];
            if (logit <= top_approx[255].logit) continue;

            var insert_pos: usize = 255;
            while (insert_pos > 0 and logit > top_approx[insert_pos - 1].logit) {
                top_approx[insert_pos] = top_approx[insert_pos - 1];
                insert_pos -= 1;
            }
            top_approx[insert_pos] = .{ .token = tok, .logit = logit, .weight = 0.0 };
        }
        self.zml_handler.toc(&self.zml_handler.timers.svd_top_256);
        
        // compute their exact logits
        self.zml_handler.tic(&self.zml_handler.timers.svd_dense_256);
        for (0..256) |tok_pos| {
            const tok = top_approx[tok_pos].token;
            self.top256_candidates[tok_pos] = .{ .token = tok, .logit = self.logitFast(y, tok), .weight = 1.0 };
        }
        self.zml_handler.toc(&self.zml_handler.timers.svd_dense_256);
    }

    pub fn logitFast(self: *SvdSampler, query: []const f32, tok: usize) f32 {
        const rows = self.m.constItems(f32);
        const row = rows[tok * self.d ..][0..self.d];

        const simd_len = 32;
        std.debug.assert(self.d % simd_len == 0);
        const Vec = @Vector(simd_len, f32);
        var acc: Vec = @splat(0);

        var i: usize = 0;
        while (i + simd_len <= self.d) : (i += simd_len) {
            const query_vec: Vec = query[i..][0..simd_len].*;
            const row_vec: Vec = row[i..][0..simd_len].*;
            acc = @mulAdd(Vec, query_vec, row_vec, acc);
        }
        return @reduce(.Add, acc);
    }

    
    pub fn sampleSafe(self: *SvdSampler, y: []const f32, random: std.Random) !u32 {
        std.log.info("##### sampleSafe", .{});
        return self.sample(y, random, .safe);
    }

    pub fn sampleUnsafe(self: *SvdSampler, y: []const f32, random: std.Random) !u32 {
        std.log.info("##### sampleUnsafe", .{});
        return self.sample(y, random, .unsafe);
    }

    pub fn sampleFull(self: *SvdSampler, y: []const f32, random: std.Random) !u32 {
        std.log.info("##### sampleFull", .{});
        @memset(self.active, true);
        @memset(self.exact_logits, 0.0);
        const token = self.sampleExact(y, random);
        return token;
    }

    pub fn sampleTruncated(self: *SvdSampler, y: []const f32, random: std.Random, n: usize) !u32 {
        std.log.info("##### sampleTruncated {d}", .{n});
        if (y.len != self.d) return error.InvalidEmbeddingShape;
        const m_t_items = self.m_t.constItems(f32);
        var candidate_count: usize = 0;
        for (0..self.vocab_size) |token| {
            var logit: f32 = 0.0;
            for (0..n) |coord| {
                logit += m_t_items[coord * self.vocab_size + token] * y[coord];
            }
            self.sampling_candidates[candidate_count] = .{
                .token = token,
                .logit = logit / self.temp,
                .weight = 0.0,
            };
            candidate_count += 1;
        }
        return self.sampleCandidates(random, candidate_count);
    }


    fn computeRowNorms(self: *SvdSampler) void {
        const m_items = self.m.constItems(f32);
        for (0..self.vocab_size) |token| {
            const row = m_items[token * self.d ..][0..self.d];
            var sum_sq: f64 = 0.0;
            for (row) |value| {
                const value_f64: f64 = @floatCast(value);
                sum_sq += value_f64 * value_f64;
            }
            self.row_norms[token] = @floatCast(@sqrt(sum_sq));
        }
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

        var tot_y_abs: f32 = 0.0;
        for (y, 0..) |value, coord| {
            tot_y_abs += @abs(value);
            self.sampling_candidates[coord] = .{ .token = coord, .logit = value, .weight = 0.0 };
        }
        std.mem.sort(SamplingCandidate, self.sampling_candidates[0..self.d], {}, SamplingCandidate.absDecreasing);
        for (0..self.d) |i| {
            self.order[i] = self.sampling_candidates[i].token;
        }
        std.log.info("    min  |y| = {d:.4}, max |y| = {d:.4}", .{ @abs(y[self.order[self.d-1]]), @abs(y[self.order[0]])});
        std.log.info("    mean |y| = {d:.4}, med |y| = {d:.4}", .{ tot_y_abs / @as(f32, @floatFromInt(self.d)), @abs(y[self.order[@divExact(self.d, 2)]])});
        std.log.info("    p01  |y| = {d:.4}, p05 |y| = {d:.4}", .{ @abs(y[self.order[@divFloor(self.d, 100)]]), @abs(y[self.order[@divFloor(self.d, 20)]])});

        @memset(self.approx_logits, 0.0);
        @memset(self.active, true);
        var active_count = self.vocab_size;
        var previous_checkpoint: usize = 0;
        var processed_checkpoint_count: usize = 0;
        var total_flops: usize = 0;
        
        const m_t_items = self.m_t.constItems(f32);
        for (self.checkpoints, 0..) |checkpoint, checkpoint_index| {
            const checkpoint_flops = (checkpoint - previous_checkpoint) * active_count;
            total_flops += checkpoint_flops;
            std.log.info("    checkpoint: {d}({d:>4}), active_count: {d:>6}, flops: {d}", .{checkpoint_index, checkpoint, active_count, checkpoint_flops});
            // increment logit computation from previous checkpoint to current checkpoint
            for (previous_checkpoint..checkpoint) |coord_pos| {
                const coord = self.order[coord_pos];
                const coord_values = m_t_items[coord * self.vocab_size ..][0..self.vocab_size];
                for (0..self.vocab_size) |token| {
                    self.approx_logits[token] += coord_values[token] * y[coord];
                }
            }
            previous_checkpoint = checkpoint;
            @memcpy(self.checkpointApproxLogits(checkpoint_index), self.approx_logits);
            processed_checkpoint_count = checkpoint_index + 1;

            // we can improve this by computing it in one pass at start
            const y_tail_norm = if (self.use_svd) tailNorm(y, checkpoint) else tailNormOrder(self, y, checkpoint);
            
            const row_tail_norms = self.rowTailNorms(checkpoint_index);
            const best_lower_bound = self.bestLowerBound(row_tail_norms, y_tail_norm, checkpoint, mode);
            const prune_limit = best_lower_bound + self.prune_threshold_logit;

            const float_dim: f32 = @floatFromInt(self.d);
            const float_checkpoint: f32 = @floatFromInt(checkpoint);
            const tail_factor: f32 = 1.0 - float_checkpoint / float_dim;
            for (0..self.vocab_size) |token| {
                if (!self.active[token]) continue;
                const row_tail_norm = if (self.use_svd) row_tail_norms[token] else self.row_norms[token] * tail_factor;
                const err = self.errorBound(row_tail_norm, y_tail_norm, checkpoint, mode);
                const upper_bound = self.approx_logits[token] + err;
                if (upper_bound <= prune_limit) {
                    self.active[token] = false;
                    active_count -= 1;
                }
            }

            if (active_count == 1) break;
        }

        std.log.info("    total flops: {d}, max_flops: {d}, {d:.2}%", .{total_flops, self.vocab_size * self.d, 100.0 * @as(f32, @floatFromInt(total_flops)) / @as(f32, @floatFromInt(self.vocab_size * self.d))});
        std.log.info("    full sampling, active_count: {d}", .{active_count});
        const token = try self.sampleExact(y, random);
        self.logCheckpointApproxRanks(token, processed_checkpoint_count, y);
        return token;
    }

    fn checkpointApproxLogits(self: *SvdSampler, checkpoint_index: usize) []f32 {
        const start = checkpoint_index * self.vocab_size;
        return self.checkpoint_approx_logits[start..][0..self.vocab_size];
    }

    fn logCheckpointApproxRanks(self: *SvdSampler, final_token: u32, checkpoint_count: usize, y: []const f32) void {
        const final_token_index: usize = @intCast(final_token);
        std.log.info("{s:>16}  {s:>16}  {s:>16}  {s:>16}", .{ "checkpoint", "max approx logit", "tok approx logit", "pos in approx" });
        for (0..checkpoint_count) |checkpoint_index| {
            const logits = self.checkpointApproxLogits(checkpoint_index);
            const final_token_logit = logits[final_token_index];
            var highest_logit = -std.math.inf(f32);
            var final_token_pos: usize = 1;
            for (logits, 0..) |logit, token| {
                highest_logit = @max(highest_logit, logit);
                if (logit > final_token_logit or (logit == final_token_logit and token < final_token_index)) {
                    final_token_pos += 1;
                }
            }
            std.log.info("{d:>16}  {d:>16}  {d:>16}  {d:>16}", .{ self.checkpoints[checkpoint_index], highest_logit, final_token_logit, final_token_pos });
        }

        const final_token_logit = self.exactLogit(final_token_index, y);
        var highest_logit = -std.math.inf(f32);
        var final_token_pos: usize = 1;
        for (0..self.vocab_size) |token| {
            const logit = self.exactLogit(token, y);
            highest_logit = @max(highest_logit, logit);
            if (logit > final_token_logit or (logit == final_token_logit and token < final_token_index)) {
                final_token_pos += 1;
            }
        }
        std.log.info("{d:>16}  {d:>16}  {d:>16}  {d:>16}", .{ self.d, highest_logit, final_token_logit, final_token_pos });
    }

    fn bestLowerBound(self: *SvdSampler, row_tail_norms: []const f32, y_tail_norm: f32, checkpoint: usize, mode: BoundMode) f32 {
        var best = -std.math.inf(f32);
        const float_dim: f32 = @floatFromInt(self.d);
        const float_checkpoint: f32 = @floatFromInt(checkpoint);
        const tail_factor: f32 = 1.0 - float_checkpoint / float_dim;
        for (0..self.vocab_size) |token| {
            if (!self.active[token]) continue;
            const row_tail_norm = if (self.use_svd) row_tail_norms[token] else self.row_norms[token] * tail_factor;
            const err = self.errorBound(row_tail_norm, y_tail_norm, checkpoint, mode);
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

    fn sampleExact(self: *SvdSampler, y: []const f32, random: std.Random) !u32 {
        var candidate_count: usize = 0;
        for (0..self.vocab_size) |token| {
            if (!self.active[token]) continue;
            const logit = self.exactLogit(token, y);
            self.exact_logits[token] = logit / self.temp;
            self.sampling_candidates[candidate_count] = .{
                .token = token,
                .logit = self.exact_logits[token],
                .weight = 0.0,
            };
            candidate_count += 1;
        }
        return self.sampleCandidates(random, candidate_count);
    }

    fn sampleCandidates(self: *SvdSampler, random: std.Random, candidate_count: usize) !u32 {
        if (candidate_count == 0) return error.EmptySamplingCandidates;
        const candidates = self.sampling_candidates[0..candidate_count];
        std.mem.sort(SamplingCandidate, candidates, {}, SamplingCandidate.beforeThan);

        const cutoff_count = @min(self.top_k, candidate_count);
        const max_logit = candidates[0].logit;
        var total: f32 = 0.0;
        for (candidates[0..cutoff_count]) |*candidate| {
            candidate.weight = @exp(candidate.logit - max_logit);
            total += candidate.weight;
        }
        if (!std.math.isFinite(total)) return @intCast(candidates[0].token);

        std.log.info("Proba distrib for sampling", .{});
        for (candidates[0..cutoff_count]) |*candidate| {
            const proba = candidate.weight / total;
            if (proba < 1e-6) break;
            const token_str = try tokens.tokenString(self.tokenizer, candidate.token, self.allocator);
            defer self.allocator.free(token_str);
            std.log.info("cand: {d:>5}, proba: {d:>11}, token: {s}", .{ candidate.token, proba, token_str });
        }
        
        const threshold = random.float(f32) * total;
        var cumulative: f32 = 0.0;
        for (candidates[0..cutoff_count]) |candidate| {
            cumulative += candidate.weight;
            if (cumulative >= threshold) return @intCast(candidate.token);
        }
        std.debug.assert(false);
        return @intCast(candidates[0].token);
    }

    fn exactLogit(self: *SvdSampler, token: usize, y: []const f32) f32 {
        // TODO: if we already computed the dot product for biggest checkpoint, can start from there
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

    fn tailNormOrder(self: *SvdSampler, y: []const f32, checkpoint: usize) f32 {
        var sum_sq: f32 = 0.0;
        for (checkpoint..self.d) |coord_pos| {
            const coord = self.order[coord_pos];
            sum_sq += y[coord] * y[coord];
        }
        return @sqrt(sum_sq);
    }
};
