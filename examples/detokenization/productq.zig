const std = @import("std");
const zml = @import("zml");

const main = @import("main.zig");
const tokens = @import("tokens.zig");
const algebra = @import("algebra.zig");
const quantized_ = @import("quantized.zig");
const kmeans = @import("kmeans.zig");
const sampling = @import("sampling.zig");

const LmHeadMatrix = algebra.LmHeadMatrix;
const Quantized = quantized_.Quantized;
const Tokenizer = zml.tokenizer.Tokenizer;
const Zml_handler = main.Zml_handler;
const Allocator = std.mem.Allocator;
const KMeansCPU = kmeans.KMeansCPU;
const KMeansGPU = kmeans.KMeansGPU;
const Sampler = sampling.Sampler;
const Logit = sampling.Logit;

pub const hidden_dim: comptime_int = 4096;
pub const bucket_dim: comptime_int = 16;
pub const nb_buckets: comptime_int = hidden_dim / bucket_dim;
pub const nb_centers: comptime_int = 256;
pub const simd_len: comptime_int = 8;
pub const pq_top_k: comptime_int = 16;
pub const pq_z_score: comptime_int = 5;
pub const pq_orth_weight: comptime_float = 0.25;

pub const PQSample = struct {
    top_k: [pq_top_k]Logit,
    nb_dense_scored: usize = 0,
    nb_pq_scored: usize = 0,
    nb_pruned: usize = 0,
};

pub const ProductQuantizer = struct {
    zml_handler: *Zml_handler,
    allocator: Allocator,
    lm_head: *LmHeadMatrix,
    n: usize,

    // walsh-hadamard transform of the lm_head
    rotated_lm_head: []f32,
    // walsh-hadamard transform of the query
    rotated_query: [hidden_dim]f32,

    // the codebook is the nb_center vectors of dim bucket_dim obtained by clustering
    codebook: [bucket_dim * nb_centers]f32,
    // the PQ assignation of the rows, in row -> bucket major:
    // [row_0_bucket_0_center ; row_0_bucket_1_center ... row_0_bucket_nb_buckets-1_center ; row_1_bucket_0_center ...]
    row_buckets: []u8,
    // the dot product between the query and all centers, so that the dot product between the query and a row
    // is quantized in a bucket to query_center_scores[bucket_id * nb_centers + row_buckets[row_id * nb_buckets + bucket_id]]
    query_center_scores: [nb_buckets * nb_centers]f32,

    // error bounds
    quantization_error_factor: f32,

    // sampling utils
    logits: []Logit,

    min_bucket: f32,
    max_bucket: f32,
    min_norm: f32,
    max_norm: f32,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix) !ProductQuantizer {
        const alloc = zml_handler.allocator;
        std.debug.assert(lm_head.d == hidden_dim);
        const rotated_lm_head = try alloc.alloc(f32, lm_head.n * hidden_dim);
        const row_buckets = try alloc.alloc(u8, lm_head.n * nb_buckets);
        const logits = try alloc.alloc(Logit, lm_head.n);
        for (0..lm_head.n) |i| {
            logits[i].row = i;
            if (lm_head.is_junk[i]) logits[i].logit = -1e10;
        }
        return .{
            .zml_handler = zml_handler,
            .allocator = alloc,
            .lm_head = lm_head,
            .n = lm_head.n,
            .rotated_lm_head = rotated_lm_head,
            .rotated_query = [_]f32{0.0} ** hidden_dim,
            .codebook = [_]f32{0.0} ** (bucket_dim * nb_centers),
            .row_buckets = row_buckets,
            .query_center_scores = [_]f32{0.0} ** (nb_buckets * nb_centers),
            .quantization_error_factor = 0.0,
            .logits = logits,
            .min_bucket = 99999.0,
            .max_bucket = -99999.0,
            .min_norm = 99999.0,
            .max_norm = 0.0,
        };
    }

    pub fn deinit(self: *ProductQuantizer) void {
        self.allocator.free(self.rotated_lm_head);
        self.allocator.free(self.row_buckets);
        self.allocator.free(self.logits);
    }

    pub fn buildCodebook(self: *ProductQuantizer) !void {
        // build the rotated lm_head, don't normalize
        std.log.info("***** Rotating the lm_head", .{});
        for (0..self.n) |i| {
            const src = self.lm_head.data[i * hidden_dim ..][0..hidden_dim];
            const dst = self.rotated_lm_head[i * hidden_dim ..][0..hidden_dim];
            Quantized.walshHadamardSimdTo(dst, src, 12);
        }

        // build dataset from first bucket only, no normalization, exclude junk rows
        std.log.info("***** Building codebook dataset", .{});
        var nb_non_junk: usize = 0;
        const first_bucket_data = try self.allocator.alloc(f32, self.n * bucket_dim);
        defer self.allocator.free(first_bucket_data);
        for (0..self.n) |row| {
            if (self.lm_head.is_junk[row]) continue;
            const dst = first_bucket_data[nb_non_junk * bucket_dim ..][0..bucket_dim];
            const src = self.rotated_lm_head[row * hidden_dim ..][0..bucket_dim];
            @memcpy(dst, src);
            nb_non_junk += 1;
        }

        // build codebook by vanilla kmeans
        std.log.info("***** Building VanillaPQ codebook", .{});
        var km = try KMeansCPU.init(self.allocator, nb_non_junk, bucket_dim, nb_centers);
        defer km.deinit();
        km.solve(first_bucket_data[0 .. nb_non_junk * bucket_dim]);

        // store the codebook, assign all buckets to a center
        std.log.info("***** Assigning row buckets to centers", .{});
        @memcpy(self.codebook[0..], km.centers[0..self.codebook.len]);
        for (0..self.n) |row| {
            const row_codes = self.row_buckets[row * nb_buckets ..][0..nb_buckets];
            for (0..nb_buckets) |bucket| {
                const data = self.rotated_lm_head[(row * hidden_dim + bucket * bucket_dim)..][0..bucket_dim];
                const norm = Quantized.normL2(data);
                const closest = km.closestCenter(data, norm * norm, km.k);
                row_codes[bucket] = @intCast(closest[1]);
            }
        }

        // compute error bounding factors
        // in this vanilla setup, its been proven that the pq logits, if seen as random variables, have:
        // mean(pq_logit) = real_logit
        // var(pq_logit) = ||query||²||row_quantization_error||² / D
        // by using Shannon's rate distortion theory to bound the quantization error for kmeans after WH:
        // var(pq_logit) = K^{-2/B} * ||query||²||row||² / D,
        // K is the number of centers (low impact)
        // D is the hidden_dim (high impact)
        // B is the bucket_dim (very high impact)
        // we define the pq_logit upper bound with:
        // upper_bound(pq_logit) = pq_logit + Z * sqrt(var(pq_logit)),
        // which has 99,98% likelyhood to hold for Z = 5.
        const K: f32 = @floatFromInt(nb_centers);
        const B: f32 = @floatFromInt(bucket_dim);
        const D: f32 = @floatFromInt(hidden_dim);
        const kmeans_err = std.math.pow(f32, K, -1.0 / B);
        self.quantization_error_factor = pq_z_score * kmeans_err / @sqrt(D);
        std.log.info("K-means relative theoretical erorr = {d}", .{kmeans_err});
        std.log.info("PQ statistical Cauchy-Schwarz factor = {d}", .{self.quantization_error_factor});
    }

    pub fn sample(self: *ProductQuantizer, query: []const f32) PQSample {
        // step 0 : compute geometric information for error bounding
        const query_norm = Quantized.normL2(query);
        const error_bound_scale = query_norm * self.quantization_error_factor;
        self.min_norm = @min(self.min_norm, query_norm);
        self.max_norm = @max(self.max_norm, query_norm);

        // step 1 : rotate query, don't normalize it
        self.zml_handler.tic(&self.zml_handler.timers.pq_rotate);
        const src = query[0..hidden_dim];
        const dst = self.rotated_query[0..];
        Quantized.walshHadamardSimdTo(dst, src, 12);
        self.zml_handler.toc(&self.zml_handler.timers.pq_rotate);

        // step 2 : split the query into each bucket, score that against each center. normalize the scores
        self.zml_handler.tic(&self.zml_handler.timers.pq_lut);
        for (0..nb_buckets) |bucket| {
            const query_data = self.rotated_query[bucket * bucket_dim ..][0..bucket_dim];
            for (0..nb_centers) |center| {
                const center_data = self.codebook[center * bucket_dim ..][0..bucket_dim];
                var dot: f32 = 0.0;
                for (0..bucket_dim) |coord| {
                    dot += center_data[coord] * query_data[coord];
                }
                dot /= query_norm;
                self.min_bucket = @min(self.min_bucket, @abs(dot));
                self.max_bucket = @max(self.max_bucket, @abs(dot));
                self.query_center_scores[bucket * nb_centers + center] = dot;
            }
        }
        self.zml_handler.toc(&self.zml_handler.timers.pq_lut);

        // step 3 : compute logits row by row using contiguous bucket codes, unnormalize the scores
        self.zml_handler.tic(&self.zml_handler.timers.pq_score);
        for (0..self.n) |row| {
            if (self.lm_head.is_junk[row]) continue;
            const row_codes = self.row_buckets[row * nb_buckets ..][0..nb_buckets];
            var logit: f32 = 0.0;
            for (0..nb_buckets) |bucket| {
                const bucket_lut = self.query_center_scores[bucket * nb_centers ..][0..nb_centers];
                logit += bucket_lut[row_codes[bucket]];
            }
            logit *= query_norm;
            self.logits[row].logit = logit;
            self.logits[row].upper_bound = logit + error_bound_scale * self.lm_head.row_norms[row];
        }
        self.zml_handler.toc(&self.zml_handler.timers.pq_score);

        // step 4 : find and return topK
        self.zml_handler.tic(&self.zml_handler.timers.pq_top_k);
        const empty_logit: Logit = .{ .logit = -std.math.floatMax(f32), .row = 0, .upper_bound = 0 };
        var result: PQSample = .{
            .top_k = [_]Logit{empty_logit} ** pq_top_k,
            .nb_dense_scored = 0,
            .nb_pq_scored = 0,
            .nb_pruned = 0,
        };
        for (0..self.n) |row| {
            const score = self.logits[row].logit;
            if (score <= result.top_k[pq_top_k - 1].logit) continue;
            var insert_pos: usize = pq_top_k - 1;
            while (insert_pos > 0 and score > result.top_k[insert_pos - 1].logit) {
                result.top_k[insert_pos] = result.top_k[insert_pos - 1];
                insert_pos -= 1;
            }
            result.top_k[insert_pos] = self.logits[row];
        }
        self.zml_handler.toc(&self.zml_handler.timers.pq_top_k);
        return result;
    }

    pub fn sampleLog(self: *ProductQuantizer, sampler: *Sampler, query: []const f32) void {
        std.log.info("*****", .{});
        std.log.info("***** new sampling", .{});
        std.log.info("*****", .{});
        var approx_sampling = self.sample(query);
        sampler.sample(query);
        sampler.logRealVsApproxSampling(self.logits, "PQ logits");
        sampler.logApproxVsRealSampling(&approx_sampling.top_k, "PQ logit");
    }
};

/// ScaNN-style anisotropic PQ with one codebook shared by every Walsh-Hadamard
/// bucket. `base` owns the search-time representation; only codebook training and
/// row assignment differ from `ProductQuantizer`.
pub const AnisotropicProductQuantizer = struct {
    base: ProductQuantizer,

    const RotatedLmHead = struct {
        n: usize,
        d: usize,
        data: []const f32,
        is_junk: []const bool,
    };

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix) !AnisotropicProductQuantizer {
        return .{ .base = try ProductQuantizer.init(zml_handler, lm_head) };
    }

    pub fn deinit(self: *AnisotropicProductQuantizer) void {
        self.base.deinit();
    }

    pub fn buildCodebook(self: *AnisotropicProductQuantizer) !void {
        const pq = &self.base;

        std.log.info("***** Rotating the lm_head for anisotropic PQ", .{});
        for (0..pq.n) |row| {
            const src = pq.lm_head.data[row * hidden_dim ..][0..hidden_dim];
            const dst = pq.rotated_lm_head[row * hidden_dim ..][0..hidden_dim];
            Quantized.walshHadamardSimdTo(dst, src, 12);
        }

        // Seed the tied codebook with a stratified sample. Each non-junk row
        // contributes one block, cycling through all bucket positions.
        var nb_non_junk: usize = 0;
        for (pq.lm_head.is_junk) |is_junk| nb_non_junk += @intFromBool(!is_junk);
        const training_data = try pq.allocator.alloc(f32, nb_non_junk * bucket_dim);
        defer pq.allocator.free(training_data);
        var sample_index: usize = 0;
        for (0..pq.n) |row| {
            if (pq.lm_head.is_junk[row]) continue;
            const bucket = sample_index % nb_buckets;
            const src = pq.rotated_lm_head[row * hidden_dim + bucket * bucket_dim ..][0..bucket_dim];
            const dst = training_data[sample_index * bucket_dim ..][0..bucket_dim];
            @memcpy(dst, src);
            sample_index += 1;
        }

        std.log.info("***** Initializing shared codebook from all buckets on the GPU", .{});
        var km = try KMeansGPU.init(pq.zml_handler, nb_non_junk, bucket_dim, nb_centers);
        defer km.deinit();
        try km.solve(training_data);

        std.log.info("***** Jointly optimizing full-row ScaNN anisotropic loss on the GPU", .{});
        const rotated: RotatedLmHead = .{
            .n = pq.n,
            .d = hidden_dim,
            .data = pq.rotated_lm_head,
            .is_junk = pq.lm_head.is_junk,
        };
        try km.solveAnisotropic(rotated, pq.row_buckets, pq_orth_weight);
        @memcpy(pq.codebook[0..], km.centers[0..pq.codebook.len]);

        // Use a deterministic Cauchy-Schwarz bound for APQ. Unlike the vanilla
        // rate-distortion estimate, this remains valid for the anisotropic loss.
        var max_relative_error: f32 = 0.0;
        for (0..pq.n) |row| {
            if (pq.lm_head.is_junk[row]) continue;
            var error_norm2: f32 = 0.0;
            const row_codes = pq.row_buckets[row * nb_buckets ..][0..nb_buckets];
            for (0..nb_buckets) |bucket| {
                const x = pq.rotated_lm_head[row * hidden_dim + bucket * bucket_dim ..][0..bucket_dim];
                const center = pq.codebook[@as(usize, row_codes[bucket]) * bucket_dim ..][0..bucket_dim];
                for (0..bucket_dim) |coord| {
                    const err = x[coord] - center[coord];
                    error_norm2 += err * err;
                }
            }
            const row_norm = pq.lm_head.row_norms[row];
            max_relative_error = @max(max_relative_error, @sqrt(error_norm2) / row_norm);
        }
        pq.quantization_error_factor = max_relative_error;
        std.log.info("Anisotropic PQ maximum relative reconstruction error = {d}", .{max_relative_error});
    }

    pub fn sample(self: *AnisotropicProductQuantizer, query: []const f32) PQSample {
        return self.base.sample(query);
    }

    pub fn sampleLog(self: *AnisotropicProductQuantizer, sampler: *Sampler, query: []const f32) void {
        self.base.sampleLog(sampler, query);
    }
};
