const std = @import("std");
const zml = @import("zml");
const builtin = @import("builtin");

const main = @import("main.zig");
const tokens = @import("tokens.zig");
const algebra = @import("algebra.zig");
const kmeans = @import("kmeans.zig");
const sampling = @import("sampling.zig");
const quantization = @import("quantization.zig");

const LmHeadMatrix = algebra.LmHeadMatrix;
const Tokenizer = zml.tokenizer.Tokenizer;
const Zml_handler = main.Zml_handler;
const Allocator = std.mem.Allocator;
const KMeansCPU = kmeans.KMeansCPU;
const Sampler = sampling.Sampler;
const Logit = sampling.Logit;
const SamplingResult = sampling.SamplingResult;

const sampling_top_k = sampling.sampling_top_k;

pub const hidden_dim: comptime_int = 4096;
pub const bucket_dim: comptime_int = 4;
pub const nb_buckets: comptime_int = hidden_dim / bucket_dim;
pub const nb_centers: comptime_int = 256;
pub const simd_len: comptime_int = 8;
pub const pq_z_score: comptime_int = 5;
pub const pq_orth_weight: comptime_float = 0.25;
pub const pq_lut_zero_point: comptime_int = 128;
pub const pq_lut_qmax: comptime_int = 127;

inline fn walshHadamardTo(dst: []f32, src: []const f32, comptime k: comptime_int) void {
    @memcpy(dst, src);
    quantization.walshHadamard(dst, k);
}

pub const ProductQuantizationMode = enum {
    vanilla,
    anisotropic,
};

const RotatedLmHead = struct {
    n: usize,
    d: usize,
    data: []const f32,
    is_junk: []const bool,
};

fn buildCodebookAndAssign(
    allocator: Allocator,
    lm_head: *LmHeadMatrix,
    rotated_lm_head: []const f32,
    mode: ProductQuantizationMode,
    center_count: usize,
    codebook: []f32,
    row_codes: []u8,
) !void {
    var nb_non_junk: usize = 0;
    for (lm_head.is_junk) |is_junk| nb_non_junk += @intFromBool(!is_junk);

    const training_data = try allocator.alloc(f32, nb_non_junk * bucket_dim);
    defer allocator.free(training_data);
    var sample_index: usize = 0;
    for (0..lm_head.n) |row| {
        if (lm_head.is_junk[row]) continue;
        const bucket = switch (mode) {
            .vanilla => 0,
            .anisotropic => sample_index % nb_buckets,
        };
        const src = rotated_lm_head[row * hidden_dim + bucket * bucket_dim ..][0..bucket_dim];
        const dst = training_data[sample_index * bucket_dim ..][0..bucket_dim];
        @memcpy(dst, src);
        sample_index += 1;
    }

    std.log.info("***** Initializing {s} shared PQ codebook", .{@tagName(mode)});
    var km = try KMeansCPU.init(allocator, nb_non_junk, bucket_dim, center_count);
    defer km.deinit();
    km.solve(training_data);

    switch (mode) {
        .vanilla => {
            std.log.info("***** Assigning vanilla PQ row codes", .{});
            for (0..lm_head.n) |row| {
                if (lm_head.is_junk[row]) continue;
                for (0..nb_buckets) |bucket| {
                    const data = rotated_lm_head[row * hidden_dim + bucket * bucket_dim ..][0..bucket_dim];
                    const norm = quantization.normL2(data);
                    const closest = km.closestCenter(data, norm * norm, center_count);
                    row_codes[row * nb_buckets + bucket] = @intCast(closest[1]);
                }
            }
        },
        .anisotropic => {
            std.log.info("***** Jointly optimizing full-row ScaNN anisotropic loss", .{});
            const rotated: RotatedLmHead = .{
                .n = lm_head.n,
                .d = hidden_dim,
                .data = rotated_lm_head,
                .is_junk = lm_head.is_junk,
            };
            try km.solveAnisotropic(rotated, row_codes, pq_orth_weight);
        },
    }
    @memcpy(codebook, km.centers[0..codebook.len]);
}

pub const ProductQuantizer = struct {
    zml_handler: *Zml_handler,
    allocator: Allocator,
    lm_head: *LmHeadMatrix,
    n: usize,
    mode: ProductQuantizationMode,

    // walsh-hadamard transform of the lm_head
    rotated_lm_head: []f32,
    // walsh-hadamard transform of the query
    rotated_query: [hidden_dim]f32,

    // the codebook is the nb_center vectors of dim bucket_dim obtained by clustering
    codebook: [bucket_dim * nb_centers]f32,
    // the PQ assignation of the rows, in row -> bucket major:
    // [row_0_bucket_0_center ; row_0_bucket_1_center ... row_0_bucket_nb_buckets-1_center ; row_1_bucket_0_center ...]
    row_buckets: []u8,
    query_center_scores: [nb_buckets * nb_centers]f32,

    // Per-row norm term and global directional factor used by the statistical
    // logit-error bound. For anisotropic PQ the row term is the exact
    // sqrt(||e_parallel||^2 + w ||e_orthogonal||^2).
    row_error_norms: []f32,
    quantization_error_factor: f32,

    // sampling utils
    logits: []f32,
    candidates: [sampling_top_k]Logit,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, mode: ProductQuantizationMode) !ProductQuantizer {
        const alloc = zml_handler.allocator;
        std.debug.assert(lm_head.d == hidden_dim);
        const rotated_lm_head = try alloc.alloc(f32, lm_head.n * hidden_dim);
        errdefer alloc.free(rotated_lm_head);
        const row_buckets = try alloc.alloc(u8, lm_head.n * nb_buckets);
        errdefer alloc.free(row_buckets);
        const row_error_norms = try alloc.alloc(f32, lm_head.n);
        errdefer alloc.free(row_error_norms);
        @memset(row_error_norms, 0.0);
        const logits = try alloc.alloc(f32, lm_head.n);
        for (0..lm_head.n) |i| {
            if (lm_head.is_junk[i]) logits[i] = -1e10;
        }
        return .{
            .zml_handler = zml_handler,
            .allocator = alloc,
            .lm_head = lm_head,
            .n = lm_head.n,
            .mode = mode,
            .rotated_lm_head = rotated_lm_head,
            .rotated_query = [_]f32{0.0} ** hidden_dim,
            .codebook = [_]f32{0.0} ** (bucket_dim * nb_centers),
            .row_buckets = row_buckets,
            .query_center_scores = [_]f32{0.0} ** (nb_buckets * nb_centers),
            .row_error_norms = row_error_norms,
            .quantization_error_factor = 0.0,
            .logits = logits,
            .candidates = [_]Logit{.{}} ** sampling_top_k,
        };
    }

    pub fn deinit(self: *ProductQuantizer) void {
        self.allocator.free(self.rotated_lm_head);
        self.allocator.free(self.row_buckets);
        self.allocator.free(self.row_error_norms);
        self.allocator.free(self.logits);
    }

    pub fn buildCodebook(self: *ProductQuantizer) !void {
        // Build the unnormalized rotated lm_head.
        std.log.info("***** Rotating the lm_head", .{});
        for (0..self.n) |row| {
            const src = self.lm_head.data[row * hidden_dim ..][0..hidden_dim];
            const dst = self.rotated_lm_head[row * hidden_dim ..][0..hidden_dim];
            walshHadamardTo(dst, src, 12);
        }

        try buildCodebookAndAssign(
            self.allocator,
            self.lm_head,
            self.rotated_lm_head,
            self.mode,
            nb_centers,
            self.codebook[0..],
            self.row_buckets,
        );

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
        // This is a normal/directional-concentration approximation rather than
        // a deterministic bound. Under an exact normal model, one-sided Z = 5
        // coverage is approximately 99.99997%.
        switch (self.mode) {
            .vanilla => {
                const K: f32 = @floatFromInt(nb_centers);
                const B: f32 = @floatFromInt(bucket_dim);
                const D: f32 = @floatFromInt(hidden_dim);
                const kmeans_err = std.math.pow(f32, K, -1.0 / B);
                self.quantization_error_factor = pq_z_score * kmeans_err / @sqrt(D);
                for (0..self.n) |row| {
                    self.row_error_norms[row] = self.lm_head.row_norms[row];
                }
                std.log.info("K-means relative theoretical error = {d}", .{kmeans_err});
                std.log.info("PQ statistical Cauchy-Schwarz factor = {d}", .{self.quantization_error_factor});
            },
            .anisotropic => {
                const D: f32 = @floatFromInt(hidden_dim);
                const BucketVec = @Vector(bucket_dim, f32);
                self.quantization_error_factor = pq_z_score / @sqrt(D);
                for (0..self.n) |row| {
                    if (self.lm_head.is_junk[row]) continue;
                    var residual_norm2: f32 = 0.0;
                    var residual_dot_row: f32 = 0.0;
                    var row_norm2: f32 = 0.0;
                    const row_codes = self.row_buckets[row * nb_buckets ..][0..nb_buckets];
                    for (0..nb_buckets) |bucket| {
                        const data = self.rotated_lm_head[row * hidden_dim + bucket * bucket_dim ..][0..bucket_dim];
                        const center = self.codebook[@as(usize, row_codes[bucket]) * bucket_dim ..][0..bucket_dim];
                        const data_vec: BucketVec = data.*;
                        const center_vec: BucketVec = center.*;
                        const residual = data_vec - center_vec;
                        residual_norm2 += @reduce(.Add, residual * residual);
                        residual_dot_row += @reduce(.Add, residual * data_vec);
                        row_norm2 += @reduce(.Add, data_vec * data_vec);
                    }
                    // The optimizer's w * ||e||^2 + (1-w) * ||e_parallel||^2
                    // is exactly ||e_parallel||^2 + w * ||e_orthogonal||^2.
                    const parallel_norm2 = @min(
                        residual_norm2,
                        residual_dot_row * residual_dot_row / @max(row_norm2, 1e-20),
                    );
                    const orthogonal_norm2 = @max(0.0, residual_norm2 - parallel_norm2);
                    self.row_error_norms[row] = @sqrt(parallel_norm2 + pq_orth_weight * orthogonal_norm2);
                }
                std.log.info("Anisotropic PQ statistical Z-score factor = {d}", .{self.quantization_error_factor});
            },
        }
    }

    pub fn sample(self: *ProductQuantizer, query: []const f32) SamplingResult {
        // step 0 : compute geometric information for error bounding
        //const query_norm = quantization.normL2(query);
        //const error_bound_scale = query_norm * self.quantization_error_factor;

        // step 1 : rotate query, don't normalize it
        //self.zml_handler.tic(&self.zml_handler.timers.pq_rotate);
        const src = query[0..hidden_dim];
        const dst = self.rotated_query[0..];
        walshHadamardTo(dst, src, 12);
        //self.zml_handler.toc(&self.zml_handler.timers.pq_rotate);

        // step 2 : score every bucket against the shared codebook.
        //self.zml_handler.tic(&self.zml_handler.timers.pq_lut);
        for (0..nb_buckets) |bucket| {
            const query_data = self.rotated_query[bucket * bucket_dim ..][0..bucket_dim];
            for (0..nb_centers) |center| {
                const center_data = self.codebook[center * bucket_dim ..][0..bucket_dim];
                var dot: f32 = 0.0;
                for (0..bucket_dim) |coord| dot += center_data[coord] * query_data[coord];
                self.query_center_scores[bucket * nb_centers + center] = dot;
            }
        }
        //self.zml_handler.toc(&self.zml_handler.timers.pq_lut);

        // step 3 : gather floating-point coefficients and unnormalize the sum.
        //self.zml_handler.tic(&self.zml_handler.timers.pq_score);
        for (0..self.n) |row| {
            if (self.lm_head.is_junk[row]) continue;
            const row_codes = self.row_buckets[row * nb_buckets ..][0..nb_buckets];
            var logit: f32 = 0.0;
            for (0..nb_buckets) |bucket| {
                const bucket_lut = self.query_center_scores[bucket * nb_centers ..][0..nb_centers];
                logit += bucket_lut[row_codes[bucket]];
            }
            self.logits[row] = logit;
        }
        //self.zml_handler.toc(&self.zml_handler.timers.pq_score);

        // step 4 : find and return topK
        //self.zml_handler.tic(&self.zml_handler.timers.pq_top_k);
        sampling.findTopK(self.logits, self.candidates[0..sampling_top_k]);
        //self.zml_handler.toc(&self.zml_handler.timers.pq_top_k);
        return .{ .candidates = self.candidates, .nb = 0 };
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

/// Vanilla PQ laid out for SIMD in-register LUT lookups. The 16-center
/// codebook makes every code a nibble; codes for two consecutive buckets are
/// packed into one byte and transposed in groups of 64 rows. At query time a
/// 16-byte LUT is loaded into a vector register and `tbl`/`pshufb` scores 16
/// rows in parallel.
pub const ProductQuantizerFastScan = struct {
    zml_handler: *Zml_handler,
    allocator: Allocator,
    lm_head: *LmHeadMatrix,
    n: usize,
    nb_scan_groups: usize,
    mode: ProductQuantizationMode,

    rotated_lm_head: []f32,
    rotated_query: [hidden_dim]f32,

    codebook: [bucket_dim * center_count]f32,
    // Layout: [scan_group][bucket_pair][register][lane]. Each byte contains
    // the low-bucket code in bits 0..3 and the high-bucket code in bits 4..7.
    packed_row_buckets: []u8,
    query_center_scores: [nb_buckets * center_count]u8,
    logits: []f32,
    candidates: [sampling_top_k]Logit,

    // One 16-byte shuffle table is the defining constraint of FastScan.
    const center_count: comptime_int = 16;
    const register_lanes: comptime_int = 16;
    const registers_per_group: comptime_int = 4;
    const rows_per_group: comptime_int = register_lanes * registers_per_group;
    const bucket_pair_count: comptime_int = nb_buckets / 2;
    const buckets_per_u16_chunk: comptime_int = 256;

    const Vec16u8 = @Vector(register_lanes, u8);
    const Vec16u16 = @Vector(register_lanes, u16);
    const Vec16u32 = @Vector(register_lanes, u32);
    const Vec16u3 = @Vector(register_lanes, u3);

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, mode: ProductQuantizationMode) !ProductQuantizerFastScan {
        const alloc = zml_handler.allocator;
        std.debug.assert(lm_head.d == hidden_dim);

        const nb_scan_groups = (lm_head.n + rows_per_group - 1) / rows_per_group;
        const rotated_lm_head = try alloc.alloc(f32, lm_head.n * hidden_dim);
        errdefer alloc.free(rotated_lm_head);
        const packed_row_buckets = try alloc.alloc(u8, nb_scan_groups * bucket_pair_count * registers_per_group * register_lanes);
        errdefer alloc.free(packed_row_buckets);
        @memset(packed_row_buckets, 0);
        const logits = try alloc.alloc(f32, lm_head.n);
        for (0..lm_head.n) |i| {
            if (lm_head.is_junk[i]) logits[i] = -1e10;
        }

        return .{
            .zml_handler = zml_handler,
            .allocator = alloc,
            .lm_head = lm_head,
            .n = lm_head.n,
            .nb_scan_groups = nb_scan_groups,
            .mode = mode,
            .rotated_lm_head = rotated_lm_head,
            .rotated_query = [_]f32{0.0} ** hidden_dim,
            .codebook = [_]f32{0.0} ** (bucket_dim * center_count),
            .packed_row_buckets = packed_row_buckets,
            .query_center_scores = [_]u8{pq_lut_zero_point} ** (nb_buckets * center_count),
            .logits = logits,
            .candidates = [_]Logit{.{}} ** sampling_top_k,
        };
    }

    pub fn deinit(self: *ProductQuantizerFastScan) void {
        self.allocator.free(self.rotated_lm_head);
        self.allocator.free(self.packed_row_buckets);
        self.allocator.free(self.logits);
    }

    pub fn buildCodebook(self: *ProductQuantizerFastScan) !void {
        std.log.info("***** Rotating the lm_head for PQ FastScan", .{});
        for (0..self.n) |row| {
            const src = self.lm_head.data[row * hidden_dim ..][0..hidden_dim];
            const dst = self.rotated_lm_head[row * hidden_dim ..][0..hidden_dim];
            walshHadamardTo(dst, src, 12);
        }

        const row_codes = try self.allocator.alloc(u8, self.n * nb_buckets);
        defer self.allocator.free(row_codes);
        try buildCodebookAndAssign(
            self.allocator,
            self.lm_head,
            self.rotated_lm_head,
            self.mode,
            center_count,
            self.codebook[0..],
            row_codes,
        );

        std.log.info("***** Packing and transposing PQ FastScan row codes", .{});
        @memset(self.packed_row_buckets, 0);
        for (0..self.n) |row| {
            if (self.lm_head.is_junk[row]) continue;
            const scan_group = row / rows_per_group;
            const register = (row % rows_per_group) / register_lanes;
            const lane = row % register_lanes;
            for (0..nb_buckets) |bucket| {
                const code = row_codes[row * nb_buckets + bucket];
                const bucket_pair = bucket / 2;
                const shift: u3 = @intCast((bucket & 1) * 4);
                const offset = packedCodeOffset(scan_group, bucket_pair, register, lane);
                self.packed_row_buckets[offset] |= code << shift;
            }
        }
    }

    inline fn packedCodeOffset(scan_group: usize, bucket_pair: usize, register: usize, lane: usize) usize {
        return (((scan_group * bucket_pair_count + bucket_pair) * registers_per_group + register) * register_lanes) + lane;
    }

    inline fn normalizedQueryCenterScore(self: *ProductQuantizerFastScan, bucket: usize, center: usize, query_norm: f32) f32 {
        const query_data = self.rotated_query[bucket * bucket_dim ..][0..bucket_dim];
        const center_data = self.codebook[center * bucket_dim ..][0..bucket_dim];
        var dot: f32 = 0.0;
        for (0..bucket_dim) |coord| dot += center_data[coord] * query_data[coord];
        return dot / query_norm;
    }

    inline fn quantizeLutScore(score: f32, inverse_scale: f32) u8 {
        const rounded = @round(score * inverse_scale);
        const clipped = std.math.clamp(rounded, -@as(f32, pq_lut_qmax), @as(f32, pq_lut_qmax));
        const signed: i32 = @intFromFloat(clipped);
        return @intCast(signed + pq_lut_zero_point);
    }

    inline fn portableLookup16(table: Vec16u8, indices: Vec16u8) Vec16u8 {
        var result: Vec16u8 = undefined;
        inline for (0..register_lanes) |lane| result[lane] = table[indices[lane]];
        return result;
    }

    /// Dynamic 16-entry byte lookup. AArch64 `tbl` and x86 SSSE3 `pshufb`
    /// both interpret each byte lane as an index into the table register.
    inline fn registerLookup16(table: Vec16u8, indices: Vec16u8) Vec16u8 {
        if (comptime builtin.cpu.arch == .aarch64 and builtin.zig_backend != .stage2_c) {
            return asm (
                \\ tbl %[result].16b, {%[table].16b}, %[indices].16b
                : [result] "=&x" (-> Vec16u8),
                : [table] "x" (table),
                  [indices] "x" (indices),
            );
        }
        if (comptime builtin.cpu.arch == .x86_64 and
            builtin.zig_backend != .stage2_c and
            builtin.cpu.has(.x86, .ssse3))
        {
            return asm (
                \\ movdqa %[table], %[result]
                \\ pshufb %[indices], %[result]
                : [result] "=&x" (-> Vec16u8),
                : [table] "x" (table),
                  [indices] "x" (indices),
            );
        }
        return portableLookup16(table, indices);
    }

    pub fn sample(self: *ProductQuantizerFastScan, query: []const f32) SamplingResult {
        const query_norm = quantization.normL2(query);

        //self.zml_handler.tic(&self.zml_handler.timers.pq_rotate);
        walshHadamardTo(self.rotated_query[0..], query[0..hidden_dim], 12);
        //self.zml_handler.toc(&self.zml_handler.timers.pq_rotate);

        //self.zml_handler.tic(&self.zml_handler.timers.pq_lut);
        var max_abs_score: f32 = 0.0;
        for (0..nb_buckets) |bucket| {
            for (0..center_count) |center| {
                const score = self.normalizedQueryCenterScore(bucket, center, query_norm);
                const abs_score = @abs(score);
                max_abs_score = @max(max_abs_score, abs_score);
            }
        }
        const lut_scale = max_abs_score / @as(f32, pq_lut_qmax);
        const inverse_lut_scale = @as(f32, pq_lut_qmax) / max_abs_score;
        for (0..nb_buckets) |bucket| {
            for (0..center_count) |center| {
                const score = self.normalizedQueryCenterScore(bucket, center, query_norm);
                self.query_center_scores[bucket * center_count + center] = quantizeLutScore(score, inverse_lut_scale);
            }
        }
        //self.zml_handler.toc(&self.zml_handler.timers.pq_lut);

        //self.zml_handler.tic(&self.zml_handler.timers.pq_score);
        const low_nibble: Vec16u8 = @splat(0x0f);
        const high_shift: Vec16u3 = @splat(4);
        const accumulator_bias: i32 = pq_lut_zero_point * nb_buckets;
        const final_lut_scale = lut_scale * query_norm;

        for (0..self.nb_scan_groups) |scan_group| {
            var accumulators: [registers_per_group]Vec16u32 = [_]Vec16u32{@splat(0)} ** registers_per_group;
            var chunk_accumulators: [registers_per_group]Vec16u16 = [_]Vec16u16{@splat(0)} ** registers_per_group;

            for (0..bucket_pair_count) |bucket_pair| {
                const low_bucket = bucket_pair * 2;
                const low_lut: Vec16u8 = self.query_center_scores[low_bucket * center_count ..][0..center_count].*;
                const high_lut: Vec16u8 = self.query_center_scores[(low_bucket + 1) * center_count ..][0..center_count].*;

                inline for (0..registers_per_group) |register| {
                    const codes_offset = packedCodeOffset(scan_group, bucket_pair, register, 0);
                    const packed_codes: Vec16u8 = self.packed_row_buckets[codes_offset..][0..register_lanes].*;
                    const low_scores = registerLookup16(low_lut, packed_codes & low_nibble);
                    chunk_accumulators[register] += @as(Vec16u16, @intCast(low_scores));
                    const high_scores = registerLookup16(high_lut, packed_codes >> high_shift);
                    chunk_accumulators[register] += @as(Vec16u16, @intCast(high_scores));
                }

                const buckets_done = @min((bucket_pair + 1) * 2, nb_buckets);
                if (buckets_done % buckets_per_u16_chunk == 0 or buckets_done == nb_buckets) {
                    inline for (0..registers_per_group) |register| {
                        accumulators[register] += @as(Vec16u32, @intCast(chunk_accumulators[register]));
                        chunk_accumulators[register] = @splat(0);
                    }
                }
            }

            inline for (0..registers_per_group) |register| {
                inline for (0..register_lanes) |lane| {
                    const row = scan_group * rows_per_group + register * register_lanes + lane;
                    if (row < self.n and !self.lm_head.is_junk[row]) {
                        const centered_accumulator: i32 = @as(i32, @intCast(accumulators[register][lane])) - accumulator_bias;
                        const logit = @as(f32, @floatFromInt(centered_accumulator)) * final_lut_scale;
                        self.logits[row] = logit;
                    }
                }
            }
        }
        //self.zml_handler.toc(&self.zml_handler.timers.pq_score);

        //self.zml_handler.tic(&self.zml_handler.timers.pq_top_k);
        sampling.findTopK(self.logits, self.candidates[0..sampling_top_k]);
        //self.zml_handler.toc(&self.zml_handler.timers.pq_top_k);
        return .{ .candidates = self.candidates, .nb = 0 };
    }

    pub fn sampleLog(self: *ProductQuantizerFastScan, sampler: *Sampler, query: []const f32) void {
        std.log.info("*****", .{});
        std.log.info("***** new PQ FastScan sampling", .{});
        std.log.info("*****", .{});
        var approx_sampling = self.sample(query);
        sampler.sample(query);
        sampler.logRealVsApproxSampling(self.logits, "PQ FastScan logits");
        sampler.logApproxVsRealSampling(&approx_sampling.top_k, "PQ FastScan logit");
    }
};
