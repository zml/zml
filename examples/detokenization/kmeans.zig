const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");

const simd_len: comptime_int = 8;
const max_iters: comptime_int = 256;
const anisotropic_cg_max_iters: comptime_int = 64;
const anisotropic_cg_relative_tolerance: comptime_float = 1e-6;
const anisotropic_ridge: comptime_float = 1e-6;
const anisotropic_max_iters: comptime_int = 8;
const Vec = @Vector(simd_len, f32);

pub const KMeansCPU = struct {
    allocator: std.mem.Allocator,

    n_max: usize,
    d: usize,
    k: usize,

    data: []f32,
    centers: []f32,
    assigned: []usize,
    nb_assigned: []usize,
    weights: []f32,
    data_norms2: []f32,
    center_norms: []f32,
    has_centers: bool = false,

    pub fn init(allocator: std.mem.Allocator, n_max: usize, d: usize, k: usize) !KMeansCPU {
        const centers = try allocator.alloc(f32, k * d);
        errdefer allocator.free(centers);

        const assigned = try allocator.alloc(usize, n_max);
        errdefer allocator.free(assigned);

        const nb_assigned = try allocator.alloc(usize, k);
        errdefer allocator.free(nb_assigned);

        const weights = try allocator.alloc(f32, n_max);
        errdefer allocator.free(weights);

        const data_norms2 = try allocator.alloc(f32, n_max);
        errdefer allocator.free(data_norms2);

        const center_norms = try allocator.alloc(f32, k);
        errdefer allocator.free(center_norms);

        return .{
            .allocator = allocator,
            .n_max = n_max,
            .d = d,
            .k = k,
            .data = undefined,
            .centers = centers,
            .assigned = assigned,
            .nb_assigned = nb_assigned,
            .weights = weights,
            .data_norms2 = data_norms2,
            .center_norms = center_norms,
        };
    }

    pub fn deinit(self: *KMeansCPU) void {
        self.allocator.free(self.centers);
        self.allocator.free(self.assigned);
        self.allocator.free(self.nb_assigned);
        self.allocator.free(self.weights);
        self.allocator.free(self.data_norms2);
        self.allocator.free(self.center_norms);
    }

    inline fn addTo(dst: []f32, src: []const f32, d: usize) void {
        var i: usize = 0;
        while (i + simd_len <= d) : (i += simd_len) {
            const x: Vec = dst[i..][0..simd_len].*;
            const y: Vec = src[i..][0..simd_len].*;
            dst[i..][0..simd_len].* = x + y;
        }
        while (i < d) : (i += 1) {
            dst[i] += src[i];
        }
    }

    inline fn addWeightedTo(dst: []f32, src: []const f32, weight: f32, d: usize) void {
        const w: Vec = @splat(weight);
        var i: usize = 0;
        while (i + simd_len <= d) : (i += simd_len) {
            const x: Vec = dst[i..][0..simd_len].*;
            const y: Vec = src[i..][0..simd_len].*;
            dst[i..][0..simd_len].* = @mulAdd(Vec, w, y, x);
        }
        while (i < d) : (i += 1) {
            dst[i] += weight * src[i];
        }
    }

    inline fn scaleInPlace(dst: []f32, scalar: f32, d: usize) void {
        const s: Vec = @splat(scalar);
        var i: usize = 0;
        while (i + simd_len <= d) : (i += simd_len) {
            const x: Vec = dst[i..][0..simd_len].*;
            dst[i..][0..simd_len].* = x * s;
        }
        while (i < d) : (i += 1) {
            dst[i] *= scalar;
        }
    }

    inline fn dotSlices(lhs: []const f32, rhs: []const f32, d: usize) f32 {
        var acc: Vec = @splat(0.0);
        var i: usize = 0;
        while (i + simd_len <= d) : (i += simd_len) {
            const x: Vec = lhs[i..][0..simd_len].*;
            const y: Vec = rhs[i..][0..simd_len].*;
            acc = @mulAdd(Vec, x, y, acc);
        }
        var result = @reduce(.Add, acc);
        while (i < d) : (i += 1) {
            result += lhs[i] * rhs[i];
        }
        return result;
    }

    inline fn distance2Slices(lhs: []const f32, rhs: []const f32, d: usize) f32 {
        var acc: Vec = @splat(0.0);
        var i: usize = 0;
        while (i + simd_len <= d) : (i += simd_len) {
            const x: Vec = lhs[i..][0..simd_len].*;
            const y: Vec = rhs[i..][0..simd_len].*;
            const diff = x - y;
            acc = @mulAdd(Vec, diff, diff, acc);
        }
        var result = @reduce(.Add, acc);
        while (i < d) : (i += 1) {
            const diff = lhs[i] - rhs[i];
            result += diff * diff;
        }
        return result;
    }

    inline fn computeDataNorms(self: *KMeansCPU, N: usize) void {
        for (0..N) |point| {
            const pt = self.data[point * self.d .. (point + 1) * self.d];
            self.data_norms2[point] = dotSlices(pt, pt, self.d);
        }
    }

    inline fn updateCenterNorm(self: *KMeansCPU, center: usize) void {
        const ce = self.centers[center * self.d .. (center + 1) * self.d];
        self.center_norms[center] = dotSlices(ce, ce, self.d);
    }

    pub inline fn computeCenterNorms(self: *KMeansCPU) void {
        for (0..self.k) |center| {
            self.updateCenterNorm(center);
        }
    }

    pub fn solve(self: *KMeansCPU, data: []f32) void {
        // vanilla KMeans, assign point to nearest center, centers are averages of clusters
        self.data = data;
        const N = data.len / self.d;
        self.computeDataNorms(N);
        // init centers with deterministic kmeans++ like selection
        @memcpy(self.centers[0..self.d], data[0..self.d]); // first center is pt0
        self.updateCenterNorm(0);
        @memset(self.assigned, self.k);
        var nb_centers: usize = 1;
        while (nb_centers < self.k) {
            // next center is the point in the dataset that has maximal distance to closest center
            var max_min_dist2: f32 = 0.0;
            var next_center: usize = 0;
            for (0..N) |point| {
                const min_dist2, _ = self.closestCenter(self.data[point * self.d ..][0..self.d], self.data_norms2[point], nb_centers);
                if (min_dist2 > max_min_dist2) {
                    max_min_dist2 = min_dist2;
                    next_center = point;
                }
            }
            @memcpy(self.centers[nb_centers * self.d .. (nb_centers + 1) * self.d], data[next_center * self.d .. (next_center + 1) * self.d]);
            self.updateCenterNorm(nb_centers);
            nb_centers += 1;
        }
        std.log.info("{s: >6} {s: >12} {s: >8}", .{ "iter", "sos", "chg" });
        for (0..max_iters) |it| {
            var sos_err: f32 = 0.0;
            var nb_changed: u32 = 0;
            // find the assigned center of each datapoint
            for (0..N) |point| {
                const min_dist2, const min_center = self.closestCenter(self.data[point * self.d ..][0..self.d], self.data_norms2[point], self.k);
                nb_changed += @intFromBool(self.assigned[point] != min_center);
                self.assigned[point] = min_center;
                sos_err += min_dist2;
            }
            sos_err /= @floatFromInt(N);
            // update the centers
            @memset(self.centers, 0.0);
            @memset(self.nb_assigned, 0);
            for (0..N) |point| {
                const center = self.assigned[point];
                const pt = self.data[point * self.d .. (point + 1) * self.d];
                const ce = self.centers[center * self.d .. (center + 1) * self.d];
                addTo(ce, pt, self.d);
                self.nb_assigned[center] += 1;
            }
            for (0..self.k) |center| {
                const inv_card = 1.0 / @as(f32, @floatFromInt(self.nb_assigned[center]));
                const ce = self.centers[center * self.d .. (center + 1) * self.d];
                scaleInPlace(ce, inv_card, self.d);
                self.updateCenterNorm(center);
            }
            std.log.info("{d: >6} {d: >12.6} {d: >8}", .{ it, sos_err, nb_changed });
            if (nb_changed == 0) break;
        }
        self.has_centers = true;
    }

    pub inline fn distance2(self: *KMeansCPU, point: []const f32, point_norm2: f32, center: usize) f32 {
        const ce = self.centers[center * self.d .. (center + 1) * self.d];
        return point_norm2 + self.center_norms[center] - 2.0 * dotSlices(point, ce, self.d);
    }

    pub inline fn closestCenter(self: *KMeansCPU, point: []const f32, point_norm2: f32, nb_center: usize) struct { f32, usize } {
        var min_dist2 = std.math.floatMax(f32);
        var min_center: usize = self.k;
        for (0..nb_center) |center| {
            const d2 = self.distance2(point, point_norm2, center);
            if (d2 < min_dist2) {
                min_dist2 = d2;
                min_center = center;
            }
        }
        return .{ min_dist2, min_center };
    }

    pub fn solveSpherical(self: *KMeansCPU, data: []f32) void {
        // vanilla spherical kmeans. each point in data is assumed to be unit norm
        // each point assigned to most similar center, centers are averages of clusters
        self.data = data;
        const N = data.len / self.d;
        // init centers with deterministic kmeans++ like selection
        @memcpy(self.centers[0..self.d], data[0..self.d]); // first center is pt0
        @memset(self.assigned, self.k);
        var nb_centers: usize = 1;
        while (nb_centers < self.k) {
            // next center is the point in the dataset that has minimal similarity to most similar center
            var min_max_sim: f32 = 1.0;
            var next_center: usize = 0;
            for (0..N) |point| {
                const max_sim, _ = self.mostSimilarCenter(point, nb_centers);
                if (max_sim < min_max_sim) {
                    min_max_sim = max_sim;
                    next_center = point;
                }
            }
            @memcpy(self.centers[nb_centers * self.d .. (nb_centers + 1) * self.d], data[next_center * self.d .. (next_center + 1) * self.d]);
            nb_centers += 1;
        }
        std.log.info("{s: >6} {s: >12} {s: >8}", .{ "iter", "sim", "chg" });
        for (0..max_iters) |it| {
            var avg_sim: f32 = 0.0;
            var nb_changed: u32 = 0;
            // find the assigned center of each datapoint
            for (0..N) |point| {
                const max_sim, const max_center = self.mostSimilarCenter(point, self.k);
                nb_changed += @intFromBool(self.assigned[point] != max_center);
                self.assigned[point] = max_center;
                avg_sim += max_sim;
            }
            avg_sim /= @floatFromInt(N);
            // update the centers : average the clusters
            @memset(self.centers, 0.0);
            @memset(self.nb_assigned, 0);
            for (0..N) |point| {
                const center = self.assigned[point];
                const pt = self.data[point * self.d .. (point + 1) * self.d];
                const ce = self.centers[center * self.d .. (center + 1) * self.d];
                addTo(ce, pt, self.d);
                self.nb_assigned[center] += 1;
            }
            // update the centers : normalize the centers
            for (0..self.k) |center| {
                self.normalizeCenter(center);
            }
            std.log.info("{d: >6} {d: >12.6} {d: >8}", .{ it, avg_sim, nb_changed });
            if (nb_changed == 0) break;
        }
        self.has_centers = true;
    }

    pub fn solveSphericalWeighted(self: *KMeansCPU, data: []f32) void {
        // weighted sperical kmeans. do not assumes data contains unit norm points
        // points are assigned to most similar center (centers are unit norm, point norm doesn't change assignation)
        // centers are weighted averages of clusters, weights being the norm of the points, then normalized.
        // this corresponds to initial norms squared weighting for the normalized data points.
        self.data = data;
        const N = data.len / self.d;
        // init weights with the norms of the points (not squared, as we leave the points un-normalized)
        for (0..N) |point| {
            const pt = self.data[point * self.d .. (point + 1) * self.d];
            self.weights[point] = @sqrt(dotSlices(pt, pt, self.d));
        }
        // init centers with deterministic kmeans++ like selection
        @memcpy(self.centers[0..self.d], data[0..self.d]); // first center is pt0
        self.normalizeCenter(0);
        @memset(self.assigned, self.k);
        var nb_centers: usize = 1;
        while (nb_centers < self.k) {
            // next center is the point in the dataset that has minimal similarity to most similar center
            var min_max_sim: f32 = std.math.floatMax(f32);
            var next_center: usize = 0;
            for (0..N) |point| {
                const max_sim, _ = self.mostSimilarCenter(point, nb_centers);
                if (max_sim < min_max_sim) {
                    min_max_sim = max_sim;
                    next_center = point;
                }
            }
            @memcpy(self.centers[nb_centers * self.d .. (nb_centers + 1) * self.d], data[next_center * self.d .. (next_center + 1) * self.d]);
            self.normalizeCenter(nb_centers);
            nb_centers += 1;
        }
        std.log.info("{s: >6} {s: >12} {s: >8}", .{ "iter", "sim", "chg" });
        for (0..max_iters) |it| {
            var avg_sim: f32 = 0.0;
            var nb_changed: u32 = 0;
            // find the assigned center of each datapoint (same as un-weighted version)
            for (0..N) |point| {
                const max_sim, const max_center = self.mostSimilarCenter(point, self.k);
                nb_changed += @intFromBool(self.assigned[point] != max_center);
                self.assigned[point] = max_center;
                avg_sim += max_sim / self.weights[point];
            }
            avg_sim /= @floatFromInt(N);
            // update the centers : weight average the clusters
            @memset(self.centers, 0.0);
            @memset(self.nb_assigned, 0);
            for (0..N) |point| {
                const w = self.weights[point];
                const center = self.assigned[point];
                const pt = self.data[point * self.d .. (point + 1) * self.d];
                const ce = self.centers[center * self.d .. (center + 1) * self.d];
                addWeightedTo(ce, pt, w, self.d);
                self.nb_assigned[center] += 1;
            }
            // update the centers : normalize the centers
            for (0..self.k) |center| {
                self.normalizeCenter(center);
            }
            std.log.info("{d: >6} {d: >12.6} {d: >8}", .{ it, avg_sim, nb_changed });
            if (nb_changed == 0) break;
        }
        self.has_centers = true;
    }

    pub inline fn normalizeCenter(self: *KMeansCPU, center: usize) void {
        const ce = self.centers[center * self.d .. (center + 1) * self.d];
        const inv_norm = 1.0 / @sqrt(dotSlices(ce, ce, self.d));
        scaleInPlace(ce, inv_norm, self.d);
    }

    pub inline fn dot(self: *KMeansCPU, point: usize, center: usize) f32 {
        const pt = self.data[point * self.d .. (point + 1) * self.d];
        const ce = self.centers[center * self.d .. (center + 1) * self.d];
        return dotSlices(pt, ce, self.d);
    }

    pub inline fn mostSimilarCenter(self: *KMeansCPU, point: usize, nb_center: usize) struct { f32, u8 } {
        var max_sim = -std.math.floatMax(f32);
        var max_center: usize = self.k;
        for (0..nb_center) |center| {
            const sim = self.dot(point, center);
            if (sim > max_sim) {
                max_sim = sim;
                max_center = center;
            }
        }
        return .{ max_sim, @intCast(max_center) };
    }

    const CgResult = struct {
        relative_residual: f32,
        iterations: usize,
    };

    fn initializeAnisotropicCodes(self: *KMeansCPU, lm_head: anytype, row_codes: []u8, nb_buckets: usize) void {
        self.computeCenterNorms();
        @memset(row_codes, 0);
        for (0..lm_head.n) |row| {
            if (lm_head.is_junk[row]) continue;
            for (0..nb_buckets) |bucket| {
                const x = lm_head.data[row * lm_head.d + bucket * self.d ..][0..self.d];
                const x_norm2 = dotSlices(x, x, self.d);
                const closest = self.closestCenter(x, x_norm2, self.k);
                row_codes[row * nb_buckets + bucket] = @intCast(closest[1]);
            }
        }
    }

    fn computeAnisotropicRowState(
        self: *KMeansCPU,
        lm_head: anytype,
        row_codes: []const u8,
        nb_buckets: usize,
        squared_error: []f32,
        parallel_error: []f32,
    ) void {
        for (0..lm_head.n) |row| {
            squared_error[row] = 0.0;
            parallel_error[row] = 0.0;
            if (lm_head.is_junk[row]) continue;
            for (0..nb_buckets) |bucket| {
                const x = lm_head.data[row * lm_head.d + bucket * self.d ..][0..self.d];
                const center: usize = row_codes[row * nb_buckets + bucket];
                const c = self.centers[center * self.d .. (center + 1) * self.d];
                const x_norm2 = dotSlices(x, x, self.d);
                const dot_xc = dotSlices(x, c, self.d);
                squared_error[row] += x_norm2 + self.center_norms[center] - 2.0 * dot_xc;
                parallel_error[row] += x_norm2 - dot_xc;
            }
        }
    }

    fn buildAnisotropicTarget(
        self: *KMeansCPU,
        lm_head: anytype,
        row_codes: []const u8,
        nb_buckets: usize,
        target: []f32,
        counts: []usize,
    ) void {
        @memset(target, 0.0);
        @memset(counts, 0);
        for (0..lm_head.n) |row| {
            if (lm_head.is_junk[row]) continue;
            for (0..nb_buckets) |bucket| {
                const center: usize = row_codes[row * nb_buckets + bucket];
                const x = lm_head.data[row * lm_head.d + bucket * self.d ..][0..self.d];
                addTo(target[center * self.d .. (center + 1) * self.d], x, self.d);
                counts[center] += 1;
            }
        }
    }

    /// Applies the full shared-codebook normal matrix without materializing its
    /// (K * B) x (K * B) entries:
    /// A = w R^T R + (1-w) sum_i(g_i g_i^T / ||x_i||^2) + ridge I.
    fn anisotropicNormalMatVec(
        self: *KMeansCPU,
        lm_head: anytype,
        row_codes: []const u8,
        row_norms2: []const f32,
        counts: []const usize,
        orth_weight: f32,
        vector: []const f32,
        output: []f32,
        nb_buckets: usize,
    ) void {
        for (0..self.k) |center| {
            const diagonal: Vec = @splat(orth_weight * @as(f32, @floatFromInt(counts[center])) + anisotropic_ridge);
            var coord: usize = 0;
            while (coord + simd_len <= self.d) : (coord += simd_len) {
                const v: Vec = vector[center * self.d + coord ..][0..simd_len].*;
                output[center * self.d + coord ..][0..simd_len].* = diagonal * v;
            }
            const diagonal_scalar = orth_weight * @as(f32, @floatFromInt(counts[center])) + anisotropic_ridge;
            while (coord < self.d) : (coord += 1) {
                output[center * self.d + coord] = diagonal_scalar * vector[center * self.d + coord];
            }
        }

        for (0..lm_head.n) |row| {
            if (lm_head.is_junk[row]) continue;
            var g_dot_vector: f32 = 0.0;
            for (0..nb_buckets) |bucket| {
                const center: usize = row_codes[row * nb_buckets + bucket];
                const x = lm_head.data[row * lm_head.d + bucket * self.d ..][0..self.d];
                const v = vector[center * self.d .. (center + 1) * self.d];
                g_dot_vector += dotSlices(x, v, self.d);
            }
            const scale = (1.0 - orth_weight) * g_dot_vector / row_norms2[row];
            for (0..nb_buckets) |bucket| {
                const center: usize = row_codes[row * nb_buckets + bucket];
                const x = lm_head.data[row * lm_head.d + bucket * self.d ..][0..self.d];
                addWeightedTo(output[center * self.d .. (center + 1) * self.d], x, scale, self.d);
            }
        }
    }

    fn solveAnisotropicCentersCG(
        self: *KMeansCPU,
        lm_head: anytype,
        row_codes: []const u8,
        row_norms2: []const f32,
        counts: []usize,
        target: []f32,
        residual: []f32,
        direction: []f32,
        normal_direction: []f32,
        orth_weight: f32,
        nb_buckets: usize,
    ) CgResult {
        const parameter_count = self.k * self.d;
        self.buildAnisotropicTarget(lm_head, row_codes, nb_buckets, target, counts);
        self.anisotropicNormalMatVec(lm_head, row_codes, row_norms2, counts, orth_weight, self.centers, normal_direction, nb_buckets);

        var coord: usize = 0;
        while (coord < parameter_count) : (coord += simd_len) {
            const b: Vec = target[coord..][0..simd_len].*;
            const ax: Vec = normal_direction[coord..][0..simd_len].*;
            residual[coord..][0..simd_len].* = b - ax;
        }
        @memcpy(direction, residual);

        const target2 = @max(dotSlices(target, target, parameter_count), 1e-20);
        var residual2 = dotSlices(residual, residual, parameter_count);
        var relative_residual = @sqrt(residual2 / target2);
        if (relative_residual <= anisotropic_cg_relative_tolerance) {
            return .{ .relative_residual = relative_residual, .iterations = 0 };
        }

        const iteration_limit = @min(anisotropic_cg_max_iters, parameter_count);
        for (0..iteration_limit) |iteration| {
            self.anisotropicNormalMatVec(lm_head, row_codes, row_norms2, counts, orth_weight, direction, normal_direction, nb_buckets);
            const denominator = dotSlices(direction, normal_direction, parameter_count);
            if (!(denominator > 1e-20) or !std.math.isFinite(denominator)) {
                return .{ .relative_residual = relative_residual, .iterations = iteration };
            }

            const alpha = residual2 / denominator;
            addWeightedTo(self.centers, direction, alpha, parameter_count);
            addWeightedTo(residual, normal_direction, -alpha, parameter_count);
            const next_residual2 = dotSlices(residual, residual, parameter_count);
            relative_residual = @sqrt(next_residual2 / target2);
            if (relative_residual <= anisotropic_cg_relative_tolerance) {
                return .{ .relative_residual = relative_residual, .iterations = iteration + 1 };
            }

            const beta: Vec = @splat(next_residual2 / @max(residual2, 1e-20));
            var i: usize = 0;
            while (i < parameter_count) : (i += simd_len) {
                const r: Vec = residual[i..][0..simd_len].*;
                const p: Vec = direction[i..][0..simd_len].*;
                direction[i..][0..simd_len].* = @mulAdd(Vec, beta, p, r);
            }
            residual2 = next_residual2;
        }
        return .{ .relative_residual = relative_residual, .iterations = iteration_limit };
    }

    /// Full-row ScaNN anisotropic optimization with a shared B-dimensional
    /// codebook. Centers must first be initialized by solve/solveSpherical.
    pub fn solveAnisotropic(self: *KMeansCPU, lm_head: anytype, row_codes: []u8, orth_weight: f32) !void {
        std.debug.assert(orth_weight > 0.0 and orth_weight <= 1.0);
        std.debug.assert(self.has_centers and self.k <= 256);
        std.debug.assert(lm_head.d % self.d == 0);
        std.debug.assert(lm_head.data.len == lm_head.n * lm_head.d);
        const nb_buckets = lm_head.d / self.d;
        std.debug.assert(row_codes.len == lm_head.n * nb_buckets);

        const parameter_count = self.k * self.d;
        std.debug.assert(parameter_count % simd_len == 0);
        const row_norms2 = try self.allocator.alloc(f32, lm_head.n);
        defer self.allocator.free(row_norms2);
        const squared_error = try self.allocator.alloc(f32, lm_head.n);
        defer self.allocator.free(squared_error);
        const parallel_error = try self.allocator.alloc(f32, lm_head.n);
        defer self.allocator.free(parallel_error);
        const counts = try self.allocator.alloc(usize, self.k);
        defer self.allocator.free(counts);
        const cg_storage = try self.allocator.alloc(f32, parameter_count * 4);
        defer self.allocator.free(cg_storage);
        const target = cg_storage[0 * parameter_count .. 1 * parameter_count];
        const residual = cg_storage[1 * parameter_count .. 2 * parameter_count];
        const direction = cg_storage[2 * parameter_count .. 3 * parameter_count];
        const normal_direction = cg_storage[3 * parameter_count .. 4 * parameter_count];

        var active_count: usize = 0;
        for (0..lm_head.n) |row| {
            if (lm_head.is_junk[row]) {
                row_norms2[row] = 1.0;
                continue;
            }
            const x = lm_head.data[row * lm_head.d .. (row + 1) * lm_head.d];
            row_norms2[row] = @max(dotSlices(x, x, lm_head.d), 1e-20);
            active_count += 1;
        }
        std.debug.assert(active_count > 0);

        self.initializeAnisotropicCodes(lm_head, row_codes, nb_buckets);
        std.log.info("CPU full anisotropic PQ: rows={d} buckets={d} centers={d} block_dim={d} system={d}x{d}", .{ lm_head.n, nb_buckets, self.k, self.d, parameter_count, parameter_count });
        std.log.info("{s: >6} {s: >12} {s: >12} {s: >8} {s: >6}", .{ "iter", "aniso", "cg-resid", "chg", "cg-it" });

        for (0..anisotropic_max_iters) |iteration| {
            self.computeAnisotropicRowState(lm_head, row_codes, nb_buckets, squared_error, parallel_error);
            var total_changed: usize = 0;

            // Sequential coordinate descent over buckets. Each candidate is scored
            // with the full-row loss while all other bucket codes remain fixed.
            for (0..nb_buckets) |bucket| {
                for (0..lm_head.n) |row| {
                    if (lm_head.is_junk[row]) continue;
                    const code_index = row * nb_buckets + bucket;
                    const old_center: usize = row_codes[code_index];
                    const x = lm_head.data[row * lm_head.d + bucket * self.d ..][0..self.d];
                    const x_norm2 = dotSlices(x, x, self.d);
                    const old_c = self.centers[old_center * self.d .. (old_center + 1) * self.d];
                    const old_dot = dotSlices(x, old_c, self.d);
                    const old_squared = x_norm2 + self.center_norms[old_center] - 2.0 * old_dot;
                    const old_parallel = x_norm2 - old_dot;
                    const base_squared = squared_error[row] - old_squared;
                    const base_parallel = parallel_error[row] - old_parallel;

                    var best_center = old_center;
                    var best_squared = old_squared;
                    var best_parallel = old_parallel;
                    var best_loss = std.math.floatMax(f32);
                    for (0..self.k) |center| {
                        const c = self.centers[center * self.d .. (center + 1) * self.d];
                        const dot_xc = dotSlices(x, c, self.d);
                        const candidate_squared = x_norm2 + self.center_norms[center] - 2.0 * dot_xc;
                        const candidate_parallel = x_norm2 - dot_xc;
                        const full_parallel = base_parallel + candidate_parallel;
                        const loss = orth_weight * (base_squared + candidate_squared) +
                            (1.0 - orth_weight) * full_parallel * full_parallel / row_norms2[row];
                        if (loss < best_loss) {
                            best_loss = loss;
                            best_center = center;
                            best_squared = candidate_squared;
                            best_parallel = candidate_parallel;
                        }
                    }

                    total_changed += @intFromBool(best_center != old_center);
                    row_codes[code_index] = @intCast(best_center);
                    squared_error[row] = base_squared + best_squared;
                    parallel_error[row] = base_parallel + best_parallel;
                }
            }

            var loss: f64 = 0.0;
            for (0..lm_head.n) |row| {
                if (lm_head.is_junk[row]) continue;
                loss += orth_weight * @as(f64, squared_error[row]) +
                    (1.0 - orth_weight) * @as(f64, parallel_error[row]) * @as(f64, parallel_error[row]) / @as(f64, row_norms2[row]);
            }
            loss /= @floatFromInt(active_count);

            const cg = self.solveAnisotropicCentersCG(
                lm_head,
                row_codes,
                row_norms2,
                counts,
                target,
                residual,
                direction,
                normal_direction,
                orth_weight,
                nb_buckets,
            );
            self.computeCenterNorms();
            std.log.info("{d: >6} {d: >12.6} {d: >12.6} {d: >8} {d: >6}", .{ iteration, loss, cg.relative_residual, total_changed, cg.iterations });
            if (iteration > 0 and total_changed == 0) break;
        }
    }
};

const GpuKMeansMode = enum {
    euclidean,
    spherical,
    spherical_weighted,
};

const GpuKMeansModel = struct {
    mode: GpuKMeansMode,
    k: i64,
    block_dim: i64,
    nb_buckets: i64 = 1,
    orth_weight: f32 = 1.0,

    fn distances(data_: zml.Tensor, centers_: zml.Tensor) zml.Tensor {
        const data = data_.withTags(.{ .point, .coord });
        const centers = centers_.withTags(.{ .center, .coord });
        const dots = data.dot(centers, .coord);
        const data_norm2 = data.mul(data).sum(.coord).squeeze(.coord).insertAxes(.last, .{.center}).broad(dots.shape());
        const center_norm2 = centers.mul(centers).sum(.coord).squeeze(.coord).insertAxes(.center, .{.point}).broad(dots.shape());
        return data_norm2.add(center_norm2).sub(dots.scale(2.0));
    }

    pub fn iteration(self: @This(), data_: zml.Tensor, centers_: zml.Tensor, old_assigned_: zml.Tensor, weights_: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        const data = data_.withTags(.{ .point, .coord });
        const centers = centers_.withTags(.{ .center, .coord });
        const old_assigned = old_assigned_.withTags(.{.point});
        const weights = weights_.withTags(.{.point});

        const scores = switch (self.mode) {
            .euclidean => distances(data, centers).negate(),
            .spherical, .spherical_weighted => data.dot(centers, .coord),
        };
        const best = scores.argMax(.center);
        const assigned = best.indices.squeeze(.center).convert(.u32);
        const metric = switch (self.mode) {
            .euclidean => best.values.negate().mean(.point).asScalar(),
            .spherical => best.values.mean(.point).asScalar(),
            .spherical_weighted => best.values.squeeze(.center).div(weights).mean(.point).asScalar(),
        };
        const nb_changed = old_assigned.cmp(.NE, assigned).convert(.u32).sum(.point).asScalar();

        const point_weights = switch (self.mode) {
            .euclidean, .spherical => zml.Tensor.scalar(1.0, .f32).broad(weights.shape()),
            .spherical_weighted => weights,
        };
        const updates = data.mul(point_weights.insertAxes(.last, .{.coord}).broad(data.shape()));
        const sums = zml.Tensor.scalar(0.0, .f32).broad(centers.shape()).scatterSlices(
            .{ .center = assigned },
            updates,
            .{ .update_fn = zml.Tensor.ScatterOpts.increment },
        );
        const counts = zml.Tensor.scalar(0.0, .f32).broad(zml.Shape.init(.{ .center = self.k }, .f32)).scatterSlices(
            .{ .center = assigned },
            zml.Tensor.scalar(1.0, .f32).broad(weights.shape()),
            .{ .update_fn = zml.Tensor.ScatterOpts.increment },
        );
        const non_empty = counts.cmp(.GT, zml.Tensor.scalar(0.0, .f32));
        const empty_count = non_empty.not().convert(.u32).sum(.center).asScalar();
        const safe_counts = counts.maximum(zml.Tensor.scalar(1.0, .f32));
        var updated = sums.div(safe_counts.insertAxes(.last, .{.coord}).broad(sums.shape()));
        if (self.mode != .euclidean) {
            const norms = updated.mul(updated).sum(.coord).sqrt().maximum(zml.Tensor.scalar(1e-20, .f32));
            updated = updated.div(norms.broad(updated.shape()));
        }
        const keep_new = non_empty.insertAxes(.last, .{.coord}).broad(updated.shape());
        const new_centers = keep_new.select(updated, centers);
        return .{ new_centers, assigned, metric, nb_changed, empty_count };
    }

    fn fullData(data_: zml.Tensor) zml.Tensor {
        return data_.withTags(.{ .row, .bucket, .coord });
    }

    fn codebook(centers_: zml.Tensor) zml.Tensor {
        return centers_.withTags(.{ .center, .coord });
    }

    fn bucketSlice(x: zml.Tensor, bucket: zml.Tensor) zml.Tensor {
        return x.dynamicSlice1d(x.axis(.bucket), .{ .start = bucket, .len = 1 }).squeeze(.bucket);
    }

    fn blockDistances(x: zml.Tensor, centers: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const dots = x.dot(centers, .coord);
        const x_norm2 = x.mul(x).sum(.coord).squeeze(.coord);
        const center_norm2 = centers.mul(centers).sum(.coord).squeeze(.coord);
        const dist = x_norm2.insertAxes(.last, .{.center}).broad(dots.shape())
            .add(center_norm2.insertAxes(.center, .{.row}).broad(dots.shape()))
            .sub(dots.scale(2.0));
        return .{ dist, x_norm2.insertAxes(.last, .{.center}).broad(dots.shape()).sub(dots) };
    }

    pub fn initializeCodes(self: @This(), data_: zml.Tensor, centers_: zml.Tensor, codes_: zml.Tensor, bucket: zml.Tensor) zml.Tensor {
        _ = self;
        const data = fullData(data_);
        const centers = codebook(centers_);
        const codes = codes_.withTags(.{ .row, .bucket });
        const x = bucketSlice(data, bucket);
        const dist, _ = blockDistances(x, centers);
        const selected = dist.negate().argMax(.center).indices.squeeze(.center).convert(.u32);
        return codes.dynamicUpdateSlice(.{ .bucket = bucket }, selected.insertAxes(.last, .{.bucket}));
    }

    pub fn rowState(self: @This(), data_: zml.Tensor, centers_: zml.Tensor, codes_: zml.Tensor, is_junk_: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        const data = fullData(data_);
        const centers = codebook(centers_);
        const codes = codes_.withTags(.{ .row, .bucket });
        const is_junk = is_junk_.withTags(.{.row});
        const selected = centers.gather(.{ .center = codes }, .{});
        const reconstruction_error = data.sub(selected);
        var squared_error = reconstruction_error.mul(reconstruction_error).sum(.coord).squeeze(.coord).sum(.bucket).squeeze(.bucket);
        var parallel_error = data.mul(reconstruction_error).sum(.coord).squeeze(.coord).sum(.bucket).squeeze(.bucket);
        const active = is_junk.not();
        squared_error = active.select(squared_error, zml.Tensor.scalar(0.0, .f32));
        parallel_error = active.select(parallel_error, zml.Tensor.scalar(0.0, .f32));
        const norm2 = data.mul(data).sum(.coord).squeeze(.coord).sum(.bucket).squeeze(.bucket).maximum(zml.Tensor.scalar(1e-20, .f32));
        const per_row = squared_error.scale(self.orth_weight).add(parallel_error.mul(parallel_error).div(norm2).scale(1.0 - self.orth_weight));
        const active_count = active.convert(.f32).sum(.row).asScalar().maximum(zml.Tensor.scalar(1.0, .f32));
        const loss = per_row.sum(.row).asScalar().div(active_count);
        return .{ squared_error, parallel_error, loss };
    }

    pub fn assignAnisotropicBucket(self: @This(), data_: zml.Tensor, centers_: zml.Tensor, codes_: zml.Tensor, squared_error_: zml.Tensor, parallel_error_: zml.Tensor, bucket: zml.Tensor, is_junk_: zml.Tensor, total_changed_: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        const data = fullData(data_);
        const centers = codebook(centers_);
        const codes = codes_.withTags(.{ .row, .bucket });
        const squared_error = squared_error_.withTags(.{.row});
        const parallel_error = parallel_error_.withTags(.{.row});
        const is_junk = is_junk_.withTags(.{.row});
        const active = is_junk.not();

        const x = bucketSlice(data, bucket);
        const old_codes = bucketSlice(codes, bucket);
        const old_centers = centers.gather(.{ .center = old_codes }, .{});
        const old_error = x.sub(old_centers);
        const old_squared = old_error.mul(old_error).sum(.coord).squeeze(.coord);
        const old_parallel = x.mul(old_error).sum(.coord).squeeze(.coord);
        const candidate_squared, const candidate_parallel = blockDistances(x, centers);

        const candidate_shape = candidate_squared.shape();
        const base_squared = squared_error.sub(old_squared).insertAxes(.last, .{.center}).broad(candidate_shape);
        const base_parallel = parallel_error.sub(old_parallel).insertAxes(.last, .{.center}).broad(candidate_shape);
        const row_norm2 = data.mul(data).sum(.coord).squeeze(.coord).sum(.bucket).squeeze(.bucket)
            .maximum(zml.Tensor.scalar(1e-20, .f32));
        const norm2 = row_norm2.insertAxes(.last, .{.center}).broad(candidate_shape);
        const new_parallel = base_parallel.add(candidate_parallel);
        var candidate_loss = base_squared.add(candidate_squared).scale(self.orth_weight)
            .add(new_parallel.mul(new_parallel).div(norm2).scale(1.0 - self.orth_weight));
        candidate_loss = active.insertAxes(.last, .{.center}).broad(candidate_shape).select(
            candidate_loss,
            zml.Tensor.scalar(std.math.inf(f32), .f32),
        );
        const selected = candidate_loss.negate().argMax(.center);
        const proposed_codes = selected.indices.squeeze(.center).convert(.u32);
        const new_codes = active.select(proposed_codes, old_codes);
        const selected_squared = candidate_squared.gather(.{ .center = new_codes }, .{});
        const selected_parallel = candidate_parallel.gather(.{ .center = new_codes }, .{});
        const next_squared = active.select(squared_error.sub(old_squared).add(selected_squared), squared_error);
        const next_parallel = active.select(parallel_error.sub(old_parallel).add(selected_parallel), parallel_error);
        const next_codes = codes.dynamicUpdateSlice(.{ .bucket = bucket }, new_codes.insertAxes(.last, .{.bucket}));
        const changed = total_changed_.add(active.select(old_codes.cmp(.NE, new_codes), zml.Tensor.scalar(false, .bool)).convert(.u32).sum(.row).asScalar());
        const per_row = next_squared.scale(self.orth_weight)
            .add(next_parallel.mul(next_parallel).div(row_norm2).scale(1.0 - self.orth_weight));
        const active_count = active.convert(.f32).sum(.row).asScalar().maximum(zml.Tensor.scalar(1.0, .f32));
        const loss = per_row.sum(.row).asScalar().div(active_count);
        return .{ next_codes, next_squared, next_parallel, changed, loss };
    }

    fn normalMatVec(a: zml.Tensor, x: zml.Tensor) zml.Tensor {
        return a.dot(x.rename(.{ .param = .param_col }), .param_col).rename(.{ .param_row = .param });
    }

    pub fn updateAnisotropicCenters(self: @This(), data_: zml.Tensor, codes_: zml.Tensor, is_junk_: zml.Tensor, centers_: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const data = fullData(data_);
        const codes = codes_.withTags(.{ .row, .bucket });
        const is_junk = is_junk_.withTags(.{.row});
        const centers = codebook(centers_);
        const n = data.dim(.row);
        const parameter_count = self.k * self.block_dim;
        const active = is_junk.not();
        const active_f = active.convert(.f32);

        // g_i = R_i^T x_i. With a shared codebook this is a K x B vector per row,
        // and the rank-one g_i g_i^T term couples every center and every bucket.
        const scattered = zml.Tensor.scalar(0.0, .f32)
            .broad(zml.Shape.init(.{ .center = self.k, .row = n, .coord = self.block_dim }, .f32))
            .scatterSlices(
            .{ .center = codes },
            data,
            .{ .update_fn = zml.Tensor.ScatterOpts.increment },
        );
        var g = scattered.transpose(.{ .row, .center, .coord });
        g = g.mul(active_f.insertAxes(.last, .{ .center, .coord }).broad(g.shape()));
        const norm2 = data.mul(data).sum(.coord).squeeze(.coord).sum(.bucket).squeeze(.bucket).maximum(zml.Tensor.scalar(1e-20, .f32));
        const scale = norm2.rsqrt().scale(@sqrt(1.0 - self.orth_weight)).mul(active_f);
        const g_flat = g.reshape(.{ .row = n, .param = parameter_count });
        const weighted_g = g_flat.mul(scale.insertAxes(.last, .{.param}).broad(g_flat.shape()));
        const lhs = weighted_g.transpose(.{ .param, .row }).rename(.{ .param = .param_row });
        const rhs = weighted_g.rename(.{ .param = .param_col });
        const rank_one = lhs.dot(rhs, .row);

        const occurrence_count = n * self.nb_buckets;
        const flat_codes = codes.reshape(.{ .occurrence = occurrence_count });
        const active_occurrences = active_f.insertAxes(.last, .{.bucket}).broad(codes.shape()).reshape(.{ .occurrence = occurrence_count });
        const counts = zml.Tensor.scalar(0.0, .f32).broad(zml.Shape.init(.{ .center = self.k }, .f32)).scatterSlices(
            .{ .center = flat_codes },
            active_occurrences,
            .{ .update_fn = zml.Tensor.ScatterOpts.increment },
        );
        const parameter_counts = counts.insertAxes(.last, .{.coord})
            .broad(zml.Shape.init(.{ .center = self.k, .coord = self.block_dim }, .f32))
            .reshape(.{ .param = parameter_count });
        const diagonal = parameter_counts.scale(self.orth_weight).addConstant(anisotropic_ridge).toDiagonal(.param, .{ .param_row, .param_col });
        const normal = rank_one.add(diagonal);
        const target = g_flat.sum(.row).squeeze(.row);

        // The full system is SPD. Conjugate gradients avoids decomposing it into the
        // incorrect independent B x B systems used by the old CPU implementation.
        var solution = centers.reshape(.{ .param = parameter_count });
        var residual = target.sub(normalMatVec(normal, solution));
        var direction = residual;
        var residual2 = residual.dot(residual, .param).asScalar();
        const target2 = target.dot(target, .param).asScalar().maximum(zml.Tensor.scalar(1e-20, .f32));
        for (0..anisotropic_cg_max_iters) |_| {
            const normal_direction = normalMatVec(normal, direction);
            const denominator = direction.dot(normal_direction, .param).asScalar().maximum(zml.Tensor.scalar(1e-20, .f32));
            const alpha = residual2.div(denominator);
            solution = solution.add(direction.mul(alpha));
            residual = residual.sub(normal_direction.mul(alpha));
            const next_residual2 = residual.dot(residual, .param).asScalar();
            const beta = next_residual2.div(residual2.maximum(zml.Tensor.scalar(1e-20, .f32)));
            direction = residual.add(direction.mul(beta));
            residual2 = next_residual2;
        }
        const relative_residual = residual2.div(target2).sqrt();
        return .{ solution.reshape(.{ .center = self.k, .coord = self.block_dim }), relative_residual };
    }
};

pub const KMeansGPU = struct {
    zml_handler: *main.Zml_handler,
    allocator: std.mem.Allocator,
    n_max: usize,
    d: usize,
    k: usize,
    centers: []f32,
    assigned: []u32,
    has_centers: bool = false,

    pub fn init(zml_handler: *main.Zml_handler, n_max: usize, d: usize, k: usize) !KMeansGPU {
        const centers = try zml_handler.allocator.alloc(f32, k * d);
        errdefer zml_handler.allocator.free(centers);
        const assigned = try zml_handler.allocator.alloc(u32, n_max);
        errdefer zml_handler.allocator.free(assigned);
        return .{
            .zml_handler = zml_handler,
            .allocator = zml_handler.allocator,
            .n_max = n_max,
            .d = d,
            .k = k,
            .centers = centers,
            .assigned = assigned,
        };
    }

    pub fn deinit(self: *KMeansGPU) void {
        self.allocator.free(self.centers);
        self.allocator.free(self.assigned);
    }

    pub fn solve(self: *KMeansGPU, data: []const f32) !void {
        try self.solveStandard(data, .euclidean);
    }

    pub fn solveSpherical(self: *KMeansGPU, data: []const f32) !void {
        try self.solveStandard(data, .spherical);
    }

    pub fn solveSphericalWeighted(self: *KMeansGPU, data: []const f32) !void {
        try self.solveStandard(data, .spherical_weighted);
    }

    fn solveStandard(self: *KMeansGPU, data: []const f32, comptime mode: GpuKMeansMode) !void {
        const n = data.len / self.d;
        std.debug.assert(n <= self.n_max and n >= self.k and data.len == n * self.d);
        const model: GpuKMeansModel = .{ .mode = mode, .k = @intCast(self.k), .block_dim = @intCast(self.d) };
        const data_shape = zml.Shape.init(.{ .point = n, .coord = self.d }, .f32);
        const center_shape = zml.Shape.init(.{ .center = self.k, .coord = self.d }, .f32);
        const assigned_shape = zml.Shape.init(.{ .point = n }, .u32);
        const weights_shape = zml.Shape.init(.{ .point = n }, .f32);
        const exe = try self.zml_handler.platform.compile(
            self.allocator,
            self.zml_handler.io,
            model,
            .iteration,
            .{ zml.Tensor.fromShape(data_shape), zml.Tensor.fromShape(center_shape), zml.Tensor.fromShape(assigned_shape), zml.Tensor.fromShape(weights_shape) },
            .{},
        );
        defer exe.deinit();
        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);

        // Deterministic, well-spread initialization. All expensive assignment and
        // reduction work below stays on the accelerator.
        const weights = try self.allocator.alloc(f32, n);
        defer self.allocator.free(weights);
        for (0..n) |point| {
            const x = data[point * self.d .. (point + 1) * self.d];
            weights[point] = if (mode == .spherical_weighted) @sqrt(KMeansCPU.dotSlices(x, x, self.d)) else 1.0;
        }
        for (0..self.k) |center| {
            const point = center * n / self.k;
            const src = data[point * self.d .. (point + 1) * self.d];
            const dst = self.centers[center * self.d .. (center + 1) * self.d];
            @memcpy(dst, src);
            if (mode != .euclidean) {
                const inv_norm = 1.0 / @sqrt(@max(KMeansCPU.dotSlices(dst, dst, self.d), 1e-20));
                KMeansCPU.scaleInPlace(dst, inv_norm, self.d);
            }
        }
        @memset(self.assigned[0..n], std.math.maxInt(u32));

        const data_slice = zml.Slice.init(data_shape, std.mem.sliceAsBytes(data));
        const center_slice = zml.Slice.init(center_shape, std.mem.sliceAsBytes(self.centers));
        const assigned_slice = zml.Slice.init(assigned_shape, std.mem.sliceAsBytes(self.assigned[0..n]));
        const weights_slice = zml.Slice.init(weights_shape, std.mem.sliceAsBytes(weights));
        var data_buffer = try zml.Buffer.fromSlice(self.zml_handler.io, self.zml_handler.platform, data_slice, .replicated);
        defer data_buffer.deinit();
        var centers_buffer = try zml.Buffer.fromSlice(self.zml_handler.io, self.zml_handler.platform, center_slice, .replicated);
        defer centers_buffer.deinit();
        var assigned_buffer = try zml.Buffer.fromSlice(self.zml_handler.io, self.zml_handler.platform, assigned_slice, .replicated);
        defer assigned_buffer.deinit();
        var weights_buffer = try zml.Buffer.fromSlice(self.zml_handler.io, self.zml_handler.platform, weights_slice, .replicated);
        defer weights_buffer.deinit();

        const metric_name = if (mode == .euclidean) "sos" else "sim";
        std.log.info("GPU KMeans ({s})", .{@tagName(mode)});
        std.log.info("{s: >6} {s: >12} {s: >8} {s: >8}", .{ "iter", metric_name, "chg", "empty" });
        for (0..max_iters) |it| {
            args.set(.{ data_buffer, centers_buffer, assigned_buffer, weights_buffer });
            exe.call(args, &results);
            var next_centers: zml.Buffer = undefined;
            var next_assigned: zml.Buffer = undefined;
            var metric_buffer: zml.Buffer = undefined;
            var changed_buffer: zml.Buffer = undefined;
            var empty_buffer: zml.Buffer = undefined;
            results.fill(.{ &next_centers, &next_assigned, &metric_buffer, &changed_buffer, &empty_buffer });
            const metric = try metric_buffer.getValue(f32, self.zml_handler.io);
            const nb_changed = try changed_buffer.getValue(u32, self.zml_handler.io);
            const nb_empty = try empty_buffer.getValue(u32, self.zml_handler.io);
            metric_buffer.deinit();
            changed_buffer.deinit();
            empty_buffer.deinit();
            centers_buffer.deinit();
            assigned_buffer.deinit();
            centers_buffer = next_centers;
            assigned_buffer = next_assigned;
            std.log.info("{d: >6} {d: >12.6} {d: >8} {d: >8}", .{ it, metric, nb_changed, nb_empty });
            if (nb_changed == 0) break;
        }
        try centers_buffer.toSlice(self.zml_handler.io, center_slice);
        try assigned_buffer.toSlice(self.zml_handler.io, assigned_slice);
        self.has_centers = true;
    }

    pub fn solveAnisotropic(self: *KMeansGPU, lm_head: anytype, row_codes: []u8, orth_weight: f32) !void {
        std.debug.assert(orth_weight > 0.0 and orth_weight <= 1.0);
        std.debug.assert(lm_head.d % self.d == 0 and lm_head.data.len == lm_head.n * lm_head.d);
        const n: usize = lm_head.n;
        const nb_buckets = lm_head.d / self.d;
        std.debug.assert(row_codes.len == n * nb_buckets and self.has_centers);

        const model: GpuKMeansModel = .{
            .mode = .euclidean,
            .k = @intCast(self.k),
            .block_dim = @intCast(self.d),
            .nb_buckets = @intCast(nb_buckets),
            .orth_weight = orth_weight,
        };
        const data_shape = zml.Shape.init(.{ .row = n, .bucket = nb_buckets, .coord = self.d }, .f32);
        const center_shape = zml.Shape.init(.{ .center = self.k, .coord = self.d }, .f32);
        const codes_shape = zml.Shape.init(.{ .row = n, .bucket = nb_buckets }, .u32);
        const row_shape = zml.Shape.init(.{ .row = n }, .f32);
        const junk_shape = zml.Shape.init(.{ .row = n }, .bool);
        const bucket_shape = zml.Shape.scalar(.u32);

        const init_exe = try self.zml_handler.platform.compile(self.allocator, self.zml_handler.io, model, .initializeCodes, .{ zml.Tensor.fromShape(data_shape), zml.Tensor.fromShape(center_shape), zml.Tensor.fromShape(codes_shape), zml.Tensor.fromShape(bucket_shape) }, .{});
        defer init_exe.deinit();
        var init_args = try init_exe.args(self.allocator);
        defer init_args.deinit(self.allocator);
        var init_results = try init_exe.results(self.allocator);
        defer init_results.deinit(self.allocator);

        const state_exe = try self.zml_handler.platform.compile(self.allocator, self.zml_handler.io, model, .rowState, .{ zml.Tensor.fromShape(data_shape), zml.Tensor.fromShape(center_shape), zml.Tensor.fromShape(codes_shape), zml.Tensor.fromShape(junk_shape) }, .{});
        defer state_exe.deinit();
        var state_args = try state_exe.args(self.allocator);
        defer state_args.deinit(self.allocator);
        var state_results = try state_exe.results(self.allocator);
        defer state_results.deinit(self.allocator);

        const assign_exe = try self.zml_handler.platform.compile(self.allocator, self.zml_handler.io, model, .assignAnisotropicBucket, .{ zml.Tensor.fromShape(data_shape), zml.Tensor.fromShape(center_shape), zml.Tensor.fromShape(codes_shape), zml.Tensor.fromShape(row_shape), zml.Tensor.fromShape(row_shape), zml.Tensor.fromShape(bucket_shape), zml.Tensor.fromShape(junk_shape), zml.Tensor.fromShape(zml.Shape.scalar(.u32)) }, .{});
        defer assign_exe.deinit();
        var assign_args = try assign_exe.args(self.allocator);
        defer assign_args.deinit(self.allocator);
        var assign_results = try assign_exe.results(self.allocator);
        defer assign_results.deinit(self.allocator);

        const update_exe = try self.zml_handler.platform.compile(self.allocator, self.zml_handler.io, model, .updateAnisotropicCenters, .{ zml.Tensor.fromShape(data_shape), zml.Tensor.fromShape(codes_shape), zml.Tensor.fromShape(junk_shape), zml.Tensor.fromShape(center_shape) }, .{});
        defer update_exe.deinit();
        var update_args = try update_exe.args(self.allocator);
        defer update_args.deinit(self.allocator);
        var update_results = try update_exe.results(self.allocator);
        defer update_results.deinit(self.allocator);

        const codes_u32 = try self.allocator.alloc(u32, row_codes.len);
        defer self.allocator.free(codes_u32);
        @memset(codes_u32, 0);
        const data_slice = zml.Slice.init(data_shape, std.mem.sliceAsBytes(lm_head.data));
        const center_slice = zml.Slice.init(center_shape, std.mem.sliceAsBytes(self.centers));
        const codes_slice = zml.Slice.init(codes_shape, std.mem.sliceAsBytes(codes_u32));
        const junk_slice = zml.Slice.init(junk_shape, std.mem.sliceAsBytes(lm_head.is_junk));
        var data_buffer = try zml.Buffer.fromSlice(self.zml_handler.io, self.zml_handler.platform, data_slice, .replicated);
        defer data_buffer.deinit();
        var centers_buffer = try zml.Buffer.fromSlice(self.zml_handler.io, self.zml_handler.platform, center_slice, .replicated);
        defer centers_buffer.deinit();
        var codes_buffer = try zml.Buffer.fromSlice(self.zml_handler.io, self.zml_handler.platform, codes_slice, .replicated);
        defer codes_buffer.deinit();
        var junk_buffer = try zml.Buffer.fromSlice(self.zml_handler.io, self.zml_handler.platform, junk_slice, .replicated);
        defer junk_buffer.deinit();

        std.log.info("GPU full anisotropic PQ: rows={d} buckets={d} centers={d} block_dim={d} system={d}x{d}", .{ n, nb_buckets, self.k, self.d, self.k * self.d, self.k * self.d });
        std.log.info("Initializing all PQ codes on the GPU", .{});
        for (0..nb_buckets) |bucket| {
            var bucket_buffer = try zml.Buffer.scalar(self.zml_handler.io, self.zml_handler.platform, @as(u32, @intCast(bucket)), .u32);
            init_args.set(.{ data_buffer, centers_buffer, codes_buffer, bucket_buffer });
            init_exe.call(init_args, &init_results);
            var next_codes: zml.Buffer = undefined;
            init_results.fill(.{&next_codes});
            bucket_buffer.deinit();
            codes_buffer.deinit();
            codes_buffer = next_codes;
        }

        std.log.info("{s: >6} {s: >12} {s: >12} {s: >8}", .{ "iter", "aniso", "cg-resid", "chg" });
        for (0..max_iters) |it| {
            state_args.set(.{ data_buffer, centers_buffer, codes_buffer, junk_buffer });
            state_exe.call(state_args, &state_results);
            var squared_buffer: zml.Buffer = undefined;
            var parallel_buffer: zml.Buffer = undefined;
            var state_loss_buffer: zml.Buffer = undefined;
            state_results.fill(.{ &squared_buffer, &parallel_buffer, &state_loss_buffer });
            state_loss_buffer.deinit();

            var changed_total_buffer = try zml.Buffer.scalar(self.zml_handler.io, self.zml_handler.platform, @as(u32, 0), .u32);
            var last_loss_buffer: ?zml.Buffer = null;
            for (0..nb_buckets) |bucket| {
                var bucket_buffer = try zml.Buffer.scalar(self.zml_handler.io, self.zml_handler.platform, @as(u32, @intCast(bucket)), .u32);
                assign_args.set(.{ data_buffer, centers_buffer, codes_buffer, squared_buffer, parallel_buffer, bucket_buffer, junk_buffer, changed_total_buffer });
                assign_exe.call(assign_args, &assign_results);
                var next_codes: zml.Buffer = undefined;
                var next_squared: zml.Buffer = undefined;
                var next_parallel: zml.Buffer = undefined;
                var changed_buffer: zml.Buffer = undefined;
                var loss_buffer: zml.Buffer = undefined;
                assign_results.fill(.{ &next_codes, &next_squared, &next_parallel, &changed_buffer, &loss_buffer });
                bucket_buffer.deinit();
                codes_buffer.deinit();
                squared_buffer.deinit();
                parallel_buffer.deinit();
                changed_total_buffer.deinit();
                if (last_loss_buffer) |*buffer| buffer.deinit();
                codes_buffer = next_codes;
                squared_buffer = next_squared;
                parallel_buffer = next_parallel;
                changed_total_buffer = changed_buffer;
                last_loss_buffer = loss_buffer;
            }
            const total_changed = try changed_total_buffer.getValue(u32, self.zml_handler.io);
            const loss = try last_loss_buffer.?.getValue(f32, self.zml_handler.io);
            changed_total_buffer.deinit();
            if (last_loss_buffer) |*buffer| buffer.deinit();
            squared_buffer.deinit();
            parallel_buffer.deinit();

            update_args.set(.{ data_buffer, codes_buffer, junk_buffer, centers_buffer });
            update_exe.call(update_args, &update_results);
            var next_centers: zml.Buffer = undefined;
            var residual_buffer: zml.Buffer = undefined;
            update_results.fill(.{ &next_centers, &residual_buffer });
            const relative_residual = try residual_buffer.getValue(f32, self.zml_handler.io);
            residual_buffer.deinit();
            centers_buffer.deinit();
            centers_buffer = next_centers;
            std.log.info("{d: >6} {d: >12.6} {d: >12.6} {d: >8}", .{ it, loss, relative_residual, total_changed });
            if (it > 0 and total_changed == 0) break;
        }

        try centers_buffer.toSlice(self.zml_handler.io, center_slice);
        try codes_buffer.toSlice(self.zml_handler.io, codes_slice);
        for (codes_u32, row_codes) |code, *dst| dst.* = @intCast(code);
        self.has_centers = true;
    }
};
