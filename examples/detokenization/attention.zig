const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");
const save_load = @import("saveload.zig");
const quantization = @import("quantization.zig");

const Zml_handler = main.Zml_handler;

const head_dim = 128;
const head_dim_log2 = 7;
const num_queries = 4;
const num_heads = 8;

const use_1bit = false;
const use_2bit = false;
const use_3bit = true;

pub fn runTests(zml_handler: *Zml_handler) !void {
    const layers = [_]usize{ 5, 15, 25, 35 };
    const heads = [_]usize{ 0, 2, 4 };
    const tokens = [_]usize{29000};
    for (layers) |layer| {
        for (heads) |head| {
            var attn: Attention = try .init(zml_handler, layer, head);
            defer Attention.deinit(&attn);
            zml_handler.ticLog();
            attn.quantize();
            zml_handler.tocLog("quantization");
            std.log.info("Testing head {d} at layer {d}", .{ head, layer });
            for (tokens) |token| {
                std.log.info("     - token {d}", .{token});
                zml_handler.ticLog();
                //try attn.analyze(token);
                //try attn.normalize(token);
                //try attn.truncatedAttention(token);
                try attn.sampledAttention(token);
                zml_handler.tocLog("attention");
            }
        }
    }
}

pub fn runBenchs(zml_handler: *Zml_handler) !void {
    const layers = [_]usize{ 0, 10, 20, 30 };
    const heads = [_]usize{ 0, 2, 4, 6 };
    const tokens = [_]usize{ 25000, 26000, 27000, 28000, 29000, 30000, 31000 };
    for (layers) |layer| {
        for (heads) |head| {
            var attn: Attention = try .init(zml_handler, layer, head);
            defer Attention.deinit(&attn);
            zml_handler.ticLog();
            attn.quantize();
            zml_handler.tocLog("quantization");
            std.log.info("Testing head {d} at layer {d}", .{ head, layer });
            for (tokens) |token| {
                std.log.info("     - token {d}", .{token});
                zml_handler.ticLog();
                try attn.computeAttention(token);
                zml_handler.tocLog("attention");
            }
        }
    }
}

pub const Score = struct {
    token: usize,
    approx: f32,
    lo: f32,
    hi: f32,
    real: f32,
    proba: f32,

    fn decreasingApprox(_: void, lhs: Score, rhs: Score) bool {
        return lhs.approx > rhs.approx or (lhs.approx == rhs.approx and lhs.token < rhs.token);
    }

    fn decreasingReal(_: void, lhs: Score, rhs: Score) bool {
        return lhs.real > rhs.real or (lhs.real == rhs.real and lhs.token < rhs.token);
    }
};

const WeightedToken = struct {
    token: usize,
    weight: f32,

    fn decreasingWeight(_: void, lhs: WeightedToken, rhs: WeightedToken) bool {
        return lhs.weight > rhs.weight or (lhs.weight == rhs.weight and lhs.token < rhs.token);
    }
};

pub const Attention = struct {
    allocator: std.mem.Allocator,

    q: []const f32, // [q0s, q1s, q2s, q3s]
    k: []const f32,
    v: []const f32,

    q_norms: []f32,
    k_norms: []f32,
    v_norms: []f32,

    buffer: []f32,

    q_qjl: []u128, // [q0s, q1s, q2s, q3s]
    k_qjl: []u128,
    v_qjl: []u128,

    q_lsb: []u128,
    q_msb: []u128,
    k_lsb: []u128,
    k_msb: []u128,
    v_lsb: []u128,
    v_msb: []u128,

    q_scales: []f32,
    k_scales: []f32,
    v_scales: []f32,

    q_3bit_lsb: []u128,
    q_3bit_hsb: []u128,
    q_3bit_msb: []u128,
    k_3bit_lsb: []u128,
    k_3bit_hsb: []u128,
    k_3bit_msb: []u128,
    v_3bit_lsb: []u128,
    v_3bit_hsb: []u128,
    v_3bit_msb: []u128,

    q_3bit_scales: []f32,
    k_3bit_scales: []f32,
    v_3bit_scales: []f32,

    sink: usize,

    pub fn init(zml_handler: *Zml_handler, layer_id: usize, head_id: usize) !Attention {
        if (head_id >= num_heads) return error.InvalidHeadId;

        const allocator = zml_handler.allocator;
        const filename = try std.fmt.allocPrint(allocator, "layer{d}_head_{d}.safetensors", .{ layer_id, head_id });
        defer allocator.free(filename);

        const q0 = try loadTensorF32(zml_handler, filename, "q_0");
        defer allocator.free(q0);
        if (q0.len % head_dim != 0) return error.InvalidTensorShape;

        const q = try allocator.alloc(f32, num_queries * q0.len);
        errdefer allocator.free(q);
        @memcpy(q[0..q0.len], q0);

        for (1..num_queries) |query_id| {
            const tensor_name = try std.fmt.allocPrint(allocator, "q_{d}", .{query_id});
            defer allocator.free(tensor_name);

            const query = try loadTensorF32(zml_handler, filename, tensor_name);
            defer allocator.free(query);
            if (query.len != q0.len) return error.InvalidTensorShape;
            @memcpy(q[query_id * q0.len .. (query_id + 1) * q0.len], query);
        }

        const k = try loadTensorF32(zml_handler, filename, "k");
        errdefer allocator.free(k);
        if (k.len % head_dim != 0) return error.InvalidTensorShape;

        const v = try loadTensorF32(zml_handler, filename, "v");
        errdefer allocator.free(v);
        if (v.len != k.len) return error.InvalidTensorShape;

        const buffer = try allocator.alloc(f32, head_dim);
        errdefer allocator.free(buffer);

        const q_qjl = try allocator.alloc(u128, q.len / head_dim);
        errdefer allocator.free(q_qjl);
        const k_qjl = try allocator.alloc(u128, k.len / head_dim);
        errdefer allocator.free(k_qjl);
        const v_qjl = try allocator.alloc(u128, v.len / head_dim);
        errdefer allocator.free(v_qjl);
        const q_norms = try allocator.alloc(f32, q_qjl.len);
        errdefer allocator.free(q_norms);
        const k_norms = try allocator.alloc(f32, k_qjl.len);
        errdefer allocator.free(k_norms);
        const v_norms = try allocator.alloc(f32, v_qjl.len);
        errdefer allocator.free(v_norms);
        const q_lsb = try allocator.alloc(u128, q_qjl.len);
        errdefer allocator.free(q_lsb);
        const q_msb = try allocator.alloc(u128, q_qjl.len);
        errdefer allocator.free(q_msb);
        const k_lsb = try allocator.alloc(u128, k_qjl.len);
        errdefer allocator.free(k_lsb);
        const k_msb = try allocator.alloc(u128, k_qjl.len);
        errdefer allocator.free(k_msb);
        const v_lsb = try allocator.alloc(u128, v_qjl.len);
        errdefer allocator.free(v_lsb);
        const v_msb = try allocator.alloc(u128, v_qjl.len);
        errdefer allocator.free(v_msb);
        const q_scales = try allocator.alloc(f32, q_qjl.len);
        errdefer allocator.free(q_scales);
        const k_scales = try allocator.alloc(f32, k_qjl.len);
        errdefer allocator.free(k_scales);
        const v_scales = try allocator.alloc(f32, v_qjl.len);
        errdefer allocator.free(v_scales);
        const q_3bit_lsb = try allocator.alloc(u128, q_qjl.len);
        errdefer allocator.free(q_3bit_lsb);
        const q_3bit_hsb = try allocator.alloc(u128, q_qjl.len);
        errdefer allocator.free(q_3bit_hsb);
        const q_3bit_msb = try allocator.alloc(u128, q_qjl.len);
        errdefer allocator.free(q_3bit_msb);
        const k_3bit_lsb = try allocator.alloc(u128, k_qjl.len);
        errdefer allocator.free(k_3bit_lsb);
        const k_3bit_hsb = try allocator.alloc(u128, k_qjl.len);
        errdefer allocator.free(k_3bit_hsb);
        const k_3bit_msb = try allocator.alloc(u128, k_qjl.len);
        errdefer allocator.free(k_3bit_msb);
        const v_3bit_lsb = try allocator.alloc(u128, v_qjl.len);
        errdefer allocator.free(v_3bit_lsb);
        const v_3bit_hsb = try allocator.alloc(u128, v_qjl.len);
        errdefer allocator.free(v_3bit_hsb);
        const v_3bit_msb = try allocator.alloc(u128, v_qjl.len);
        errdefer allocator.free(v_3bit_msb);
        const q_3bit_scales = try allocator.alloc(f32, q_qjl.len);
        errdefer allocator.free(q_3bit_scales);
        const k_3bit_scales = try allocator.alloc(f32, k_qjl.len);
        errdefer allocator.free(k_3bit_scales);
        const v_3bit_scales = try allocator.alloc(f32, v_qjl.len);
        errdefer allocator.free(v_3bit_scales);

        return .{
            .allocator = allocator,
            .q = q,
            .k = k,
            .v = v,
            .buffer = buffer,
            .q_qjl = q_qjl,
            .k_qjl = k_qjl,
            .v_qjl = v_qjl,
            .q_norms = q_norms,
            .k_norms = k_norms,
            .v_norms = v_norms,
            .q_lsb = q_lsb,
            .q_msb = q_msb,
            .k_lsb = k_lsb,
            .k_msb = k_msb,
            .v_lsb = v_lsb,
            .v_msb = v_msb,
            .q_scales = q_scales,
            .k_scales = k_scales,
            .v_scales = v_scales,
            .q_3bit_lsb = q_3bit_lsb,
            .q_3bit_hsb = q_3bit_hsb,
            .q_3bit_msb = q_3bit_msb,
            .k_3bit_lsb = k_3bit_lsb,
            .k_3bit_hsb = k_3bit_hsb,
            .k_3bit_msb = k_3bit_msb,
            .v_3bit_lsb = v_3bit_lsb,
            .v_3bit_hsb = v_3bit_hsb,
            .v_3bit_msb = v_3bit_msb,
            .q_3bit_scales = q_3bit_scales,
            .k_3bit_scales = k_3bit_scales,
            .v_3bit_scales = v_3bit_scales,
            .sink = 0,
        };
    }

    pub fn deinit(self: *Attention) void {
        std.log.info("sink {d}", .{self.sink});
        self.allocator.free(self.q);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.allocator.free(self.buffer);
        self.allocator.free(self.q_qjl);
        self.allocator.free(self.k_qjl);
        self.allocator.free(self.v_qjl);
        self.allocator.free(self.q_norms);
        self.allocator.free(self.k_norms);
        self.allocator.free(self.v_norms);
        self.allocator.free(self.q_lsb);
        self.allocator.free(self.q_msb);
        self.allocator.free(self.k_lsb);
        self.allocator.free(self.k_msb);
        self.allocator.free(self.v_lsb);
        self.allocator.free(self.v_msb);
        self.allocator.free(self.q_scales);
        self.allocator.free(self.k_scales);
        self.allocator.free(self.v_scales);
        self.allocator.free(self.q_3bit_lsb);
        self.allocator.free(self.q_3bit_hsb);
        self.allocator.free(self.q_3bit_msb);
        self.allocator.free(self.k_3bit_lsb);
        self.allocator.free(self.k_3bit_hsb);
        self.allocator.free(self.k_3bit_msb);
        self.allocator.free(self.v_3bit_lsb);
        self.allocator.free(self.v_3bit_hsb);
        self.allocator.free(self.v_3bit_msb);
        self.allocator.free(self.q_3bit_scales);
        self.allocator.free(self.k_3bit_scales);
        self.allocator.free(self.v_3bit_scales);
    }

    pub fn quantize(self: *Attention) void {
        quantizeHistory1bit(self.q, self.q_qjl, self.q_norms, self.buffer);
        quantizeHistory1bit(self.k, self.k_qjl, self.k_norms, self.buffer);
        quantizeHistory1bit(self.v, self.v_qjl, self.v_norms, self.buffer);
        quantizeHistory2bit(self.q, self.q_lsb, self.q_msb, self.q_scales, self.buffer);
        quantizeHistory2bit(self.k, self.k_lsb, self.k_msb, self.k_scales, self.buffer);
        quantizeHistory2bit(self.v, self.v_lsb, self.v_msb, self.v_scales, self.buffer);
        quantizeHistory3bit(self.q, self.q_3bit_lsb, self.q_3bit_hsb, self.q_3bit_msb, self.q_3bit_scales, self.buffer);
        quantizeHistory3bit(self.k, self.k_3bit_lsb, self.k_3bit_hsb, self.k_3bit_msb, self.k_3bit_scales, self.buffer);
        quantizeHistory3bit(self.v, self.v_3bit_lsb, self.v_3bit_hsb, self.v_3bit_msb, self.v_3bit_scales, self.buffer);
    }

    pub fn analyze(self: *Attention, token_id: usize) !void {
        const query_id = try self.queryIndex(token_id);
        const scores = try self.allocator.alloc(Score, token_id + 1);
        defer self.allocator.free(scores);

        const q_128 = self.q_qjl[query_id];
        const q_norm = self.q_norms[query_id];
        const q_lsb = self.q_lsb[query_id];
        const q_msb = self.q_msb[query_id];
        const q_scale = self.q_scales[query_id];
        const q_3bit_lsb = self.q_3bit_lsb[query_id];
        const q_3bit_hsb = self.q_3bit_hsb[query_id];
        const q_3bit_msb = self.q_3bit_msb[query_id];
        const q_3bit_scale = self.q_3bit_scales[query_id];

        const std_dev_1 = @sqrt(std.math.pi * 0.5) / @sqrt(@as(f32, head_dim));
        const std_dev_2 = 0.47 / @sqrt(@as(f32, head_dim));
        const std_dev_3 = 0.23 / @sqrt(@as(f32, head_dim));
        const z_score = 2.5;
        const sfm_scale = 1.0 / @sqrt(@as(f32, head_dim));

        for (0..token_id + 1) |tok| {
            var real: f32 = 0.0;
            for (0..head_dim) |coord| {
                real += self.k[coord + head_dim * tok] * self.q[coord + head_dim * query_id];
            }
            var approx: f32 = 0.0;
            var err: f32 = 0.0;
            if (use_1bit) {
                // 1 bit
                approx = q_norm * self.k_norms[tok] * qjl_dot_lut[@popCount(q_128 ^ self.k_qjl[tok])];
                err = q_norm * self.k_norms[tok] * std_dev_1 * z_score;
            } else if (use_2bit) {
                // 2 bits
                const pop_mm = @popCount(q_msb ^ self.k_msb[tok]);
                const pop_ml = @popCount(q_msb ^ self.k_lsb[tok]);
                const pop_lm = @popCount(q_lsb ^ self.k_msb[tok]);
                const pop_ll = @popCount(q_lsb ^ self.k_lsb[tok]);
                const pos: i32 = 9 * head_dim;
                const neg = (@as(u32, pop_mm) << 3) + ((@as(u32, pop_ml) + @as(u32, pop_lm)) << 2) + (@as(u32, pop_ll) << 1);
                approx = q_scale * self.k_scales[tok] * @as(f32, @floatFromInt(pos - @as(i32, @intCast(neg))));
                err = q_norm * self.k_norms[tok] * std_dev_2 * z_score;
            } else if (use_3bit) {
                const pop_mm = @popCount(q_3bit_msb ^ self.k_3bit_msb[tok]);
                const pop_mh = @popCount(q_3bit_msb ^ self.k_3bit_hsb[tok]);
                const pop_hm = @popCount(q_3bit_hsb ^ self.k_3bit_msb[tok]);
                const pop_ml = @popCount(q_3bit_msb ^ self.k_3bit_lsb[tok]);
                const pop_lm = @popCount(q_3bit_lsb ^ self.k_3bit_msb[tok]);
                const pop_hh = @popCount(q_3bit_hsb ^ self.k_3bit_hsb[tok]);
                const pop_hl = @popCount(q_3bit_hsb ^ self.k_3bit_lsb[tok]);
                const pop_lh = @popCount(q_3bit_lsb ^ self.k_3bit_hsb[tok]);
                const pop_ll = @popCount(q_3bit_lsb ^ self.k_3bit_lsb[tok]);
                const pos: i32 = 49 * head_dim;
                const neg = (@as(u32, pop_mm) << 5) +
                    ((@as(u32, pop_mh) + @as(u32, pop_hm)) << 4) +
                    ((@as(u32, pop_ml) + @as(u32, pop_lm) + @as(u32, pop_hh)) << 3) +
                    ((@as(u32, pop_hl) + @as(u32, pop_lh)) << 2) +
                    (@as(u32, pop_ll) << 1);
                const quantized_dot = pos - @as(i32, @intCast(neg));
                approx = q_3bit_scale * self.k_3bit_scales[tok] * @as(f32, @floatFromInt(quantized_dot));
                err = q_norm * self.k_norms[tok] * std_dev_3 * z_score;
            } else {
                unreachable;
            }
            scores[tok] = .{
                .token = tok,
                .approx = sfm_scale * approx,
                .lo = sfm_scale * (approx - err),
                .hi = sfm_scale * (approx + err),
                .real = sfm_scale * real,
                .proba = 0.0,
            };
        }

        var max_real = scores[0].real;
        for (scores[1..]) |score| max_real = @max(max_real, score.real);

        var proba_sum: f32 = 0.0;
        for (scores) |*score| {
            score.proba = @exp(score.real - max_real);
            proba_sum += score.proba;
        }
        for (scores) |*score| score.proba /= proba_sum;

        const print_count = @min(256, scores.len);
        printScores("first 256 tokens", scores[0..print_count]);
        printScores("last 256 tokens", scores[scores.len - print_count ..]);

        std.mem.sort(Score, scores, {}, Score.decreasingApprox);
        printScores("top 256 by approximate score", scores[0..print_count]);

        std.mem.sort(Score, scores, {}, Score.decreasingReal);
        printScores("top 256 by real score", scores[0..print_count]);

        var total: f32 = 0.0;
        var threshold: f32 = 0.5;
        var count: usize = 0;
        for (scores) |score| {
            total += score.proba;
            count += 1;
            if (total > threshold) {
                std.log.info("Tokens for sfm mass {d} : {d}", .{ threshold, count });
                threshold = (1.0 + threshold) / 2.0;
            }
        }
    }

    pub fn truncatedAttention(self: *Attention, token_id: usize) !void {
        const query_id = try self.queryIndex(token_id);
        if (token_id >= self.v.len / head_dim) return error.InvalidTokenId;

        const count = token_id + 1;
        const weighted_tokens = try self.allocator.alloc(WeightedToken, count);
        defer self.allocator.free(weighted_tokens);

        const query = self.q[query_id * head_dim ..][0..head_dim];
        const softmax_scale = 1.0 / @sqrt(@as(f32, head_dim));
        var max_score = -std.math.inf(f32);
        for (weighted_tokens, 0..) |*weighted, token| {
            const key = self.k[token * head_dim ..][0..head_dim];
            var score: f32 = 0.0;
            for (query, key) |q_value, k_value| score += q_value * k_value;
            score *= softmax_scale;
            weighted.* = .{ .token = token, .weight = score };
            max_score = @max(max_score, score);
        }

        var weight_sum: f32 = 0.0;
        for (weighted_tokens) |*weighted| {
            weighted.weight = @exp(weighted.weight - max_score);
            weight_sum += weighted.weight;
        }
        for (weighted_tokens) |*weighted| weighted.weight /= weight_sum;

        var attention: [head_dim]f32 = @splat(0.0);
        for (weighted_tokens) |weighted| {
            const value = self.v[weighted.token * head_dim ..][0..head_dim];
            for (&attention, value) |*result, component| result.* += weighted.weight * component;
        }
        const attention_norm = quantization.normL2(&attention);
        const attention_norm_squared = attention_norm * attention_norm;

        std.mem.sort(WeightedToken, weighted_tokens, {}, WeightedToken.decreasingWeight);
        var mass_estimated_attention: [head_dim]f32 = @splat(0.0);

        std.log.info("\nTruncated attention", .{});
        std.log.info("{s:>10} {s:>14} {s:>14} {s:>18} {s:>18} {s:>18} {s:>18}", .{
            "tokens", "a norm", "b norm", "rel magnitude err", "rel direction err", "rel rescale L2", "rel mass L2",
        });
        std.log.info("{s:>10} {s:>14} {s:>14} {s:>18} {s:>18} {s:>18} {s:>18}", .{
            "----------", "--------------", "--------------", "------------------", "------------------", "------------------", "------------------",
        });

        var cutoff: usize = 1;
        var retained_mass: f32 = 0.0;
        for (weighted_tokens, 0..) |weighted, index| {
            const value = self.v[weighted.token * head_dim ..][0..head_dim];
            for (&mass_estimated_attention, value) |*result, component| result.* += weighted.weight * component;
            retained_mass += weighted.weight;

            const included = index + 1;
            if (included != cutoff or cutoff > token_id) continue;

            var truncated_attention: [head_dim]f32 = undefined;
            for (&truncated_attention, mass_estimated_attention) |*result, component| {
                result.* = component / retained_mass;
            }
            const truncated_norm = quantization.normL2(&truncated_attention);
            const magnitude_delta = attention_norm - truncated_norm;
            const relative_magnitude_error = if (attention_norm_squared > 0.0)
                magnitude_delta * magnitude_delta / attention_norm_squared
            else
                0.0;

            var cosine: f32 = 1.0;
            if (attention_norm > 0.0 and truncated_norm > 0.0) {
                var dot: f32 = 0.0;
                for (attention, truncated_attention) |a, b| dot += a * b;
                cosine = std.math.clamp(dot / (attention_norm * truncated_norm), -1.0, 1.0);
            }
            const relative_directional_error = 2.0 * (1.0 - cosine);

            var rescaled_l2_squared: f32 = 0.0;
            if (truncated_norm > 0.0) {
                const rescale = attention_norm / truncated_norm;
                for (attention, truncated_attention) |a, b| {
                    const delta = a - rescale * b;
                    rescaled_l2_squared += delta * delta;
                }
            } else {
                rescaled_l2_squared = attention_norm_squared;
            }

            var mass_estimated_l2_squared: f32 = 0.0;
            for (attention, mass_estimated_attention) |a, b| {
                const delta = a - b;
                mass_estimated_l2_squared += delta * delta;
            }
            const relative_rescaled_l2 = if (attention_norm > 0.0)
                @sqrt(@max(0.0, rescaled_l2_squared)) / attention_norm
            else
                0.0;
            const relative_mass_estimated_l2 = if (attention_norm > 0.0)
                @sqrt(@max(0.0, mass_estimated_l2_squared)) / attention_norm
            else
                0.0;

            std.log.info("{d:>10} {d:>14.6} {d:>14.6} {d:>18.8} {d:>18.8} {d:>18.8} {d:>18.8}", .{
                cutoff,
                attention_norm,
                truncated_norm,
                relative_magnitude_error,
                relative_directional_error,
                relative_rescaled_l2,
                relative_mass_estimated_l2,
            });

            if (cutoff > token_id / 2) break;
            cutoff *= 2;
        }
    }

    pub fn sampledAttention(self: *Attention, token_id: usize) !void {
        const query_id = try self.queryIndex(token_id);
        if (token_id >= self.v.len / head_dim) return error.InvalidTokenId;

        const first_window = 256;
        const last_window = 512;
        const top_k = 256;
        const sample_budget = 1024;
        const count = token_id + 1;

        const first_end = @min(first_window, count);
        const last_start = count - @min(last_window, count);
        const middle_start = first_end;
        const middle_end = @max(middle_start, last_start);
        const disjoint_last_start = @max(first_end, last_start);
        const middle_count = middle_end - middle_start;

        const middle = try self.allocator.alloc(WeightedToken, middle_count);
        defer self.allocator.free(middle);

        const query = self.q[query_id * head_dim ..][0..head_dim];
        const softmax_scale = 1.0 / @sqrt(@as(f32, head_dim));
        const full_weights = try self.allocator.alloc(f32, count);
        defer self.allocator.free(full_weights);
        var full_max_score = -std.math.inf(f32);
        for (full_weights, 0..) |*score, token| {
            const key = self.k[token * head_dim ..][0..head_dim];
            score.* = 0.0;
            for (query, key) |q_value, k_value| score.* += q_value * k_value;
            score.* *= softmax_scale;
            full_max_score = @max(full_max_score, score.*);
        }
        var full_mass: f32 = 0.0;
        for (full_weights) |*weight| {
            weight.* = @exp(weight.* - full_max_score);
            full_mass += weight.*;
        }
        for (full_weights) |*weight| weight.* /= full_mass;

        var real_attention: [head_dim]f32 = @splat(0.0);
        for (full_weights, 0..) |weight, token| {
            const value = self.v[token * head_dim ..][0..head_dim];
            for (&real_attention, value) |*result, component| result.* += weight * component;
        }

        for (middle, middle_start..) |*candidate, token| {
            const key = self.k[token * head_dim ..][0..head_dim];
            var score: f32 = 0.0;
            for (query, key) |q_value, k_value| score += q_value * k_value;
            candidate.* = .{ .token = token, .weight = score * softmax_scale };
        }
        std.mem.sort(WeightedToken, middle, {}, WeightedToken.decreasingWeight);

        const top_count = @min(top_k, middle_count);
        const excluded_count = middle_count - top_count;
        const sample_count = @min(sample_budget, excluded_count);
        const sampled_candidates = middle[top_count..];
        var prng = std.Random.DefaultPrng.init(@as(u64, @intCast(token_id)) ^ 0x9e3779b97f4a7c15);
        const random = prng.random();
        for (0..sample_count) |i| {
            const selected_offset = random.uintLessThan(usize, excluded_count - i);
            std.mem.swap(WeightedToken, &sampled_candidates[i], &sampled_candidates[i + selected_offset]);
        }

        const fixed_count = first_end + count - disjoint_last_start;
        const selected_count = fixed_count + top_count + sample_count;
        const selected = try self.allocator.alloc(WeightedToken, selected_count);
        defer self.allocator.free(selected);

        var selected_index: usize = 0;
        for (0..first_end) |token| {
            const key = self.k[token * head_dim ..][0..head_dim];
            var score: f32 = 0.0;
            for (query, key) |q_value, k_value| score += q_value * k_value;
            selected[selected_index] = .{ .token = token, .weight = score * softmax_scale };
            selected_index += 1;
        }
        for (disjoint_last_start..count) |token| {
            const key = self.k[token * head_dim ..][0..head_dim];
            var score: f32 = 0.0;
            for (query, key) |q_value, k_value| score += q_value * k_value;
            selected[selected_index] = .{ .token = token, .weight = score * softmax_scale };
            selected_index += 1;
        }
        for (middle[0..top_count]) |candidate| {
            selected[selected_index] = candidate;
            selected_index += 1;
        }
        for (sampled_candidates[0..sample_count]) |candidate| {
            selected[selected_index] = candidate;
            selected_index += 1;
        }
        std.debug.assert(selected_index == selected.len);

        const exact_count = fixed_count + top_count;
        const sampling_factor = if (sample_count > 0)
            @as(f32, @floatFromInt(excluded_count)) / @as(f32, @floatFromInt(sample_count))
        else
            0.0;
        var true_retained_mass: f32 = 0.0;
        for (selected) |candidate| true_retained_mass += full_weights[candidate.token];
        var estimated_full_mass: f32 = 0.0;
        for (selected[0..exact_count]) |candidate| estimated_full_mass += full_weights[candidate.token];
        for (selected[exact_count..]) |candidate| estimated_full_mass += sampling_factor * full_weights[candidate.token];

        var max_score = -std.math.inf(f32);
        for (selected) |candidate| max_score = @max(max_score, candidate.weight);
        for (selected) |*candidate| {
            candidate.weight = @exp(candidate.weight - max_score);
        }

        var exact_mass: f32 = 0.0;
        var sampled_mass: f32 = 0.0;
        var exact_numerator: [head_dim]f32 = @splat(0.0);
        var sampled_numerator: [head_dim]f32 = @splat(0.0);
        for (selected, 0..) |candidate, index| {
            const value = self.v[candidate.token * head_dim ..][0..head_dim];
            if (index < exact_count) {
                exact_mass += candidate.weight;
                for (&exact_numerator, value) |*result, component| result.* += candidate.weight * component;
            } else {
                sampled_mass += candidate.weight;
                for (&sampled_numerator, value) |*result, component| result.* += candidate.weight * component;
            }
        }

        const estimated_mass = exact_mass + sampling_factor * sampled_mass;
        var sampled_attention: [head_dim]f32 = undefined;
        for (&sampled_attention, exact_numerator, sampled_numerator) |*result, exact, sampled| {
            result.* = (exact + sampling_factor * sampled) / estimated_mass;
        }

        const real_norm = quantization.normL2(&real_attention);
        const sampled_norm = quantization.normL2(&sampled_attention);
        var dot: f32 = 0.0;
        var l2_squared: f32 = 0.0;
        for (real_attention, sampled_attention) |real, sampled| {
            dot += real * sampled;
            const delta = real - sampled;
            l2_squared += delta * delta;
        }
        const cosine = if (real_norm > 0.0 and sampled_norm > 0.0)
            std.math.clamp(dot / (real_norm * sampled_norm), -1.0, 1.0)
        else
            1.0;
        const magnitude_error = @abs(real_norm - sampled_norm);
        const directional_error = @sqrt(@max(0.0, 2.0 * real_norm * sampled_norm * (1.0 - cosine)));
        const l2_error = @sqrt(@max(0.0, l2_squared));
        const relative_magnitude_error = if (real_norm > 0.0) magnitude_error / real_norm else 0.0;
        const relative_directional_error = if (real_norm > 0.0) directional_error / real_norm else 0.0;
        const relative_l2_error = if (real_norm > 0.0) l2_error / real_norm else 0.0;

        std.log.info("Sampled attention: fixed={d}, top={d}, sampled={d}, selected={d}/{d}", .{
            fixed_count,
            top_count,
            sample_count,
            selected_count,
            count,
        });
        std.log.info("{s:>10} {s:>10} {s:>10} {s:>10} {s:>12} {s:>12} {s:>12} {s:>12} {s:>12} {s:>12} {s:>12}", .{
            "kept mass", "est mass", "cosine", "real norm", "sample norm", "mag abs", "mag rel", "dir abs", "dir rel", "L2 abs", "L2 rel",
        });
        std.log.info("{d:>10.6} {d:>10.6} {d:>10.6} {d:>10.6} {d:>12.6} {d:>12.6} {d:>12.6} {d:>12.6} {d:>12.6} {d:>12.6} {d:>12.6}", .{
            true_retained_mass,
            estimated_full_mass,
            cosine,
            real_norm,
            sampled_norm,
            magnitude_error,
            relative_magnitude_error,
            directional_error,
            relative_directional_error,
            l2_error,
            relative_l2_error,
        });
    }

    pub fn normalize(self: *Attention, token_id: usize) !void {
        const token_count = self.k.len / head_dim;
        if (token_count < head_dim or token_id > token_count - head_dim) return error.InvalidTokenId;

        const element_count = head_dim * head_dim;
        const before = try self.allocator.alloc(f32, element_count);
        defer self.allocator.free(before);
        const after = try self.allocator.alloc(f32, element_count);
        defer self.allocator.free(after);
        const log_token_scales = try self.allocator.alloc(f32, head_dim);
        defer self.allocator.free(log_token_scales);
        const log_coord_scales = try self.allocator.alloc(f32, head_dim);
        defer self.allocator.free(log_coord_scales);
        const best_log_token_scales = try self.allocator.alloc(f32, head_dim);
        defer self.allocator.free(best_log_token_scales);
        const best_log_coord_scales = try self.allocator.alloc(f32, head_dim);
        defer self.allocator.free(best_log_coord_scales);
        const token_scales = try self.allocator.alloc(f32, head_dim);
        defer self.allocator.free(token_scales);
        const coord_scales = try self.allocator.alloc(f32, head_dim);
        defer self.allocator.free(coord_scales);
        const token_variances = try self.allocator.alloc(f32, head_dim);
        defer self.allocator.free(token_variances);
        const coord_variances = try self.allocator.alloc(f32, head_dim);
        defer self.allocator.free(coord_variances);

        const source = self.k[token_id * head_dim ..][0..element_count];
        @memcpy(before, source);
        for (0..head_dim) |token| {
            const vector = before[token * head_dim ..][0..head_dim];
            //const norm = quantization.normL2(vector);
            //if (norm > 0.0) {
            //    const inv_norm = 1.0 / norm;
            //    for (vector) |*value| value.* *= inv_norm;
            //}
            quantization.walshHadamard(vector, head_dim_log2);
        }

        @memset(log_token_scales, 0.0);
        @memset(log_coord_scales, 0.0);
        @memset(best_log_token_scales, 0.0);
        @memset(best_log_coord_scales, 0.0);

        const epsilon: f32 = 1e-8;
        const min_log_scale: f32 = -40.0;
        const max_log_scale: f32 = 40.0;
        var best_score: f32 = std.math.inf(f32);

        for (0..8) |_| {
            normalizeMatrix(before, after, log_token_scales, log_coord_scales);
            columnVariances(after, coord_variances);
            for (log_coord_scales, coord_variances) |*log_scale, variance| {
                log_scale.* = std.math.clamp(log_scale.* + 0.5 * @log(@max(variance, epsilon)), min_log_scale, max_log_scale);
            }

            normalizeMatrix(before, after, log_token_scales, log_coord_scales);
            rowVariances(after, token_variances);
            for (log_token_scales, token_variances) |*log_scale, variance| {
                log_scale.* = std.math.clamp(log_scale.* + 0.5 * @log(@max(variance, epsilon)), min_log_scale, max_log_scale);
            }

            normalizeMatrix(before, after, log_token_scales, log_coord_scales);
            rowVariances(after, token_variances);
            columnVariances(after, coord_variances);
            const score = varianceImbalance(token_variances, epsilon) + varianceImbalance(coord_variances, epsilon);
            if (score < best_score) {
                best_score = score;
                @memcpy(best_log_token_scales, log_token_scales);
                @memcpy(best_log_coord_scales, log_coord_scales);
            }
        }

        var mean_log_token_scale: f32 = 0.0;
        for (best_log_token_scales) |log_scale| mean_log_token_scale += log_scale;
        mean_log_token_scale /= @as(f32, head_dim);
        for (0..head_dim) |i| {
            token_scales[i] = @exp(best_log_token_scales[i] - mean_log_token_scale);
            coord_scales[i] = @exp(best_log_coord_scales[i] + mean_log_token_scale);
        }
        normalizeMatrixWithScales(before, after, token_scales, coord_scales);

        std.log.info("\nDual variance scales (score {d:.2})", .{best_score});
        std.log.info("{s:>5} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10}", .{
            "i", "tok scale", "coord scl", "tok max b", "tok max a", "crd max b", "crd max a", "tok min b", "tok min a", "crd min b", "crd min a",
        });
        std.log.info("{s:>5} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10}", .{
            "-----", "----------", "----------", "----------", "----------", "----------", "----------", "----------", "----------", "----------", "----------",
        });
        for (0..head_dim) |i| {
            const token_before = columnAbsRange(before, i);
            const token_after = columnAbsRange(after, i);
            const coord_before = rowAbsRange(before, i);
            const coord_after = rowAbsRange(after, i);
            std.log.info("{d:>5} {d:>10.2} {d:>10.2} {d:>10.2} {d:>10.2} {d:>10.2} {d:>10.2} {d:>10.2} {d:>10.2} {d:>10.2} {d:>10.2}", .{
                i,
                token_scales[i],
                coord_scales[i],
                token_before.max,
                token_after.max,
                coord_before.max,
                coord_after.max,
                token_before.min,
                token_after.min,
                coord_before.min,
                coord_after.min,
            });
        }

        printMatrix16("FWHT block before dual normalization", before);
        printMatrix16("FWHT block after dual normalization", after);
    }

    pub fn computeAttention(self: *Attention, token_id: usize) !void {
        var max_score: f32 = -1e10;
        var best_tok: usize = 0;
        var score: f32 = undefined;

        const query_id = try self.queryIndex(token_id);
        const q_128 = self.q_qjl[query_id];
        const q_norm = self.q_norms[query_id];
        const q_lsb = self.q_lsb[query_id];
        const q_msb = self.q_msb[query_id];
        const q_scale = self.q_scales[query_id];
        const q_3bit_lsb = self.q_3bit_lsb[query_id];
        const q_3bit_hsb = self.q_3bit_hsb[query_id];
        const q_3bit_msb = self.q_3bit_msb[query_id];
        const q_3bit_scale = self.q_3bit_scales[query_id];

        for (0..token_id + 1) |tok| {
            if (use_1bit) {
                score = q_norm * self.k_norms[tok] * qjl_dot_lut[@popCount(q_128 ^ self.k_qjl[tok])];
            } else if (use_2bit) {
                const pop_mm = @popCount(q_msb ^ self.k_msb[tok]);
                const pop_ml = @popCount(q_msb ^ self.k_lsb[tok]);
                const pop_lm = @popCount(q_lsb ^ self.k_msb[tok]);
                const pop_ll = @popCount(q_lsb ^ self.k_lsb[tok]);
                const pos: i32 = 9 * head_dim;
                const neg = (@as(u32, pop_mm) << 3) + ((@as(u32, pop_ml) + @as(u32, pop_lm)) << 2) + (@as(u32, pop_ll) << 1);
                score = q_scale * self.k_scales[tok] * @as(f32, @floatFromInt(pos - @as(i32, @intCast(neg))));
            } else if (use_3bit) {
                const pop_mm = @popCount(q_3bit_msb ^ self.k_3bit_msb[tok]);
                const pop_mh = @popCount(q_3bit_msb ^ self.k_3bit_hsb[tok]);
                const pop_hm = @popCount(q_3bit_hsb ^ self.k_3bit_msb[tok]);
                const pop_ml = @popCount(q_3bit_msb ^ self.k_3bit_lsb[tok]);
                const pop_lm = @popCount(q_3bit_lsb ^ self.k_3bit_msb[tok]);
                const pop_hh = @popCount(q_3bit_hsb ^ self.k_3bit_hsb[tok]);
                const pop_hl = @popCount(q_3bit_hsb ^ self.k_3bit_lsb[tok]);
                const pop_lh = @popCount(q_3bit_lsb ^ self.k_3bit_hsb[tok]);
                const pop_ll = @popCount(q_3bit_lsb ^ self.k_3bit_lsb[tok]);
                const pos: i32 = 49 * head_dim;
                const neg = (@as(u32, pop_mm) << 5) +
                    ((@as(u32, pop_mh) + @as(u32, pop_hm)) << 4) +
                    ((@as(u32, pop_ml) + @as(u32, pop_lm) + @as(u32, pop_hh)) << 3) +
                    ((@as(u32, pop_hl) + @as(u32, pop_lh)) << 2) +
                    (@as(u32, pop_ll) << 1);
                const quantized_dot = pos - @as(i32, @intCast(neg));
                score = q_3bit_scale * self.k_3bit_scales[tok] * @as(f32, @floatFromInt(quantized_dot));
            } else {
                unreachable;
            }
            if (score > max_score) {
                max_score = score;
                best_tok = tok;
            }
        }
        //std.log.info("Best tok = {d}, with score = {d}", .{ best_tok, max_score });
        self.sink += best_tok;
    }

    fn queryIndex(self: Attention, token_id: usize) !usize {
        const query_count = self.q_qjl.len / num_queries;
        if (query_count > self.k_qjl.len) return error.InvalidTensorShape;

        const prompt_length = self.k_qjl.len - query_count;
        if (token_id < prompt_length or token_id >= self.k_qjl.len) return error.QueryNotExported;
        return token_id - prompt_length;
    }

    fn makeQjlDotLut(comptime coord_count: usize) [coord_count + 1]f32 {
        @setEvalBranchQuota(8192);
        var lut: [coord_count + 1]f32 = undefined;
        for (0..(coord_count + 1)) |mismatch_count| {
            const dot: i32 = @as(i32, @intCast(coord_count)) - 2 * @as(i32, @intCast(mismatch_count));
            const w = @as(f32, @floatFromInt(dot)) / @as(f32, @floatFromInt(coord_count));
            lut[mismatch_count] = @sin(@as(f32, @floatCast(0.5 * std.math.pi)) * w);
        }
        return lut;
    }

    pub const qjl_dot_lut = makeQjlDotLut(head_dim);
};

fn normalizeMatrix(matrix: []const f32, normalized: []f32, log_token_scales: []const f32, log_coord_scales: []const f32) void {
    for (0..head_dim) |token| {
        const token_scale = @exp(log_token_scales[token]);
        for (0..head_dim) |coord| {
            normalized[token * head_dim + coord] = matrix[token * head_dim + coord] / (token_scale * @exp(log_coord_scales[coord]));
        }
    }
}

fn normalizeMatrixWithScales(matrix: []const f32, normalized: []f32, token_scales: []const f32, coord_scales: []const f32) void {
    for (0..head_dim) |token| {
        for (0..head_dim) |coord| {
            normalized[token * head_dim + coord] = matrix[token * head_dim + coord] / (token_scales[token] * coord_scales[coord]);
        }
    }
}

fn rowVariances(matrix: []const f32, variances: []f32) void {
    for (0..head_dim) |token| {
        const row = matrix[token * head_dim ..][0..head_dim];
        var sum: f32 = 0.0;
        var sum_squared: f32 = 0.0;
        for (row) |value| {
            sum += value;
            sum_squared += value * value;
        }
        const mean = sum / @as(f32, head_dim);
        variances[token] = @max(0.0, sum_squared / @as(f32, head_dim) - mean * mean);
    }
}

fn columnVariances(matrix: []const f32, variances: []f32) void {
    for (0..head_dim) |coord| {
        var sum: f32 = 0.0;
        var sum_squared: f32 = 0.0;
        for (0..head_dim) |token| {
            const value = matrix[token * head_dim + coord];
            sum += value;
            sum_squared += value * value;
        }
        const mean = sum / @as(f32, head_dim);
        variances[coord] = @max(0.0, sum_squared / @as(f32, head_dim) - mean * mean);
    }
}

fn varianceImbalance(variances: []const f32, epsilon: f32) f32 {
    var min_variance = std.math.inf(f32);
    var max_variance: f32 = 0.0;
    for (variances) |variance| {
        min_variance = @min(min_variance, variance);
        max_variance = @max(max_variance, variance);
    }
    return max_variance / @max(min_variance, epsilon);
}

const AbsRange = struct {
    min: f32,
    max: f32,
};

fn rowAbsRange(matrix: []const f32, token: usize) AbsRange {
    var result: AbsRange = .{ .min = std.math.inf(f32), .max = 0.0 };
    for (matrix[token * head_dim ..][0..head_dim]) |value| {
        const absolute = @abs(value);
        result.min = @min(result.min, absolute);
        result.max = @max(result.max, absolute);
    }
    return result;
}

fn columnAbsRange(matrix: []const f32, coord: usize) AbsRange {
    var result: AbsRange = .{ .min = std.math.inf(f32), .max = 0.0 };
    for (0..head_dim) |token| {
        const absolute = @abs(matrix[token * head_dim + coord]);
        result.min = @min(result.min, absolute);
        result.max = @max(result.max, absolute);
    }
    return result;
}

fn printMatrix16(title: []const u8, matrix: []const f32) void {
    std.log.info("\n{s}", .{title});
    std.log.info("{s:>5} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8} {d:>8}", .{
        "row", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    });
    for (0..16) |row| {
        const values = matrix[row * head_dim ..][0..16];
        std.log.info("{d:>5} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2} {d:>8.2}", .{
            row,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            values[8],
            values[9],
            values[10],
            values[11],
            values[12],
            values[13],
            values[14],
            values[15],
        });
    }
}

fn printScores(title: []const u8, scores: []const Score) void {
    std.log.info("\n{s}", .{title});
    std.log.info("{s:>10}  {s:>14}  {s:>14}  {s:>14}  {s:>14}  {s:>14}", .{ "token", "approx", "lo", "hi", "real", "proba" });
    std.log.info("{s:>10}  {s:>14}  {s:>14}  {s:>14}  {s:>14}  {s:>14}", .{ "----------", "--------------", "--------------", "--------------", "--------------", "--------------" });
    for (scores) |score| {
        std.log.info("{d:>10}  {d:>14.6}  {d:>14.6}  {d:>14.6}  {d:>14.6}  {d:>14.8}", .{
            score.token,
            score.approx,
            score.lo,
            score.hi,
            score.real,
            score.proba,
        });
    }
}

fn quantizeHistory1bit(history: []const f32, qjl: []u128, norms: []f32, buffer: []f32) void {
    std.debug.assert(buffer.len == head_dim);
    std.debug.assert(history.len == qjl.len * head_dim);
    std.debug.assert(norms.len == qjl.len);
    for (qjl, 0..) |*bits, vector_id| {
        const vector = history[vector_id * head_dim ..][0..head_dim];
        norms[vector_id] = quantization.normL2(vector);
        @memcpy(buffer, vector);
        quantization.walshHadamard(buffer, head_dim_log2);
        bits.* = 0;
        for (buffer, 0..) |value, coordinate| {
            if (value > 0) {
                const shift: std.math.Log2Int(u128) = @intCast(coordinate);
                bits.* |= @as(u128, 1) << shift;
            }
        }
    }
}

fn quantizeHistory2bit(history: []const f32, lsb: []u128, msb: []u128, scales: []f32, buffer: []f32) void {
    std.debug.assert(buffer.len == head_dim);
    std.debug.assert(history.len == lsb.len * head_dim);
    std.debug.assert(msb.len == lsb.len);
    std.debug.assert(scales.len == lsb.len);

    const inv_sqrt_dim = 1.0 / @sqrt(@as(f32, head_dim));
    for (0..lsb.len) |vector_id| {
        const vector = history[vector_id * head_dim ..][0..head_dim];
        const threshold = quantization.normL2(vector) * inv_sqrt_dim;

        @memcpy(buffer, vector);
        quantization.walshHadamard(buffer, head_dim_log2);

        var lsb_bits: u128 = 0;
        var msb_bits: u128 = 0;
        var dot: f32 = 0.0;
        var norm_squared: f32 = 0.0;

        for (buffer, 0..) |value, coordinate| {
            const msb_positive = value > 0.0;
            const lsb_positive = if (msb_positive) value > threshold else value > -threshold;
            const shift: std.math.Log2Int(u128) = @intCast(coordinate);

            if (msb_positive) msb_bits |= @as(u128, 1) << shift;
            if (lsb_positive) lsb_bits |= @as(u128, 1) << shift;

            const m: f32 = if (msb_positive) 1.0 else -1.0;
            const l: f32 = if (lsb_positive) 1.0 else -1.0;
            const quantized = 2.0 * m + l;
            dot += value * quantized;
            norm_squared += quantized * quantized;
        }

        lsb[vector_id] = lsb_bits;
        msb[vector_id] = msb_bits;
        scales[vector_id] = dot / norm_squared;
    }
}

fn quantizeHistory3bit(history: []const f32, lsb: []u128, hsb: []u128, msb: []u128, scales: []f32, buffer: []f32) void {
    std.debug.assert(buffer.len == head_dim);
    std.debug.assert(history.len == lsb.len * head_dim);
    std.debug.assert(hsb.len == lsb.len);
    std.debug.assert(msb.len == lsb.len);
    std.debug.assert(scales.len == lsb.len);

    const inv_sqrt_dim = 1.0 / @sqrt(@as(f32, head_dim));
    for (0..lsb.len) |vector_id| {
        const vector = history[vector_id * head_dim ..][0..head_dim];
        const sigma = quantization.normL2(vector) * inv_sqrt_dim;
        const threshold_step = 0.58601946 * sigma;
        const threshold_0 = threshold_step;
        const threshold_1 = 2.0 * threshold_step;
        const threshold_2 = 3.0 * threshold_step;

        @memcpy(buffer, vector);
        quantization.walshHadamard(buffer, head_dim_log2);

        var lsb_bits: u128 = 0;
        var hsb_bits: u128 = 0;
        var msb_bits: u128 = 0;
        var dot: f32 = 0.0;
        var norm_squared: f32 = 0.0;

        for (buffer, 0..) |value, coordinate| {
            const level_index: usize = if (value > threshold_2)
                7
            else if (value > threshold_1)
                6
            else if (value > threshold_0)
                5
            else if (value > 0.0)
                4
            else if (value > -threshold_0)
                3
            else if (value > -threshold_1)
                2
            else if (value > -threshold_2)
                1
            else
                0;
            const shift: std.math.Log2Int(u128) = @intCast(coordinate);

            if ((level_index & 1) != 0) lsb_bits |= @as(u128, 1) << shift;
            if ((level_index & 2) != 0) hsb_bits |= @as(u128, 1) << shift;
            if ((level_index & 4) != 0) msb_bits |= @as(u128, 1) << shift;

            const quantized: f32 = @floatFromInt(@as(i32, @intCast(2 * level_index)) - 7);
            dot += value * quantized;
            norm_squared += quantized * quantized;
        }

        lsb[vector_id] = lsb_bits;
        hsb[vector_id] = hsb_bits;
        msb[vector_id] = msb_bits;
        scales[vector_id] = dot / norm_squared;
    }
}

fn loadTensorF32(zml_handler: *Zml_handler, filename: []const u8, tensor_name: []const u8) ![]f32 {
    const slice = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.qkv, filename, tensor_name);
    defer slice.free(zml_handler.allocator);
    if (slice.shape.rank() != 2 or slice.shape.dim(1) != head_dim) return error.InvalidTensorShape;
    const result = try zml_handler.allocator.alloc(f32, slice.shape.count());
    errdefer zml_handler.allocator.free(result);
    switch (slice.dtype()) {
        .bf16 => {
            for (result, slice.constItems(zml.floats.BFloat16)) |*dst, src| dst.* = src.toF32();
        },
        .f32 => @memcpy(result, slice.constItems(f32)),
        else => return error.UnsupportedTensorType,
    }
    return result;
}
