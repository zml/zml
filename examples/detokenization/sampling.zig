const std = @import("std");
const zml = @import("zml");

const main = @import("main.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const tokens = @import("tokens.zig");
const algebra = @import("algebra.zig");
const quantization = @import("quantization.zig");
const graph_ = @import("graph.zig");

const Zml_handler = main.Zml_handler;
const Tokenizer = zml.tokenizer.Tokenizer;
const LmHeadMatrix = algebra.LmHeadMatrix;
const QuantizationInt8 = quantization.QuantizationInt8;
const QuantizationInt4 = quantization.QuantizationInt4;
const QuantizationQJL1 = quantization.QuantizationQJL1;
const QuantizationQJL2 = quantization.QuantizationQJL2;
const VectorQJL1 = quantization.VectorQJL1;
const VectorQJL2 = quantization.VectorQJL2;
const Graph = graph_.Graph;

pub const hidden_dim = quantization.hidden_dim;

pub const sampling_simd_len: comptime_int = 16;
pub const sampling_top_k: comptime_int = 64;
pub const sampling_top_p: comptime_float = 0.99;

const sampling_truncate_coords: comptime_int = 256;
const sampling_truncate_dense: comptime_int = 256;

pub const Logit = struct {
    row: usize = std.math.maxInt(usize),
    logit: f32 = 0,
    proba: f32 = 0,
    upper_bound: f32 = 0,

    pub fn decreasingOrder(_: void, lhs: Logit, rhs: Logit) bool {
        return lhs.logit > rhs.logit;
    }
};

pub const SamplingResult = struct {
    candidates: [sampling_top_k]Logit,
    nb: usize,
};

pub const SamplingReference = struct {
    ref: []SamplingResult,
};

pub const Sampler = struct {
    allocator: std.mem.Allocator,
    tokenizer: Tokenizer,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    logits: []f32,
    candidates: [sampling_top_k]Logit,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix) !Sampler {
        const v: usize = lm_head.n;
        const d: usize = lm_head.d;

        const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
        const tokenizer = try llm_.Llm_handler.loadTokenizer(zml_handler, repo);

        const logits = try zml_handler.allocator.alloc(f32, v);

        return .{
            .allocator = zml_handler.allocator,
            .tokenizer = tokenizer,
            .lm_head = lm_head,
            .d = d,
            .vocab_size = v,
            .logits = logits,
            .candidates = [_]Logit{.{}} ** sampling_top_k,
        };
    }

    pub fn deinit(self: *Sampler) void {
        self.tokenizer.deinit();
        self.allocator.free(self.logits);
    }

    pub fn sample(self: *Sampler, y: []const f32) SamplingResult {
        self.computeLogits(y);
        findTopK(self.logits, self.candidates[0..sampling_top_k]);
        computeProbas(&self.candidates);
        return .{ .candidates = self.candidates, .nb = self.nbInTopP() };
    }

    pub fn computeLogits(self: *Sampler, y: []const f32) void {
        for (0..self.vocab_size) |token| {
            self.logits[token] = computeLogit(self.lm_head, token, y);
        }
    }

    pub inline fn computeLogit(lm_head: *LmHeadMatrix, token: usize, y: []const f32) f32 {
        const Vec = @Vector(sampling_simd_len, f32);
        const row = lm_head.data[token * lm_head.d ..][0..lm_head.d];
        std.debug.assert(y.len >= lm_head.d and lm_head.d % sampling_simd_len == 0);

        var sum: Vec = @splat(0.0);
        var i: usize = 0;
        while (i < lm_head.d) : (i += sampling_simd_len) {
            const weights: Vec = row[i..][0..sampling_simd_len].*;
            const values: Vec = y[i..][0..sampling_simd_len].*;
            sum = @mulAdd(Vec, weights, values, sum);
        }
        return @reduce(.Add, sum);
    }

    pub fn computeProbas(candidates: *[sampling_top_k]Logit) void {
        var max_logit: f32 = candidates[0].logit;
        for (1..sampling_top_k) |i| {
            max_logit = @max(max_logit, candidates[i].logit);
        }
        var total: f32 = 0.0;
        for (0..sampling_top_k) |i| {
            const proba = @exp(candidates[i].logit - max_logit);
            candidates[i].proba = proba;
            total += proba;
        }
        const inv_total = 1.0 / total;
        for (0..sampling_top_k) |i| {
            candidates[i].proba *= inv_total;
        }
    }

    pub fn nbInTopP(self: *Sampler) usize {
        var total: f32 = 0.0;
        for (0..sampling_top_k) |i| {
            total += self.candidates[i].proba;
            if (total > sampling_top_p) return i;
        }
        return sampling_top_k;
    }
    
    pub fn logSampling(self: *Sampler) !void {
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s:>14}  {s}", .{ "rank", "token_id", "logit", "proba", "cumul", "token" });
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s:>14}  {s}", .{ "------", "----------", "--------------", "--------------", "--------------", "-----" });
        var cumul: f32 = 0.0;
        for (0..sampling_top_k) |i| {
            const row = self.candidates[i].row;
            const logit = self.candidates[i].logit;
            const proba = self.candidates[i].proba;
            cumul += proba;
            var decoded_buf: [512]u8 = undefined;
            var escaped_buf: [512]u8 = undefined;
            const token_text = self.escapedToken(row, &decoded_buf, &escaped_buf);
            std.log.info("{d:>6}  {d:>10}  {d:>14.6}  {d:>14.6}  {d:>14.6}  {s}", .{ i + 1, row, logit, proba, cumul, token_text });
            if (cumul > sampling_top_p) return;
        }
    }

    pub fn logRealVsApproxSampling(self: *Sampler, approx_logits: []Logit, label: []const u8) void {
        const count = self.top_k;
        std.log.info("***** Real logits top-k", .{});
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14} {s:>14}  {s}", .{ "rank", "token_id", "logit", "proba", label, "token" });
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s:>14}  {s}", .{ "------", "----------", "--------------", "--------------", "--------------", "-----" });
        for (self.sampling_candidates[0..count], 0..) |candidate, rank| {
            var decoded_buf: [512]u8 = undefined;
            var escaped_buf: [512]u8 = undefined;
            const token_text = self.escapedToken(candidate.token, &decoded_buf, &escaped_buf);
            std.log.info("{d:>6}  {d:>10}  {d:>14.6}  {d:>14.6}  {d:>14.6}  {s}", .{ rank + 1, candidate.token, candidate.logit, candidate.weight, approx_logits[candidate.token].logit, token_text });
        }
    }

    pub fn logApproxVsRealSampling(self: *Sampler, approx_logits: []Logit, label: []const u8) void {
        const count = @min(self.top_k, approx_logits.len);
        std.log.info("***** Approx logits top-k", .{});
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s:>14}  {s}", .{ "rank", "token_id", label, "upper bnd", "real_logit", "token" });
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s:>14}  {s}", .{ "------", "----------", "--------------", "--------------", "--------------", "-----" });
        for (0..count) |rank| {
            const logit = approx_logits[rank];
            const token = logit.row;
            var decoded_buf: [512]u8 = undefined;
            var escaped_buf: [512]u8 = undefined;
            const token_text = self.escapedToken(token, &decoded_buf, &escaped_buf);
            std.log.info("{d:>6}  {d:>10}  {d:>14.6}  {d:>14.6}  {d:>14.6}  {s}", .{ rank + 1, token, logit.logit, logit.upper_bound, self.realLogitForToken(token), token_text });
        }
    }

    fn realLogitForToken(self: *Sampler, token: usize) f32 {
        for (self.sampling_candidates[0..self.vocab_size]) |candidate| {
            if (candidate.token == token) return candidate.logit;
        }
        return -std.math.floatMax(f32);
    }

    fn escapedToken(self: *Sampler, token: usize, decoded_buf: []u8, escaped_buf: []u8) []const u8 {
        const decoded = tokens.decodeToken(self.tokenizer, @intCast(token), decoded_buf) catch return "<decode-error>";
        return tokens.escapeTokenText(decoded, escaped_buf);
    }
};

pub const AngularSampler = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    logits: []f32,
    candidates: [sampling_top_k]Logit,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix) !AngularSampler {
        const v: usize = lm_head.n;
        const d: usize = lm_head.d;
        const logits = try zml_handler.allocator.alloc(f32, v);
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .d = d,
            .vocab_size = v,
            .logits = logits,
            .candidates = [_]Logit{.{}} ** sampling_top_k,
        };
    }

    pub fn deinit(self: *AngularSampler) void {
        self.allocator.free(self.logits);
    }

    pub fn sample(self: *AngularSampler, y: []const f32) SamplingResult {
        for (0..self.vocab_size) |token| {
            self.logits[token] = computeLogit(self.lm_head, token, y) / self.lm_head.row_norms[token];
        }
        findTopK(self.logits, self.candidates[0..sampling_top_k]);
        return .{ .candidates = self.candidates, .nb = 0 };
    }

    pub inline fn computeLogit(lm_head: *LmHeadMatrix, token: usize, y: []const f32) f32 {
        const Vec = @Vector(sampling_simd_len, f32);
        const row = lm_head.data[token * lm_head.d ..][0..lm_head.d];
        std.debug.assert(y.len >= lm_head.d and lm_head.d % sampling_simd_len == 0);

        var sum: Vec = @splat(0.0);
        var i: usize = 0;
        while (i < lm_head.d) : (i += sampling_simd_len) {
            const weights: Vec = row[i..][0..sampling_simd_len].*;
            const values: Vec = y[i..][0..sampling_simd_len].*;
            sum = @mulAdd(Vec, weights, values, sum);
        }
        return @reduce(.Add, sum);
    }
    
};

pub inline fn findTopK(logits: []const f32, candidates: []Logit) void {
    for (0..candidates.len) |i| { candidates[i].logit = -std.math.floatMax(f32); }
    for (0..logits.len) |token| {
        const logit = logits[token];
        var insert_pos: usize = candidates.len - 1;
        if (logit <= candidates[insert_pos].logit) continue;
        while (insert_pos > 0 and logit > candidates[insert_pos - 1].logit) {
            candidates[insert_pos] = candidates[insert_pos - 1];
            insert_pos -= 1;
        }
        candidates[insert_pos] = .{ .row = token, .logit = logit };
    }
}

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

        findTopK(self.logits, self.candidates[0..sampling_truncate_dense]);

        // dense score them and return top k
        for (0..sampling_truncate_dense) |i| {
            self.candidates[i].logit = Sampler.computeLogit(self.lm_head, self.candidates[i].row, query);
        }
        std.mem.sort(Logit, &self.candidates, {}, Logit.decreasingOrder);
        return .{ .candidates = self.candidates[0..sampling_top_k].*, .nb = 0 };
    }
};

pub const Int8Sampler = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    logits: []f32,
    candidates: [sampling_top_k]Logit,
    quantizer: *QuantizationInt8,
    quantized_query: []i8,

    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix, quant: *QuantizationInt8) !Int8Sampler {
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .vocab_size = lm_head.n,
            .d = lm_head.d,
            .logits = try zml_handler.allocator.alloc(f32, lm_head.n),
            .candidates = [_]Logit{.{}} ** sampling_top_k,
            .quantizer = quant,
            .quantized_query = try zml_handler.allocator.alloc(i8, lm_head.d),
        };
    }

    pub fn deinit(self: *Int8Sampler) void {
        self.allocator.free(self.logits);
        self.allocator.free(self.quantized_query);
    }

    pub fn sample(self: *Int8Sampler, query: []const f32) SamplingResult {
        const query_norm = quantization.normL2(query);
        const query_quant_scale = QuantizationInt8.quantizeVector(query, self.quantizer.buffer, self.quantized_query, query_norm);

        // compute logits with int8 dot products
        for (0..self.vocab_size) |row| {
            const quantized_row = self.quantizer.lm_head_quantized[row * self.d ..][0..self.d];
            const res = QuantizationInt8.int8dot(self.quantized_query, quantized_row);
            self.logits[row] = @as(f32, @floatFromInt(res)) * self.quantizer.row_quant_scale[row] * query_quant_scale;
        }

        findTopK(self.logits, self.candidates[0..sampling_top_k]);
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};

pub const Int8x4Sampler = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    logits: []f32,
    candidates: [sampling_top_k]Logit,
    quantizer: *QuantizationInt4,
    quantized_query: []i8, // quantized to int8 : no throughput loss, better accuracy

    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix, quant: *QuantizationInt4) !Int8x4Sampler {
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .vocab_size = lm_head.n,
            .d = lm_head.d,
            .logits = try zml_handler.allocator.alloc(f32, lm_head.n),
            .candidates = [_]Logit{.{}} ** sampling_top_k,
            .quantizer = quant,
            .quantized_query = try zml_handler.allocator.alloc(i8, lm_head.d),
        };
    }

    pub fn deinit(self: *Int8x4Sampler) void {
        self.allocator.free(self.logits);
        self.allocator.free(self.quantized_query);
    }

    pub fn sample(self: *Int8x4Sampler, query: []const f32) SamplingResult {
        const query_norm = quantization.normL2(query);
        const query_quant_scale = QuantizationInt8.quantizeVector(query, self.quantizer.buffer, self.quantized_query, query_norm);

        // Compute logits between the int8 query and packed int4 rows.
        const packed_d = self.d / 2;
        for (0..self.vocab_size) |row| {
            const quantized_row = self.quantizer.lm_head_quantized[row * packed_d ..][0..packed_d];
            const res = QuantizationInt4.int8x4dot(self.quantized_query, quantized_row);
            self.logits[row] = @as(f32, @floatFromInt(res)) * self.quantizer.row_quant_scale[row] * query_quant_scale;
        }

        findTopK(self.logits, self.candidates[0..sampling_top_k]);
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};

pub const Int4Sampler = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    logits: []f32,
    candidates: [sampling_top_k]Logit,
    quantizer: *QuantizationInt4,
    quantized_query: []i8,

    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix, quant: *QuantizationInt4) !Int4Sampler {
        std.debug.assert(lm_head.d % 2 == 0);
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .vocab_size = lm_head.n,
            .d = lm_head.d,
            .logits = try zml_handler.allocator.alloc(f32, lm_head.n),
            .candidates = [_]Logit{.{}} ** sampling_top_k,
            .quantizer = quant,
            .quantized_query = try zml_handler.allocator.alloc(i8, lm_head.d / 2),
        };
    }

    pub fn deinit(self: *Int4Sampler) void {
        self.allocator.free(self.logits);
        self.allocator.free(self.quantized_query);
    }

    pub fn sample(self: *Int4Sampler, query: []const f32) SamplingResult {
        const query_norm = quantization.normL2(query);
        const query_quant_scale = QuantizationInt4.quantizeVector(query, self.quantizer.buffer, self.quantized_query, query_norm);

        const packed_d = self.d / 2;
        for (0..self.vocab_size) |row| {
            const quantized_row = self.quantizer.lm_head_quantized[row * packed_d ..][0..packed_d];
            const res = QuantizationInt4.int4dot(self.quantized_query, quantized_row);
            self.logits[row] = @as(f32, @floatFromInt(res)) * self.quantizer.row_quant_scale[row] * query_quant_scale;
        }

        findTopK(self.logits, self.candidates[0..sampling_top_k]);
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};

pub const QJL1Sampler = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    logits: []f32,
    candidates: [sampling_top_k]Logit,
    quantizer: *QuantizationQJL1,
    quantized_query: VectorQJL1,

    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix, quant: *QuantizationQJL1) !QJL1Sampler {
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .vocab_size = lm_head.n,
            .d = lm_head.d,
            .logits = try zml_handler.allocator.alloc(f32, lm_head.n),
            .candidates = [_]Logit{.{}} ** sampling_top_k,
            .quantizer = quant,
            .quantized_query = quantization.makeVectorQJL1(),
        };
    }

    pub fn deinit(self: *QJL1Sampler) void {
        self.allocator.free(self.logits);
    }

    pub fn sample(self: *QJL1Sampler, query: []const f32) SamplingResult {
        const query_norm = quantization.normL2(query);
        _ = QuantizationQJL1.quantizeVector(query, self.quantizer.buffer, &self.quantized_query, query_norm);

        for (0..self.vocab_size) |row| {
            const quantized_row = &self.quantizer.lm_head_quantized[row];
            const res = QuantizationQJL1.qjl1dot(&self.quantized_query, quantized_row);
            self.logits[row] = res * self.lm_head.row_norms[row] * query_norm;
        }

        findTopK(self.logits, self.candidates[0..sampling_top_k]);
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};

pub const QJL2Sampler = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    logits: []f32,
    candidates: [sampling_top_k]Logit,
    quantizer: *QuantizationQJL2,
    quantized_query: VectorQJL2,

    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix, quant: *QuantizationQJL2) !QJL2Sampler {
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .vocab_size = lm_head.n,
            .d = lm_head.d,
            .logits = try zml_handler.allocator.alloc(f32, lm_head.n),
            .candidates = [_]Logit{.{}} ** sampling_top_k,
            .quantizer = quant,
            .quantized_query = quantization.makeVectorQJL2(),
        };
    }

    pub fn deinit(self: *QJL2Sampler) void {
        self.allocator.free(self.logits);
    }

    pub fn sample(self: *QJL2Sampler, query: []const f32) SamplingResult {
        const query_norm = quantization.normL2(query);
        const query_quant_scale = QuantizationQJL2.quantizeVector(query, self.quantizer.buffer, &self.quantized_query, query_norm);

        for (0..self.vocab_size) |row| {
            const quantized_row = &self.quantizer.lm_head_quantized[row];
            const res = QuantizationQJL2.qjl2dot(&self.quantized_query, quantized_row);
            self.logits[row] = @as(f32, @floatFromInt(res)) * self.quantizer.row_quant_scale[row] * query_quant_scale;
        }

        findTopK(self.logits, self.candidates[0..sampling_top_k]);
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};

pub const QJL2x1Sampler = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    logits: []f32,
    candidates: [sampling_top_k]Logit,
    quantizer: *QuantizationQJL1,
    quantized_query: VectorQJL2,

    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix, quant: *QuantizationQJL1) !QJL2x1Sampler {
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .vocab_size = lm_head.n,
            .d = lm_head.d,
            .logits = try zml_handler.allocator.alloc(f32, lm_head.n),
            .candidates = [_]Logit{.{}} ** sampling_top_k,
            .quantizer = quant,
            .quantized_query = quantization.makeVectorQJL2(),
        };
    }

    pub fn deinit(self: *QJL2x1Sampler) void {
        self.allocator.free(self.logits);
    }

    pub fn sample(self: *QJL2x1Sampler, query: []const f32) SamplingResult {
        const query_norm = quantization.normL2(query);
        const query_quant_scale = QuantizationQJL2.quantizeVector(query, self.quantizer.buffer, &self.quantized_query, query_norm);
        for (0..self.vocab_size) |row| {
            const quantized_row = &self.quantizer.lm_head_quantized[row];
            const res = QuantizationQJL2.qjl2x1dot(&self.quantized_query, quantized_row);
            self.logits[row] = @as(f32, @floatFromInt(res)) * self.quantizer.row_quant_scale[row] * query_quant_scale;
        }
        findTopK(self.logits, self.candidates[0..sampling_top_k]);
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};

pub const QJLNx1Sampler = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    logits: []f32,
    candidates: [sampling_top_k]Logit,
    quantizer: *QuantizationQJL1,
    rotated_query: []f32,
    query_lut: [hidden_dim / 8 * 256]f32,

    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix, quant: *QuantizationQJL1) !QJLNx1Sampler {
        const allocator = zml_handler.allocator;
        const logits = try allocator.alloc(f32, lm_head.n);
        errdefer allocator.free(logits);
        const rotated_query = try allocator.alloc(f32, lm_head.d);
        errdefer allocator.free(rotated_query);

        return .{
            .allocator = allocator,
            .lm_head = lm_head,
            .vocab_size = lm_head.n,
            .d = lm_head.d,
            .logits = logits,
            .candidates = [_]Logit{.{}} ** sampling_top_k,
            .quantizer = quant,
            .rotated_query = rotated_query,
            .query_lut = [_]f32{0} ** (hidden_dim / 8 * 256),
        };
    }

    pub fn deinit(self: *QJLNx1Sampler) void {
        self.allocator.free(self.logits);
        self.allocator.free(self.rotated_query);
    }

    pub fn sample(self: *QJLNx1Sampler, query: []const f32) SamplingResult {
        @memcpy(self.rotated_query, query);
        quantization.walshHadamard(self.rotated_query, quantization.hidden_dim_log2);
        const query_sum = self.precomputeLut();
        for (0..self.vocab_size) |row| {
            const quantized_row = &self.quantizer.lm_head_quantized[row];
            const res = QuantizationQJL1.qjlNx1dot(&self.query_lut, quantized_row, query_sum);
            self.logits[row] = res * self.quantizer.row_quant_scale[row];
        }
        findTopK(self.logits, self.candidates[0..sampling_top_k]);
        return .{ .candidates = self.candidates, .nb = 0 };
    }

    pub fn precomputeLut(self: *QJLNx1Sampler) f32 {
        var query_sum: f32 = 0.0;
        for (0..hidden_dim / 8) |byte_i| {
            const values = self.rotated_query[byte_i * 8 ..][0..8];
            const lut = self.query_lut[byte_i * 256 ..][0..256];
            lut[0] = 0.0;
            for (values) |value| query_sum += value;
            for (1..256) |mask| {
                const previous_mask = mask & (mask - 1);
                const bit_i: usize = @ctz(mask);
                lut[mask] = lut[previous_mask] + values[bit_i];
            }
        }
        return query_sum;
    }
};

pub const GraphSampler = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    graph: Graph,
    d: usize,
    vocab_size: usize,
    candidates: [sampling_top_k]Logit,

    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix) !GraphSampler {
        zml_handler.tic(&zml_handler.timers.similarity_matrix);
        var similarity_matrix = try algebra.loadSimilarityMatrix(zml_handler, true);
        defer similarity_matrix.deinit(zml_handler.allocator);
        zml_handler.toc(&zml_handler.timers.similarity_matrix);

        const ang_params: graph_.GraphParams = .{ .graph_type = .Angular };

        var gr: Graph = undefined;
        if (false) {
            std.log.info("Init KNN-angular graph", .{});
            zml_handler.tic(&zml_handler.timers.knn_graph);
            var g_knn: Graph = try .init(zml_handler, lm_head, &similarity_matrix, ang_params);
            g_knn.consolidateNearestPrune();
            zml_handler.toc(&zml_handler.timers.knn_graph);
            gr = g_knn;
        } else {
            std.log.info("Init NSW-angular graph", .{});
            zml_handler.tic(&zml_handler.timers.nsw_graph);
            var g_nsw: Graph = try .init(zml_handler, lm_head, &similarity_matrix, ang_params);
            g_nsw.consolidateNearestPrune();
            try g_nsw.extendToNsw();
            try g_nsw.fixNswExtention();
            g_nsw.consolidateNearestPrune();
            zml_handler.toc(&zml_handler.timers.nsw_graph);
            gr = g_nsw;
        }
        
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .graph = gr,
            .vocab_size = lm_head.n,
            .d = lm_head.d,
            .candidates = [_]Logit{.{}} ** sampling_top_k,
        };
    }

    pub fn deinit(self: *GraphSampler) void {
        self.graph.deinit();
    }

    pub fn sample(self: *GraphSampler, query: []const f32) SamplingResult {
        self.graph.greedySearchPrefetch(query, 16384);
        for (0..sampling_top_k) |i| {
            self.candidates[i].row = self.graph.visited[i].node;
            self.candidates[i].logit = self.graph.visited[i].similarity;
        }
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};