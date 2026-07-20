const std = @import("std");
const zml = @import("zml");
const builtin = @import("builtin");

const main = @import("main.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const tokens = @import("tokens.zig");
const algebra = @import("algebra.zig");
const quantization = @import("quantization.zig");

const Zml_handler = main.Zml_handler;
const Tokenizer = zml.tokenizer.Tokenizer;
const LmHeadMatrix = algebra.LmHeadMatrix;
const QuantizationInt8 = quantization.QuantizationInt8;
const QuantizationInt4 = quantization.QuantizationInt4;
const QuantizationQJL1 = quantization.QuantizationQJL1;
const QuantizationQJL2 = quantization.QuantizationQJL2;
const VectorQJL1 = quantization.VectorQJL1;
const VectorQJL2 = quantization.VectorQJL2;

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
        self.hardTopK();
        self.computeProbas();
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

    pub fn hardTopK(self: *Sampler) void {
        for (0..sampling_top_k) |i| {
            self.candidates[i].logit = -std.math.floatMax(f32);
        }
        for (0..self.vocab_size) |token| {
            const logit = self.logits[token];
            var insert_pos: usize = sampling_top_k - 1;
            if (logit <= self.candidates[insert_pos].logit) continue;
            while (insert_pos > 0 and logit > self.candidates[insert_pos - 1].logit) {
                self.candidates[insert_pos] = self.candidates[insert_pos - 1];
                insert_pos -= 1;
            }
            self.candidates[insert_pos] = .{ .row = token, .logit = logit };
        }
    }

    pub fn computeProbas(self: *Sampler) void {
        var max_logit: f32 = self.candidates[0].logit;
        for (1..sampling_top_k) |i| {
            max_logit = @max(max_logit, self.candidates[i].logit);
        }
        var total: f32 = 0.0;
        for (0..sampling_top_k) |i| {
            const proba = @exp(self.candidates[i].logit - max_logit);
            self.candidates[i].proba = proba;
            total += proba;
        }
        const inv_total = 1.0 / total;
        for (0..sampling_top_k) |i| {
            self.candidates[i].proba *= inv_total;
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
        // quantize query
        var query_norm: f32 = 0.0;
        for (0..self.d) |coord| {
            query_norm += query[coord] * query[coord];
        }
        query_norm = @sqrt(query_norm);
        const query_quant_scale = QuantizationInt8.quantizeVector(query, self.quantizer.buffer, self.quantized_query, query_norm);

        // compute logits with int8 dot products
        for (0..self.vocab_size) |row| {
            const quantized_row = self.quantizer.lm_head_quantized[row * self.d ..][0..self.d];
            const res = int8DotProduct(self.quantized_query, quantized_row);
            self.logits[row] = @as(f32, @floatFromInt(res)) * self.quantizer.row_quant_scale[row] * query_quant_scale;
        }

        // find and return top sampling_top_k logits
        for (0..sampling_top_k) |i| {
            self.candidates[i].logit = -std.math.floatMax(f32);
        }
        for (0..self.vocab_size) |token| {
            const logit = self.logits[token];
            var insert_pos: usize = sampling_top_k - 1;
            if (logit <= self.candidates[insert_pos].logit) continue;
            while (insert_pos > 0 and logit > self.candidates[insert_pos - 1].logit) {
                self.candidates[insert_pos] = self.candidates[insert_pos - 1];
                insert_pos -= 1;
            }
            self.candidates[insert_pos] = .{ .row = token, .logit = logit };
        }
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
        // quantize query
        var query_norm: f32 = 0.0;
        for (0..self.d) |coord| {
            query_norm += query[coord] * query[coord];
        }
        query_norm = @sqrt(query_norm);
        const query_quant_scale = QuantizationInt8.quantizeVector(query, self.quantizer.buffer, self.quantized_query, query_norm);

        // Compute logits between the int8 query and packed int4 rows.
        const packed_d = self.d / 2;
        for (0..self.vocab_size) |row| {
            const quantized_row = self.quantizer.lm_head_quantized[row * packed_d ..][0..packed_d];
            const res = int8x4DotProduct(self.quantized_query, quantized_row);
            self.logits[row] = @as(f32, @floatFromInt(res)) * self.quantizer.row_quant_scale[row] * query_quant_scale;
        }

        // find and return top sampling_top_k logits
        for (0..sampling_top_k) |i| {
            self.candidates[i].logit = -std.math.floatMax(f32);
        }
        for (0..self.vocab_size) |token| {
            const logit = self.logits[token];
            var insert_pos: usize = sampling_top_k - 1;
            if (logit <= self.candidates[insert_pos].logit) continue;
            while (insert_pos > 0 and logit > self.candidates[insert_pos - 1].logit) {
                self.candidates[insert_pos] = self.candidates[insert_pos - 1];
                insert_pos -= 1;
            }
            self.candidates[insert_pos] = .{ .row = token, .logit = logit };
        }
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
        var query_norm: f32 = 0.0;
        for (0..self.d) |coord| {
            query_norm += query[coord] * query[coord];
        }
        query_norm = @sqrt(query_norm);
        const query_quant_scale = QuantizationInt4.quantizeVector(query, self.quantizer.buffer, self.quantized_query, query_norm);

        const packed_d = self.d / 2;
        for (0..self.vocab_size) |row| {
            const quantized_row = self.quantizer.lm_head_quantized[row * packed_d ..][0..packed_d];
            const res = int4DotProduct(self.quantized_query, quantized_row);
            self.logits[row] = @as(f32, @floatFromInt(res)) * self.quantizer.row_quant_scale[row] * query_quant_scale;
        }

        for (0..sampling_top_k) |i| {
            self.candidates[i].logit = -std.math.floatMax(f32);
        }
        for (0..self.vocab_size) |token| {
            const logit = self.logits[token];
            var insert_pos: usize = sampling_top_k - 1;
            if (logit <= self.candidates[insert_pos].logit) continue;
            while (insert_pos > 0 and logit > self.candidates[insert_pos - 1].logit) {
                self.candidates[insert_pos] = self.candidates[insert_pos - 1];
                insert_pos -= 1;
            }
            self.candidates[insert_pos] = .{ .row = token, .logit = logit };
        }
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};

const useNeonSdot = builtin.cpu.arch == .aarch64 and
    builtin.zig_backend != .stage2_c and
    (builtin.os.tag == .macos or builtin.cpu.has(.aarch64, .dotprod));

const Vec4i32 = @Vector(4, i32);
const Vec16i8 = @Vector(16, i8);
const Vec16i32 = @Vector(16, i32);
const int4_block_len: comptime_int = 32;
const int4_packed_block_len: comptime_int = int4_block_len / 2;

inline fn neonSdot(acc: Vec4i32, a: Vec16i8, b: Vec16i8) Vec4i32 {
    var result = acc;
    asm volatile (
        \\ sdot %[result].4s, %[a].16b, %[b].16b
        : [result] "+w" (result),
        : [a] "w" (a),
          [b] "w" (b),
    );
    return result;
}

const Int8x4DotAccumulators = struct {
    sum0: Vec4i32,
    sum1: Vec4i32,
    sum2: Vec4i32,
    sum3: Vec4i32,
};

const Int4DotAccumulators = struct {
    sum0: Vec4i32,
    sum1: Vec4i32,
    sum2: Vec4i32,
    sum3: Vec4i32,
};

inline fn signExtendInt4(nibble: u8) i8 {
    const shifted: i8 = @bitCast(nibble << 4);
    return shifted >> 4;
}

/// Unpacks two packed 32-coordinate blocks and accumulates their four SDOTs in
/// one assembly block, keeping all intermediate vectors in SIMD registers.
inline fn neonSdotInt8x4x64(
    sum0: Vec4i32,
    sum1: Vec4i32,
    sum2: Vec4i32,
    sum3: Vec4i32,
    a0: Vec16i8,
    a1: Vec16i8,
    a2: Vec16i8,
    a3: Vec16i8,
    packed0: Vec16i8,
    packed1: Vec16i8,
) Int8x4DotAccumulators {
    var result0 = sum0;
    var result1 = sum1;
    var result2 = sum2;
    var result3 = sum3;
    var high: Vec16i8 = undefined;
    var low: Vec16i8 = undefined;
    asm volatile (
        \\ sshr %[high].16b, %[packed0].16b, #4
        \\ shl %[low].16b, %[packed0].16b, #4
        \\ sshr %[low].16b, %[low].16b, #4
        \\ sdot %[result0].4s, %[a0].16b, %[high].16b
        \\ sdot %[result1].4s, %[a1].16b, %[low].16b
        \\ sshr %[high].16b, %[packed1].16b, #4
        \\ shl %[low].16b, %[packed1].16b, #4
        \\ sshr %[low].16b, %[low].16b, #4
        \\ sdot %[result2].4s, %[a2].16b, %[high].16b
        \\ sdot %[result3].4s, %[a3].16b, %[low].16b
        : [result0] "+&w" (result0),
          [result1] "+&w" (result1),
          [result2] "+&w" (result2),
          [result3] "+&w" (result3),
          [high] "=&w" (high),
          [low] "=&w" (low),
        : [a0] "w" (a0),
          [a1] "w" (a1),
          [a2] "w" (a2),
          [a3] "w" (a3),
          [packed0] "w" (packed0),
          [packed1] "w" (packed1),
    );
    return .{ .sum0 = result0, .sum1 = result1, .sum2 = result2, .sum3 = result3 };
}

/// Computes a dot product between a full int8 vector and an int4 vector packed
/// two coordinates per byte.
inline fn int8x4DotProduct(a: []const i8, b: []const i8) i32 {
    std.debug.assert(a.len == b.len * 2);
    std.debug.assert(a.len >= 64 and a.len % 64 == 0);

    if (comptime useNeonSdot) {
        var sum0: Vec4i32 = @splat(0);
        var sum1: Vec4i32 = @splat(0);
        var sum2: Vec4i32 = @splat(0);
        var sum3: Vec4i32 = @splat(0);

        var i: usize = 0;
        while (i < a.len) : (i += 64) {
            const a0: Vec16i8 = a[i..][0..16].*;
            const a1: Vec16i8 = a[i + 16 ..][0..16].*;
            const a2: Vec16i8 = a[i + 32 ..][0..16].*;
            const a3: Vec16i8 = a[i + 48 ..][0..16].*;

            const packed0: Vec16i8 = b[i / 2 ..][0..int4_packed_block_len].*;
            const packed1: Vec16i8 = b[i / 2 + int4_packed_block_len ..][0..int4_packed_block_len].*;
            const sums = neonSdotInt8x4x64(sum0, sum1, sum2, sum3, a0, a1, a2, a3, packed0, packed1);
            sum0 = sums.sum0;
            sum1 = sums.sum1;
            sum2 = sums.sum2;
            sum3 = sums.sum3;
        }
        return @reduce(.Add, sum0 + sum1 + sum2 + sum3);
    }

    var sum: i32 = 0;
    for (0..a.len / int4_block_len) |block| {
        const coord_base = block * int4_block_len;
        const packed_base = block * int4_packed_block_len;
        for (0..int4_packed_block_len) |lane| {
            const byte: u8 = @bitCast(b[packed_base + lane]);
            const high: i32 = signExtendInt4(byte >> 4);
            const low: i32 = signExtendInt4(byte & 0x0f);
            sum += @as(i32, a[coord_base + lane]) * high;
            sum += @as(i32, a[coord_base + int4_packed_block_len + lane]) * low;
        }
    }
    return sum;
}

/// Computes four SDOTs from two packed int4 query blocks and two packed int4
/// weight blocks without materializing either operand as int8 in memory.
inline fn neonSdotInt4x4x64(
    sum0: Vec4i32,
    sum1: Vec4i32,
    sum2: Vec4i32,
    sum3: Vec4i32,
    packedA0: Vec16i8,
    packedA1: Vec16i8,
    packedB0: Vec16i8,
    packedB1: Vec16i8,
) Int4DotAccumulators {
    var result0 = sum0;
    var result1 = sum1;
    var result2 = sum2;
    var result3 = sum3;
    var aHigh: Vec16i8 = undefined;
    var aLow: Vec16i8 = undefined;
    var bHigh: Vec16i8 = undefined;
    var bLow: Vec16i8 = undefined;
    asm volatile (
        \\ sshr %[a_high].16b, %[packed_a0].16b, #4
        \\ shl %[a_low].16b, %[packed_a0].16b, #4
        \\ sshr %[a_low].16b, %[a_low].16b, #4
        \\ sshr %[b_high].16b, %[packed_b0].16b, #4
        \\ shl %[b_low].16b, %[packed_b0].16b, #4
        \\ sshr %[b_low].16b, %[b_low].16b, #4
        \\ sdot %[result0].4s, %[a_high].16b, %[b_high].16b
        \\ sdot %[result1].4s, %[a_low].16b, %[b_low].16b
        \\ sshr %[a_high].16b, %[packed_a1].16b, #4
        \\ shl %[a_low].16b, %[packed_a1].16b, #4
        \\ sshr %[a_low].16b, %[a_low].16b, #4
        \\ sshr %[b_high].16b, %[packed_b1].16b, #4
        \\ shl %[b_low].16b, %[packed_b1].16b, #4
        \\ sshr %[b_low].16b, %[b_low].16b, #4
        \\ sdot %[result2].4s, %[a_high].16b, %[b_high].16b
        \\ sdot %[result3].4s, %[a_low].16b, %[b_low].16b
        : [result0] "+&w" (result0),
          [result1] "+&w" (result1),
          [result2] "+&w" (result2),
          [result3] "+&w" (result3),
          [a_high] "=&w" (aHigh),
          [a_low] "=&w" (aLow),
          [b_high] "=&w" (bHigh),
          [b_low] "=&w" (bLow),
        : [packed_a0] "w" (packedA0),
          [packed_a1] "w" (packedA1),
          [packed_b0] "w" (packedB0),
          [packed_b1] "w" (packedB1),
    );
    return .{ .sum0 = result0, .sum1 = result1, .sum2 = result2, .sum3 = result3 };
}

/// Computes a dot product between two vectors using the shared packed int4
/// nibble-plane layout.
inline fn int4DotProduct(a: []const i8, b: []const i8) i32 {
    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len >= 2 * int4_packed_block_len and a.len % (2 * int4_packed_block_len) == 0);

    if (comptime useNeonSdot) {
        var sum0: Vec4i32 = @splat(0);
        var sum1: Vec4i32 = @splat(0);
        var sum2: Vec4i32 = @splat(0);
        var sum3: Vec4i32 = @splat(0);

        var i: usize = 0;
        while (i < a.len) : (i += 2 * int4_packed_block_len) {
            const packedA0: Vec16i8 = a[i..][0..int4_packed_block_len].*;
            const packedA1: Vec16i8 = a[i + int4_packed_block_len ..][0..int4_packed_block_len].*;
            const packedB0: Vec16i8 = b[i..][0..int4_packed_block_len].*;
            const packedB1: Vec16i8 = b[i + int4_packed_block_len ..][0..int4_packed_block_len].*;
            const sums = neonSdotInt4x4x64(sum0, sum1, sum2, sum3, packedA0, packedA1, packedB0, packedB1);
            sum0 = sums.sum0;
            sum1 = sums.sum1;
            sum2 = sums.sum2;
            sum3 = sums.sum3;
        }
        return @reduce(.Add, sum0 + sum1 + sum2 + sum3);
    }

    var sum: i32 = 0;
    for (a, b) |packedA, packedB| {
        const aByte: u8 = @bitCast(packedA);
        const bByte: u8 = @bitCast(packedB);
        const aHigh: i32 = signExtendInt4(aByte >> 4);
        const aLow: i32 = signExtendInt4(aByte & 0x0f);
        const bHigh: i32 = signExtendInt4(bByte >> 4);
        const bLow: i32 = signExtendInt4(bByte & 0x0f);
        sum += aHigh * bHigh + aLow * bLow;
    }
    return sum;
}

inline fn int8DotProduct(a: []const i8, b: []const i8) i32 {
    if (comptime useNeonSdot) {
        const unrollLen = 4 * sampling_simd_len;
        std.debug.assert(a.len == b.len and a.len >= unrollLen and a.len % unrollLen == 0);

        var sum0: Vec4i32 = @splat(0);
        var sum1: Vec4i32 = @splat(0);
        var sum2: Vec4i32 = @splat(0);
        var sum3: Vec4i32 = @splat(0);
        var i: usize = 0;
        while (i < a.len) : (i += unrollLen) {
            const a0: Vec16i8 = a[i..][0..sampling_simd_len].*;
            const a1: Vec16i8 = a[i + sampling_simd_len ..][0..sampling_simd_len].*;
            const a2: Vec16i8 = a[i + 2 * sampling_simd_len ..][0..sampling_simd_len].*;
            const a3: Vec16i8 = a[i + 3 * sampling_simd_len ..][0..sampling_simd_len].*;
            const b0: Vec16i8 = b[i..][0..sampling_simd_len].*;
            const b1: Vec16i8 = b[i + sampling_simd_len ..][0..sampling_simd_len].*;
            const b2: Vec16i8 = b[i + 2 * sampling_simd_len ..][0..sampling_simd_len].*;
            const b3: Vec16i8 = b[i + 3 * sampling_simd_len ..][0..sampling_simd_len].*;
            sum0 = neonSdot(sum0, a0, b0);
            sum1 = neonSdot(sum1, a1, b1);
            sum2 = neonSdot(sum2, a2, b2);
            sum3 = neonSdot(sum3, a3, b3);
        }
        return @reduce(.Add, sum0 + sum1 + sum2 + sum3);
    }

    var acc: Vec16i32 = @splat(0);
    var i: usize = 0;
    while (i + sampling_simd_len <= a.len) : (i += sampling_simd_len) {
        const a_vec: Vec16i8 = a[i..][0..sampling_simd_len].*;
        const b_vec: Vec16i8 = b[i..][0..sampling_simd_len].*;
        const a_i32: Vec16i32 = @intCast(a_vec);
        const b_i32: Vec16i32 = @intCast(b_vec);
        acc += a_i32 * b_i32;
    }
    return @reduce(.Add, acc);
}

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
        var query_norm: f32 = 0.0;
        for (0..self.d) |coord| {
            query_norm += query[coord] * query[coord];
        }
        query_norm = @sqrt(query_norm);
        const query_quant_scale = QuantizationQJL1.quantizeVector(query, self.quantizer.buffer, &self.quantized_query, query_norm);
        _ = query_quant_scale;

        for (0..self.vocab_size) |row| {
            const quantized_row = &self.quantizer.lm_head_quantized[row];
            const res = qjl1dot(&self.quantized_query, quantized_row);
            self.logits[row] = res * self.lm_head.row_norms[row] * query_norm;
        }

        for (0..sampling_top_k) |i| {
            self.candidates[i].logit = -std.math.floatMax(f32);
        }
        for (0..self.vocab_size) |token| {
            const logit = self.logits[token];
            var insert_pos: usize = sampling_top_k - 1;
            if (logit <= self.candidates[insert_pos].logit) continue;
            while (insert_pos > 0 and logit > self.candidates[insert_pos - 1].logit) {
                self.candidates[insert_pos] = self.candidates[insert_pos - 1];
                insert_pos -= 1;
            }
            self.candidates[insert_pos] = .{ .row = token, .logit = logit };
        }
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};

inline fn qjl1dot(a: *const VectorQJL1, b: *const VectorQJL1) f32 {
    // dot(u,v) approx = ||u|| * ||v|| * sin(pi/2 * dot(q(u), q(v)) / d)
    // dot(q(u), q(v)) = d - 2 * bit_mismatches,
    // XOR returns 1 where bits differ, 0 where they match
    // this returns sin(pi/2 * dot(q(u), q(v)) / d)
    // we use a LUT for f(x) = sin(pi / 2 * (d - 2 * x) / d),
    // for x integer in [-d, d], as sin is about as expensive as
    // counting the mismatches
    const mismatches = popcountXor(a, b);
    return qjl_dot_lut[mismatches];
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

const qjl_dot_lut = makeQjlDotLut(hidden_dim);

inline fn popcountXor(a: *const VectorQJL1, b: *const VectorQJL1) u32 {
    // Cast the arrays directly to raw byte pointers
    const ptr_a = @as([*]const u8, @ptrCast(a));
    const ptr_b = @as([*]const u8, @ptrCast(b));

    // Define a 128-bit vector type
    const VecType = @Vector(16, u8);
    const vec_a = @as([*]const VecType, @ptrCast(@alignCast(ptr_a)));
    const vec_b = @as([*]const VecType, @ptrCast(@alignCast(ptr_b)));

    // 4 independent 16-bit accumulators to prevent overflow
    var sum0: @Vector(16, u16) = @splat(0);
    var sum1: @Vector(16, u16) = @splat(0);
    var sum2: @Vector(16, u16) = @splat(0);
    var sum3: @Vector(16, u16) = @splat(0);

    var i: usize = 0;
    var vec_idx: usize = 0;

    // Process 512 bits per iteration
    while (i + 64 <= @sizeOf(VectorQJL1)) {

        // Inner 8-bit accumulators
        var inner_sum0: VecType = @splat(0);
        var inner_sum1: VecType = @splat(0);
        var inner_sum2: VecType = @splat(0);
        var inner_sum3: VecType = @splat(0);

        var inner_iters: usize = 0;

        // Accumulate up to 30 times safely without 8-bit overflow (30 * 8 = 240 < 255)
        while (inner_iters < 30 and i + 64 <= @sizeOf(VectorQJL1)) {
            const a0 = vec_a[vec_idx + 0];
            const a1 = vec_a[vec_idx + 1];
            const a2 = vec_a[vec_idx + 2];
            const a3 = vec_a[vec_idx + 3];

            const b0 = vec_b[vec_idx + 0];
            const b1 = vec_b[vec_idx + 1];
            const b2 = vec_b[vec_idx + 2];
            const b3 = vec_b[vec_idx + 3];

            inner_sum0 += @popCount(a0 ^ b0);
            inner_sum1 += @popCount(a1 ^ b1);
            inner_sum2 += @popCount(a2 ^ b2);
            inner_sum3 += @popCount(a3 ^ b3);

            inner_iters += 1;
            vec_idx += 4;
            i += 64;
        }

        // 3. Widen the inner sums to 16-bit and add to main accumulators
        // This only executes once every 1,920 bytes, saving CPU time
        sum0 += @as(@Vector(16, u16), inner_sum0);
        sum1 += @as(@Vector(16, u16), inner_sum1);
        sum2 += @as(@Vector(16, u16), inner_sum2);
        sum3 += @as(@Vector(16, u16), inner_sum3);
    }

    // 4. Horizontal reduction of the 16-bit vector accumulators
    const total_vec = sum0 + sum1 + sum2 + sum3;
    return @reduce(.Add, total_vec);
}

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
        var query_norm: f32 = 0.0;
        for (0..self.d) |coord| {
            query_norm += query[coord] * query[coord];
        }
        query_norm = @sqrt(query_norm);
        const query_quant_scale = QuantizationQJL2.quantizeVector(query, self.quantizer.buffer, &self.quantized_query, query_norm);

        for (0..self.vocab_size) |row| {
            const quantized_row = &self.quantizer.lm_head_quantized[row];
            const res = qjl2dot(&self.quantized_query, quantized_row);
            self.logits[row] = @as(f32, @floatFromInt(res)) * self.quantizer.row_quant_scale[row] * query_quant_scale;
        }

        for (0..sampling_top_k) |i| {
            self.candidates[i].logit = -std.math.floatMax(f32);
        }
        for (0..self.vocab_size) |token| {
            const logit = self.logits[token];
            var insert_pos: usize = sampling_top_k - 1;
            if (logit <= self.candidates[insert_pos].logit) continue;
            while (insert_pos > 0 and logit > self.candidates[insert_pos - 1].logit) {
                self.candidates[insert_pos] = self.candidates[insert_pos - 1];
                insert_pos -= 1;
            }
            self.candidates[insert_pos] = .{ .row = token, .logit = logit };
        }
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};

pub inline fn qjl2dot(a: *const VectorQJL2, b: *const VectorQJL2) i32 {
    // A quantized value is represented as 2 * sign(msb[i]) + 1 * sign(lsb[i])
    // dot(a, b) = dot(2 * sign(a.msb[i]) + 1 * sign(a.lsb[i]), 2 * sign(b.msb[i]) + 1 * sign(b.lsb[i]))
    // dot(a,b) = 4 * dot1(a.msb, b.msb) + 2 * dot1(a.msb, b.lsb) + 2 * dot1(a.lsb, b.msb) + dot1(a.lsb, b.lsb)

    const pop_mm = popcountXor(&@bitCast(a.msb), &@bitCast(b.msb));
    const pop_ml = popcountXor(&@bitCast(a.msb), &@bitCast(b.lsb));
    const pop_lm = popcountXor(&@bitCast(a.lsb), &@bitCast(b.msb));
    const pop_ll = popcountXor(&@bitCast(a.lsb), &@bitCast(b.lsb));

    // dot(a,b) = 4 * (d - 2 * pop_mm) + 2 * (d - 2 * pop_ml + d - 2 * pop_lm) + (d - 2 * pop_ll)
    // dot(a,b) = 9 * d - 8 * pop_mm - 4 * pop_ml - 4 * pop_lm - 2 * pop_ll

    const pos: i32 = 9 * hidden_dim;
    const neg = (pop_mm << 3) + ((pop_ml + pop_lm) << 2) + (pop_ll << 1);
    return pos - @as(i32, @intCast(neg));
}

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
        var query_norm: f32 = 0.0;
        for (0..self.d) |coord| {
            query_norm += query[coord] * query[coord];
        }
        query_norm = @sqrt(query_norm);
        const query_quant_scale = QuantizationQJL2.quantizeVector(query, self.quantizer.buffer, &self.quantized_query, query_norm);
        for (0..self.vocab_size) |row| {
            const quantized_row = &self.quantizer.lm_head_quantized[row];
            const res = qjl2x1dot(&self.quantized_query, quantized_row);
            self.logits[row] = @as(f32, @floatFromInt(res)) * self.quantizer.row_quant_scale[row] * query_quant_scale;
        }
        for (0..sampling_top_k) |i| {
            self.candidates[i].logit = -std.math.floatMax(f32);
        }
        for (0..self.vocab_size) |token| {
            const logit = self.logits[token];
            var insert_pos: usize = sampling_top_k - 1;
            if (logit <= self.candidates[insert_pos].logit) continue;
            while (insert_pos > 0 and logit > self.candidates[insert_pos - 1].logit) {
                self.candidates[insert_pos] = self.candidates[insert_pos - 1];
                insert_pos -= 1;
            }
            self.candidates[insert_pos] = .{ .row = token, .logit = logit };
        }
        return .{ .candidates = self.candidates, .nb = 0 };
    }
};

pub inline fn qjl2x1dot(a: *const VectorQJL2, b: *const VectorQJL1) i32 {
    // A 2 bits quantized value is represented as 2 * sign(msb[i]) + 1 * sign(lsb[i])
    // dot(a, b) = dot(2 * sign(a.msb[i]) + 1 * sign(a.lsb[i]), sign(b))
    // dot(a,b) = 2 * dot1(a.msb, b) + 2 * dot1(a.lsb, b)

    const pop_m = popcountXor(&@bitCast(a.msb), b);
    const pop_l = popcountXor(&@bitCast(a.lsb), b);

    // dot(a,b) = 2 * (d - 2 * pop_m) + 2 * (d - 2 * pop_l)
    // dot(a,b) = 4 * d - 4 * pop_m - 4 * pop_l

    const pos: i32 = 4 * hidden_dim;
    const neg = (pop_m + pop_l) << 2;
    return pos - @as(i32, @intCast(neg));
}

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
            const res = qjlNx1dot(&self.query_lut, quantized_row, query_sum);
            self.logits[row] = res * self.quantizer.row_quant_scale[row];
        }
        for (0..sampling_top_k) |i| {
            self.candidates[i].logit = -std.math.floatMax(f32);
        }
        for (0..self.vocab_size) |token| {
            const logit = self.logits[token];
            var insert_pos: usize = sampling_top_k - 1;
            if (logit <= self.candidates[insert_pos].logit) continue;
            while (insert_pos > 0 and logit > self.candidates[insert_pos - 1].logit) {
                self.candidates[insert_pos] = self.candidates[insert_pos - 1];
                insert_pos -= 1;
            }
            self.candidates[insert_pos] = .{ .row = token, .logit = logit };
        }
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

pub inline fn qjlNx1dot(query_lut: []const f32, b: *const VectorQJL1, query_sum: f32) f32 {
    // dot(q, v) ~= ||rot(v)||_1 / D * dot(q, quant(rot(v)))
    // dot(q, quant(rot(v))) is hard to compute, the current
    // version is about 30% slower than the f32xf32 dot product.
    // with "usine à gaz" simd/masking/lut/assembly injection,
    // we can only get on par with the baseline.
    // TODO: on AVX512 CPUs, there is bitmasking available that
    // would allow to solve the issue at the hardware level,
    // effectively making this being a SIMD vectorized sum,
    // saving the multiplications of the f32xf32.
    var positive_sum: f32 = 0.0;
    const b_bytes: *const [hidden_dim / 8]u8 = @ptrCast(b);
    for (b_bytes, 0..) |byte, byte_i| {
        positive_sum += query_lut[byte_i * 256 + byte];
    }
    return 2.0 * positive_sum - query_sum;
}
