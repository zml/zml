const std = @import("std");
const zml = @import("zml");

const main = @import("main.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const tokens = @import("tokens.zig");
const algebra = @import("algebra.zig");

const Zml_handler = main.Zml_handler;
const Tokenizer = zml.tokenizer.Tokenizer;
const LmHeadMatrix = algebra.LmHeadMatrix;

pub const sampling_simd_len: comptime_int = 16;
pub const sampling_top_k: comptime_int = 64;
pub const sampling_top_p: comptime_float = 0.99;

pub const Logit = struct {
    row: usize = std.math.maxInt(usize),
    logit: f32 = 0,
    proba: f32 = 0,
    upper_bound: f32 = 0,

    pub fn decreasingOrder(_: void, lhs: Logit, rhs: Logit) bool { return lhs.logit > rhs.logit; }
};

pub const SamplingResult = struct {
    candidates: [sampling_top_k]Logit,
    nb: usize,
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
        var acc: Vec = @splat(0.0);
        var i: usize = 0;
        while (i < lm_head.d) : (i += sampling_simd_len) {
            const weights: Vec = row[i..][0..sampling_simd_len].*;
            const values: Vec = y[i..][0..sampling_simd_len].*;
            acc = @mulAdd(Vec, weights, values, acc);
        }
        return @reduce(.Add, acc);
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
