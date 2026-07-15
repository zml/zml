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

const simd_len: comptime_int = 8;
const Vec = @Vector(simd_len, f32);

pub const Logit = struct {
    row: usize,
    logit: f32,
    upper_bound: f32,
};

pub const Sampler = struct {
    const SamplingCandidate = struct {
        token: usize,
        logit: f32,
        weight: f32,

        fn decreasingLogits(_: void, lhs: SamplingCandidate, rhs: SamplingCandidate) bool {
            return lhs.logit > rhs.logit;
        }
    };

    allocator: std.mem.Allocator,
    tokenizer: Tokenizer,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    sampling_candidates: []SamplingCandidate,
    top_k: usize,
    

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix) !Sampler {
        const v: usize = lm_head.n;
        const d: usize = lm_head.d;

        const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
        const tokenizer = try llm_.Llm_handler.loadTokenizer(zml_handler, repo);

        const sampling_candidates = try zml_handler.allocator.alloc(SamplingCandidate, v);

        return .{
            .allocator = zml_handler.allocator,
            .tokenizer = tokenizer,
            .lm_head = lm_head,
            .d = d,
            .vocab_size = v,
            .sampling_candidates = sampling_candidates,
            .top_k = 16,
        };
    }

    pub fn deinit(self: *Sampler) void {
        self.tokenizer.deinit();
        self.allocator.free(self.sampling_candidates);
    }

    
    pub fn sample(self: *Sampler, y: []const f32) void {
        // computes all exact logits,
        // computes probability distribution for self.top_k tokens
        for (0..self.vocab_size) |token| {
            self.sampling_candidates[token] = .{
                .token = token,
                .logit = self.computeLogit(token, y),
                .weight = 0.0,
            };
        }
        const candidates = self.sampling_candidates[0..self.vocab_size];
        std.mem.sort(SamplingCandidate, candidates, {}, SamplingCandidate.decreasingLogits);
        const cutoff_count = self.top_k;
        const max_logit = candidates[0].logit;
        var total: f32 = 0.0;
        for (candidates[0..cutoff_count]) |*candidate| {
            candidate.weight = @exp(candidate.logit - max_logit);
            total += candidate.weight;
        }
        const inv_total = 1.0 / total;
        for (candidates[0..cutoff_count]) |*candidate| {
            candidate.weight *= inv_total;
        }
    }

    pub fn computeLogit(self: *Sampler, token: usize, y: []const f32) f32 {
        const row = self.lm_head.data[token * self.d ..][0..self.d];
        var acc: Vec = @splat(0.0);
        var i: usize = 0;
        while (i < self.d) : (i += simd_len) {
            const weights: Vec = row[i..][0..simd_len].*;
            const values: Vec = y[i..][0..simd_len].*;
            acc = @mulAdd(Vec, weights, values, acc);
        }
        return @reduce(.Add, acc);
    }

    
    pub fn logSampling(self: *Sampler) !void {
        const count = @min(self.top_k, self.nb_candidates);
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s}", .{ "rank", "token_id", "logit", "proba", "token" });
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s}", .{ "------", "----------", "--------------", "--------------", "-----" });
        for (self.sampling_candidates[0..count], 0..) |candidate, rank| {
            var decoded_buf: [512]u8 = undefined;
            var escaped_buf: [512]u8 = undefined;
            const token_text = self.escapedToken(candidate.token, &decoded_buf, &escaped_buf);
            std.log.info("{d:>6}  {d:>10}  {d:>14.6}  {d:>14.6}  {s}", .{ rank + 1, candidate.token, candidate.logit, candidate.weight, token_text });
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
