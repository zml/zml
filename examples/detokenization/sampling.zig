const std = @import("std");
const zml = @import("zml");

const main = @import("main.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const tokens = @import("tokens.zig");

const Zml_handler = main.Zml_handler;
const Tokenizer = zml.tokenizer.Tokenizer;

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
    lm_head: zml.Slice,
    d: usize,
    vocab_size: usize,
    sampling_candidates: []SamplingCandidate,
    nb_candidates: usize,
    top_k: usize,

    pub fn init(zml_handler: *Zml_handler, lm_head: zml.Slice) !Sampler {
        const v: usize = @intCast(lm_head.shape.dim(0));
        const d: usize = @intCast(lm_head.shape.dim(1));

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
            .nb_candidates = 0,
            .top_k = 16,
        };
    }

    pub fn deinit(self: *Sampler) void {
        self.tokenizer.deinit();
        self.allocator.free(self.sampling_candidates);
    }

    
    pub fn sample(self: *Sampler, y: []const f32, active_parent: ?[]usize) !void {
        std.log.info("Full sampling", .{});
        self.nb_candidates = 0;
        for (0..self.vocab_size) |token| {
            self.sampling_candidates[self.nb_candidates] = .{
                .token = token,
                .logit = self.computeLogit(token, y),
                .weight = 0.0,
            };
            self.nb_candidates += 1;
        }
        try self.showProbabilityDistribution(active_parent);
    }

    pub fn sampleCoarse(self: *Sampler, y: []const f32, active_parent: []usize) !void {
        std.log.info("Coarse sampling", .{});
        self.nb_candidates = 0;
        for (0..self.vocab_size) |token| {
            if (active_parent[token] != token) continue;
            self.sampling_candidates[self.nb_candidates] = .{
                .token = token,
                .logit = self.computeLogit(token, y),
                .weight = 0.0,
            };
            self.nb_candidates += 1;
        }
        try self.showProbabilityDistribution(null);
    }

    
    pub fn computeLogit(self: *Sampler, token: usize, y: []const f32) f32 {
        const row = self.lm_head.constItems(f32)[token * self.d ..][0..self.d];
        var dot: f32 = 0.0;
        var norm2: f32 = 0.0;
        for (row, y) |weight, value| {
            dot += weight * value;
            norm2 += weight * weight;
        }
        return dot / @sqrt(norm2);
    }

    pub fn showProbabilityDistribution(self: *Sampler, active_parent: ?[]usize) !void {
        const candidates = self.sampling_candidates[0..self.nb_candidates];
        std.mem.sort(SamplingCandidate, candidates, {}, SamplingCandidate.decreasingLogits);

        const cutoff_count = @min(self.top_k, self.nb_candidates);
        const max_logit = candidates[0].logit;
        var total: f32 = 0.0;
        for (candidates[0..cutoff_count]) |*candidate| {
            candidate.weight = @exp(candidate.logit - max_logit);
            total += candidate.weight;
        }

        std.log.info("{s:>10}  {s:>11} {s:>13}  {s}", .{ "token_id", "proba", "logit", "token" });
        std.log.info("{s:>10}  {s:>11} {s:>13}  {s}", .{ "----------", "-----------", "-------------", "-----" });
        for (candidates[0..cutoff_count]) |*candidate| {
            const proba = candidate.weight / total;
            if (proba < 1e-6) break;
            const token_str = try tokens.tokenString(self.tokenizer, candidate.token, self.allocator);
            defer self.allocator.free(token_str);
            if (active_parent) |parent| {
                std.log.info("{d:>10}  {d:>11.3} {d:>13.3}  {s} ({d})", .{ candidate.token, proba, candidate.logit, token_str, parent[candidate.token] });
            } else {
                std.log.info("{d:>10}  {d:>11.3} {d:>13.3}  {s}", .{ candidate.token, proba, candidate.logit, token_str });
            }
        }
    }
};
