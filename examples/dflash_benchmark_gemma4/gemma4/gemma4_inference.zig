const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const model = @import("gemma4_model.zig");

pub const TargetLayers = struct {
    ids: [32]u32 = undefined,
    len: usize = 0,

    pub fn init(ids: []const u32) TargetLayers {
        stdx.debug.assert(ids.len <= 32, "Gemma 4 DFlash supports at most 32 target layers, got {}", .{ids.len});
        var res: TargetLayers = .{};
        res.len = ids.len;
        @memcpy(res.ids[0..ids.len], ids);
        return res;
    }

    pub fn slice(self: *const TargetLayers) []const u32 {
        return self.ids[0..self.len];
    }
};

const TargetPrefillContext = struct {
    len: u32,
};

const TargetVerifyContext = struct {
    hidden_len: u32,
};

pub fn targetHiddenTensor(target_model: model.Model, config: model.Config, target_layers: TargetLayers, hidden_len: u32) zml.Tensor {
    return .init(.{
        .s = hidden_len,
        .d = target_layers.len * config.hidden_size,
    }, target_model.model.embed_tokens.embed_tokens.weight.dtype());
}

pub fn compileTargetPrefill(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    target_model: model.Model,
    target_layers: TargetLayers,
    prompt_len: u32,
    input_tokens: zml.Tensor,
    active_len: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
    shardings: []const zml.Sharding,
) !zml.Exe {
    return platform.compileFn(
        allocator,
        io,
        targetPrefill,
        .{
            target_model,
            target_layers,
            TargetPrefillContext{ .len = prompt_len },
            input_tokens,
            active_len,
            token_index,
            target_kv_cache,
            rng,
            sampling,
        },
        .{ .shardings = shardings },
    );
}

pub fn compileTargetVerify(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    target_model: model.Model,
    target_layers: TargetLayers,
    hidden_len: u32,
    block_tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const draft_logits = zml.Tensor.init(
        .{ .s = block_tokens.dim(.s), .voc = target_model.model.embed_tokens.embed_tokens.weight.dim(.voc) },
        target_model.model.embed_tokens.embed_tokens.weight.dtype(),
    );
    return platform.compileFn(
        allocator,
        io,
        targetVerify,
        .{
            target_model,
            target_layers,
            TargetVerifyContext{ .hidden_len = hidden_len },
            block_tokens,
            draft_logits,
            token_index,
            target_kv_cache,
            rng,
            sampling,
        },
        .{ .shardings = shardings },
    );
}

fn targetPrefill(
    target_model: model.Model,
    target_layers: TargetLayers,
    context: TargetPrefillContext,
    tokens: zml.Tensor,
    active_len: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
) struct { zml.Tensor, zml.Tensor, model.KvCache, zml.Tensor.Rng } {
    const prefill_tokens = tokens.withPartialTags(.{.s}).slice1d(.s, .{
        .start = 0,
        .end = context.len,
    });
    const target_hidden, const out, const updated_kv_cache = target_model.model.forward(
        prefill_tokens,
        token_index,
        target_kv_cache,
        target_layers.slice(),
    );
    const last_idx = active_len.sub(zml.Tensor.scalar(@as(u32, 1), active_len.dtype()));
    const last = out.withPartialTags(.{ .s, .d }).gather(.{ .s = last_idx }, .{}).reshape(.{ .s = 1, .d = out.dim(.d) });
    const logits = target_model.logitsForward(last).withPartialTags(.{.voc});
    const topk: u32 = if (sampling.temperature < 0.00001) 1 else @intCast(logits.dim(.voc));
    const target_token, const updated_rng = zml.nn.sampleTokens(logits, .{ .topk = topk, .temperature = sampling.temperature }, rng);
    return .{ padTargetHidden(target_hidden, tokens.dim(.s)), target_token.convert(.u32), updated_kv_cache, updated_rng };
}

fn targetVerify(
    target_model: model.Model,
    target_layers: TargetLayers,
    context: TargetVerifyContext,
    tokens: zml.Tensor,
    draft_logits: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    sampling: model.SamplingConfig,
) struct { zml.Tensor, zml.Tensor, zml.Tensor, model.KvCache, zml.Tensor.Rng } {
    const target_hidden, const valid_draft_tokens, const correction_token, const updated_kv_cache, const updated_rng = target_model.verifyForward(
        tokens,
        draft_logits,
        token_index,
        target_kv_cache,
        target_layers.slice(),
        sampling,
        rng,
    );
    return .{ padTargetHidden(target_hidden, context.hidden_len), valid_draft_tokens, correction_token, updated_kv_cache, updated_rng };
}

fn padTargetHidden(target_hidden_: zml.Tensor, hidden_len: i64) zml.Tensor {
    const target_hidden = target_hidden_.withPartialTags(.{ .s, .d });
    if (target_hidden.dim(.s) == hidden_len) return target_hidden;

    stdx.debug.assert(target_hidden.dim(.s) < hidden_len, "target hidden length {} exceeds DFlash block size {}", .{ target_hidden.dim(.s), hidden_len });
    const padding = zml.Tensor.constant(target_hidden.dtype().zero()).broad(.init(.{
        .s = hidden_len - target_hidden.dim(.s),
        .d = target_hidden.dim(.d),
    }, target_hidden.dtype()));
    return zml.Tensor.concatenate(&.{ target_hidden, padding }, .s);
}
