const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const attention = zml.attention.attention;

const model = @import("llama_model.zig");

pub const TargetLayers = struct {
    ids: [32]u32 = undefined,
    len: usize = 0,

    pub fn init(ids: []const u32) TargetLayers {
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

pub const TargetAttention = struct {
    backend: attention.Backend,
    metadata: attention.Metadata,
    parameters: attention.Parameters,

    pub fn init(platform: *const zml.Platform, cache_seq_len: u32, num_attention_heads: u32) TargetAttention {
        const backend = attention.Backend.auto(platform);
        return .{
            .backend = backend,
            .metadata = .init(.fromBackend(backend, cache_seq_len, num_attention_heads)),
            .parameters = .init(.fromBackend(backend)),
        };
    }
};

pub fn targetHiddenTensor(target_model: model.Model, config: model.Config, target_layers: TargetLayers, hidden_len: u32) zml.Tensor {
    return .init(.{
        .s = hidden_len,
        .d = target_layers.len * config.hidden_size,
    }, target_model.model.embed_tokens.weight.dtype());
}

pub fn compileTargetPrefill(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    target_model: model.Model,
    target_layers: TargetLayers,
    prompt_len: u32,
    input_tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    target_attention: TargetAttention,
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
            token_index,
            target_kv_cache,
            target_attention.metadata,
            target_attention.parameters,
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
    target_attention: TargetAttention,
    shardings: []const zml.Sharding,
) !zml.Exe {
    return platform.compileFn(
        allocator,
        io,
        targetVerify,
        .{
            target_model,
            target_layers,
            TargetVerifyContext{ .hidden_len = hidden_len },
            block_tokens,
            token_index,
            target_kv_cache,
            target_attention.metadata,
            target_attention.parameters,
        },
        .{ .shardings = shardings },
    );
}

fn targetPrefill(
    target_model: model.Model,
    target_layers: TargetLayers,
    context: TargetPrefillContext,
    tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
) struct { zml.Tensor, zml.Tensor, model.KvCache } {
    const prefill_tokens = tokens.withPartialTags(.{.s}).slice1d(.s, .{
        .start = 0,
        .end = context.len,
    });
    const target_hidden, const target_token, const updated_kv_cache = target_model.dflashPrefill(
        prefill_tokens,
        token_index,
        target_kv_cache,
        attention_metadata,
        attention_parameters,
        target_layers.slice(),
    );
    return .{ padTargetHidden(target_hidden, tokens.dim(.s)), target_token, updated_kv_cache };
}

fn targetVerify(
    target_model: model.Model,
    target_layers: TargetLayers,
    context: TargetVerifyContext,
    tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: model.KvCache,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
) struct { zml.Tensor, zml.Tensor, model.KvCache } {
    const target_hidden, const target_token, const updated_kv_cache = target_model.dflashVerify(
        tokens,
        token_index,
        target_kv_cache,
        attention_metadata,
        attention_parameters,
        target_layers.slice(),
    );
    return .{ padTargetHidden(target_hidden, context.hidden_len), target_token, updated_kv_cache };
}

pub fn padTargetHidden(target_hidden_: zml.Tensor, hidden_len: i64) zml.Tensor {
    const target_hidden = target_hidden_.withPartialTags(.{ .s, .d });
    if (target_hidden.dim(.s) == hidden_len) return target_hidden;

    stdx.debug.assert(target_hidden.dim(.s) < hidden_len, "target hidden length {} exceeds DFlash block size {}", .{ target_hidden.dim(.s), hidden_len });
    const padding = zml.Tensor.constant(target_hidden.dtype().zero()).broad(.init(.{
        .s = hidden_len - target_hidden.dim(.s),
        .d = target_hidden.dim(.d),
    }, target_hidden.dtype()));
    return zml.Tensor.concatenate(&.{ target_hidden, padding }, .s);
}
