const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;
const main = @import("../main.zig");
const dialects = @import("mlir/dialects");

pub const TokenIds = stdx.json.Union(union(enum) {
    int: u32,
    ints: []u32,
});

pub fn dupeTokenIds(token_ids: TokenIds, allocator: std.mem.Allocator) !TokenIds {
    return switch (token_ids.value) {
        .int => .{ .value = .{ .int = token_ids.value.int } },
        .ints => .{ .value = .{ .ints = try allocator.dupe(u32, token_ids.value.ints) } },
    };
}

pub fn deinitTokenIds(token_ids: TokenIds, allocator: std.mem.Allocator) void {
    switch (token_ids.value) {
        .int => {},
        .ints => allocator.free(token_ids.value.ints),
    }
}

pub fn tokenIdsContain(token_ids: TokenIds, token_id: u32) bool {
    return switch (token_ids.value) {
        .int => |eos| token_id == eos,
        .ints => |eos_list| for (eos_list) |eos| {
            if (token_id == eos) break true;
        } else false,
    };
}


pub const LayerType = enum {
    sliding_attention,
    full_attention,
};

pub const RopeParameters = struct {
    full_attention: zml.nn.RopeOpts.Scaling,
    sliding_attention: zml.nn.RopeOpts.Scaling,
};

pub const ModelConfig = struct {
    text_config: Config,
};

pub const Config = struct {
    bos_token_id: u32,
    eos_token_id: TokenIds,
    pad_token_id: u32,
    head_dim: u32,
    global_head_dim: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    num_global_key_value_heads: u32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    attention_k_eq_v: bool,
    sliding_window: u32,
    final_logit_softcapping: ?f32 = null,
    layer_types: []LayerType,
    rope_parameters: RopeParameters,
    vocab_size: u32,

    pub fn dupe(self: Config, allocator: std.mem.Allocator) !Config {
        return .{
            .bos_token_id = self.bos_token_id,
            .eos_token_id = try dupeTokenIds(self.eos_token_id, allocator),
            .pad_token_id = self.pad_token_id,
            .head_dim = self.head_dim,
            .global_head_dim = self.global_head_dim,
            .hidden_size = self.hidden_size,
            .intermediate_size = self.intermediate_size,
            .num_hidden_layers = self.num_hidden_layers,
            .num_attention_heads = self.num_attention_heads,
            .num_key_value_heads = self.num_key_value_heads,
            .num_global_key_value_heads = self.num_global_key_value_heads,
            .max_position_embeddings = self.max_position_embeddings,
            .rms_norm_eps = self.rms_norm_eps,
            .tie_word_embeddings = self.tie_word_embeddings,
            .attention_k_eq_v = self.attention_k_eq_v,
            .sliding_window = self.sliding_window,
            .final_logit_softcapping = self.final_logit_softcapping,
            .layer_types = try allocator.dupe(LayerType, self.layer_types),
            .rope_parameters = self.rope_parameters,
            .vocab_size = self.vocab_size,
        };
    }

    pub fn deinit(self: Config, allocator: std.mem.Allocator) void {
        deinitTokenIds(self.eos_token_id, allocator);
        allocator.free(self.layer_types);
    }
};

pub const GenerationConfig = struct {
    bos_token_id: u32,
    do_sample: bool = false,
    eos_token_id: TokenIds,
    pad_token_id: u32,
    temperature: f32 = 1.0,
    top_k: u32 = 50,
    top_p: f32 = 1.0,
    min_p: f32 = 0.0,
    suppress_tokens: []u32 = &.{},

    pub const max_sampling_top_k: u32 = 64;

    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, model_config: Config) !GenerationConfig {
        const file = dir.openFile(io, "generation_config.json", .{}) catch |err| {
            if (err == error.FileNotFound) return GenerationConfig.fromModelConfig(model_config, allocator);
            return err;
        };
        defer file.close(io);

        var buffer: [256]u8 = undefined;
        var file_reader = file.reader(io, &buffer);
        var reader: std.json.Reader = .init(allocator, &file_reader.interface);
        defer reader.deinit();

        const parsed = try std.json.parseFromTokenSource(GenerationConfig, allocator, &reader, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        return parsed.value.dupe(allocator);
    }

    pub fn fromModelConfig(model_config: Config, allocator: std.mem.Allocator) !GenerationConfig {
        return .{
            .bos_token_id = model_config.bos_token_id,
            .do_sample = false,
            .eos_token_id = try dupeTokenIds(model_config.eos_token_id, allocator),
            .pad_token_id = model_config.pad_token_id,
            .suppress_tokens = try allocator.alloc(u32, 0),
        };
    }

    pub fn dupe(self: GenerationConfig, allocator: std.mem.Allocator) !GenerationConfig {
        return .{
            .bos_token_id = self.bos_token_id,
            .do_sample = self.do_sample,
            .eos_token_id = try dupeTokenIds(self.eos_token_id, allocator),
            .pad_token_id = self.pad_token_id,
            .temperature = self.temperature,
            .top_k = self.top_k,
            .top_p = self.top_p,
            .min_p = self.min_p,
            .suppress_tokens = try allocator.dupe(u32, self.suppress_tokens),
        };
    }

    pub fn deinit(self: GenerationConfig, allocator: std.mem.Allocator) void {
        deinitTokenIds(self.eos_token_id, allocator);
        allocator.free(self.suppress_tokens);
    }

    pub fn samplingStrategy(self: GenerationConfig) zml.nn.SamplingStrategy {
        // zml.nn.SamplingStrategy currently supports top-k and temperature; top_p is parsed for config fidelity.
        if (!self.do_sample) return .{};
        return .{
            .topk = self.top_k,
            .temperature = self.temperature,
        };
    }

    pub fn samplingStrategyBuffers(self: GenerationConfig, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(zml.nn.DynamicSamplingStrategy) {
        const top_k = if (self.do_sample) @min(self.top_k, max_sampling_top_k) else 1;
        return try zml.nn.DynamicSamplingStrategy.makeBuffers(io, platform, .f32, .{
            .top_k = top_k,
            .temperature = self.temperature,
            .top_p = if (self.do_sample) self.top_p else 1.0,
            .min_p = if (self.do_sample) self.min_p else 0.0,
        });
    }

    pub fn isEosToken(self: GenerationConfig, token_id: u32) bool {
        return tokenIdsContain(self.eos_token_id, token_id);
    }
};

pub const Options = struct {
    seq_len: u32,
    hidden_size: u32,
    intermediate_size: u32,
    voc_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    num_global_key_value_heads: u32,
    global_head_dim: u32,
    num_local_layers: u32,
    num_global_layers: u32,
};


pub const KvCache = struct {
    local_k: zml.Tensor,
    local_v: zml.Tensor,
    global_k: zml.Tensor,
    global_v: zml.Tensor,

    pub fn init(local_shape: zml.Shape, global_shape: zml.Shape) KvCache {
        return .{
            .local_k = .fromShape(local_shape),
            .local_v = .fromShape(local_shape),
            .global_k = .fromShape(global_shape),
            .global_v = .fromShape(global_shape),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(KvCache) {
        var local_k = try zml.Buffer.uninitialized(io, platform, self.local_k.shape(), sharding, .{});
        errdefer local_k.deinit();
        var local_v = try zml.Buffer.uninitialized(io, platform, self.local_v.shape(), sharding, .{});
        errdefer local_v.deinit();
        var global_k = try zml.Buffer.uninitialized(io, platform, self.global_k.shape(), sharding, .{});
        errdefer global_k.deinit();
        return .{
            .local_k = local_k,
            .local_v = local_v,
            .global_k = global_k,
            .global_v = try zml.Buffer.uninitialized(io, platform, self.global_v.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.local_k.deinit();
        self.local_v.deinit();
        self.global_k.deinit();
        self.global_v.deinit();
    }

    pub fn keys(self: KvCache, is_global: bool, layer_index: zml.Tensor) zml.Tensor {
        const k = if (is_global) self.global_k else self.local_k;
        return k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache, is_global: bool, layer_index: zml.Tensor) zml.Tensor {
        const v = if (is_global) self.global_v else self.local_v;
        return v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, is_global: bool, layer_index: zml.Tensor, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
        const old_k = if (is_global) self.global_k else self.local_k;
        const old_v = if (is_global) self.global_v else self.local_v;
        const k_shape = old_k.shape().drop(.layer);
        const v_shape = old_v.shape().drop(.layer);
        var layer = layer_index;
        layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;
        const updated_k = if (token_index) |idx|
            old_k.scatterSlices(.{ .layer = layer, .k = idx }, new_k.convert(old_k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(old_k)
        else
            old_k.scatterSlices(.{ .layer = layer }, new_k.convert(old_k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(old_k);
        const updated_v = if (token_index) |idx|
            old_v.scatterSlices(.{ .layer = layer, .k = idx }, new_v.convert(old_v.dtype()).transpose(v_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(old_v)
        else
            old_v.scatterSlices(.{ .layer = layer }, new_v.convert(old_v.dtype()).transpose(v_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(old_v);

        var result = self;
        if (is_global) {
            result.global_k = updated_k;
            result.global_v = updated_v;
        } else {
            result.local_k = updated_k;
            result.local_v = updated_v;
        }
        return result;
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .local_k = self.local_k.reuseBuffer(other.local_k),
            .local_v = self.local_v.reuseBuffer(other.local_v),
            .global_k = self.global_k.reuseBuffer(other.global_k),
            .global_v = self.global_v.reuseBuffer(other.global_v),
        };
    }
};

pub const ActivationCache = struct {
    pub const capacity: u32 = 256;
    pub const Slices = zml.meta.MapRestrict(zml.Tensor, zml.Slice).map(ActivationCache);

    local_layer_input: zml.Tensor,
    local_input_norm: zml.Tensor,
    local_q: zml.Tensor,
    local_k: zml.Tensor,
    local_v: zml.Tensor,
    local_attention_context: zml.Tensor,
    local_attention_output: zml.Tensor,
    local_post_attention_residual: zml.Tensor,
    local_pre_ff_norm: zml.Tensor,
    local_gate: zml.Tensor,
    local_up: zml.Tensor,
    local_geglu: zml.Tensor,
    local_post_mlp_residual: zml.Tensor,

    global_layer_input: zml.Tensor,
    global_input_norm: zml.Tensor,
    global_q: zml.Tensor,
    global_k: zml.Tensor,
    global_v: zml.Tensor,
    global_attention_context: zml.Tensor,
    global_attention_output: zml.Tensor,
    global_post_attention_residual: zml.Tensor,
    global_pre_ff_norm: zml.Tensor,
    global_gate: zml.Tensor,
    global_up: zml.Tensor,
    global_geglu: zml.Tensor,
    global_post_mlp_residual: zml.Tensor,

    pub const Values = struct {
        layer_input: zml.Tensor,
        input_norm: zml.Tensor,
        q: zml.Tensor,
        k: zml.Tensor,
        v: zml.Tensor,
        attention_context: zml.Tensor,
        attention_output: zml.Tensor,
        post_attention_residual: zml.Tensor,
        pre_ff_norm: zml.Tensor,
        gate: zml.Tensor,
        up: zml.Tensor,
        geglu: zml.Tensor,
        post_mlp_residual: zml.Tensor,
    };

    pub fn init(options: Options) ActivationCache {
        const local_hidden = zml.Shape.init(.{ .layer = options.num_local_layers, .a = capacity, .d = options.hidden_size }, .bf16);
        const local_q = zml.Shape.init(.{ .layer = options.num_local_layers, .a = capacity, .h = options.num_attention_heads, .hd = options.head_dim }, .bf16);
        const local_kv = zml.Shape.init(.{ .layer = options.num_local_layers, .a = capacity, .h = options.num_key_value_heads, .hd = options.head_dim }, .bf16);
        const local_intermediate = zml.Shape.init(.{ .layer = options.num_local_layers, .a = capacity, .i = options.intermediate_size }, .bf16);

        const global_hidden = zml.Shape.init(.{ .layer = options.num_global_layers, .a = capacity, .d = options.hidden_size }, .bf16);
        const global_q = zml.Shape.init(.{ .layer = options.num_global_layers, .a = capacity, .h = options.num_attention_heads, .hd = options.global_head_dim }, .bf16);
        const global_kv = zml.Shape.init(.{ .layer = options.num_global_layers, .a = capacity, .h = options.num_global_key_value_heads, .hd = options.global_head_dim }, .bf16);
        const global_intermediate = zml.Shape.init(.{ .layer = options.num_global_layers, .a = capacity, .i = options.intermediate_size }, .bf16);

        return .{
            .local_layer_input = .fromShape(local_hidden),
            .local_input_norm = .fromShape(local_hidden),
            .local_q = .fromShape(local_q),
            .local_k = .fromShape(local_kv),
            .local_v = .fromShape(local_kv),
            .local_attention_context = .fromShape(local_q),
            .local_attention_output = .fromShape(local_hidden),
            .local_post_attention_residual = .fromShape(local_hidden),
            .local_pre_ff_norm = .fromShape(local_hidden),
            .local_gate = .fromShape(local_intermediate),
            .local_up = .fromShape(local_intermediate),
            .local_geglu = .fromShape(local_intermediate),
            .local_post_mlp_residual = .fromShape(local_hidden),

            .global_layer_input = .fromShape(global_hidden),
            .global_input_norm = .fromShape(global_hidden),
            .global_q = .fromShape(global_q),
            .global_k = .fromShape(global_kv),
            .global_v = .fromShape(global_kv),
            .global_attention_context = .fromShape(global_q),
            .global_attention_output = .fromShape(global_hidden),
            .global_post_attention_residual = .fromShape(global_hidden),
            .global_pre_ff_norm = .fromShape(global_hidden),
            .global_gate = .fromShape(global_intermediate),
            .global_up = .fromShape(global_intermediate),
            .global_geglu = .fromShape(global_intermediate),
            .global_post_mlp_residual = .fromShape(global_hidden),
        };
    }

    pub fn initBuffer(self: ActivationCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(ActivationCache) {
        @setEvalBranchQuota(10_000);
        var result: zml.Bufferized(ActivationCache) = undefined;
        var initialized: usize = 0;
        errdefer inline for (std.meta.fields(ActivationCache), 0..) |field, i| {
            if (initialized > i) @field(result, field.name).deinit();
        };
        inline for (std.meta.fields(ActivationCache)) |field| {
            const tensor = @field(self, field.name);
            @field(result, field.name) = try zml.Buffer.uninitialized(io, platform, tensor.shape(), sharding, .{});
            initialized += 1;
        }
        return result;
    }

    pub fn deinitBuffer(self: *zml.Bufferized(ActivationCache)) void {
        @setEvalBranchQuota(10_000);
        inline for (std.meta.fields(ActivationCache)) |field| {
            @field(self, field.name).deinit();
        }
    }

    pub fn copyToHost(self: ActivationCache, buffers: *const zml.Bufferized(ActivationCache), allocator: std.mem.Allocator, io: std.Io) !Slices {
        @setEvalBranchQuota(10_000);
        var result: Slices = undefined;
        var initialized: usize = 0;
        errdefer inline for (std.meta.fields(ActivationCache), 0..) |field, i| {
            if (initialized > i) @field(result, field.name).free(allocator);
        };
        inline for (std.meta.fields(ActivationCache)) |field| {
            const slice = try zml.Slice.alloc(allocator, @field(self, field.name).shape());
            @field(result, field.name) = slice;
            initialized += 1;
            try @field(buffers, field.name).toSlice(io, slice);
        }
        return result;
    }

    pub fn deinitSlices(slices: *Slices, allocator: std.mem.Allocator) void {
        @setEvalBranchQuota(10_000);
        inline for (std.meta.fields(ActivationCache)) |field| {
            @field(slices, field.name).free(allocator);
        }
    }

    pub fn update(self: ActivationCache, is_global: bool, layer_index: zml.Tensor, activation_index: zml.Tensor, values: Values) ActivationCache {
        var result = self;
        if (is_global) {
            result.global_layer_input = updateTensor(self.global_layer_input, values.layer_input.rename(.{ .s = .a }), layer_index, activation_index);
            result.global_input_norm = updateTensor(self.global_input_norm, values.input_norm.rename(.{ .s = .a }), layer_index, activation_index);
            result.global_q = updateTensor(self.global_q, values.q.rename(.{ .q = .a }), layer_index, activation_index);
            result.global_k = updateTensor(self.global_k, values.k.rename(.{ .k = .a }), layer_index, activation_index);
            result.global_v = updateTensor(self.global_v, values.v.rename(.{ .k = .a }), layer_index, activation_index);
            result.global_attention_context = updateTensor(self.global_attention_context, values.attention_context.rename(.{ .q = .a }), layer_index, activation_index);
            result.global_attention_output = updateTensor(self.global_attention_output, values.attention_output.rename(.{ .s = .a }), layer_index, activation_index);
            result.global_post_attention_residual = updateTensor(self.global_post_attention_residual, values.post_attention_residual.rename(.{ .s = .a }), layer_index, activation_index);
            result.global_pre_ff_norm = updateTensor(self.global_pre_ff_norm, values.pre_ff_norm.rename(.{ .s = .a }), layer_index, activation_index);
            result.global_gate = updateTensor(self.global_gate, values.gate.rename(.{ .s = .a, .d_out = .i }), layer_index, activation_index);
            result.global_up = updateTensor(self.global_up, values.up.rename(.{ .s = .a, .d_out = .i }), layer_index, activation_index);
            result.global_geglu = updateTensor(self.global_geglu, values.geglu.rename(.{ .s = .a, .d_out = .i }), layer_index, activation_index);
            result.global_post_mlp_residual = updateTensor(self.global_post_mlp_residual, values.post_mlp_residual.rename(.{ .s = .a }), layer_index, activation_index);
        } else {
            result.local_layer_input = updateTensor(self.local_layer_input, values.layer_input.rename(.{ .s = .a }), layer_index, activation_index);
            result.local_input_norm = updateTensor(self.local_input_norm, values.input_norm.rename(.{ .s = .a }), layer_index, activation_index);
            result.local_q = updateTensor(self.local_q, values.q.rename(.{ .q = .a }), layer_index, activation_index);
            result.local_k = updateTensor(self.local_k, values.k.rename(.{ .k = .a }), layer_index, activation_index);
            result.local_v = updateTensor(self.local_v, values.v.rename(.{ .k = .a }), layer_index, activation_index);
            result.local_attention_context = updateTensor(self.local_attention_context, values.attention_context.rename(.{ .q = .a }), layer_index, activation_index);
            result.local_attention_output = updateTensor(self.local_attention_output, values.attention_output.rename(.{ .s = .a }), layer_index, activation_index);
            result.local_post_attention_residual = updateTensor(self.local_post_attention_residual, values.post_attention_residual.rename(.{ .s = .a }), layer_index, activation_index);
            result.local_pre_ff_norm = updateTensor(self.local_pre_ff_norm, values.pre_ff_norm.rename(.{ .s = .a }), layer_index, activation_index);
            result.local_gate = updateTensor(self.local_gate, values.gate.rename(.{ .s = .a, .d_out = .i }), layer_index, activation_index);
            result.local_up = updateTensor(self.local_up, values.up.rename(.{ .s = .a, .d_out = .i }), layer_index, activation_index);
            result.local_geglu = updateTensor(self.local_geglu, values.geglu.rename(.{ .s = .a, .d_out = .i }), layer_index, activation_index);
            result.local_post_mlp_residual = updateTensor(self.local_post_mlp_residual, values.post_mlp_residual.rename(.{ .s = .a }), layer_index, activation_index);
        }
        return result;
    }

    pub fn updateTensor(old: zml.Tensor, new_: zml.Tensor, layer_index: zml.Tensor, activation_index: zml.Tensor) zml.Tensor {
        const new = if (new_.dim(.a) > capacity)
            new_.slice1d(.a, .{ .start = 0, .end = capacity })
        else
            new_;
        const target_shape = old.shape().drop(.layer);
        const layer = layer_index.broad(activation_index.shape());
        return old.scatterSlices(
            .{ .layer = layer, .a = activation_index },
            new.convert(old.dtype()).transpose(target_shape),
            .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
        ).reuseBuffer(old);
    }

    pub fn reuseBuffer(self: ActivationCache, other: ActivationCache) ActivationCache {
        @setEvalBranchQuota(10_000);
        var result = self;
        inline for (std.meta.fields(ActivationCache)) |field| {
            @field(result, field.name) = @field(self, field.name).reuseBuffer(@field(other, field.name));
        }
        return result;
    }
}; 
