const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");

const hz_type = .f32;
const cfg: f32 = 1.0;

const dialects = @import("mlir/dialects");


pub const AceLlm_handler = struct {
    model: AceLlm,
    params: Params,
    config: Config,
    options: Options,
    phase: Phase,
    tokenizer: zml.tokenizer.Tokenizer,
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
    prefill_exe_cfg: zml.Exe,
    decode_exe_cfg: zml.Exe,
    model_buffers: zml.Bufferized(AceLlm),
    kv_cache_buffers: zml.Bufferized(KvCache),
    // this models supports CFG generation: two kv caches are used for conditional and unconditional generation
    kv_cache_buffers_cfg: zml.Bufferized(KvCache),
    
    pub fn initFromFile(zml_handler: main.Zml_handler) !AceLlm_handler {
        const model_path = "//Users//sboulmier//zml//examples//acestep//models//acestep-5Hz-lm-0.6B//model.safetensors";
        const token_path = "//Users//sboulmier//zml//examples//acestep//models//acestep-5Hz-lm-0.6B//tokenizer.json";
        const config_path = "//Users//sboulmier//zml//examples//acestep//models//acestep-5Hz-lm-0.6B//config.json";
        var registry: zml.safetensors.TensorRegistry = try .fromPath(zml_handler.allocator, zml_handler.io, model_path);
        defer registry.deinit();
            
        //try main.printSafetensors(zml_handler.allocator, zml_handler.io, model_path);
    
        std.log.info("5Hz parse config and safetensors", .{});
        const parsed_config = try parseConfig(zml_handler, config_path);
        defer parsed_config.deinit();
        const config = try parsed_config.value.dupe(zml_handler.allocator);
        const acellm_options: Options = .{
            .seq_len = 256,
        };
        std.log.info("5Hz parsed", .{});
        
        std.log.info("5Hz initialize model", .{});
        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();
        const model: AceLlm = try .init(zml_handler.allocator, store.view().withPrefix("model"), config);
        std.log.info("5Hz initialized", .{});
    
        // Specify shapes of input arguments
        const dtype = model.embed_tokens.weight.dtype();
        const voc_size: usize = @intCast(model.embed_tokens.weight.dim(.voc));
        const params: Params = .{
            .prefill_tokens = .init(.{ .s = acellm_options.seq_len }, .u32),
            .decode_tokens = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{ .s = 1 }, .u32),
            .kv_cache = .init(zml.Shape.init(.{
                .layer = model.layers.len,
                .k = acellm_options.seq_len,
                .h = config.num_key_value_heads,
                .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
            }, dtype), zml_handler.io),
            .phase_mask = .init( .{ .voc = voc_size }, .f32),
            .rng = .init(),
            .shardings = try .init(zml_handler.platform),
        };
        
        const prefill_exe, const decode_exe = try compileModel(zml_handler, model, params);
        const tokenizer = try zml.tokenizer.Tokenizer.fromFile(zml_handler.allocator, zml_handler.io, token_path);
        const phase = try buildTokenMasks(zml_handler.allocator, tokenizer, voc_size);
        
        std.log.info("5Hz load buffers", .{});
        const model_buffers = try model.load(zml_handler, &store, &params.shardings.all());
        std.log.info("5Hz model loaded", .{});
        
        return .{
            .model = model,
            .params = params,
            .config = config,
            .options = acellm_options,
            .phase = phase,
            .tokenizer = tokenizer,
            .prefill_exe = prefill_exe,
            .decode_exe = decode_exe,
            .model_buffers = model_buffers,
            .kv_cache_buffers = try params.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, params.shardings.model),
            .kv_cache_buffers_cfg = try params.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, params.shardings.model),
        };
    }
    
    pub fn parseConfig(zml_handler: main.Zml_handler, path: []const u8) !std.json.Parsed(Config) {
        const parsed_config = blk: {
            const config_json_file = try std.Io.Dir.openFileAbsolute(zml_handler.io, path, .{});
            defer config_json_file.close(zml_handler.io);
            var config_json_buffer: [256]u8 = undefined;
            var config_reader = config_json_file.reader(zml_handler.io, &config_json_buffer);
            var reader = std.json.Reader.init(zml_handler.allocator, &config_reader.interface);
            defer reader.deinit();
            break :blk try std.json.parseFromTokenSource(Config, zml_handler.allocator, &reader, .{ .ignore_unknown_fields = true });
        };
        errdefer parsed_config.deinit();
        return parsed_config;
    }
    
    pub fn compileModel(zml_handler: main.Zml_handler, model: AceLlm, params: Params) !struct { zml.Exe, zml.Exe } {
        const shardings_arr = params.shardings.all();
        const opts: zml.module.CompilationOptions = .{
            .shardings = &shardings_arr,
        };
        // Compile the model twice, one for prefill, one for generation.
        std.log.info("5Hz compile prefill", .{});
        const prefill_exe = try zml_handler.platform.compile(zml_handler.allocator, zml_handler.io, model, .forward, .{ params.prefill_tokens, params.token_index, params.kv_cache, params.rng, params.phase_mask }, opts);
        std.log.info("5Hz compile decode", .{});
        const decode_exe = try zml_handler.platform.compile(zml_handler.allocator, zml_handler.io, model, .forward, .{ params.decode_tokens, params.token_index, params.kv_cache, params.rng, params.phase_mask }, opts);
        std.log.info("5Hz compiled models", .{});
        return .{ prefill_exe, decode_exe };
    }

    pub fn resetKvCache(self: *AceLlm_handler, zml_handler: main.Zml_handler) !void {
        KvCache.deinitBuffer(&self.kv_cache_buffers);
        self.kv_cache_buffers = try self.params.kv_cache.initBuffer(zml_handler.io, zml_handler.platform, self.params.shardings.model);
    }
    
    pub fn unloadBuffers(self: *AceLlm_handler) void {
        AceLlm.unloadBuffers(&self.model_buffers);
        KvCache.deinitBuffer(&self.kv_cache_buffers);
        KvCache.deinitBuffer(&self.kv_cache_buffers_cfg);
    }
    
    pub fn deinit(self: *AceLlm_handler, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.config.deinit(allocator);
        self.phase.deinit(allocator);
        self.tokenizer.deinit();
        self.prefill_exe.deinit();
        self.decode_exe.deinit();
    }
    
};

pub const AceCfg_handler = struct {
    model: AceLlm,
    params: Params,
    config: Config,
    options: Options,
    phase: Phase,
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
    model_buffers: zml.Bufferized(AceLlm),
    kv_cache_buffers: zml.Bufferized(KvCache),
    // this models supports CFG generation: two kv caches are used for conditional and unconditional generation
    kv_cache_buffers_cfg: zml.Bufferized(KvCache),
    
    pub fn initFromLlm(acellm: AceLlm_handler) !AceCfg_handler {
        
    }
    
   
    pub fn compileCfgModel(zml_handler: main.Zml_handler, model: AceLlm, params: Params) !struct { zml.Exe, zml.Exe } {
        const shardings_arr = params.shardings.all();
        const opts: zml.module.CompilationOptions = .{
            .shardings = &shardings_arr,
        };
        // Compile the model twice, one for prefill, one for generation.
        std.log.info("5Hz compile cfg prefill", .{});
        const prefill_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .forwardCfg,
            .{ params.prefill_tokens, params.token_index, params.kv_cache, params.rng, params.phase_mask },
            opts,
        );
        std.log.info("5Hz compile cfg decode", .{});
        const decode_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .forwardCfg,
            .{ params.decode_tokens, params.token_index, params.kv_cache, params.rng, params.phase_mask },
            opts,
        );
        std.log.info("5Hz compiled models", .{});
        return .{ prefill_exe, decode_exe };
    }

    
    pub fn unloadBuffers(self: *AceLlm_handler) void {
        AceLlm.unloadBuffers(&self.model_buffers);
        KvCache.deinitBuffer(&self.kv_cache_buffers);
        KvCache.deinitBuffer(&self.kv_cache_buffers_cfg);
    }
    
    pub fn deinit(self: *AceLlm_handler, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.config.deinit(allocator);
        self.phase.deinit(allocator);
        self.tokenizer.deinit();
        self.prefill_exe.deinit();
        self.decode_exe.deinit();
    }
};


pub const Params = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: KvCache,
    phase_mask: zml.Tensor,
    rng: zml.Tensor.Rng,
    shardings: main.Shardings,
    // cfg generation
};

pub const Config = struct {
    bos_token_id: u32,
    eos_token_id: stdx.json.Union(union(enum) {
        int: u32,
        ints: []u32,
    }),
    head_dim: ?u32 = null,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    rope_theta: f32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool = false,
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    // todo vocab_size: u32,

    pub fn dupe(self: Config, allocator: std.mem.Allocator) !Config {
        return .{
            .bos_token_id = self.bos_token_id,
            .eos_token_id = switch (self.eos_token_id.value) {
                .int => .{ .value = .{ .int = self.eos_token_id.value.int } },
                .ints => .{ .value = .{ .ints = try allocator.dupe(u32, self.eos_token_id.value.ints) } },
            },
            .head_dim = self.head_dim,
            .hidden_size = self.hidden_size,
            .num_hidden_layers = self.num_hidden_layers,
            .num_attention_heads = self.num_attention_heads,
            .num_key_value_heads = self.num_key_value_heads,
            .rope_theta = self.rope_theta,
            .max_position_embeddings = self.max_position_embeddings,
            .rms_norm_eps = self.rms_norm_eps,
            .tie_word_embeddings = self.tie_word_embeddings,
            .rope_scaling = self.rope_scaling,
        };
    }

    pub fn deinit(self: Config, allocator: std.mem.Allocator) void {
        switch (self.eos_token_id.value) {
            .int => {},
            .ints => allocator.free(self.eos_token_id.value.ints),
        }
    }
};

pub const Options = struct {
    seq_len: u32 = 256,
};

pub const Phase = struct {
    phase1_mask: []f32,
    phase2_mask: []f32,
    eos_id: u32,

    pub fn deinit(self: Phase, allocator: std.mem.Allocator) void {
        allocator.free(self.phase1_mask);
        allocator.free(self.phase2_mask);
    }
};


pub fn buildTokenMasks(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, voc_size: usize) !Phase {
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();
    var tokens_audio: std.ArrayList(f32) = try .initCapacity(allocator, voc_size);
    var tokens_text: std.ArrayList(f32) = try .initCapacity(allocator, voc_size);
    const token_slice = try allocator.alloc(u32, 1);
    defer allocator.free(token_slice);
    for (0..voc_size) |i| {
        token_slice[0] = @intCast(i);
        const chunk = try tokenizer_decoder.decode(token_slice);
        const l = chunk.len;
        if (l < 16) {
            try tokens_audio.append(allocator, -1e10);
            try tokens_text.append(allocator, 0.0);
        } else {
            if (std.mem.eql(u8, chunk[0..12], "<|audio_code")) {
                try tokens_audio.append(allocator, 0.0);
                try tokens_text.append(allocator, -1e10);
            } else {
                try tokens_audio.append(allocator, -1e10);
                try tokens_text.append(allocator, 0.0);
            }
        }
    }
    const eos_id = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
    // in audio tokens mode, the eos is constrained at pos 5 * duration
    tokens_audio.items[eos_id] = 1e-10;
    return .{
        .phase1_mask = try tokens_text.toOwnedSlice(allocator),
        .phase2_mask = try tokens_audio.toOwnedSlice(allocator),
        .eos_id = eos_id,
    };
}


pub const AceLlm = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,
 
    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !AceLlm {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("layers").withLayer(i), config);
        }
        return .{
            .embed_tokens = .{ .weight = store.createTensor(
                "embed_tokens.weight",
                .{ .voc, .d },
                .{ .voc = .replicated, .d = .model },
            ) },
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config),
        };
    }

    pub fn load(self: *const AceLlm, zml_handler: main.Zml_handler, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(AceLlm) {
        return zml.io.load(AceLlm, self, zml_handler.allocator, zml_handler.io, zml_handler.platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 128 * 1024 * 1024,
        });
    }

    pub fn deinit(self: *const AceLlm, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AceLlm)) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(self: AceLlm, tokens: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, rng: zml.Tensor.Rng, phase_mask: zml.Tensor) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        const logits, const updated_kv_cache = self.computeLogits(tokens, token_index, kv_cache, phase_mask);
        const next_tokens, const new_rng = sampleNucleus(logits, rng);
        return .{ next_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache.reuseBuffer(kv_cache), new_rng };
    }
    
    pub fn forwardCfg(self: AceLlm,
        tokens_cond: zml.Tensor, tokens_uncond: zml.Tensor,
        token_index_cond: zml.Tensor, token_index_uncond: zml.Tensor,
        kv_cache_cond: KvCache, kv_cache_uncond: KvCache, 
        rng: zml.Tensor.Rng,
        phase_mask: zml.Tensor
    ) struct { zml.Tensor, KvCache, KvCache, zml.Tensor.Rng } {
        const logits_cond, const updated_kv_cache_cond = self.computeLogits(tokens_cond, token_index_cond, kv_cache_cond, phase_mask);
        const logits_uncond, const updated_kv_cache_uncond = self.computeLogits(tokens_uncond, token_index_uncond, kv_cache_uncond, phase_mask);
        
        // todo : use a tensor passed as argument to extract the right slice of logits
        const next_logits_cond = logits_cond.choose1d(.s, tokens_cond.dim(.s) - 1);
        const next_logits_uncond = logits_uncond.choose1d(.s, tokens_uncond.dim(.s) - 1);
        
        const next_logits = next_logits_uncond.add((next_logits_cond.sub(next_logits_uncond)).scale(cfg));
        const next_tokens, const new_rng = sampleNucleus(next_logits, rng);
        
        return .{
            next_tokens.convert(tokens_cond.dtype()).reuseBuffer(tokens_cond),
            updated_kv_cache_cond.reuseBuffer(kv_cache_cond),
            updated_kv_cache_uncond.reuseBuffer(kv_cache_uncond),
            new_rng
        };
    }
    
    pub fn computeLogits(self: AceLlm, tokens: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, phase_mask: zml.Tensor) struct { zml.Tensor, KvCache } {
        const tok_id = token_index.squeeze(.s);
        var updated_kv_cache = kv_cache;
        // embbed input tokens
        var output = self.embed_tokens.convert(hz_type).forward(tokens.withPartialTags(.{ .s })).withPartialTags(.{ .d });
        // forward pass on the neural network layers
        for (self.layers, 0..) |layer, i| {
            output, updated_kv_cache = layer.forward(output, tok_id, updated_kv_cache.atLayer(i));
        }
        output = self.norm.forward(output);
        // compute logits for the output tokens
        const logits = self.embed_tokens.weight.withTags(.{ .voc, .d }).convert(hz_type).dot(output, .d);
        // apply phase mask to select tokens from the valid subset
        const masked_logits = logits.add(phase_mask.convert(hz_type).broad(logits.shape()));
        return .{ masked_logits.convert(.f32), updated_kv_cache.reuseBuffer(kv_cache) };
    }
    
    pub fn sampleNucleus(logits: zml.Tensor, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const top_p = 0.9;
        const temperature = 0.85;

        const sorted = logits.sort(.voc, .{ .descending = true });
        var sorted_logits = sorted.values;
        sorted_logits = sorted_logits.scale(1.0 / temperature);

        // softmax implementation is resilient to overflows
        const probs = sorted_logits.softmax(.voc);
        const cumulative_probs = probs.cumulativeSum(.voc);
        // shift cumulative sum to the right by 1, padding with 0 at the start
        // this allows the first token that makes the sum exceed top_p to be included
        // this also makes so that the most probable token is included even if its proba exceeds top_p
        const padding: zml.Tensor.Pad = .{ .low = 1, .high = 0, .interior = 0 };
        const size_slice_voc: zml.Tensor.Slice = .{ .start = 0, .end = logits.dim(.voc) };
        const size_slice_s: zml.Tensor.Slice = .{ .start = 0, .end = logits.dim(.s) };
        const shifted_cum_probs = cumulative_probs.pad(0, .{ .voc = padding }).slice(&.{ size_slice_voc, size_slice_s });

        const top_p_tensor = zml.Tensor.scalar(top_p, .f32).broad(probs.shape());
        const is_in_top_p = shifted_cum_probs.cmp(.LE, top_p_tensor);

        const zero = zml.Tensor.scalar(0.0, .f32).broad(probs.shape());
        const filtered_probs = is_in_top_p.select(probs, zero);
        const filtered_sum = filtered_probs.sum(.voc);
        const normalized_probs = filtered_probs.div(filtered_sum.broad(filtered_probs.shape()));

        const next_rng, const uniform_sample = rng.uniform(zml.Shape.init(.{}, .f32), .{});
        const sample_threshold = uniform_sample.broad(normalized_probs.shape());
        const normalized_cdf = normalized_probs.cumulativeSum(.voc);
        // we compare the cumulative probas to the threshold: we get a tensor [0 0 ... 0 1 1 ... 1]
        // and the sampled token is the position of the first 1. argMax returns the first max value
        const sampled_token = normalized_cdf.cmp(.GE, sample_threshold).argMax(.voc).indices;
        const next_token = sorted.indices.gather(.{ .voc = sampled_token.squeeze(.voc) }, .{});
        return .{ next_token, next_rng };
    }
};


const EmbedWrapper = struct {
    embed: zml.nn.TokenEmbedding,

    pub fn forward(wrapper: EmbedWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s }).squeeze(.b);
        const output = wrapper.embed.convert(hz_type).forward(tagged_input).withPartialTags(.{.d});
        return output.insertAxes(0, .{ .b });
    }
};

const TransformerLayer = struct {
    id: u8,
    input_norm: RmsNorm,
    att_layer: AttLayer,
    post_att_norm: RmsNorm,
    mlp_layer: MlpLayer,

    pub fn init(id_: u8, store: zml.io.TensorStore.View, config: Config) !TransformerLayer {
        return .{
            .id = id_,
            .input_norm = .init(store.withPrefix("input_layernorm"), config),
            .att_layer = try .init(store.withPrefix("self_attn"), config),
            .post_att_norm = .init(store.withPrefix("post_attention_layernorm"), config),
            .mlp_layer = try .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_norm);
        AttLayer.unloadBuffers(&self.att_layer);
        RmsNorm.unloadBuffers(&self.post_att_norm);
        MlpLayer.unloadBuffers(&self.mlp_layer);
    }

    pub fn forward(self: TransformerLayer, x0: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache) struct { zml.Tensor, KvCache } {
        std.log.debug("FWD : transformer layer {d}", .{self.id});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_norm.forward(x0_replicated);
        const delta0, const updated_kv_cache = self.att_layer.forward(
            x0_normalized,
            token_index,
            kv_cache,
        );

        // Fully Connected
        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });
        const x1_normalized = self.post_att_norm.forward(x1);
        const x2 = self.mlp_layer.forward(x1_normalized)
            .withPartitioning(.{ .d = .replicated })
            .add(x1)
            .withPartitioning(.{ .d = .replicated });

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

pub const TransWrapper = struct {
    trans: TransformerLayer,
    tok_id: zml.Tensor,
    kv_cache: KvCache,

    pub fn forward(wrapper: TransWrapper, input: zml.Tensor, att_mask: zml.Tensor, pos_id: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        _ = att_mask;
        _ = pos_id;
        _ = cos;
        _ = sin;
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output, _ = wrapper.trans.forward(tagged_input, wrapper.tok_id, wrapper.kv_cache.atLayer(0));
        return output.insertAxes(0, .{ .b });
    }
};

const AttLayer = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !AttLayer {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, .{ .d = .model }), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), config),
            .k_norm = .init(store.withPrefix("k_norm"), config),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = .sequential,
                .scaling = rope_scaling,
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AttLayer)) void {
        self.q_proj.weight.deinit();
        self.k_proj.weight.deinit();
        self.v_proj.weight.deinit();
        self.o_proj.weight.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    /// Self zml.attention.attention.
    ///   - If token_index is set, x is assumed to be the representation of one new token,
    /// and kv_cache will be read for the previous tokens.
    ///   - If token_index is not set, x is assumed to be the representation of all tokens
    /// since the beginning of the sequence, and kv_cache won't be read.
    /// In both case, kv_cache will be updated with the computed key and value.
    /// x: {.b, .s, .d } -> .{.b, .s, .d}
    pub fn forward(
        self: AttLayer,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        // Make hidden state replicated once and reuse it across q/k/v projections.
        // This avoids paying gather-style collectives independently for each projection.
        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q = self.q_proj.convert(hz_type).forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.convert(hz_type).forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.convert(hz_type).forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });
        q = q.withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);
        k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const attn_output = zml.attention.attention.attention(
            q,
            k,
            v,
            token_index,
            .vanilla,
            .vanilla,
        ).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.convert(hz_type).forward(attn)
            .rename(.{ .d_out = .d })
            .withPartitioning(.{ .d = .replicated });
        return .{ delta, new_kv_cache };
    }
};

const AttWrapper = struct {
    att: AttLayer,
    tok_id: zml.Tensor,
    kv_cache: KvCache,

    pub fn forward(wrapper: AttWrapper, input: zml.Tensor, att_mask: zml.Tensor, pos_id: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        _ = att_mask;
        _ = pos_id;
        _ = cos;
        _ = sin;
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output, _ = wrapper.att.forward(tagged_input, wrapper.tok_id, wrapper.kv_cache);
        return output.insertAxes(0, .{ .b });
    }
};

const MlpLayer = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) !MlpLayer {
        return .{
            .up_proj = store.createTensor("up_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }),
            .gate_proj = store.createTensor("gate_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }),
            .down_proj = store.createTensor("down_proj.weight", .{ .d, .d_out }, .{ .d = .model }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MlpLayer)) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: MlpLayer, input: zml.Tensor) zml.Tensor {
        std.log.debug("FWD : mlp", .{});
        const up_projection = input.dot(self.up_proj.convert(hz_type), .d);
        const gate_projection = input.dot(self.gate_proj.convert(hz_type), .d);
        const activation = gate_projection.silu().mul(up_projection);
        const output = activation.dot(self.down_proj.convert(hz_type), .d_out);
        return output;
    }
};

const MlpWrapper = struct {
    mlp: MlpLayer,

    pub fn forward(wrapper: MlpWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = wrapper.mlp.forward(tagged_input);
        return output.insertAxes(0, .{ .b });
    }
};

const TensorWrapper = struct {
    tensor: zml.Tensor,

    pub fn forward(wrapper: TensorWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = tagged_input.dot(wrapper.tensor.convert(hz_type), .d);
        return output.insertAxes(0, .{ .b });
    }
};

const TensorWrapperI = struct {
    tensor: zml.Tensor,

    pub fn forward(wrapper: TensorWrapperI, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d_out }).squeeze(.b);
        const output = tagged_input.dot(wrapper.tensor.convert(hz_type), .d_out);
        return output.insertAxes(0, .{ .b });
    }
};

const RmsNorm = struct {
    weights: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) RmsNorm {
        return .{
            .weights = store.createTensor("weight", .{.d_out }, null),
            .eps = config.rms_norm_eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weights.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        //std.log.debug("FWD : norm", .{});
        // normalize along d axis each one of the s input tokens
        const normalized = zml.nn.rmsNorm(input, .d, self.eps);
        // scale with learned weights by broadening weights to match s input tokens
        return normalized.mul(self.weights.convert(hz_type).withTags(.{.d}).broad(input.shape()));
    }
};

pub const RmsWrapper = struct {
    norm: RmsNorm,

    pub fn forward(wrapper: RmsWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = wrapper.norm.forward(tagged_input);
        return output.insertAxes(0, .{ .b });
    }
};

pub const RmshWrapper = struct {
    norm: RmsNorm,

    pub fn forward(wrapper: RmshWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .h, .d }).squeeze(.b);
        const output = wrapper.norm.forward(tagged_input);
        return output.insertAxes(0, .{ .b });
    }
};

pub const ProjWrapper = struct {
    proj: zml.nn.Linear,

    pub fn forward(wrapper: ProjWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = wrapper.proj.convert(hz_type).forward(tagged_input);
        return output.insertAxes(0, .{ .b });
    }
};

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: zml.Tensor,
    io: std.Io,

    pub fn init(kv_shape: zml.Shape, io: std.Io) KvCache {
        const sharded_shape = kv_shape.withPartitioning(.{ .h = .model });

        return .{
            .k = .fromShape(sharded_shape),
            .v = .fromShape(sharded_shape),
            .layer_index = .init(.{}, .u32),
            .io = io,
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), sharding, .{}),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32, sharding),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
        self.layer_index.deinit();
    }

    pub fn keys(self: KvCache) zml.Tensor {
        return self.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache) zml.Tensor {
        return self.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer);
        var layer = self.layer_index;
        layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;

        return if (token_index) |idx|
            .{
                .k = self.k.scatterSlices(.{ .layer = layer, .k = idx }, new_k.convert(self.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
                .v = self.v.scatterSlices(.{ .layer = layer, .k = idx }, new_v.convert(self.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
                .layer_index = self.layer_index,
                .io = self.io,
            }
        else
            .{
                .k = self.k.scatterSlices(.{ .layer = layer }, new_k.convert(self.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
                .v = self.v.scatterSlices(.{ .layer = layer }, new_v.convert(self.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
                .layer_index = self.layer_index,
                .io = self.io,
            };
    }

    pub fn atLayer(self: KvCache, layer_index: usize) KvCache {
        return .{
            .k = self.k,
            .v = self.v,
            .layer_index = zml.Tensor.scalar(layer_index, .u32),
            .io = self.io,
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
            .layer_index = self.layer_index.reuseBuffer(other.layer_index),
            .io = self.io,
        };
    }
};


pub fn testModel(zml_handler: main.Zml_handler, acellm: AceLlm_handler) !void {
    const allocator = zml_handler.allocator;
    const io = zml_handler.io;
    const platform = zml_handler.platform;
    const parameters = acellm.params;
    const acellm_model = acellm.model;
    const acellm_buffers = acellm.model_buffers;
    const kv_cache_buffers = acellm.kv_cache_buffers;

    const activations_path = "//Users//sboulmier//zml//examples//acestep//models//acestep-5Hz-lm-0.6B//activations.safetensors";

    try main.printSafetensors(zml_handler.allocator, zml_handler.io, activations_path);
    
    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(zml_handler.allocator, zml_handler.io, activations_path);
    defer activations_registry.deinit();

    var activations_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry);
    defer activations_store.deinit();
    var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, parameters.shardings.model);
    defer prefill_token_pos_buffer.deinit();
    const tok_id = zml.Tensor.fromShape(.init(.{}, .u32));

    std.log.info("Test activations : embed layer", .{});
    const wrapper_embed: EmbedWrapper = .{
        .embed = acellm_model.embed_tokens,
    };
    const wrapper_buffers_embed: zml.Bufferized(EmbedWrapper) = .{
        .embed = acellm_buffers.embed_tokens,
    };
    const layer_embed = "model.model.embed_tokens";
    try zml.testing.testLayer(allocator, io, platform, wrapper_embed, .forward, activations_store.view(), layer_embed, wrapper_buffers_embed, &parameters.shardings.all(), .{});

    std.log.info("Test activations : rms layer input norm", .{});
    const wrapper_rms: RmsWrapper = .{
        .norm = acellm_model.layers[0].input_norm,
    };
    const wrapper_buffers_rms: zml.Bufferized(RmsWrapper) = .{
        .norm = acellm_buffers.layers[0].input_norm,
    };
    const layer_rms = "model.model.layers.0.input_layernorm";
    try zml.testing.testLayer(allocator, io, platform, wrapper_rms, .forward, activations_store.view(), layer_rms, wrapper_buffers_rms, &parameters.shardings.all(), .{});

    std.log.info("Test activations : q_proj", .{});
    const wrapper_q_proj: ProjWrapper = .{
        .proj = acellm_model.layers[0].att_layer.q_proj,
    };
    const wrapper_buffers_q_proj: zml.Bufferized(ProjWrapper) = .{
        .proj = acellm_buffers.layers[0].att_layer.q_proj,
    };
    const layer_q_proj = "model.model.layers.0.self_attn.q_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_q_proj, .forward, activations_store.view(), layer_q_proj, wrapper_buffers_q_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : k_proj", .{});
    const wrapper_k_proj: ProjWrapper = .{
        .proj = acellm_model.layers[0].att_layer.k_proj,
    };
    const wrapper_buffers_k_proj: zml.Bufferized(ProjWrapper) = .{
        .proj = acellm_buffers.layers[0].att_layer.k_proj,
    };
    const layer_k_proj = "model.model.layers.0.self_attn.k_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_k_proj, .forward, activations_store.view(), layer_k_proj, wrapper_buffers_k_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : v_proj", .{});
    const wrapper_v_proj: ProjWrapper = .{
        .proj = acellm_model.layers[0].att_layer.v_proj,
    };
    const wrapper_buffers_v_proj: zml.Bufferized(ProjWrapper) = .{
        .proj = acellm_buffers.layers[0].att_layer.v_proj,
    };
    const layer_v_proj = "model.model.layers.0.self_attn.v_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_v_proj, .forward, activations_store.view(), layer_v_proj, wrapper_buffers_v_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : o_proj", .{});
    const wrapper_o_proj: ProjWrapper = .{
        .proj = acellm_model.layers[0].att_layer.o_proj,
    };
    const wrapper_buffers_o_proj: zml.Bufferized(ProjWrapper) = .{
        .proj = acellm_buffers.layers[0].att_layer.o_proj,
    };
    const layer_o_proj = "model.model.layers.0.self_attn.o_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_o_proj, .forward, activations_store.view(), layer_o_proj, wrapper_buffers_o_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : k_norm", .{});
    const wrapper_k_norm: RmshWrapper = .{
        .norm = acellm_model.layers[0].att_layer.k_norm,
    };
    const wrapper_buffers_k_norm: zml.Bufferized(RmshWrapper) = .{
        .norm = acellm_buffers.layers[0].att_layer.k_norm,
    };
    const layer_k_norm = "model.model.layers.0.self_attn.k_norm";
    try zml.testing.testLayer(allocator, io, platform, wrapper_k_norm, .forward, activations_store.view(), layer_k_norm, wrapper_buffers_k_norm, &parameters.shardings.all(), .{});

    std.log.info("Test activations : q_norm", .{});
    const wrapper_q_norm: RmshWrapper = .{
        .norm = acellm_model.layers[0].att_layer.q_norm,
    };
    const wrapper_buffers_q_norm: zml.Bufferized(RmshWrapper) = .{
        .norm = acellm_buffers.layers[0].att_layer.q_norm,
    };
    const layer_q_norm = "model.model.layers.0.self_attn.q_norm";
    try zml.testing.testLayer(allocator, io, platform, wrapper_q_norm, .forward, activations_store.view(), layer_q_norm, wrapper_buffers_q_norm, &parameters.shardings.all(), .{});

    std.log.info("Test activations : self attention", .{});
    const wrapper_att: AttWrapper = .{
        .att = acellm_model.layers[0].att_layer,
        .tok_id = tok_id,
        .kv_cache = parameters.kv_cache,
    };
    const wrapper_buffers_att: zml.Bufferized(AttWrapper) = .{
        .att = acellm_buffers.layers[0].att_layer,
        .tok_id = prefill_token_pos_buffer,
        .kv_cache = kv_cache_buffers,
    };
    const layer_att = "model.model.layers.0.self_attn";
    try zml.testing.testLayer(allocator, io, platform, wrapper_att, .forward, activations_store.view(), layer_att, wrapper_buffers_att, &parameters.shardings.all(), .{});

    std.log.info("Test activations : up_proj", .{});
    const wrapper_up_proj: TensorWrapper = .{
        .tensor = acellm_model.layers[0].mlp_layer.up_proj,
    };
    const wrapper_buffers_up_proj: zml.Bufferized(TensorWrapper) = .{
        .tensor = acellm_buffers.layers[0].mlp_layer.up_proj,
    };
    const layer_up_proj = "model.model.layers.0.mlp.up_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_up_proj, .forward, activations_store.view(), layer_up_proj, wrapper_buffers_up_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : gate_proj", .{});
    const wrapper_gate_proj: TensorWrapper = .{
        .tensor = acellm_model.layers[0].mlp_layer.gate_proj,
    };
    const wrapper_buffers_gate_proj: zml.Bufferized(TensorWrapper) = .{
        .tensor = acellm_buffers.layers[0].mlp_layer.gate_proj,
    };
    const layer_gate_proj = "model.model.layers.0.mlp.gate_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_gate_proj, .forward, activations_store.view(), layer_gate_proj, wrapper_buffers_gate_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : down_proj", .{});
    const wrapper_down_proj: TensorWrapperI = .{
        .tensor = acellm_model.layers[0].mlp_layer.down_proj,
    };
    const wrapper_buffers_down_proj: zml.Bufferized(TensorWrapperI) = .{
        .tensor = acellm_buffers.layers[0].mlp_layer.down_proj,
    };
    const layer_down_proj = "model.model.layers.0.mlp.down_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_down_proj, .forward, activations_store.view(), layer_down_proj, wrapper_buffers_down_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : mlp", .{});
    const wrapper_mlp: MlpWrapper = .{
        .mlp = acellm_model.layers[0].mlp_layer,
    };
    const wrapper_buffers_mlp: zml.Bufferized(MlpWrapper) = .{
        .mlp = acellm_buffers.layers[0].mlp_layer,
    };
    const layer_mlp = "model.model.layers.0.mlp";
    try zml.testing.testLayer(allocator, io, platform, wrapper_mlp, .forward, activations_store.view(), layer_mlp, wrapper_buffers_mlp, &parameters.shardings.all(), .{});

    std.log.info("Test activations : transformer layer", .{});
    const wrapper: TransWrapper = .{
        .trans = acellm_model.layers[0],
        .tok_id = tok_id,
        .kv_cache = parameters.kv_cache,
    };
    const wrapper_buffers: zml.Bufferized(TransWrapper) = .{
        .trans = acellm_buffers.layers[0],
        .tok_id = prefill_token_pos_buffer,
        .kv_cache = kv_cache_buffers,
    };
    const layer = "model.model.layers.0";
    try zml.testing.testLayer(allocator, io, platform, wrapper, .forward, activations_store.view(), layer, wrapper_buffers, &parameters.shardings.all(), .{});
}

pub fn print(x: zml.Tensor, n: u8) void {
    const name = &.{ n };
    const sh = x._shape;
    const nb_dims = sh.rank();
    switch (nb_dims) {
        0 => std.log.info("{s} is scalar", .{ name }),
        1 => std.log.info("{s} is dim 1 : {s} = {d}", .{ name, sh.tag(0), sh.dim(0) }),
        2 => std.log.info("{s} is dim 2 : {s} x {s} = {d} x {d}", .{ name, sh.tag(0), sh.tag(1), sh.dim(0), sh.dim(1) }),
        3 => std.log.info("{s} is dim 3 : {s} x {s} x {s} = {d} x {d} x {d}", .{ name, sh.tag(0), sh.tag(1), sh.tag(2), sh.dim(0), sh.dim(1), sh.dim(2) }),
        4 => std.log.info("{s} is dim 4 : {s} x {s} x {s} x {s} = {d} x {d} x {d} x {d}", .{ name, sh.tag(0), sh.tag(1), sh.tag(2), sh.tag(3), sh.dim(0), sh.dim(1), sh.dim(2), sh.dim(3) }),
        else => std.log.info("{s} is rank >= 5", .{ name }),
    }
}