const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const Qwen2TokenizerFast = @import("tokenization_qwen2_fast.zig").Qwen2TokenizerFast;
const tools = @import("tools.zig");
const utils = @import("utils.zig");

const log = std.log.scoped(.modeling_qwen3);

pub const Config = struct {
    vocab_size: i64 = 152064,
    hidden_size: i64 = 2048,
    intermediate_size: i64 = 11008,
    num_hidden_layers: i64 = 32,
    num_attention_heads: i64 = 16,
    num_key_value_heads: i64 = 2,
    max_position_embeddings: i64 = 32768,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-6,
    use_cache: bool = true,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 1000000.0,
    attention_dropout: f32 = 0.0,
    attention_bias: bool = true,
};

fn unloadWeights(allocator: std.mem.Allocator, weights: anytype) void {
    const T = @TypeOf(weights.*);
    const type_info = @typeInfo(T);
    switch (type_info) {
        .@"struct" => |info| {
            if (T == zml.Buffer) {
                weights.deinit();
                return;
            }
            inline for (info.fields) |field| {
                unloadWeights(allocator, &@field(weights, field.name));
            }
        },
        .optional => {
            if (weights.*) |*w| {
                unloadWeights(allocator, w);
            }
        },
        .pointer => |info| {
            if (info.size == .slice) {
                for (weights.*) |*item| {
                    unloadWeights(allocator, item);
                }
                allocator.free(weights.*);
            }
        },
        else => {},
    }
}

pub const Linear = struct {
    inner: zml.nn.Linear,

    pub fn init(weight: zml.Tensor, bias: ?zml.Tensor) Linear {
        const w = weight.withTags(.{ .out, .d });
        return .{
            .inner = zml.nn.Linear.init(w, bias, .out),
        };
    }

    pub fn forward(self: Linear, x: zml.Tensor) zml.Tensor {
        const input_dtype = x.dtype();
        const x_f32 = x.convert(.f32).withPartialTags(.{.d});
        const weight_f32 = self.inner.weight.convert(.f32);

        var y = x_f32.dot(weight_f32, .d);
        if (self.inner.bias) |bias| {
            y = y.add(bias.convert(.f32).broad(y.shape()));
        }

        const out = y.rename(.{ .out = .d });
        return out.convert(input_dtype);
    }
};

pub const Qwen3RMSNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) Qwen3RMSNorm {
        return .{
            .weight = store.createTensor("weight"),
            .eps = eps,
        };
    }

    pub fn forward(self: Qwen3RMSNorm, hidden_states: zml.Tensor) zml.Tensor {
        // Match Python: internal computation in f32
        // Assumes hidden_states is already f32 for efficiency, but convert() is safe anyway
        const hs_f32 = hidden_states.convert(.f32);
        const out_f32 = zml.nn.rmsNorm(hs_f32, -1, self.eps);
        const weight_f32 = self.weight.convert(.f32).broadcastLeft(out_f32.shape());
        return out_f32.mul(weight_f32);
    }
};

pub const Qwen3MLP = struct {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,

    pub fn init(store: zml.io.TensorStore.View) Qwen3MLP {
        return .{
            .gate_proj = Linear.init(store.createTensor("gate_proj.weight"), null),
            .up_proj = Linear.init(store.createTensor("up_proj.weight"), null),
            .down_proj = Linear.init(store.createTensor("down_proj.weight"), null),
        };
    }

    pub fn forward(self: Qwen3MLP, x: zml.Tensor) zml.Tensor {
        // Input and Output maintained in f32 for high precision chain
        const gate = self.gate_proj.forward(x);
        const up = self.up_proj.forward(x);
        const act = gate.silu();
        return self.down_proj.forward(act.mul(up));
    }
};

pub const Qwen3Attention = struct {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen3RMSNorm,
    k_norm: Qwen3RMSNorm,

    head_dim: i64,
    config: Config,
    layer_idx: usize,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_idx: usize) Qwen3Attention {
        const bias_q = if (config.attention_bias) store.maybeCreateTensor("q_proj.bias") else null;
        const bias_k = if (config.attention_bias) store.maybeCreateTensor("k_proj.bias") else null;
        const bias_v = if (config.attention_bias) store.maybeCreateTensor("v_proj.bias") else null;
        const bias_o = if (config.attention_bias) store.maybeCreateTensor("o_proj.bias") else null;

        const q_proj = Linear.init(store.createTensor("q_proj.weight"), bias_q);

        // Dynamic head_dim detection:
        // q_proj weight is [out, in]. We tagged it {.out, .d}.
        // The underlying tensor dimensions are still 0 and 1.
        // We can access inner.weight.shape().dim(0) for output features.
        const head_dim = @divExact(q_proj.inner.weight.shape().dim(0), config.num_attention_heads);

        return .{
            .q_proj = q_proj,
            .k_proj = Linear.init(store.createTensor("k_proj.weight"), bias_k),
            .v_proj = Linear.init(store.createTensor("v_proj.weight"), bias_v),
            .o_proj = Linear.init(store.createTensor("o_proj.weight"), bias_o),
            .q_norm = Qwen3RMSNorm.init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = Qwen3RMSNorm.init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .head_dim = head_dim,
            .config = config,
            .layer_idx = layer_idx,
        };
    }

    pub fn forward(
        self: Qwen3Attention,
        hidden_states: zml.Tensor,
        rope: utils.RoPE,
        attention_mask: ?zml.Tensor,
    ) zml.Tensor {
        const b = hidden_states.shape().dim(0);
        const s = hidden_states.shape().dim(1);
        const head_dim = self.head_dim;

        const q_tmp = self.q_proj.forward(hidden_states);
        const k_tmp = self.k_proj.forward(hidden_states);
        const v_tmp = self.v_proj.forward(hidden_states);

        const q_shape = zml.Shape.init(.{ b, s, self.config.num_attention_heads, head_dim }, hidden_states.dtype());
        const k_shape = zml.Shape.init(.{ b, s, self.config.num_key_value_heads, head_dim }, hidden_states.dtype());
        const v_shape = zml.Shape.init(.{ b, s, self.config.num_key_value_heads, head_dim }, hidden_states.dtype());

        var query_states = q_tmp.reshape(q_shape);
        var key_states = k_tmp.reshape(k_shape);
        const value_states_reshaped = v_tmp.reshape(v_shape);

        query_states = self.q_norm.forward(query_states);
        key_states = self.k_norm.forward(key_states);

        query_states = query_states.transpose(.{ 0, 2, 1, 3 });
        key_states = key_states.transpose(.{ 0, 2, 1, 3 });
        var value_states = value_states_reshaped.transpose(.{ 0, 2, 1, 3 });

        // RoPE slicing
        const cos = rope.cos;
        const sin = rope.sin;

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin);

        const n_rep = @divExact(self.config.num_attention_heads, self.config.num_key_value_heads);
        if (n_rep > 1) {
            key_states = repeat_kv(key_states, n_rep);
            value_states = repeat_kv(value_states, n_rep);
        }

        const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(head_dim)));

        var attn_weights = zml.Tensor.dotGeneral(
            query_states.convert(.f32),
            key_states.convert(.f32),
            &.{.{ 3, 3 }},
            &.{ .{ 0, 0 }, .{ 1, 1 } },
        );
        attn_weights = attn_weights.scale(scale);

        // Causal Masking: mask out future tokens (j > i)
        // Create [s, 1] and [1, s] index tensors, broadcast to [s, s], compare
        const i_idx = zml.Tensor.arange(.{ .end = s }, .i32).reshape(.{ s, 1 });
        const j_idx = zml.Tensor.arange(.{ .end = s }, .i32).reshape(.{ 1, s });
        // Broadcast to [s, s]
        const i_broad = i_idx.broadcastLeft(zml.Shape.init(.{ s, s }, .i32));
        const j_broad = j_idx.broadcastLeft(zml.Shape.init(.{ s, s }, .i32));
        const causal_mask = j_broad.cmp(.GT, i_broad); // bool [s, s], true where should be masked
        // Expand to [1, 1, s, s] for broadcasting with [b, h, s, s]
        const causal_4d = causal_mask.reshape(zml.Shape.init(.{ 1, 1, s, s }, .bool));
        const causal_broad = causal_4d.broadcastLeft(attn_weights.shape().withDtype(.bool));
        // Use very large negative value matching Python's torch.finfo(dtype).min behavior
        const neg_inf = zml.Tensor.scalar(-3.4e38, .f32);
        attn_weights = causal_broad.select(neg_inf.broadcastLeft(attn_weights.shape()), attn_weights);

        if (attention_mask) |mask| {
            // mask is [b, s]. 1=valid, 0=pad.
            // Expand to [b, 1, 1, s]
            const mask_i32 = mask.convert(.i32);
            const b_dim = mask_i32.shape().dim(0);
            const s_dim = mask_i32.shape().dim(1);
            const mask_4d = mask_i32.reshape(zml.Shape.init(.{ b_dim, 1, 1, s_dim }, .i32));
            // Create boolean mask: pad positions (mask==0) should be masked
            const pad_mask = mask_4d.cmp(.EQ, zml.Tensor.scalar(0, .i32));
            // Broadcast to [b, h, s, s]
            const pad_mask_broad = pad_mask.broadcastLeft(attn_weights.shape().withDtype(.bool));
            attn_weights = pad_mask_broad.select(neg_inf.broadcastLeft(attn_weights.shape()), attn_weights);
        }

        // Match Python: compute softmax in f32
        const input_dtype = hidden_states.dtype();
        attn_weights = attn_weights.softmax(-1);

        var attn_output = zml.Tensor.dotGeneral(
            attn_weights,
            value_states.convert(.f32),
            &.{.{ 3, 2 }},
            &.{ .{ 0, 0 }, .{ 1, 1 } },
        );

        attn_output = attn_output.transpose(.{ 0, 2, 1, 3 });

        const concat_dim = self.config.num_attention_heads * self.head_dim;
        attn_output = attn_output.reshape(zml.Shape.init(.{ b, s, concat_dim }, .f32));

        const out = self.o_proj.forward(attn_output);
        return out.convert(input_dtype);
    }
};

pub const Qwen3DecoderLayer = struct {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: Qwen3RMSNorm,
    post_attention_layernorm: Qwen3RMSNorm,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_idx: usize) Qwen3DecoderLayer {
        return .{
            .self_attn = Qwen3Attention.init(store.withPrefix("self_attn"), config, layer_idx),
            .mlp = Qwen3MLP.init(store.withPrefix("mlp")),
            .input_layernorm = Qwen3RMSNorm.init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .post_attention_layernorm = Qwen3RMSNorm.init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
        };
    }

    pub fn forward(
        self: Qwen3DecoderLayer,
        hidden_states: zml.Tensor, // input is f32
        rope: utils.RoPE,
        attention_mask: ?zml.Tensor,
    ) zml.Tensor { // returns f32
        var h = self.input_layernorm.forward(hidden_states);
        h = self.self_attn.forward(h, rope, attention_mask);
        h = hidden_states.add(h);

        const res2 = h;
        h = self.post_attention_layernorm.forward(h);
        h = self.mlp.forward(h);
        return res2.add(h);
    }
};

pub const Qwen3Model = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    // Use Allocator slice instead of BoundedArray to match zml.io.load path
    layers: []Qwen3DecoderLayer,
    norm: Qwen3RMSNorm,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Qwen3Model {
        const layers = try allocator.alloc(Qwen3DecoderLayer, @intCast(config.num_hidden_layers));

        for (0..@intCast(config.num_hidden_layers)) |i| {
            var buf: [32]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&buf, "layers.{}", .{i});
            layers[i] = Qwen3DecoderLayer.init(store.withPrefix(prefix), config, i);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight") },
            .layers = layers,
            .norm = Qwen3RMSNorm.init(store.withPrefix("norm"), config.rms_norm_eps),
            .config = config,
        };
    }

    pub fn deinit(self: *Qwen3Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn forward(
        self: Qwen3Model,
        input_ids: zml.Tensor,
        rope: utils.RoPE,
        attention_mask: ?zml.Tensor,
        output_hidden_states: bool,
    ) struct { last_hidden_state: zml.Tensor, hidden_states: ?stdx.BoundedArray(zml.Tensor, 64) } {
        // Initial conversion to f32. All activations stay f32 until the very end.
        var hidden_states = self.embed_tokens.forward(input_ids).convert(.f32);

        var all_hidden_states: ?stdx.BoundedArray(zml.Tensor, 64) = null;
        if (output_hidden_states) {
            all_hidden_states = stdx.BoundedArray(zml.Tensor, 64).init(0) catch unreachable;
        }

        if (output_hidden_states) {
            all_hidden_states.?.append(hidden_states) catch {}; // Index 0: embeddings (f32)
        }

        for (self.layers) |layer| {
            hidden_states = layer.forward(hidden_states, rope, attention_mask);
            if (output_hidden_states) {
                all_hidden_states.?.append(hidden_states) catch {}; // Index 1..36 (f32)
            }
        }

        hidden_states = self.norm.forward(hidden_states);
        if (output_hidden_states) {
            all_hidden_states.?.append(hidden_states) catch {}; // Index 37: final norm (f32)
        }

        return .{ .last_hidden_state = hidden_states, .hidden_states = all_hidden_states };
    }
};

pub const Qwen3ForCausalLM = struct {
    model: Qwen3Model,
    lm_head: Linear,
    config: Config,
    data_type: zml.DataType,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, data_type: zml.DataType) !Qwen3ForCausalLM {
        const model = try Qwen3Model.init(allocator, store.withPrefix("model"), config);

        var lm_head_weight = store.maybeCreateTensor("lm_head.weight");
        if (lm_head_weight == null) {
            if (config.tie_word_embeddings) {
                lm_head_weight = model.embed_tokens.weight;
            } else {
                return error.MissingWeights;
            }
        }

        return .{
            .model = model,
            .lm_head = Linear.init(lm_head_weight.?, null),
            .config = config,
            .data_type = data_type,
        };
    }

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
    }

    pub fn forward(
        self: @This(),
        input_ids: zml.Tensor,
        rope: utils.RoPE,
        attention_mask: ?zml.Tensor,
        output_hidden_states: bool,
    ) struct { logits: zml.Tensor, hidden_states: ?stdx.BoundedArray(zml.Tensor, 64) } {
        const out = self.model.forward(input_ids, rope, attention_mask, output_hidden_states);
        const logits = out.last_hidden_state;
        return .{ .logits = logits, .hidden_states = out.hidden_states };
    }

    pub const ModelContext = struct {
        model: Qwen3ForCausalLM,
        store: zml.io.TensorStore,
        registry: zml.safetensors.TensorRegistry,
        config: Config,
        weights: zml.Bufferized(Qwen3ForCausalLM),

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            unloadWeights(allocator, &self.weights);
            self.model.deinit(allocator);
            self.store.deinit();
            self.registry.deinit();
        }
    };

    pub fn loadFromFile(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, repo_dir: std.Io.Dir, parallelism_level: usize, progress: ?*std.Progress.Node, data_type: zml.DataType) !ModelContext {
        @setEvalBranchQuota(10_000);
        const subfolder = "text_encoder";

        const config_json = try tools.parseConfig(Config, allocator, io, repo_dir, .{ .subfolder = subfolder, .json_name = "config.json" });
        // log.info("Loaded config type: {s} from {s} with values {any}", .{ @typeName(Config), subfolder, config_json });
        errdefer config_json.deinit();
        const config = config_json.value;

        // std.log.info("Loaded Qwen3 Config: rms_norm_eps {any}", .{config_json});

        var tensor_registry = try zml.safetensors.TensorRegistry.fromRepo(allocator, io, try repo_dir.openDir(io, subfolder, .{}));
        defer tensor_registry.deinit();

        var tensor_store = zml.io.TensorStore.fromRegistry(allocator, &tensor_registry);
        errdefer tensor_store.deinit();
        var model = try Qwen3ForCausalLM.init(allocator, tensor_store.view(), config, data_type);
        errdefer model.deinit(allocator);

        // log.info("Hydrating Qwen3 Weights...", .{});
        var weights = try zml.io.load(
            Qwen3ForCausalLM,
            &model,
            allocator,
            io,
            platform,
            .{ .parallelism = parallelism_level, .store = &tensor_store, .dma_chunks = 4, .dma_chunk_size = 64 * 1024 * 1024, .progress = progress },
        );
        errdefer unloadWeights(allocator, &weights);
        return .{
            .model = model,
            .store = tensor_store,
            .registry = tensor_registry,
            .config = config,
            .weights = weights,
        };
    }

    const QwenEncodingStep = struct {
        pub fn forward(
            self: @This(),
            model: Qwen3ForCausalLM,
            input_ids: zml.Tensor,
            attention_mask: zml.Tensor,
            rope: utils.RoPE,
        ) zml.Tensor {
            _ = self;
            const out = model.forward(input_ids.convert(.i32), rope, attention_mask.convert(.f32), true);
            const h9 = out.hidden_states.?.get(9);
            const h18 = out.hidden_states.?.get(18);
            const h27 = out.hidden_states.?.get(27);
            return zml.Tensor.concatenate(&.{ h9, h18, h27 }, -1);
        }
    };

    pub const EmbedingOutput = struct {
        text_ids: zml.Buffer,
        text_embedding: zml.Buffer,
        pub fn deinit(self: *@This()) void {
            self.text_ids.deinit();
            self.text_embedding.deinit();
        }
    };

    pub fn pipelineRun(allocator: std.mem.Allocator, io: std.Io, repo_dir: std.Io.Dir, platform: *const zml.Platform, progress: ?*std.Progress.Node, parallelism_level: usize, data_type: zml.DataType, tokens: Qwen2TokenizerFast.TokenizeOutput) !EmbedingOutput {
        var qwen3_model_ctx = try Qwen3ForCausalLM.loadFromFile(allocator, io, platform, repo_dir, parallelism_level, progress, data_type);

        // Prepare RoPE
        const seq_len: usize = @intCast(tokens.input_ids.shape().dim(1));
        const head_dim: usize = @intCast(qwen3_model_ctx.model.model.layers[0].self_attn.head_dim);

        var rope = try utils.computeRoPE(allocator, io, platform, seq_len, head_dim, qwen3_model_ctx.config.rope_theta, data_type);
        defer rope.deinit();

        const input_shape = tokens.input_ids.shape();
        var qwen_exe = try platform.compile(allocator, io, QwenEncodingStep{}, .forward, .{
            qwen3_model_ctx.model,
            zml.Tensor.fromShape(input_shape),
            zml.Tensor.fromShape(input_shape),
            rope.forCompile(),
        });
        defer qwen_exe.deinit();

        var qwen_args = try qwen_exe.args(allocator);
        defer qwen_args.deinit(allocator);
        qwen_args.set(.{ qwen3_model_ctx.weights, tokens.input_ids, tokens.attention_mask, rope.forRuntime() });

        var qwen_res = try qwen_exe.results(allocator);
        defer qwen_res.deinit(allocator);
        qwen_exe.call(qwen_args, &qwen_res);

        var prompt_embeds = qwen_res.get(zml.Buffer);
        errdefer prompt_embeds.deinit();

        const prompt_seq_len: usize = @intCast(prompt_embeds.shape().dim(1));

        return .{ .text_ids = try utils.prepare_text_ids(allocator, io, platform, prompt_seq_len), .text_embedding = prompt_embeds };
    }
};

fn rotate_half(x: zml.Tensor) zml.Tensor {
    const last_dim = x.rank() - 1;
    const dim_val = x.shape().dim(last_dim);
    const half = @divExact(dim_val, 2);
    const chunks = x.split(last_dim, &.{ half, half });
    return zml.Tensor.concatenate(&.{ chunks[1].scale(-1.0), chunks[0] }, -1);
}

fn apply_rotary_pos_emb(q: zml.Tensor, k: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
    const cos_casted = cos.convert(q.dtype());
    const sin_casted = sin.convert(q.dtype());
    const q_embed = q.mul(cos_casted).add(rotate_half(q).mul(sin_casted));
    const k_embed = k.mul(cos_casted).add(rotate_half(k).mul(sin_casted));
    return .{ q_embed, k_embed };
}

fn repeat_kv(hidden_states: zml.Tensor, n_rep: i64) zml.Tensor {
    if (n_rep == 1) return hidden_states;
    const b = hidden_states.shape().dim(0);
    const h_kv = hidden_states.shape().dim(1);
    const s = hidden_states.shape().dim(2);
    const d = hidden_states.shape().dim(3);
    const expanded = hidden_states.reshape(zml.Shape.init(.{ b, h_kv, 1, s, d }, hidden_states.dtype()));
    const broadcasted = expanded.broad(zml.Shape.init(.{ b, h_kv, n_rep, s, d }, hidden_states.dtype()));
    return broadcasted.reshape(zml.Shape.init(.{ b, h_kv * n_rep, s, d }, hidden_states.dtype()));
}
