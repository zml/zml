const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

pub const Config = struct {
    vocab_size: i64 = 152064,
    hidden_size: i64 = 2048,
    intermediate_size: i64 = 11008,
    num_hidden_layers: i64 = 32, // Default Qwen2.5-3B, check 0.5B config if different
    num_attention_heads: i64 = 16,
    num_key_value_heads: i64 = 2,
    hidden_act: []const u8 = "silu",
    max_position_embeddings: i64 = 32768,
    initializer_range: f32 = 0.02,
    rms_norm_eps: f32 = 1e-6,
    use_cache: bool = true,
    tie_word_embeddings: bool = false,
    rope_theta: f32 = 1000000.0,
    rope_scaling: ?struct {
        type: []const u8 = "default",
    } = null,
    attention_dropout: f32 = 0.0,
    attention_bias: bool = true, // Qwen2 default is True
};

pub const RoPE = struct {
    cos: zml.Tensor,
    sin: zml.Tensor,
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
        return zml.nn.rmsNorm(hidden_states, -1, self.eps).mul(self.weight);
    }
};

pub const Qwen3MLP = struct {
    gate_proj: zml.nn.Linear,
    up_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Qwen3MLP {
        return .{
            .gate_proj = zml.nn.Linear.init(store.createTensor("gate_proj.weight"), null, .d),
            .up_proj = zml.nn.Linear.init(store.createTensor("up_proj.weight"), null, .d),
            .down_proj = zml.nn.Linear.init(store.createTensor("down_proj.weight"), null, .d),
        };
    }

    pub fn forward(self: Qwen3MLP, x: zml.Tensor) zml.Tensor {
        const gate = self.gate_proj.forward(x);
        const up = self.up_proj.forward(x);
        const act = gate.silu();
        return self.down_proj.forward(act.mul(up));
    }
};

pub const Qwen3Attention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: Qwen3RMSNorm,
    k_norm: Qwen3RMSNorm,

    config: Config,
    layer_idx: usize,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_idx: usize) Qwen3Attention {
        const head_dim = @divExact(config.hidden_size, config.num_attention_heads);
        _ = head_dim; // autofix
        const bias_q = if (config.attention_bias) store.maybeCreateTensor("q_proj.bias") else null;
        const bias_k = if (config.attention_bias) store.maybeCreateTensor("k_proj.bias") else null;
        const bias_v = if (config.attention_bias) store.maybeCreateTensor("v_proj.bias") else null;
        const bias_o = if (config.attention_bias) store.maybeCreateTensor("o_proj.bias") else null;

        return .{
            .q_proj = zml.nn.Linear.init(store.createTensor("q_proj.weight"), bias_q, .d),
            .k_proj = zml.nn.Linear.init(store.createTensor("k_proj.weight"), bias_k, .d),
            .v_proj = zml.nn.Linear.init(store.createTensor("v_proj.weight"), bias_v, .d),
            .o_proj = zml.nn.Linear.init(store.createTensor("o_proj.weight"), bias_o, .d),
            .q_norm = Qwen3RMSNorm.init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = Qwen3RMSNorm.init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .config = config,
            .layer_idx = layer_idx,
        };
    }

    pub fn forward(
        self: Qwen3Attention,
        hidden_states: zml.Tensor,
        rope: RoPE,
        attention_mask: ?zml.Tensor,
    ) zml.Tensor {
        const b = hidden_states.shape().dim(0);
        const s = hidden_states.shape().dim(1);
        const head_dim = @divExact(self.config.hidden_size, self.config.num_attention_heads);

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
        const cos = rope.cos.slice1d(2, .{ .start = 0, .end = s });
        const sin = rope.sin.slice1d(2, .{ .start = 0, .end = s });

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin);

        const n_rep = @divExact(self.config.num_attention_heads, self.config.num_key_value_heads);
        if (n_rep > 1) {
            key_states = repeat_kv(key_states, n_rep);
            value_states = repeat_kv(value_states, n_rep);
        }

        const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(head_dim)));

        var attn_weights = zml.Tensor.dotGeneral(
            query_states,
            key_states,
            &.{.{ 3, 3 }},
            &.{ .{ 0, 0 }, .{ 1, 1 } },
        );
        attn_weights = attn_weights.scale(scale);

        if (attention_mask) |mask| {
            attn_weights = attn_weights.add(mask);
        }

        attn_weights = attn_weights.softmax(-1);

        var attn_output = zml.Tensor.dotGeneral(
            attn_weights,
            value_states,
            &.{.{ 3, 2 }},
            &.{ .{ 0, 0 }, .{ 1, 1 } },
        );

        attn_output = attn_output.transpose(.{ 0, 2, 1, 3 });

        attn_output = attn_output.reshape(zml.Shape.init(.{ b, s, self.config.hidden_size }, hidden_states.dtype()));

        return self.o_proj.forward(attn_output);
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
        hidden_states: zml.Tensor,
        rope: RoPE,
        attention_mask: ?zml.Tensor,
    ) zml.Tensor {
        var residual = hidden_states;
        var h = self.input_layernorm.forward(hidden_states);
        h = self.self_attn.forward(h, rope, attention_mask);
        h = residual.add(h);

        residual = h;
        h = self.post_attention_layernorm.forward(h);
        h = self.mlp.forward(h);
        h = residual.add(h);

        return h;
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

    pub fn forward(
        self: Qwen3Model,
        input_ids: zml.Tensor,
        rope: RoPE,
        attention_mask: ?zml.Tensor,
        output_hidden_states: bool,
    ) struct { last_hidden_state: zml.Tensor, hidden_states: ?stdx.BoundedArray(zml.Tensor, 64) } {
        var hidden_states = self.embed_tokens.forward(input_ids);

        var all_hidden_states: ?stdx.BoundedArray(zml.Tensor, 64) = null;
        if (output_hidden_states) {
            all_hidden_states = stdx.BoundedArray(zml.Tensor, 64).init(0) catch unreachable;
        }

        if (output_hidden_states) {
            all_hidden_states.?.append(hidden_states) catch {};
        }

        for (self.layers) |layer| {
            hidden_states = layer.forward(hidden_states, rope, attention_mask);
            if (output_hidden_states) {
                all_hidden_states.?.append(hidden_states) catch {};
            }
        }

        hidden_states = self.norm.forward(hidden_states);
        if (output_hidden_states) {
            all_hidden_states.?.append(hidden_states) catch {};
        }

        return .{ .last_hidden_state = hidden_states, .hidden_states = all_hidden_states };
    }
};

pub const Qwen3ForCausalLM = struct {
    model: Qwen3Model,
    lm_head: zml.nn.Linear,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Qwen3ForCausalLM {
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
            .lm_head = zml.nn.Linear.init(lm_head_weight.?, null, .d),
            .config = config,
        };
    }

    pub fn forward(
        self: Qwen3ForCausalLM,
        input_ids: zml.Tensor,
        rope: RoPE,
        attention_mask: ?zml.Tensor,
        output_hidden_states: bool,
    ) struct { logits: zml.Tensor, hidden_states: ?stdx.BoundedArray(zml.Tensor, 64) } {
        const out = self.model.forward(input_ids, rope, attention_mask, output_hidden_states);
        const logits = self.lm_head.forward(out.last_hidden_state);
        return .{ .logits = logits, .hidden_states = out.hidden_states };
    }

    pub const ModelContext = struct {
        model: Qwen3ForCausalLM,
        store: zml.io.TensorStore,
        registry: zml.safetensors.TensorRegistry,
        config: Config,
        weights: zml.Bufferized(Qwen3ForCausalLM),

        pub fn deinit(self: *ModelContext, allocator: std.mem.Allocator) void {
            unloadWeights(allocator, &self.weights);
            allocator.free(self.model.model.layers);
            self.store.deinit();
            self.registry.deinit();
        }
    };

    pub fn loadFromFile(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model_path: []const u8) !ModelContext {
        @setEvalBranchQuota(10000);
        std.log.info("Loading Qwen3 from: {s}/text_encoder", .{model_path});

        const tr_dir = try std.fs.path.join(allocator, &.{ model_path, "text_encoder" });
        defer allocator.free(tr_dir);

        const tr_repo = try zml.safetensors.resolveModelRepo(io, tr_dir);

        var config = Config{};
        if (tr_repo.readFileAlloc(io, "config.json", allocator, .limited(1024 * 1024))) |bytes| {
            defer allocator.free(bytes);
            var config_parsed = try std.json.parseFromSlice(Config, allocator, bytes, .{ .ignore_unknown_fields = true });
            defer config_parsed.deinit();
            config = config_parsed.value;
            std.log.info("Loaded Qwen3 Config: {any}", .{config});
        } else |err| {
            std.log.warn("Failed to load config.json (err={}), using default Qwen3 Config", .{err});
        }

        var tr_registry = blk: {
            const index_path = try std.fs.path.join(allocator, &.{ tr_dir, "model.safetensors.index.json" });
            defer allocator.free(index_path);

            // Try to load index
            if (zml.safetensors.TensorRegistry.fromPath(allocator, io, index_path)) |reg| {
                std.log.info("Loaded Qwen3 Weights from Index: {s}", .{index_path});
                break :blk reg;
            } else |_| {
                // Fallback to single file
                const model_path_sf = try std.fs.path.join(allocator, &.{ tr_dir, "model.safetensors" });
                defer allocator.free(model_path_sf);
                std.log.info("Loading Qwen3 Weights from: {s}", .{model_path_sf});
                break :blk try zml.safetensors.TensorRegistry.fromPath(allocator, io, model_path_sf);
            }
        };

        var tr_store = zml.io.TensorStore.fromRegistry(allocator, &tr_registry);

        const model = try Qwen3ForCausalLM.init(allocator, tr_store.view(), config);

        std.log.info("Hydrating Qwen3 Weights...", .{});
        const weights = try zml.io.load(
            Qwen3ForCausalLM,
            &model,
            allocator,
            io,
            platform,
            .{ .parallelism = 1, .store = &tr_store, .dma_chunks = 4, .dma_chunk_size = 64 * 1024 * 1024 },
        );

        return .{
            .model = model,
            .store = tr_store,
            .registry = tr_registry,
            .config = config,
            .weights = weights,
        };
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
    const q_embed = q.mul(cos).add(rotate_half(q).mul(sin));
    const k_embed = k.mul(cos).add(rotate_half(k).mul(sin));
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

pub fn computeRoPE(seq_len: i64, head_dim: i64, theta: f32) RoPE {
    const half = @divExact(head_dim, 2);
    const shape_half = zml.Shape.init(.{half}, .f32);
    const idx_half = zml.Tensor.iota(shape_half, 0).convert(.f32);
    const log_v = @log(theta);
    const factor: f32 = -log_v / @as(f32, @floatFromInt(half));
    const exponents = idx_half.scale(@as(f64, @floatCast(factor)));
    const inv_freq = exponents.exp();
    const shape_seq = zml.Shape.init(.{seq_len}, .f32);
    const t = zml.Tensor.iota(shape_seq, 0).convert(.f32);

    // Explicit broadcast for outer product
    const shape_full = zml.Shape.init(.{ seq_len, half }, .f32);

    const t_reshaped = t.reshape(zml.Shape.init(.{ seq_len, 1 }, .f32));
    const inv_reshaped = inv_freq.reshape(zml.Shape.init(.{ 1, half }, .f32));

    // Use broad to broadcast to shape
    const t_broad = t_reshaped.broad(shape_full);
    const inv_broad = inv_reshaped.broad(shape_full);

    const freqs = t_broad.mul(inv_broad);

    const emb = zml.Tensor.concatenate(&.{ freqs, freqs }, -1);
    const cos = emb.cos().reshape(zml.Shape.init(.{ 1, 1, seq_len, head_dim }, .f32));
    const sin = emb.sin().reshape(zml.Shape.init(.{ 1, 1, seq_len, head_dim }, .f32));
    return .{ .cos = cos, .sin = sin };
}
